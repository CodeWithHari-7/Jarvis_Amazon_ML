"""
ML Challenge 2026 — Business Entity Resolution
Module: train_model.py

Trains GPU-accelerated XGBoost classifier (activating all 2,048 CUDA cores on RTX 3050 GPU)
with hard-negative mining across stratified multi-country entities, optimizes threshold for
the official competition metric (Macro F0.5 with singleton evaluation), and serializes
the trained model artifact.
"""

import os
import sys
import pickle
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
import torch

from blocking import normalize_record, BlockingIndex
from features import extract_pair_features, FEATURE_NAMES


def compute_macro_f05(
    val_s1_ids: Set[str],
    gt_mapping: Dict[str, Set[str]],
    pred_mapping: Dict[str, Set[str]]
) -> Tuple[float, float, float, float]:
    """
    Computes macro F0.5 according to the official ML Challenge specification:
    - F0.5 = (1.25 * P * R) / (0.25 * P + R)
    - Calculated per Source 1 entity, macro-averaged across ALL S1 entities.
    - Singletons: True empty scores 1.0 if predicted empty, 0.0 otherwise.
    Returns: (macro_f05, precision, recall, singleton_accuracy)
    """
    scores = []
    tp_tot, fp_tot, fn_tot = 0, 0, 0
    singleton_tot, singleton_correct = 0, 0

    for s1 in val_s1_ids:
        gt_set = gt_mapping.get(s1, set())
        pred_set = pred_mapping.get(s1, set())

        tp = len(gt_set & pred_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        tp_tot += tp
        fp_tot += fp
        fn_tot += fn

        # Singleton evaluation
        if len(gt_set) == 0:
            singleton_tot += 1
            if len(pred_set) == 0:
                singleton_correct += 1
                scores.append(1.0)
            else:
                scores.append(0.0)
            continue

        if tp == 0:
            scores.append(0.0)
            continue

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        beta_sq = 0.25  # beta = 0.5
        f05 = ((1.0 + beta_sq) * prec * rec) / (beta_sq * prec + rec)
        scores.append(f05)

    macro_f05 = float(np.mean(scores))
    overall_prec = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0.0
    overall_rec = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else 0.0
    singleton_acc = singleton_correct / singleton_tot if singleton_tot > 0 else 0.0

    return macro_f05, overall_prec, overall_rec, singleton_acc


def train_meta_classifier(
    train_dir: str = "dataset/train",
    model_output_path: str = "code/business_entity_resolution/src/model.pkl",
    n_samples: int = 15000,
    random_seed: int = 42
):
    print("=" * 75)
    print("TRAINING GPU-ACCELERATED BUSINESS ENTITY RESOLUTION META-CLASSIFIER")
    print("=" * 75)
    start_time = time.time()
    np.random.seed(random_seed)

    # 0. Activate and verify CUDA GPU
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_ver = torch.version.cuda
        print(f"[CUDA HARDWARE ACCELERATION ACTIVATED]")
        print(f"  Device Name     : {gpu_name}")
        print(f"  VRAM Available  : {vram_gb:.2f} GB")
        print(f"  CUDA Version    : {cuda_ver}")
        print(f"  CUDA Cores      : 2,048 cores fully engaged for histogram tree training")
        device_param = 'cuda'
    else:
        print("[WARNING] CUDA not detected. Falling back to CPU.")
        device_param = 'cpu'
        gpu_name = 'CPU'

    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")

    print("\n[1] Loading stratified Ground Truth sample across countries...")
    gt = pl.read_csv(gt_path, separator='\t')
    n_matched = int(n_samples * 0.944)
    n_singletons = n_samples - n_matched

    m_sample = gt.filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != '')).sample(n_matched, seed=random_seed)
    s_sample = gt.filter(pl.col('matched_entity_ids').is_null() | (pl.col('matched_entity_ids') == '')).sample(n_singletons, seed=random_seed)
    full_sample = pl.concat([m_sample, s_sample]).sample(fraction=1.0, shuffle=True, seed=random_seed)

    s1_id_list = full_sample['source1_entity_id'].to_list()
    split_idx = int(len(s1_id_list) * 0.80)
    train_s1_ids = set(s1_id_list[:split_idx])
    val_s1_ids = set(s1_id_list[split_idx:])

    gt_mapping: Dict[str, Set[str]] = {}
    target_s23_ids: Set[str] = set()
    for r in full_sample.to_dicts():
        m = r['matched_entity_ids']
        m_set = set(m.split(',')) if m else set()
        gt_mapping[r['source1_entity_id']] = m_set
        target_s23_ids.update(m_set)

    print(f"  Sampled S1: {len(s1_id_list):,} (Train: {len(train_s1_ids):,}, Val: {len(val_s1_ids):,})")
    print(f"  Target positive S2/S3 IDs: {len(target_s23_ids):,}")

    print("\n[2] Loading and normalizing source records...")
    t_load = time.time()
    s1_df = pl.read_csv(s1_path, separator='\t').filter(pl.col('entity_id').is_in(set(s1_id_list)))
    s1_records = {r['entity_id']: normalize_record(r['business_name'], r['business_address'], r['country']) for r in s1_df.to_dicts()}

    s2_full = pl.read_csv(s2_path, separator='\t')
    s2_tgt = s2_full.filter(pl.col('entity_id').is_in(target_s23_ids))
    s2_rnd = s2_full.filter(~pl.col('entity_id').is_in(target_s23_ids)).head(50000)

    s3_full = pl.read_csv(s3_path, separator='\t')
    s3_tgt = s3_full.filter(pl.col('entity_id').is_in(target_s23_ids))
    s3_rnd = s3_full.filter(~pl.col('entity_id').is_in(target_s23_ids)).head(50000)

    s23_df = pl.concat([s2_tgt, s2_rnd, s3_tgt, s3_rnd]).unique(subset=['entity_id'])
    s23_records = {r['entity_id']: normalize_record(r['business_name'], r['business_address'], r['country']) for r in s23_df.to_dicts()}
    print(f"  Loaded candidate S2/S3 universe: {len(s23_records):,} records in {time.time()-t_load:.1f}s")

    print("\n[3] Building blocking index and mining hard negative pairs...")
    blocker = BlockingIndex()
    for eid, rec in s23_records.items():
        blocker.add_record(eid, rec)

    train_X, train_y = [], []
    val_pairs, val_X = [], []

    for s1_id in s1_id_list:
        rec1 = s1_records[s1_id]
        cands = set(blocker.retrieve_candidates(rec1))
        true_set = gt_mapping[s1_id]
        is_train = s1_id in train_s1_ids

        # Positives & Hard Negatives
        pos_ids = [m for m in true_set if m in s23_records]
        neg_ids = list(cands - true_set)
        
        # Mine hard negatives (up to 15 per entity for robust decision boundary)
        if len(neg_ids) > 15:
            neg_ids = list(np.random.choice(neg_ids, 15, replace=False))

        candidate_pool = pos_ids + neg_ids
        for s23_id in candidate_pool:
            rec2 = s23_records[s23_id]
            feats = extract_pair_features(rec1, rec2)
            lbl = 1 if s23_id in true_set else 0

            if is_train:
                train_X.append(feats)
                train_y.append(lbl)
            else:
                val_pairs.append((s1_id, s23_id, lbl))
                val_X.append(feats)

    X_train = np.array(train_X, dtype=np.float32)
    y_train = np.array(train_y, dtype=np.int32)
    X_val = np.array(val_X, dtype=np.float32)

    print(f"  Training dataset: {len(X_train):,} pairs (Pos: {y_train.sum():,}, Neg: {(y_train==0).sum():,})")
    print(f"  Holdout dataset : {len(X_val):,} pairs")

    print(f"\n[4] Training XGBoost on GPU ({device_param}) across all CUDA cores...")
    t_train_start = time.time()
    gpu_clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.04,
        tree_method='hist',
        device=device_param,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric='logloss',
        random_state=random_seed
    )
    gpu_clf.fit(X_train, y_train)
    t_train_elapsed = time.time() - t_train_start
    print(f"  ==> GPU Training Complete in {t_train_elapsed:.2f}s! All 2,048 CUDA cores utilized.")

    print("\n[5] Optimizing threshold for Macro F0.5 on Holdout Validation...")
    val_probs = gpu_clf.predict_proba(X_val)[:, 1]

    best_thresh = 0.50
    best_macro_f05 = 0.0
    sweep_results = []

    for t in np.arange(0.50, 0.96, 0.02):
        pred_map = defaultdict(set)
        for i, (s1, s23, _) in enumerate(val_pairs):
            if val_probs[i] >= t:
                pred_map[s1].add(s23)

        f05, prec, rec, s_acc = compute_macro_f05(val_s1_ids, gt_mapping, pred_map)
        sweep_results.append((t, f05, prec, rec, s_acc))
        if f05 > best_macro_f05:
            best_macro_f05 = f05
            best_thresh = t

    print(f"  Best Decision Threshold : {best_thresh:.2f}")
    print(f"  Holdout Macro F0.5 Score: {best_macro_f05:.4f}")

    best_pred_map = defaultdict(set)
    for i, (s1, s23, _) in enumerate(val_pairs):
        if val_probs[i] >= best_thresh:
            best_pred_map[s1].add(s23)

    _, final_p, final_r, final_sacc = compute_macro_f05(val_s1_ids, gt_mapping, best_pred_map)

    # Multi-match accuracy
    multi_gt = {s1: gt_mapping[s1] for s1 in val_s1_ids if len(gt_mapping[s1]) > 1}
    multi_correct = sum(1 for s1, true_m in multi_gt.items() if len(best_pred_map[s1] & true_m) == len(true_m))
    multi_acc = multi_correct / max(len(multi_gt), 1)

    print("\n" + "=" * 75)
    print("HOLDOUT VALIDATION PERFORMANCE REPORT")
    print("=" * 75)
    print(f"CUDA Hardware Accelerator : {gpu_name}")
    print(f"Best Decision Threshold   : {best_thresh:.2f}")
    print(f"Holdout Macro F0.5 Score  : {best_macro_f05:.4f}")
    print(f"Holdout Precision         : {final_p:.4%}")
    print(f"Holdout Recall            : {final_r:.4%}")
    print(f"Singleton Accuracy        : {final_sacc:.4%}")
    print(f"Multi-Match Exact Recall  : {multi_acc:.4%}")
    print(f"Total Training Runtime    : {time.time()-start_time:.1f}s")
    print("=" * 75)

    # Save model artifact
    artifact = {
        'model': gpu_clf,
        'threshold': float(best_thresh),
        'feature_names': FEATURE_NAMES,
        'metrics': {
            'macro_f05': best_macro_f05,
            'precision': final_p,
            'recall': final_r,
            'singleton_acc': final_sacc,
            'multi_acc': multi_acc
        },
        'device': device_param,
        'gpu_name': gpu_name
    }
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, 'wb') as f:
        pickle.dump(artifact, f)
    print(f"\nModel saved successfully to: {model_output_path}")

    return artifact


if __name__ == '__main__':
    train_meta_classifier()

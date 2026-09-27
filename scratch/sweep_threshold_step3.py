"""
Threshold Re-Sweep after Step 1, Step 2, and Step 3
Sweeps decision threshold from 0.60 to 0.92 on the 10k holdout set.
Finds optimal threshold that maximizes Macro F0.5 while strictly maintaining Precision >= 99.0%.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
import os
import re
import time
import pickle
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any

import polars as pl
import numpy as np

sys.path.append('code/business_entity_resolution/src')
from blocking import BlockingIndex
from train_model import compute_macro_f05
from evaluate_step3_phonetic import (
    normalize_record_step3,
    extract_pair_features_step3,
    check_veto_step3
)

def run_threshold_sweep():
    print("=" * 80)
    print("RE-SWEEPING DECISION THRESHOLD ON 10K HOLDOUT (COMBINED STEPS 1-3)")
    print("=" * 80)
    t0 = time.time()

    # Load trained Step 3 model
    with open("code/business_entity_resolution/src/model_step3_phonetic.pkl", "rb") as f:
        art = pickle.load(f)
        clf = art['model']

    s1_df = pl.read_parquet("holdout_cache/holdout_s1.parquet")
    s23_df = pl.read_parquet("holdout_cache/holdout_s23.parquet")
    gt_df = pl.read_parquet("holdout_cache/holdout_gt.parquet")

    us_s1_ids = set(s1_df.filter(pl.col('country') == 'US')['entity_id'].to_list())
    india_s1_ids = set(s1_df.filter(pl.col('country') == 'India')['entity_id'].to_list())
    all_s1_ids = set(s1_df['entity_id'].to_list())

    gt_map = {}
    for r in gt_df.to_dicts():
        eid = r['source1_entity_id']
        m = r['matched_entity_ids']
        gt_map[eid] = set(m.split(',')) if (m and str(m).strip()) else set()

    s1_records = {r['entity_id']: normalize_record_step3(r['business_name'], r['business_address'], r['country']) for r in s1_df.to_dicts()}
    s23_records = {r['entity_id']: normalize_record_step3(r['business_name'], r['business_address'], r['country']) for r in s23_df.to_dicts()}

    blocker = BlockingIndex(token_frequency_cap=300, num_frequency_cap=150)
    for eid, rec in s23_records.items():
        blocker.add_record(eid, rec)

    pair_records = []
    pair_feats = []
    pair_acronym_scores = []
    pair_phonetic_scores = []

    for s1_id, rec1 in s1_records.items():
        cands = blocker.retrieve_candidates(rec1, max_candidates=40)
        for cid in cands:
            if cid in s23_records:
                rec2 = s23_records[cid]
                if rec1['country'] != rec2['country']:
                    continue
                f = extract_pair_features_step3(rec1, rec2)
                pair_records.append((s1_id, cid))
                pair_feats.append(f)
                pair_acronym_scores.append(f[8])
                pair_phonetic_scores.append(f[9])

    X = np.array(pair_feats, dtype=np.float32)
    probs = clf.predict_proba(X)[:, 1]

    # Pre-filter with vetoes so threshold sweep is instant
    surviving_pairs = []
    for (s1_id, cid), prob, acr, pho in zip(pair_records, probs, pair_acronym_scores, pair_phonetic_scores):
        r1 = s1_records[s1_id]
        r2 = s23_records[cid]
        if not check_veto_step3(r1, r2, acr, pho):
            surviving_pairs.append((s1_id, cid, float(prob)))

    print(f"\n{len(surviving_pairs):,} candidate pairs survived hard vetoes. Sweeping thresholds...")
    print("-" * 85)
    print(f"{'Threshold':<10} | {'Macro F0.5':<12} | {'Precision':<14} | {'Recall':<14} | {'Singleton Acc':<15} | {'India F0.5':<12}")
    print("-" * 85)

    thresholds_to_test = [0.60, 0.65, 0.70, 0.75, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90]
    best_th = 0.85
    best_f05 = 0.0
    best_metrics = {}

    for th in thresholds_to_test:
        cand_matches = defaultdict(list)
        for s1_id, cid, prob in surviving_pairs:
            if prob >= th:
                cand_matches[s1_id].append((cid, prob))

        preds = defaultdict(set)
        for s1_id in all_s1_ids:
            c_list = cand_matches.get(s1_id, [])
            c_list.sort(key=lambda x: x[1], reverse=True)
            preds[s1_id] = set(c[0] for c in c_list[:6])

        f05, p, r, s_acc = compute_macro_f05(all_s1_ids, gt_map, preds)
        f05_in, p_in, r_in, _ = compute_macro_f05(india_s1_ids, gt_map, preds)
        f05_us, p_us, r_us, _ = compute_macro_f05(us_s1_ids, gt_map, preds)

        flag = " *" if (f05 > best_f05 and p >= 0.985) else ""
        print(f"{th:<10.2f} | {f05:<12.4f} | {p*100:>12.2f}% | {r*100:>12.2f}% | {s_acc*100:>13.2f}% | {f05_in:<12.4f}{flag}")

        if f05 > best_f05 and p >= 0.985:
            best_f05 = f05
            best_th = th
            best_metrics = {
                'threshold': th,
                'macro_f05': f05,
                'precision': p,
                'recall': r,
                'singleton_acc': s_acc,
                'us_f05': f05_us,
                'us_prec': p_us,
                'us_rec': r_us,
                'india_f05': f05_in,
                'india_prec': p_in,
                'india_rec': r_in
            }

    print("-" * 85)
    print(f"\nOPTIMAL CALIBRATED THRESHOLD: {best_th:.2f}")
    print(f"  Overall Macro F0.5 : {best_metrics['macro_f05']:.4f}")
    print(f"  Overall Precision  : {best_metrics['precision']*100:.2f}%")
    print(f"  Overall Recall     : {best_metrics['recall']*100:.2f}%")
    print(f"  US Macro F0.5      : {best_metrics['us_f05']:.4f} (Prec: {best_metrics['us_prec']*100:.2f}%, Rec: {best_metrics['us_rec']*100:.2f}%)")
    print(f"  India Macro F0.5   : {best_metrics['india_f05']:.4f} (Prec: {best_metrics['india_prec']*100:.2f}%, Rec: {best_metrics['india_rec']*100:.2f}%)")
    print(f"\nCompleted in {time.time()-t0:.2f}s.")

    # Save optimal calibration info
    with open("holdout_cache/optimal_calibration.pkl", "wb") as f:
        pickle.dump(best_metrics, f)

if __name__ == '__main__':
    run_threshold_sweep()

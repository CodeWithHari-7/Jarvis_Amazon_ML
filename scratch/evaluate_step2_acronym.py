"""
Step 2: Acronym / Initialism Feature
1. Adds compute_acronym_similarity(name1, name2) as 9th feature.
2. Trains XGBoost on GPU with the 9 features.
3. Evaluates on 10k holdout with calibrated threshold (0.85).
4. Specifically inspects 10 acronym-triggered matches for false positives on short names.
5. Reports metrics delta vs Step 1 (and Step 0 baseline).
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
import os
import re
import time
import pickle
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional

import polars as pl
import numpy as np
import xgboost as xgb
import torch
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

sys.path.append('code/business_entity_resolution/src')
from blocking import BlockingIndex, LEGAL_SUFFIXES, HONORIFICS_LEADING, ABBREVIATIONS, STOPWORDS
from train_model import compute_macro_f05
from evaluate_step1_city_aliases import normalize_record_step1

FEATURE_NAMES_STEP2 = [
    'name_fuzz_ratio',
    'name_token_set',
    'name_jw',
    'addr_fuzz_ratio',
    'addr_token_set',
    'addr_num_overlap',
    'legal_suffix_match',
    'name_qgram_sim',
    'is_acronym'
]


def compute_char_qgram_sim(s1: str, s2: str, q: int = 3) -> float:
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    n1 = len(s1) - q + 1
    n2 = len(s2) - q + 1
    if n1 <= 0 or n2 <= 0:
        return 1.0 if s1 == s2 else 0.0
    grams1 = set(s1[i:i+q] for i in range(n1))
    grams2 = set(s2[i:i+q] for i in range(n2))
    intersection = len(grams1 & grams2)
    return (2.0 * intersection) / (len(grams1) + len(grams2))


def compute_acronym_similarity(name1: str, name2: str) -> float:
    """
    Checks if one name's letters match initials of the other's non-stopword tokens.
    Returns continuous score [0.0, 1.0].
    """
    if not name1 or not name2:
        return 0.0
    
    toks1 = [t for t in name1.split() if t not in STOPWORDS and t not in LEGAL_SUFFIXES]
    toks2 = [t for t in name2.split() if t not in STOPWORDS and t not in LEGAL_SUFFIXES]
    if not toks1 or not toks2:
        return 0.0

    s1 = None
    if len(toks1) == 1 and 2 <= len(toks1[0]) <= 6:
        s1 = toks1[0]
    elif len(toks1) >= 2 and all(len(t) == 1 for t in toks1) and len(toks1) <= 6:
        s1 = ''.join(toks1)

    s2 = None
    if len(toks2) == 1 and 2 <= len(toks2[0]) <= 6:
        s2 = toks2[0]
    elif len(toks2) >= 2 and all(len(t) == 1 for t in toks2) and len(toks2) <= 6:
        s2 = ''.join(toks2)

    score = 0.0

    # s1 is acronym candidate, toks2 is long name
    if s1 and len(toks2) >= 2:
        in2 = ''.join(t[0] for t in toks2)
        if s1 == in2:
            score = max(score, 1.0 if len(s1) >= 3 else 0.75)
        elif in2.startswith(s1) and len(s1) >= 3:
            score = max(score, 0.8)
        elif s1.startswith(in2) and len(in2) >= 3:
            score = max(score, 0.7)

    # s2 is acronym candidate, toks1 is long name
    if s2 and len(toks1) >= 2:
        in1 = ''.join(t[0] for t in toks1)
        if s2 == in1:
            score = max(score, 1.0 if len(s2) >= 3 else 0.75)
        elif in1.startswith(s2) and len(s2) >= 3:
            score = max(score, 0.8)
        elif s2.startswith(in1) and len(in1) >= 3:
            score = max(score, 0.7)

    return score


def extract_pair_features_step2(rec1: Dict[str, Any], rec2: Dict[str, Any]) -> List[float]:
    n1, n2 = rec1['norm_name'], rec2['norm_name']
    a1, a2 = rec1['norm_addr'], rec2['norm_addr']

    # 1. Lexical features
    name_fuzz = fuzz.ratio(n1, n2) / 100.0
    name_tok = fuzz.token_set_ratio(n1, n2) / 100.0
    name_jw = JaroWinkler.similarity(n1, n2)

    addr_fuzz = fuzz.ratio(a1, a2) / 100.0 if (a1 and a2) else 0.0
    addr_tok = fuzz.token_set_ratio(a1, a2) / 100.0 if (a1 and a2) else 0.0

    num1, num2 = rec1['nums'], rec2['nums']
    if num1 and num2:
        num_overlap = len(num1 & num2) / len(num1 | num2)
    elif not num1 and not num2:
        num_overlap = 0.8
    else:
        num_overlap = 0.0

    suf1, suf2 = rec1.get('suffix_tokens', set()), rec2.get('suffix_tokens', set())
    if suf1 and suf2:
        suffix_match = 1.0 if (suf1 & suf2) else 0.0
    elif not suf1 and not suf2:
        suffix_match = 0.5
    else:
        suffix_match = 0.3

    qgram_sim = compute_char_qgram_sim(n1, n2, q=3)

    # 2. Acronym feature (additive 9th feature)
    acronym_sim = compute_acronym_similarity(n1, n2)

    return [
        name_fuzz,
        name_tok,
        name_jw,
        addr_fuzz,
        addr_tok,
        num_overlap,
        suffix_match,
        qgram_sim,
        acronym_sim
    ]


def check_veto_step2(r1: Dict[str, Any], r2: Dict[str, Any], acronym_score: float) -> bool:
    # Rule 1: Country mismatch
    if r1.get('country') and r2.get('country') and r1['country'] != r2['country']:
        return True

    # Rule 2: Street number exact mismatch with low address similarity
    nums1, nums2 = r1.get('nums', set()), r2.get('nums', set())
    if nums1 and nums2 and not (nums1 & nums2):
        addr_sim = fuzz.token_set_ratio(r1.get('norm_addr', ''), r2.get('norm_addr', ''))
        if addr_sim < 75:
            return True

    # Rule 3: Core name token mismatch (exempt if strong acronym AND plausible address)
    core1, core2 = r1.get('core_tokens', set()), r2.get('core_tokens', set())
    if core1 and core2 and not (core1 & core2):
        if acronym_score >= 0.7:
            # For acronym match, ensure address is at least moderately consistent to prevent spurious FP
            addr_sim = fuzz.token_set_ratio(r1.get('norm_addr', ''), r2.get('norm_addr', ''))
            if addr_sim < 45:
                return True
        else:
            name_sim = fuzz.ratio(r1.get('norm_name', ''), r2.get('norm_name', ''))
            if name_sim < 65:
                return True

    return False


def run_step2():
    print("=" * 80)
    print("STEP 2: ACRONYM / INITIALISM FEATURE VALIDATION ON 10K HOLDOUT")
    print("=" * 80)
    t0 = time.time()

    # 1. Train or Load Model with 9 features
    model_path = "code/business_entity_resolution/src/model_step2_acronym.pkl"
    device_param = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.isfile(model_path):
        print(f"\n[1] Training GPU XGBoost classifier with 9 features ({device_param})...")
        t_tr = time.time()
        # Mine training pairs from train set
        gt_train = pl.read_csv("dataset/train/train_ground_truth.tsv", separator='\t').sample(12000, seed=42)
        s1_tr_df = pl.read_csv("dataset/train/train_source1.tsv", separator='\t')
        s2_tr_df = pl.read_csv("dataset/train/train_source2.tsv", separator='\t')
        s3_tr_df = pl.read_csv("dataset/train/train_source3.tsv", separator='\t')

        s1_ids = set(gt_train['source1_entity_id'].to_list())
        true_s23_map = {}
        target_s23 = set()
        for r in gt_train.to_dicts():
            m = r['matched_entity_ids']
            m_set = set(m.split(',')) if (m and str(m).strip()) else set()
            true_s23_map[r['source1_entity_id']] = m_set
            target_s23.update(m_set)

        s1_sub = s1_tr_df.filter(pl.col('entity_id').is_in(s1_ids))
        s23_sub = pl.concat([
            s2_tr_df.filter(pl.col('entity_id').is_in(target_s23)),
            s2_tr_df.filter(~pl.col('entity_id').is_in(target_s23)).head(25000),
            s3_tr_df.filter(pl.col('entity_id').is_in(target_s23)),
            s3_tr_df.filter(~pl.col('entity_id').is_in(target_s23)).head(25000),
        ]).unique(subset=['entity_id'])

        s1_norm = {r['entity_id']: normalize_record_step1(r['business_name'], r['business_address'], r['country']) for r in s1_sub.to_dicts()}
        s23_norm = {r['entity_id']: normalize_record_step1(r['business_name'], r['business_address'], r['country']) for r in s23_sub.to_dicts()}

        blocker_tr = BlockingIndex()
        for eid, rec in s23_norm.items():
            blocker_tr.add_record(eid, rec)

        train_X, train_y = [], []
        for s1_id in s1_ids:
            rec1 = s1_norm[s1_id]
            cands = blocker_tr.retrieve_candidates(rec1, max_candidates=25)
            true_set = true_s23_map[s1_id]
            pos_ids = [m for m in true_set if m in s23_norm]
            neg_ids = [c for c in cands if c not in true_set][:10]

            for cid in pos_ids:
                train_X.append(extract_pair_features_step2(rec1, s23_norm[cid]))
                train_y.append(1)
            for cid in neg_ids:
                train_X.append(extract_pair_features_step2(rec1, s23_norm[cid]))
                train_y.append(0)

        X_train = np.array(train_X, dtype=np.float32)
        y_train = np.array(train_y, dtype=np.int32)
        print(f"  Training set size: {len(X_train):,} pairs (Pos: {y_train.sum():,}, Neg: {(y_train==0).sum():,})")

        clf = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.04,
            tree_method='hist',
            device=device_param,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric='logloss',
            random_state=42
        )
        clf.fit(X_train, y_train)
        print(f"  Classifier trained in {time.time()-t_tr:.2f}s.")

        with open(model_path, 'wb') as f:
            pickle.dump({'model': clf, 'feature_names': FEATURE_NAMES_STEP2, 'threshold': 0.85}, f)
    else:
        print(f"  Loaded existing {model_path}.")
        with open(model_path, 'rb') as f:
            art = pickle.load(f)
            clf = art['model']

    # 2. Load 10k Holdout
    s1_df = pl.read_parquet("holdout_cache/holdout_s1.parquet")
    s23_df = pl.read_parquet("holdout_cache/holdout_s23.parquet")
    gt_df = pl.read_parquet("holdout_cache/holdout_gt.parquet")

    us_s1_ids = set(s1_df.filter(pl.col('country') == 'US')['entity_id'].to_list())
    india_s1_ids = set(s1_df.filter(pl.col('country') == 'India')['entity_id'].to_list())
    all_s1_ids = set(s1_df['entity_id'].to_list())

    gt_map = {}
    total_true_pairs = 0
    us_true_pairs = 0
    india_true_pairs = 0
    for r in gt_df.to_dicts():
        eid = r['source1_entity_id']
        m = r['matched_entity_ids']
        m_set = set(m.split(',')) if (m and str(m).strip()) else set()
        gt_map[eid] = m_set
        n_m = len(m_set)
        total_true_pairs += n_m
        if eid in us_s1_ids:
            us_true_pairs += n_m
        else:
            india_true_pairs += n_m

    s1_records = {}
    s1_raw_info = {}
    for r in s1_df.to_dicts():
        eid = r['entity_id']
        s1_raw_info[eid] = r
        s1_records[eid] = normalize_record_step1(r['business_name'], r['business_address'], r['country'])

    s23_records = {}
    s23_raw_info = {}
    for r in s23_df.to_dicts():
        eid = r['entity_id']
        s23_raw_info[eid] = r
        s23_records[eid] = normalize_record_step1(r['business_name'], r['business_address'], r['country'])

    blocker = BlockingIndex(token_frequency_cap=300, num_frequency_cap=150)
    for eid, rec in s23_records.items():
        blocker.add_record(eid, rec)

    retrieved_cands = {}
    captured_true = 0
    us_captured_true = 0
    india_captured_true = 0

    for s1_id, rec1 in s1_records.items():
        cands = blocker.retrieve_candidates(rec1, max_candidates=40)
        retrieved_cands[s1_id] = cands
        c_set = set(cands)
        true_set = gt_map[s1_id]
        if true_set:
            cap = len(c_set & true_set)
            captured_true += cap
            if s1_id in us_s1_ids:
                us_captured_true += cap
            else:
                india_captured_true += cap

    block_rec_all = captured_true / total_true_pairs
    block_rec_us = us_captured_true / us_true_pairs
    block_rec_in = india_captured_true / india_true_pairs

    # 3. Extract 9 features and predict
    pair_records = []
    pair_feats = []
    pair_acronym_scores = []

    for s1_id, cands in retrieved_cands.items():
        rec1 = s1_records[s1_id]
        for cid in cands:
            if cid in s23_records:
                rec2 = s23_records[cid]
                if rec1['country'] != rec2['country']:
                    continue
                f = extract_pair_features_step2(rec1, rec2)
                pair_records.append((s1_id, cid))
                pair_feats.append(f)
                pair_acronym_scores.append(f[8])

    X = np.array(pair_feats, dtype=np.float32)
    probs = clf.predict_proba(X)[:, 1]

    threshold = 0.85
    cand_matches = defaultdict(list)
    pair_prob_lookup = {}
    acronym_triggered_pairs = []

    for (s1_id, cid), prob, acr_score in zip(pair_records, probs, pair_acronym_scores):
        p_val = float(prob)
        pair_prob_lookup[(s1_id, cid)] = p_val
        if p_val >= threshold:
            r1 = s1_records[s1_id]
            r2 = s23_records[cid]
            if not check_veto_step2(r1, r2, acr_score):
                cand_matches[s1_id].append((cid, p_val))
                if acr_score >= 0.7:
                    acronym_triggered_pairs.append((s1_id, cid, acr_score, p_val))

    step2_preds = defaultdict(set)
    for s1_id in all_s1_ids:
        c_list = cand_matches.get(s1_id, [])
        c_list.sort(key=lambda x: x[1], reverse=True)
        step2_preds[s1_id] = set(c[0] for c in c_list[:6])

    f05_all, p_all, r_all, s_all = compute_macro_f05(all_s1_ids, gt_map, step2_preds)
    f05_us, p_us, r_us, s_us = compute_macro_f05(us_s1_ids, gt_map, step2_preds)
    f05_in, p_in, r_in, s_in = compute_macro_f05(india_s1_ids, gt_map, step2_preds)

    base_data = np.load("holdout_cache/baseline_preds.npz")
    with open("holdout_cache/step1_preds.pkl", "rb") as f:
        step1_preds = pickle.load(f)

    print("\n" + "=" * 80)
    print("STEP 2: METRICS COMPARISON (BASELINE vs STEP 1 vs STEP 2 ACRONYM)")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Step 0 Baseline':<16} | {'Step 1 City':<14} | {'Step 2 Acronym':<15} | {'Delta vs Base':<12}")
    print("-" * 80)
    print(f"{'Overall Blocking Ceiling':<25} | {float(base_data['block_rec_all'])*100:>14.2f}% | {block_rec_all*100:>12.2f}% | {block_rec_all*100:>13.2f}% | {(block_rec_all-float(base_data['block_rec_all']))*100:>+10.2f}%")
    print(f"{'India Blocking Ceiling':<25} | {float(base_data['block_rec_in'])*100:>14.2f}% | {block_rec_in*100:>12.2f}% | {block_rec_in*100:>13.2f}% | {(block_rec_in-float(base_data['block_rec_in']))*100:>+10.2f}%")
    print(f"{'US Blocking Ceiling':<25} | {float(base_data['block_rec_us'])*100:>14.2f}% | {block_rec_us*100:>12.2f}% | {block_rec_us*100:>13.2f}% | {(block_rec_us-float(base_data['block_rec_us']))*100:>+10.2f}%")
    print("-" * 80)
    print(f"{'Overall Macro F0.5':<25} | {float(base_data['f05']):>16.4f} | {0.9050:>14.4f} | {f05_all:>15.4f} | {f05_all-float(base_data['f05']):>+12.4f}")
    print(f"{'Overall Precision':<25} | {float(base_data['prec'])*100:>14.2f}% | {99.05:>12.2f}% | {p_all*100:>13.2f}% | {(p_all-float(base_data['prec']))*100:>+10.2f}%")
    print(f"{'Overall Recall':<25} | {float(base_data['rec'])*100:>14.2f}% | {80.73:>12.2f}% | {r_all*100:>13.2f}% | {(r_all-float(base_data['rec']))*100:>+10.2f}%")
    print("-" * 80)
    print(f"{'India Macro F0.5':<25} | {float(base_data['f05_in']):>16.4f} | {0.8509:>14.4f} | {f05_in:>15.4f} | {f05_in-float(base_data['f05_in']):>+12.4f}")
    print(f"{'India Precision':<25} | {float(base_data['prec_in'])*100:>14.2f}% | {98.97:>12.2f}% | {p_in*100:>13.2f}% | {(p_in-float(base_data['prec_in']))*100:>+10.2f}%")
    print(f"{'India Recall':<25} | {float(base_data['rec_in'])*100:>14.2f}% | {71.73:>12.2f}% | {r_in*100:>13.2f}% | {(r_in-float(base_data['rec_in']))*100:>+10.2f}%")
    print("-" * 80)
    print(f"{'US Macro F0.5':<25} | {float(base_data['f05_us']):>16.4f} | {0.9590:>14.4f} | {f05_us:>15.4f} | {f05_us-float(base_data['f05_us']):>+12.4f}")
    print(f"{'US Precision':<25} | {float(base_data['prec_us'])*100:>14.2f}% | {99.12:>12.2f}% | {p_us*100:>13.2f}% | {(p_us-float(base_data['prec_us']))*100:>+10.2f}%")
    print(f"{'US Recall':<25} | {float(base_data['rec_us'])*100:>14.2f}% | {89.84:>12.2f}% | {r_us*100:>13.2f}% | {(r_us-float(base_data['rec_us']))*100:>+10.2f}%")
    print("=" * 80)

    # 4. Mandatory User Requirement: Inspect 10 Acronym-Triggered Matches
    print(f"\n--- MANDATORY INSPECTION: 10 ACRONYM-TRIGGERED MATCHES (SHORT-NAME FP AUDIT) ---")
    print(f"Total Acronym-Triggered Matches (is_acronym >= 0.70 & prob >= 0.85): {len(acronym_triggered_pairs)}")
    
    fp_acronym_count = 0
    inspected_count = min(10, len(acronym_triggered_pairs))
    for i in range(inspected_count):
        s1_id, cid, acr_score, p_val = acronym_triggered_pairs[i]
        r1 = s1_raw_info[s1_id]
        r2 = s23_raw_info[cid]
        is_tp = cid in gt_map[s1_id]
        if not is_tp:
            fp_acronym_count += 1
        status = "TRUE POSITIVE" if is_tp else "FALSE POSITIVE (SHORT-NAME COLLISION)"
        print(f"[{i+1}] Acronym Score: {acr_score:.2f} | Confidence: {p_val:.4f} | Status: {status}")
        print(f"    S1 ({r1['entity_id']}): {r1['business_name']} | Addr: {r1['business_address']}")
        print(f"    S23 ({r2['entity_id']}): {r2['business_name']} | Addr: {r2['business_address']}\n")

    if inspected_count > 0:
        fp_rate = (fp_acronym_count / inspected_count) * 100
        print(f"Acronym Sample False Positive Rate: {fp_rate:.1f}% ({fp_acronym_count}/{inspected_count})")

    # Newly captured vs suspicious
    newly_captured = []
    suspicious = []
    for s1_id in all_s1_ids:
        gt_set = gt_map[s1_id]
        p_step2 = step2_preds[s1_id]
        p_step1 = step1_preds[s1_id]

        new_tps = (p_step2 - p_step1) & gt_set
        for cid in new_tps:
            newly_captured.append((s1_id, cid))

        fps = p_step2 - gt_set
        for cid in fps:
            suspicious.append((s1_id, cid))

    print(f"\n[DIAGNOSTICS]: Newly Captured True Matches vs Step 1: {len(newly_captured):,} | Total False Positives: {len(suspicious):,}")

    print("\n--- 5 EXAMPLES: NEWLY CAPTURED TRUE MATCHES ---")
    if newly_captured:
        for i, (s1_id, cid) in enumerate(newly_captured[:5]):
            r1 = s1_raw_info[s1_id]
            r2 = s23_raw_info[cid]
            p_val = pair_prob_lookup.get((s1_id, cid), 0.0)
            print(f"[{i+1}] S1: {r1['entity_id']} | Name: {r1['business_name']} | Addr: {r1['business_address']}")
            print(f"    S23: {r2['entity_id']} | Name: {r2['business_name']} | Addr: {r2['business_address']}")
            print(f"    Confidence: {p_val:.4f} | Status: TRUE POSITIVE\n")
    else:
        print("  None")

    print("\n--- 5 EXAMPLES: SUSPICIOUS / POTENTIAL FALSE POSITIVE MATCHES ---")
    if suspicious:
        for i, (s1_id, cid) in enumerate(suspicious[:5]):
            r1 = s1_raw_info[s1_id]
            r2 = s23_raw_info[cid]
            p_val = pair_prob_lookup.get((s1_id, cid), 0.0)
            print(f"[{i+1}] S1: {r1['entity_id']} | Name: {r1['business_name']} | Addr: {r1['business_address']}")
            print(f"    S23: {r2['entity_id']} | Name: {r2['business_name']} | Addr: {r2['business_address']}")
            print(f"    Confidence: {p_val:.4f} | Status: FLAGGED\n")

    if p_all < 0.985:
        print(f"\n[GATE FAILED]: Precision dropped to {p_all*100:.2f}% (< 98.5%). Step 2 must be reverted!")
    else:
        print(f"\n[GATE PASSED]: Precision is {p_all*100:.2f}% (>= 98.5%). Step 2 is SAFE to adopt!")

    with open("holdout_cache/step2_preds.pkl", "wb") as f:
        pickle.dump(step2_preds, f)

if __name__ == '__main__':
    run_step2()

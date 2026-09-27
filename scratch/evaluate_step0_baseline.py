"""
Evaluation Harness for 10k Stratified Holdout Set
Step 0: Baseline Evaluation (to establish exact ground-truth baseline metrics)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
import os
import re
import time
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any, Optional

import polars as pl
import numpy as np

sys.path.append('code/business_entity_resolution/src')
from blocking import normalize_record, BlockingIndex
from features import extract_pair_features, FEATURE_NAMES
from train_model import compute_macro_f05
from predict import EntityMatcher

def run_baseline_evaluation():
    print("=" * 80)
    print("RUNNING STEP 0: SAFE BASELINE VERIFICATION ON 10K HOLDOUT")
    print("=" * 80)
    t0 = time.time()

    # 1. Load cached holdout
    s1_df = pl.read_parquet("holdout_cache/holdout_s1.parquet")
    s23_df = pl.read_parquet("holdout_cache/holdout_s23.parquet")
    gt_df = pl.read_parquet("holdout_cache/holdout_gt.parquet")

    us_s1_ids = set(s1_df.filter(pl.col('country') == 'US')['entity_id'].to_list())
    india_s1_ids = set(s1_df.filter(pl.col('country') == 'India')['entity_id'].to_list())
    all_s1_ids = set(s1_df['entity_id'].to_list())

    gt_map: Dict[str, Set[str]] = {}
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

    print(f"Loaded 10k holdout: {len(all_s1_ids):,} entities ({len(us_s1_ids):,} US, {len(india_s1_ids):,} India)")
    print(f"Total True Pairs: {total_true_pairs:,} (US: {us_true_pairs:,}, India: {india_true_pairs:,})")

    # 2. Normalize records
    t_norm = time.time()
    s1_records = {}
    for r in s1_df.to_dicts():
        s1_records[r['entity_id']] = normalize_record(r['business_name'], r['business_address'], r['country'])

    s23_records = {}
    for r in s23_df.to_dicts():
        s23_records[r['entity_id']] = normalize_record(r['business_name'], r['business_address'], r['country'])
    print(f"Normalized {len(s1_records):,} S1 and {len(s23_records):,} S23 records in {time.time()-t_norm:.2f}s.")

    # 3. Build Blocking Index & Retrieve Candidates
    t_block = time.time()
    blocker = BlockingIndex(token_frequency_cap=300, num_frequency_cap=150)
    for eid, rec in s23_records.items():
        blocker.add_record(eid, rec)

    retrieved_cands: Dict[str, List[str]] = {}
    captured_true = 0
    us_captured_true = 0
    india_captured_true = 0
    total_cands_retrieved = 0

    for s1_id, rec1 in s1_records.items():
        cands = blocker.retrieve_candidates(rec1, max_candidates=40)
        retrieved_cands[s1_id] = cands
        c_set = set(cands)
        total_cands_retrieved += len(cands)
        
        true_set = gt_map[s1_id]
        if true_set:
            cap = len(c_set & true_set)
            captured_true += cap
            if s1_id in us_s1_ids:
                us_captured_true += cap
            else:
                india_captured_true += cap

    block_rec_all = captured_true / total_true_pairs if total_true_pairs else 0
    block_rec_us = us_captured_true / us_true_pairs if us_true_pairs else 0
    block_rec_in = india_captured_true / india_true_pairs if india_true_pairs else 0
    avg_cands = total_cands_retrieved / len(s1_records)

    print(f"\n--- BLOCKING CEILING (STEP 0 BASELINE) ---")
    print(f"  Overall Recall Ceiling : {block_rec_all*100:.2f}% ({captured_true:,}/{total_true_pairs:,})")
    print(f"  US Recall Ceiling      : {block_rec_us*100:.2f}% ({us_captured_true:,}/{us_true_pairs:,})")
    print(f"  India Recall Ceiling   : {block_rec_in*100:.2f}% ({india_captured_true:,}/{india_true_pairs:,})")
    print(f"  Avg Candidates/Entity  : {avg_cands:.2f} / 40.0 (Indexed in {time.time()-t_block:.2f}s)")

    # 4. Feature Extraction & Scoring with calibrated EntityMatcher (threshold=0.85)
    t_score = time.time()
    matcher = EntityMatcher("code/business_entity_resolution/src/model.pkl", threshold=0.85)

    pair_records = []
    pair_feats = []
    for s1_id, cands in retrieved_cands.items():
        rec1 = s1_records[s1_id]
        for cid in cands:
            if cid in s23_records:
                rec2 = s23_records[cid]
                if rec1['country'] != rec2['country']:
                    continue
                pair_records.append((s1_id, cid))
                pair_feats.append(extract_pair_features(rec1, rec2))

    X = np.array(pair_feats, dtype=np.float32)
    probs = matcher.predict_probs(X)

    # Apply threshold 0.85 and hard vetoes
    cand_matches = defaultdict(list)
    for (s1_id, cid), prob in zip(pair_records, probs):
        p_val = float(prob)
        if p_val >= matcher.threshold:
            r1 = s1_records[s1_id]
            r2 = s23_records[cid]
            if not matcher.check_veto(r1, r2):
                cand_matches[s1_id].append((cid, p_val))

    final_preds = defaultdict(set)
    for s1_id in all_s1_ids:
        c_list = cand_matches.get(s1_id, [])
        c_list.sort(key=lambda x: x[1], reverse=True)
        final_preds[s1_id] = set(c[0] for c in c_list[:6])

    # Compute metrics
    f05_all, p_all, r_all, s_all = compute_macro_f05(all_s1_ids, gt_map, final_preds)
    f05_us, p_us, r_us, s_us = compute_macro_f05(us_s1_ids, gt_map, final_preds)
    f05_in, p_in, r_in, s_in = compute_macro_f05(india_s1_ids, gt_map, final_preds)

    print(f"\n--- METRICS BREAKDOWN (STEP 0 BASELINE at Threshold={matcher.threshold:.2f}) ---")
    print(f"{'Region':<12} | {'Macro F0.5':<12} | {'Precision':<14} | {'Recall':<14} | {'Singleton Acc':<15}")
    print("-" * 75)
    print(f"{'OVERALL':<12} | {f05_all:<12.4f} | {p_all*100:>12.2f}% | {r_all*100:>12.2f}% | {s_all*100:>13.2f}%")
    print(f"{'US':<12} | {f05_us:<12.4f} | {p_us*100:>12.2f}% | {r_us*100:>12.2f}% | {s_us*100:>13.2f}%")
    print(f"{'India':<12} | {f05_in:<12.4f} | {p_in*100:>12.2f}% | {r_in*100:>12.2f}% | {s_in*100:>13.2f}%")
    print(f"\nBaseline evaluated in {time.time()-t0:.2f}s total.")

    # Save baseline predictions for delta analysis
    import pickle
    with open("holdout_cache/baseline_preds.pkl", "wb") as f:
        pickle.dump(final_preds, f)
    np.savez_compressed("holdout_cache/baseline_preds.npz",
                        f05=f05_all, prec=p_all, rec=r_all,
                        f05_in=f05_in, prec_in=p_in, rec_in=r_in,
                        f05_us=f05_us, prec_us=p_us, rec_us=r_us,
                        block_rec_all=block_rec_all,
                        block_rec_us=block_rec_us,
                        block_rec_in=block_rec_in)

if __name__ == '__main__':
    run_baseline_evaluation()

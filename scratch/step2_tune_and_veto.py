"""
Step 2: Grid sweep threshold, hard veto evaluation, and ablation on stratified train holdout set.
Measures macro F0.5 (with singletons included) before and after each change.
"""

import os
import sys
import re
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

sys.path.append('code/business_entity_resolution/src')
from blocking import normalize_record, BlockingIndex, LEGAL_SUFFIXES, STOPWORDS
from features import extract_pair_features
from train_model import compute_macro_f05
from predict import EntityMatcher

print("=" * 80)
print("STEP 2: THRESHOLD GRID SWEEP & HARD VETO FEATURE EVALUATION")
print("=" * 80)

# 1. Load Stratified Train Ground Truth sample (with singletons)
gt = pl.read_csv("dataset/train/train_ground_truth.tsv", separator='\t')
s1_full = pl.read_csv("dataset/train/train_source1.tsv", separator='\t')

N_VAL = 12000
n_matched = int(N_VAL * 0.9442)
n_singletons = N_VAL - n_matched

m_sample = gt.filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != '')).sample(n_matched, seed=42)
s_sample = gt.filter(pl.col('matched_entity_ids').is_null() | (pl.col('matched_entity_ids') == '')).sample(n_singletons, seed=42)
val_sample = pl.concat([m_sample, s_sample]).sample(fraction=1.0, shuffle=True, seed=42)

val_s1_ids = val_sample['source1_entity_id'].to_list()
val_s1_set = set(val_s1_ids)

gt_mapping: Dict[str, Set[str]] = {}
target_s23: Set[str] = set()
for r in val_sample.to_dicts():
    m = r['matched_entity_ids']
    m_set = set(m.split(',')) if m else set()
    gt_mapping[r['source1_entity_id']] = m_set
    target_s23.update(m_set)

print(f"Loaded {len(val_s1_ids):,} holdout S1 entities ({n_singletons:,} singletons: {n_singletons/len(val_s1_ids)*100:.2f}%)")

# Load S1 and S23 records
s1_df = s1_full.filter(pl.col('entity_id').is_in(val_s1_set))
s1_records = {r['entity_id']: normalize_record(r['business_name'], r['business_address'], r['country']) for r in s1_df.to_dicts()}

s2_full = pl.read_csv("dataset/train/train_source2.tsv", separator='\t')
s3_full = pl.read_csv("dataset/train/train_source3.tsv", separator='\t')

s2_tgt = s2_full.filter(pl.col('entity_id').is_in(target_s23))
s2_rnd = s2_full.filter(~pl.col('entity_id').is_in(target_s23)).head(40000)
s3_tgt = s3_full.filter(pl.col('entity_id').is_in(target_s23))
s3_rnd = s3_full.filter(~pl.col('entity_id').is_in(target_s23)).head(40000)

s23_df = pl.concat([s2_tgt, s2_rnd, s3_tgt, s3_rnd]).unique(subset=['entity_id'])
s23_records = {r['entity_id']: normalize_record(r['business_name'], r['business_address'], r['country']) for r in s23_df.to_dicts()}
print(f"Candidate pool: {len(s23_records):,} S2/S3 records loaded.")

# Build blocking index
blocker = BlockingIndex()
for eid, rec in s23_records.items():
    blocker.add_record(eid, rec)

matcher = EntityMatcher("code/business_entity_resolution/src/model.pkl")

# Generate candidates and base feature matrix
print("\nExtracting pairs and raw model probabilities...")
t0 = time.time()
pair_list = []
feat_list = []
s1_cand_map = defaultdict(list)

for s1_id in val_s1_ids:
    rec1 = s1_records[s1_id]
    cands = blocker.retrieve_candidates(rec1)[:40]
    for c_id in cands:
        rec2 = s23_records[c_id]
        pair_list.append((s1_id, c_id))
        feat_list.append(extract_pair_features(rec1, rec2))
        s1_cand_map[s1_id].append(c_id)

X_all = np.array(feat_list, dtype=np.float32)
probs_all = matcher.predict_probs(X_all)
print(f"Scored {len(pair_list):,} candidate pairs in {time.time()-t0:.1f}s.")

# Map pair to probability
pair_prob_map = {}
for (s1_id, c_id), p in zip(pair_list, probs_all):
    pair_prob_map[(s1_id, c_id)] = float(p)

# -----------------------------------------------------------------
# STEP 2A: THRESHOLD GRID SWEEP (0.50 to 0.95)
# -----------------------------------------------------------------
print("\n" + "-" * 80)
print("STEP 2A: THRESHOLD GRID SWEEP OPTIMIZING MACRO F0.5")
print("-" * 80)
print(f"{'Threshold':<12} | {'Macro F0.5':<12} | {'Precision':<12} | {'Recall':<12} | {'Singleton Acc':<14} | {'Avg Matches':<12}")
print("-" * 80)

sweep_thresholds = [0.50, 0.55, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
best_thresh = 0.62
best_f05 = 0.0

for thresh in sweep_thresholds:
    pred_map = defaultdict(set)
    for (s1_id, c_id), p in pair_prob_map.items():
        if p >= thresh:
            pred_map[s1_id].add(c_id)
            
    f05, prec, rec, s_acc = compute_macro_f05(val_s1_set, gt_mapping, pred_map)
    tot_m = sum(len(m) for m in pred_map.values())
    avg_m = tot_m / len(val_s1_set)
    print(f"{thresh:<12.2f} | {f05:<12.4f} | {prec*100:<11.2f}% | {rec*100:<11.2f}% | {s_acc*100:<13.2f}% | {avg_m:<12.2f}")
    if f05 > best_f05:
        best_f05 = f05
        best_thresh = thresh

print(f"\nOptimal Unconstrained Threshold: {best_thresh:.2f} (Macro F0.5 = {best_f05:.4f})")

# -----------------------------------------------------------------
# STEP 2B: HARD VETO RULES IMPLEMENTATION & ABLATION
# -----------------------------------------------------------------
print("\n" + "-" * 80)
print("STEP 2B: HARD VETO RULES ABLATION STUDY")
print("-" * 80)

# Veto 1: Street Number Exact Mismatch Veto
def check_street_num_veto(r1, r2):
    n1 = r1.get('nums', set())
    n2 = r2.get('nums', set())
    # Both have extracted street numbers, but 0 overlap
    if n1 and n2 and not (n1 & n2):
        return True
    return False

# Veto 2: Core Business Name Token Veto
def check_core_name_veto(r1, r2):
    # Strip suffixes and stopwords, check token overlap
    c1 = set(w for w in r1['norm_name'].split() if w not in LEGAL_SUFFIXES and w not in STOPWORDS and len(w) >= 3)
    c2 = set(w for w in r2['norm_name'].split() if w not in LEGAL_SUFFIXES and w not in STOPWORDS and len(w) >= 3)
    if c1 and c2 and not (c1 & c2):
        # If no common core token and fuzz ratio is low
        if fuzz.ratio(r1['norm_name'], r2['norm_name']) < 65:
            return True
    return False

# Evaluate combinations on best_thresh
for veto_num in [False, True]:
    for veto_name in [False, True]:
        for cap_matches in [None, 6, 8]:
            pred_map = defaultdict(list)
            for (s1_id, c_id), p in pair_prob_map.items():
                if p < best_thresh:
                    continue
                r1 = s1_records[s1_id]
                r2 = s23_records[c_id]
                
                if veto_num and check_street_num_veto(r1, r2):
                    continue
                if veto_name and check_core_name_veto(r1, r2):
                    continue
                pred_map[s1_id].append((c_id, p))
            
            # Apply match capping (sort by confidence, keep top N)
            final_pred_map = defaultdict(set)
            for s1_id, cand_scores in pred_map.items():
                cand_scores.sort(key=lambda x: x[1], reverse=True)
                if cap_matches:
                    cand_scores = cand_scores[:cap_matches]
                final_pred_map[s1_id] = set(c[0] for c in cand_scores)
                
            f05, prec, rec, s_acc = compute_macro_f05(val_s1_set, gt_mapping, final_pred_map)
            tot_m = sum(len(m) for m in final_pred_map.values())
            avg_m = tot_m / len(val_s1_set)
            desc = f"NumVeto={str(veto_num):<5} | NameVeto={str(veto_name):<5} | Cap={str(cap_matches):<4}"
            print(f"{desc} -> Macro F0.5: {f05:.4f} | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | AvgMatches: {avg_m:.2f}")

print("=" * 80)

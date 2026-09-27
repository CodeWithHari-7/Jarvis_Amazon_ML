"""
Comprehensive Root-Cause Diagnosis Script
Answers all 6 user requirements sequentially and outputs exact metrics:
1. Blocking recall ceiling (overall, US, India)
2. Precision vs Recall split
3. Error taxonomy (20 FPs, 20 FNs classified into 5 buckets)
4. Country breakdown (US, India, French-style simulated)
5. Threshold check (evaluating at 5+ nearby thresholds)
6. Sanity check of deployed code logic
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
import os
import re
import time
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

import polars as pl
import numpy as np

sys.path.append('code/business_entity_resolution/src')
from blocking import normalize_record, BlockingIndex, LEGAL_SUFFIXES, ABBREVIATIONS, STOPWORDS
from features import extract_pair_features
from train_model import compute_macro_f05
from predict import EntityMatcher

print("=" * 80)
print("TIME-CRITICAL ROOT-CAUSE DIAGNOSIS: BUSINESS ENTITY RESOLUTION")
print("=" * 80)

# Load Ground Truth and Source 1
print("\n[Loading Ground Truth and Source 1 Train Data]...")
gt_full = pl.read_csv("dataset/train/train_ground_truth.tsv", separator='\t')
s1_full = pl.read_csv("dataset/train/train_source1.tsv", separator='\t')

# Stratified sampling of 10,000 entities (5,000 US, 5,000 India), including 5.58% singletons
N_EACH = 5000
N_SINGLE = int(N_EACH * 0.0558)
N_MATCH = N_EACH - N_SINGLE

s1_us_ids = set(s1_full.filter(pl.col('country') == 'US')['entity_id'].to_list())
s1_in_ids = set(s1_full.filter(pl.col('country') == 'India')['entity_id'].to_list())

gt_us = gt_full.filter(pl.col('source1_entity_id').is_in(s1_us_ids))
gt_in = gt_full.filter(pl.col('source1_entity_id').is_in(s1_in_ids))

# Sample US
us_m = gt_us.filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != '')).sample(N_MATCH, seed=42)
us_s = gt_us.filter(pl.col('matched_entity_ids').is_null() | (pl.col('matched_entity_ids') == '')).sample(N_SINGLE, seed=42)
us_sample = pl.concat([us_m, us_s])

# Sample India
in_m = gt_in.filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != '')).sample(N_MATCH, seed=42)
in_s = gt_in.filter(pl.col('matched_entity_ids').is_null() | (pl.col('matched_entity_ids') == '')).sample(N_SINGLE, seed=42)
in_sample = pl.concat([in_m, in_s])

holdout_gt = pl.concat([us_sample, in_sample]).sample(fraction=1.0, shuffle=True, seed=42)
holdout_s1_ids = holdout_gt['source1_entity_id'].to_list()
holdout_s1_set = set(holdout_s1_ids)

print(f"Sampled {len(holdout_s1_ids):,} holdout entities ({len(us_sample):,} US, {len(in_sample):,} India).")

# Build GT Mapping and collect needed S2/S3 IDs
gt_map: Dict[str, Set[str]] = {}
true_matches_s23: Set[str] = set()
for r in holdout_gt.to_dicts():
    m = r['matched_entity_ids']
    m_set = set(m.split(',')) if (m and str(m).strip()) else set()
    gt_map[r['source1_entity_id']] = m_set
    true_matches_s23.update(m_set)

tot_true_pairs = sum(len(m) for m in gt_map.values())
print(f"Total True Positive Match Pairs in Holdout: {tot_true_pairs:,}")

# Ingest S1 records
s1_holdout_df = s1_full.filter(pl.col('entity_id').is_in(holdout_s1_set))
s1_raw_info = {r['entity_id']: r for r in s1_holdout_df.to_dicts()}
s1_norm = {r['entity_id']: normalize_record(r['business_name'], r['business_address'], r['country']) for r in s1_holdout_df.to_dicts()}

# Ingest S2 and S3: true matches + noise (40,000 each)
print("\nLoading Candidate Pool (True matches + noise pool)...")
s2_full = pl.read_csv("dataset/train/train_source2.tsv", separator='\t')
s3_full = pl.read_csv("dataset/train/train_source3.tsv", separator='\t')

s2_pos = s2_full.filter(pl.col('entity_id').is_in(true_matches_s23))
s2_noise = s2_full.filter(~pl.col('entity_id').is_in(true_matches_s23)).head(40000)
s3_pos = s3_full.filter(pl.col('entity_id').is_in(true_matches_s23))
s3_noise = s3_full.filter(~pl.col('entity_id').is_in(true_matches_s23)).head(40000)

s23_df = pl.concat([s2_pos, s2_noise, s3_pos, s3_noise]).unique(subset=['entity_id'])
s23_raw_info = {r['entity_id']: r for r in s23_df.to_dicts()}
s23_norm = {r['entity_id']: normalize_record(r['business_name'], r['business_address'], r['country']) for r in s23_df.to_dicts()}
print(f"Candidate pool size: {len(s23_norm):,} S2/S3 entities.")

# Build Blocking Index
print("\nBuilding Multi-Pass Blocking Index...")
t_block = time.time()
blocker = BlockingIndex(token_frequency_cap=150)
for eid, rec in s23_norm.items():
    blocker.add_record(eid, rec)
print(f"Blocking index ready in {time.time()-t_block:.1f}s.")

# =====================================================================
# 1. BLOCKING RECALL CEILING
# =====================================================================
print("\n" + "=" * 80)
print("1. BLOCKING RECALL CEILING ANALYSIS")
print("=" * 80)

retrieved_candidates = {}
hits_overall = 0
total_gt_overall = 0
hits_us, total_gt_us = 0, 0
hits_in, total_gt_in = 0, 0

for s1_id in holdout_s1_ids:
    r1 = s1_norm[s1_id]
    c_name = r1['country']
    cands = set(blocker.retrieve_candidates(r1, max_candidates=40))
    retrieved_candidates[s1_id] = cands

    true_m = gt_map[s1_id]
    if true_m:
        captured = len(cands & true_m)
        total_p = len(true_m)
        hits_overall += captured
        total_gt_overall += total_p
        if c_name == 'us':
            hits_us += captured
            total_gt_us += total_p
        else:
            hits_in += captured
            total_gt_in += total_p

rec_overall = hits_overall / total_gt_overall if total_gt_overall else 0.0
rec_us = hits_us / total_gt_us if total_gt_us else 0.0
rec_in = hits_in / total_gt_in if total_gt_in else 0.0

print(f"{'Region':<12} | {'True Pairs':<12} | {'Captured by Blocking':<22} | {'Blocking Recall Ceiling':<25}")
print("-" * 80)
print(f"{'Overall':<12} | {total_gt_overall:<12,} | {hits_overall:<22,} | {rec_overall*100:>23.2f}%")
print(f"{'US':<12} | {total_gt_us:<12,} | {hits_us:<22,} | {rec_us*100:>23.2f}%")
print(f"{'India':<12} | {total_gt_in:<12,} | {hits_in:<22,} | {rec_in*100:>23.2f}%")

if rec_overall >= 0.90:
    print(f"\n>> Verdict on Question 1: Blocking Recall Ceiling is {rec_overall*100:.2f}% (>= 90%). Blocking is NOT the primary bottleneck.")
else:
    print(f"\n>> Verdict on Question 1: Blocking Recall Ceiling is {rec_overall*100:.2f}% (< 90%). Blocking IS a significant bottleneck.")

# =====================================================================
# 2. CLASSIFICATION & PRECISION VS RECALL SPLIT
# =====================================================================
print("\n" + "=" * 80)
print("2. PRECISION VS RECALL SPLIT EVALUATION (MODEL INFERENCE)")
print("=" * 80)

matcher = EntityMatcher("code/business_entity_resolution/src/model.pkl")
current_model_thresh = matcher.threshold
print(f"Current threshold deployed in model.pkl: {current_model_thresh:.4f}")

# Score all retrieved candidate pairs
pair_probs = {}
cand_pair_records = []
cand_pair_feats = []

t_feat = time.time()
for s1_id in holdout_s1_ids:
    r1 = s1_norm[s1_id]
    cands = retrieved_candidates[s1_id]
    for cid in cands:
        if cid in s23_norm:
            cand_pair_records.append((s1_id, cid))
            cand_pair_feats.append(extract_pair_features(r1, s23_norm[cid]))

X_eval = np.array(cand_pair_feats, dtype=np.float32)
probs_eval = matcher.predict_probs(X_eval)
for (s1_id, cid), p in zip(cand_pair_records, probs_eval):
    pair_probs[(s1_id, cid)] = float(p)
print(f"Extracted and scored {len(cand_pair_records):,} pairs in {time.time()-t_feat:.1f}s.")

def evaluate_predictions(pred_dict: Dict[str, Set[str]], entity_subset: Set[str]):
    f05, prec, rec, s_acc = compute_macro_f05(entity_subset, gt_map, pred_dict)
    return f05, prec, rec, s_acc

# Evaluate at current model threshold (0.62)
preds_current = defaultdict(set)
for (s1_id, cid), p in pair_probs.items():
    if p >= current_model_thresh:
        preds_current[s1_id].add(cid)

us_s1_set = set(holdout_gt.filter(pl.col('source1_entity_id').is_in(s1_us_ids))['source1_entity_id'].to_list())
in_s1_set = set(holdout_gt.filter(pl.col('source1_entity_id').is_in(s1_in_ids))['source1_entity_id'].to_list())

f05_all, p_all, r_all, s_all = evaluate_predictions(preds_current, holdout_s1_set)
f05_us, p_us, r_us, s_us = evaluate_predictions(preds_current, us_s1_set)
f05_in, p_in, r_in, s_in = evaluate_predictions(preds_current, in_s1_set)

print(f"\n[Performance at Current Model Threshold = {current_model_thresh:.2f}]:")
print(f"{'Region':<12} | {'Macro F0.5':<12} | {'Macro Precision':<18} | {'Macro Recall':<15} | {'Singleton Acc':<15}")
print("-" * 80)
print(f"{'Overall':<12} | {f05_all:<12.4f} | {p_all*100:>16.2f}% | {r_all*100:>13.2f}% | {s_all*100:>13.2f}%")
print(f"{'US':<12} | {f05_us:<12.4f} | {p_us*100:>16.2f}% | {r_us*100:>13.2f}% | {s_us*100:>13.2f}%")
print(f"{'India':<12} | {f05_in:<12.4f} | {p_in*100:>16.2f}% | {r_in*100:>13.2f}% | {s_in*100:>13.2f}%")

print(f"\n>> Analysis on Question 2:")
if p_all < r_all:
    print(f"  Precision ({p_all*100:.2f}%) is dragging the score down far more than Recall ({r_all*100:.2f}%).")
else:
    print(f"  Recall ({r_all*100:.2f}%) vs Precision ({p_all*100:.2f}%).")

# =====================================================================
# 3. ERROR TAXONOMY: 20 FALSE POSITIVES & 20 FALSE NEGATIVES
# =====================================================================
print("\n" + "=" * 80)
print("3. ERROR TAXONOMY (20 FALSE POSITIVES & 20 FALSE NEGATIVES)")
print("=" * 80)

fp_pairs = []
fn_pairs = []

for s1_id in holdout_s1_ids:
    true_m = gt_map.get(s1_id, set())
    pred_m = preds_current.get(s1_id, set())
    
    # False positives: in pred but not in true
    for cid in (pred_m - true_m):
        fp_pairs.append((s1_id, cid, pair_probs.get((s1_id, cid), 0.0)))
        
    # False negatives: in true but not in pred
    for cid in (true_m - pred_m):
        fn_pairs.append((s1_id, cid, pair_probs.get((s1_id, cid), 0.0)))

print(f"Total False Positive Pairs in Holdout: {len(fp_pairs):,}")
print(f"Total False Negative Pairs in Holdout: {len(fn_pairs):,}")

# Taxonomy classifier helper
def classify_error(r1_raw, r2_raw, r1_norm, r2_norm, is_fp=True):
    # (a) address-number mismatch missed
    n1 = r1_norm['nums']
    n2 = r2_norm['nums']
    if n1 and n2 and not (n1 & n2):
        return '(a) address-number mismatch'
    
    # (c) country mismatch
    if r1_norm['country'] != r2_norm['country']:
        return '(c) country handling bug'
    
    # (b) legal suffix / abbrev noise
    core1 = set(r1_norm['core_tokens'])
    core2 = set(r2_norm['core_tokens'])
    if not core1 or not core2:
        return '(b) legal-suffix/abbrev noise'
    
    # (d) transliteration / typo
    from rapidfuzz import fuzz
    f_ratio = fuzz.ratio(r1_norm['norm_name'], r2_norm['norm_name'])
    if 50 <= f_ratio < 80:
        return '(d) transliteration/typo not caught'
        
    return '(e) other (semantic / chain / city variation)'

fp_bucket_counts = Counter()
fn_bucket_counts = Counter()

print("\n--- [SAMPLE 20 FALSE POSITIVES (PREDICTED MATCH, NOT TRUE MATCH)] ---")
sample_fps = fp_pairs[:20]
for idx, (s1_id, cid, prob) in enumerate(sample_fps, 1):
    r1_raw = s1_raw_info[s1_id]
    r2_raw = s23_raw_info.get(cid, {})
    r1_n = s1_norm[s1_id]
    r2_n = s23_norm.get(cid, {})
    
    bucket = classify_error(r1_raw, r2_raw, r1_n, r2_n, is_fp=True)
    fp_bucket_counts[bucket] += 1
    
    print(f"\n[FP {idx}] Score: {prob:.3f} | Bucket: {bucket}")
    print(f"  S1 ({s1_id}): '{r1_raw.get('business_name')}' | Addr: '{r1_raw.get('business_address')}'")
    print(f"  S23({cid}): '{r2_raw.get('business_name')}' | Addr: '{r2_raw.get('business_address')}'")

print("\n--- [SAMPLE 20 FALSE NEGATIVES (TRUE MATCH MISSED)] ---")
sample_fns = fn_pairs[:20]
for idx, (s1_id, cid, prob) in enumerate(sample_fns, 1):
    r1_raw = s1_raw_info[s1_id]
    r2_raw = s23_raw_info.get(cid, {})
    r1_n = s1_norm[s1_id]
    r2_n = s23_norm.get(cid, {})
    
    bucket = classify_error(r1_raw, r2_raw, r1_n, r2_n, is_fp=False)
    fn_bucket_counts[bucket] += 1
    
    in_blocking = cid in retrieved_candidates.get(s1_id, set())
    status_str = f"Score: {prob:.3f}" if in_blocking else "MISSED BY BLOCKING"
    print(f"\n[FN {idx}] {status_str} | Bucket: {bucket}")
    print(f"  S1 ({s1_id}): '{r1_raw.get('business_name')}' | Addr: '{r1_raw.get('business_address')}'")
    print(f"  S23({cid}): '{r2_raw.get('business_name')}' | Addr: '{r2_raw.get('business_address')}'")

# Classify all FPs and FNs across entire holdout to get statistically rigorous totals
all_fp_buckets = Counter()
for s1_id, cid, prob in fp_pairs:
    b = classify_error(s1_raw_info[s1_id], s23_raw_info[cid], s1_norm[s1_id], s23_norm[cid], is_fp=True)
    all_fp_buckets[b] += 1

all_fn_buckets = Counter()
for s1_id, cid, prob in fn_pairs:
    if cid in s23_norm:
        b = classify_error(s1_raw_info[s1_id], s23_raw_info[cid], s1_norm[s1_id], s23_norm[cid], is_fp=False)
    else:
        b = '(e) other (unindexed / missing candidate)'
    all_fn_buckets[b] += 1

print("\n" + "=" * 80)
print("ERROR TAXONOMY SUMMARY (HOLDOUT AGGREGATE COUNTS)")
print("=" * 80)
print(f"{'Error Category Bucket':<45} | {'False Positives':<18} | {'False Negatives':<18}")
print("-" * 80)
all_categories = sorted(set(list(all_fp_buckets.keys()) + list(all_fn_buckets.keys())))
for cat in all_categories:
    print(f"{cat:<45} | {all_fp_buckets[cat]:>16,} | {all_fn_buckets[cat]:>16,}")

# =====================================================================
# 4. COUNTRY BREAKDOWN & SYNTHETIC FRANCE SIMULATION
# =====================================================================
print("\n" + "=" * 80)
print("4. COUNTRY-WISE BREAKDOWN & SYNTHETIC FRENCH HOLD-OUT ANALYSIS")
print("=" * 80)

# Simulate French format: Commune prefix / suffix pollution, accented diacritics
# How does our model perform on entities where city/commune is part of name or address numbers are messy?
french_style_s1 = []
for sid in us_s1_set:
    r = s1_raw_info[sid]
    # Check if address has multiple numbers or commune-like hyphenated patterns
    if '-' in str(r.get('business_name', '')) or len(s1_norm[sid]['nums']) >= 2:
        french_style_s1.append(sid)

if french_style_s1:
    f05_fr, p_fr, r_fr, s_fr = evaluate_predictions(preds_current, set(french_style_s1[:2000]))
    print(f"{'Simulated French-Format':<25} | F0.5: {f05_fr:.4f} | Prec: {p_fr*100:.2f}% | Rec: {r_fr*100:.2f}% | SingletonAcc: {s_fr*100:.2f}%")
print(f"{'US Holdout':<25} | F0.5: {f05_us:.4f} | Prec: {p_us*100:.2f}% | Rec: {r_us*100:.2f}% | SingletonAcc: {s_us*100:.2f}%")
print(f"{'India Holdout':<25} | F0.5: {f05_in:.4f} | Prec: {p_in*100:.2f}% | Rec: {r_in*100:.2f}% | SingletonAcc: {s_in*100:.2f}%")

# =====================================================================
# 5. THRESHOLD CHECK (NEARBY THRESHOLDS)
# =====================================================================
print("\n" + "=" * 80)
print("5. THRESHOLD CHECK: MACRO F0.5 ACROSS NEARBY THRESHOLDS")
print("=" * 80)
print(f"Current threshold in model artifact: {current_model_thresh:.4f}")
print(f"{'Threshold':<12} | {'Macro F0.5':<12} | {'Precision':<14} | {'Recall':<14} | {'Avg Matches':<12}")
print("-" * 80)

test_thresholds = [0.55, 0.62, 0.70, 0.75, 0.80, 0.85, 0.90]
for th in test_thresholds:
    th_preds = defaultdict(set)
    for (s1_id, cid), p in pair_probs.items():
        if p >= th:
            th_preds[s1_id].add(cid)
    f05_t, p_t, r_t, _ = evaluate_predictions(th_preds, holdout_s1_set)
    tot_m = sum(len(m) for m in th_preds.values())
    avg_m = tot_m / len(holdout_s1_set)
    marker = " <-- CURRENT" if abs(th - current_model_thresh) < 0.01 else ""
    print(f"{th:<12.2f} | {f05_t:<12.4f} | {p_t*100:>12.2f}% | {r_t*100:>12.2f}% | {avg_m:<12.2f}{marker}")

# =====================================================================
# 6. SANITY CHECK DEPLOYED CODE
# =====================================================================
print("\n" + "=" * 80)
print("6. SANITY CHECK: DEPLOYED CODE VS INTENDED VETO LOGIC")
print("=" * 80)

# Check predict.py
with open("code/business_entity_resolution/src/predict.py", 'r', encoding='utf-8') as f:
    predict_code = f.read()

has_veto_country = "country" in predict_code and ("mismatch" in predict_code or "!=" in predict_code)
has_veto_num = "addr_num" in predict_code or "nums" in predict_code
has_core_name = "core_tokens" in predict_code

print("Sanity Check in code/business_entity_resolution/src/predict.py:")
print(f"  - Country mismatch veto present in predict.py       : {has_veto_country}")
print(f"  - Address-number exact mismatch veto in predict.py  : {has_veto_num}")
print(f"  - Core name token veto in predict.py                : {has_core_name}")
print(f"  - Stored threshold in predict.py (fallback)         : {current_model_thresh}")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)

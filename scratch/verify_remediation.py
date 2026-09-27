import os
import sys
sys.path.append('code/business_entity_resolution/src')
import polars as pl
import numpy as np
from collections import defaultdict
from blocking import normalize_record, LEGAL_SUFFIXES, ABBREVIATIONS, STOPWORDS
from features import extract_pair_features
from train_model import compute_macro_f05
from predict import EntityMatcher

print("=" * 75)
print("VERIFYING REMEDIATION WITH SCORED CANDIDATE PRIORITIZATION")
print("=" * 75)

# 1. Load Ground Truth Sample
gt_path = "dataset/train/train_ground_truth.tsv"
s1_path = "dataset/train/train_source1.tsv"
s2_path = "dataset/train/train_source2.tsv"
s3_path = "dataset/train/train_source3.tsv"

gt = pl.read_csv(gt_path, separator='\t')
matched_gt = gt.filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != '')).sample(1888, seed=42)
singleton_gt = gt.filter(pl.col('matched_entity_ids').is_null() | (pl.col('matched_entity_ids') == '')).sample(112, seed=42)
sample_gt = pl.concat([matched_gt, singleton_gt]).sample(fraction=1.0, shuffle=True, seed=42)

s1_ids = set(sample_gt['source1_entity_id'].to_list())
gt_map = {}
needed_s23 = set()
for r in sample_gt.to_dicts():
    m = r['matched_entity_ids']
    m_set = set(m.split(',')) if m else set()
    gt_map[r['source1_entity_id']] = m_set
    needed_s23.update(m_set)

# 2. Ingest S1 & Candidate S2/S3
s1_df = pl.read_csv(s1_path, separator='\t').filter(pl.col('entity_id').is_in(list(s1_ids)))
s1_records = {r['entity_id']: normalize_record(r['business_name'], r['business_address'], r['country']) for r in s1_df.to_dicts()}

s2_full = pl.read_csv(s2_path, separator='\t')
s2_pos = s2_full.filter(pl.col('entity_id').is_in(list(needed_s23)))
s2_noise = s2_full.filter(~pl.col('entity_id').is_in(list(needed_s23))).head(40000)

s3_full = pl.read_csv(s3_path, separator='\t')
s3_pos = s3_full.filter(pl.col('entity_id').is_in(list(needed_s23)))
s3_noise = s3_full.filter(~pl.col('entity_id').is_in(list(needed_s23))).head(40000)

s23_df = pl.concat([s2_pos, s2_noise, s3_pos, s3_noise]).unique(subset=['entity_id'])
s23_records = {r['entity_id']: normalize_record(r['business_name'], r['business_address'], r['country']) for r in s23_df.to_dicts()}
print(f"Candidate pool built: {len(s23_records):,} S2/S3 entities.")

# 3. Build Indices
idx_pfx = defaultdict(list)
idx_tok = defaultdict(list)
idx_num = defaultdict(list)

for eid, n2 in s23_records.items():
    if n2.get('prefix4'): idx_pfx[n2['prefix4']].append(eid)
    if n2.get('sorted_tokens'): idx_tok[n2['sorted_tokens']].append(eid)
    for num in n2['nums']: idx_num[num].append(eid)

matcher = EntityMatcher(threshold=0.70)

# 4. Predict
pred_map = defaultdict(set)
for sid in s1_ids:
    r1 = s1_records[sid]
    cand_scores = defaultdict(int)
    if r1.get('prefix4'):
        for cid in idx_pfx.get(r1['prefix4'], []): cand_scores[cid] += 10
    if r1.get('sorted_tokens'):
        for cid in idx_tok.get(r1['sorted_tokens'], []): cand_scores[cid] += 10
    for num in r1['nums']:
        n_m = idx_num.get(num, [])
        if len(n_m) <= 150:
            for cid in n_m: cand_scores[cid] += 5
            
    if len(cand_scores) > 40:
        cands = sorted(cand_scores.keys(), key=lambda x: cand_scores[x], reverse=True)[:40]
    else:
        cands = list(cand_scores.keys())
        
    cand_recs = [(cid, s23_records[cid]) for cid in cands if cid in s23_records]
    if not cand_recs:
        continue
        
    scored = matcher.score_candidates(r1, cand_recs)
    matches = [cid for cid, p in scored if p >= 0.70]
    top_cid, top_p = max(scored, key=lambda x: x[1]) if scored else (None, 0.0)
    if not matches and top_p >= 0.50:
        matches = [top_cid]
        
    pred_map[sid] = set(matches)

# 5. Evaluate
f05, p, r, sacc = compute_macro_f05(s1_ids, gt_map, pred_map)

print("\n" + "=" * 70)
print("BENCHMARK RESULTS ON 2,000 HOLDOUT S1 ENTITIES")
print("=" * 70)
print(f"  Macro F0.5         : {f05:.4f}")
print(f"  Precision          : {p:.2%}")
print(f"  Recall             : {r:.2%}")
print(f"  Singleton Accuracy : {sacc:.2%}")
print("=" * 70)

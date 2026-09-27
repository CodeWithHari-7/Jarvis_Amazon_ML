import os
import sys
import time
from collections import defaultdict
import polars as pl
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.append('code/business_entity_resolution/src')
from blocking import normalize_record, BlockingIndex

print("=" * 80)
print("STEP 2: MEASURING INDIA CANDIDATE RECALL RECOVERY (UNIFIED VS A-M/N-Z SPLIT)")
print("=" * 80)

# 1. Load Ground Truth and 1,000 holdout India entities
gt = pl.read_csv("dataset/train/train_ground_truth.tsv", separator='\t')
s1 = pl.read_csv("dataset/train/train_source1.tsv", separator='\t')

gt_matched = gt.filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != ''))
s1_in_ids = set(s1.filter(pl.col('country') == 'India')['entity_id'].to_list())

sample_in_gt = gt_matched.filter(pl.col('source1_entity_id').is_in(s1_in_ids)).sample(1000, seed=42)
s1_eval_ids = set(sample_in_gt['source1_entity_id'].to_list())
s1_eval_df = s1.filter(pl.col('entity_id').is_in(s1_eval_ids))

gt_map = {}
for r in sample_in_gt.to_dicts():
    gt_map[r['source1_entity_id']] = set(r['matched_entity_ids'].split(','))

tot_true_matches = sum(len(v) for v in gt_map.values())
print(f"Sampled 1,000 India S1 entities. Total true matches to find: {tot_true_matches:,}")

# 2. Load Train S2 and S3 for India
print("\nLoading Full Train S2 and S3 for India...")
t0 = time.time()
s2 = pl.read_csv("dataset/train/train_source2.tsv", separator='\t', columns=['entity_id', 'business_name', 'business_address', 'country']).filter(pl.col('country') == 'India')
s3 = pl.read_csv("dataset/train/train_source3.tsv", separator='\t', columns=['entity_id', 'business_name', 'business_address', 'country']).filter(pl.col('country') == 'India')
s23 = pl.concat([s2, s3]).unique(subset=['entity_id'])
del s2, s3
n_pool = len(s23)
print(f"Loaded {n_pool:,} unique India candidate records in {time.time()-t0:.1f}s.")

p1_chars = set('0123456789abcdefghijklm')

# Check how many true matches in this 1000 sample cross the A-M vs N-Z line
s1_names = {r['entity_id']: r['business_name'] or '' for r in s1_eval_df.to_dicts()}
# Fetch target names from s23
target_eids = set()
for s in gt_map.values():
    target_eids.update(s)

target_names = {}
for r in s23.filter(pl.col('entity_id').is_in(target_eids)).to_dicts():
    target_names[r['entity_id']] = r['business_name'] or ''

cross_pairs_count = 0
for s1_id, targets in gt_map.items():
    c1 = (s1_names.get(s1_id, '')[:1]).lower() in p1_chars
    for tid in targets:
        if tid in target_names:
            c2 = (target_names[tid][:1]).lower() in p1_chars
            if c1 != c2:
                cross_pairs_count += 1

print(f"In this 1,000 sample: {cross_pairs_count:,} / {tot_true_matches:,} true matches ({cross_pairs_count/tot_true_matches*100:.2f}%) cross the A-M vs N-Z boundary!")

# 3. Build Unified Index using updated BlockingIndex
print("\nBuilding Unified BlockingIndex on all 4.9M India candidate records...")
t_idx = time.time()
blocker_unified = BlockingIndex()

eids = s23['entity_id'].to_list()
names = s23['business_name'].to_list()
addrs = s23['business_address'].to_list()
countries = s23['country'].to_list()

for i in range(n_pool):
    rec = normalize_record(names[i], addrs[i], countries[i])
    blocker_unified.add_record(eids[i], rec)
    if (i + 1) % 1500000 == 0:
        print(f"  Indexed {i+1:,}/{n_pool:,} records ({time.time()-t_idx:.1f}s)...")

print(f"Unified India Index built in {time.time()-t_idx:.1f}s.")

# 4. Evaluate Recall: Unified vs Simulated A-M / N-Z Split
s1_records = {r['entity_id']: normalize_record(r['business_name'], r['business_address'], r['country']) 
              for r in s1_eval_df.to_dicts()}

hits_unified = 0
hits_split = 0
zero_hits_unified = 0
zero_hits_split = 0

for s1_id, r1 in s1_records.items():
    true_m = gt_map[s1_id]
    s1_raw_name = s1_names.get(s1_id, '')
    s1_is_p1 = (s1_raw_name[:1]).lower() in p1_chars

    # Unified retrieval
    cands_unified = set(blocker_unified.retrieve_candidates(r1, max_candidates=40))
    h_u = len(cands_unified & true_m)
    hits_unified += h_u
    if h_u == 0:
        zero_hits_unified += 1

    # Simulated split: candidate must be in the same first-letter partition
    # (i.e. if candidate's first letter doesn't match s1_is_p1, it could not have been in the partition)
    cands_split = set()
    for cid in cands_unified:
        c_raw = target_names.get(cid, '')
        if c_raw:
            c_is_p1 = (c_raw[:1]).lower() in p1_chars
            if c_is_p1 == s1_is_p1:
                cands_split.add(cid)
        else:
            cands_split.add(cid)

    h_s = len(cands_split & true_m)
    hits_split += h_s
    if h_s == 0:
        zero_hits_split += 1

rec_unified = (hits_unified / tot_true_matches) * 100
rec_split = (hits_split / tot_true_matches) * 100

print("\n" + "=" * 80)
print("STEP 2 RESULTS ON INDIA (FULL 4.9M CANDIDATE CORPUS):")
print("=" * 80)
print(f"Old Split Candidate Recall (A-M / N-Z) : {rec_split:.2f}% (Zero-hit entities: {zero_hits_split/len(s1_records)*100:.2f}%)")
print(f"New Unified Candidate Recall           : {rec_unified:.2f}% (Zero-hit entities: {zero_hits_unified/len(s1_records)*100:.2f}%)")
print(f"Net Candidate Recall Improvement       : +{rec_unified - rec_split:.2f}%")
print(f"Zero-Hit Entities Recovered            : {zero_hits_split - zero_hits_unified} entities")
print("=" * 80)

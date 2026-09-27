import os
import sys
import time
from collections import defaultdict
import polars as pl
import numpy as np

sys.path.append('code/business_entity_resolution/src')
from blocking import normalize_record, BlockingIndex

print("=" * 80)
print("TESTING BLOCKING RECALL CEILING ON FULL S2/S3 CORPUS")
print("=" * 80)

# Load 1,000 random US entities and 1,000 random India entities from train
gt = pl.read_csv("dataset/train/train_ground_truth.tsv", separator='\t')
s1 = pl.read_csv("dataset/train/train_source1.tsv", separator='\t')

# Sample 1000 US and 1000 India entities with matches
gt_with_m = gt.filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != ''))
s1_us_ids = set(s1.filter(pl.col('country') == 'US')['entity_id'].to_list())
s1_in_ids = set(s1.filter(pl.col('country') == 'India')['entity_id'].to_list())

sample_us = gt_with_m.filter(pl.col('source1_entity_id').is_in(s1_us_ids)).sample(1000, seed=42)
sample_in = gt_with_m.filter(pl.col('source1_entity_id').is_in(s1_in_ids)).sample(1000, seed=42)

gt_test = pl.concat([sample_us, sample_in])
s1_eval_ids = set(gt_test['source1_entity_id'].to_list())
s1_eval_df = s1.filter(pl.col('entity_id').is_in(s1_eval_ids))

gt_map = {}
for r in gt_test.to_dicts():
    gt_map[r['source1_entity_id']] = set(r['matched_entity_ids'].split(','))

print(f"Sampled {len(s1_eval_ids)} S1 evaluation entities (1000 US, 1000 India).")
tot_true_matches = sum(len(v) for v in gt_map.values())
print(f"Total true match pairs to find: {tot_true_matches:,}")

# Now build index on FULL S2 and S3 for US (to test real scale!)
print("\nLoading FULL Train S2 and S3 for US...")
s2_us = pl.read_csv("dataset/train/train_source2.tsv", separator='\t').filter(pl.col('country') == 'US')
s3_us = pl.read_csv("dataset/train/train_source3.tsv", separator='\t').filter(pl.col('country') == 'US')
s23_us = pl.concat([s2_us, s3_us]).unique(subset=['entity_id'])
del s2_us, s3_us

n_pool = len(s23_us)
print(f"Full US Candidate pool size: {n_pool:,} records.")

# 1. Test current BlockingIndex (token_frequency_cap=300)
print("\n[Test 1] Building BlockingIndex with token_frequency_cap=300...")
t0 = time.time()
blocker_300 = BlockingIndex(token_frequency_cap=300, num_frequency_cap=150)

eids = s23_us['entity_id'].to_list()
names = s23_us['business_name'].to_list()
addrs = s23_us['business_address'].to_list()
countries = s23_us['country'].to_list()
del s23_us

for i in range(n_pool):
    rec = normalize_record(names[i], addrs[i], countries[i])
    blocker_300.add_record(eids[i], rec)
print(f"Index built in {time.time()-t0:.1f}s.")

# Check token frequencies
print(f"Unique any_tok keys: {len(blocker_300.idx_any_tok):,}")
tok_lens = [len(v) for v in blocker_300.idx_any_tok.values()]
tok_lens = np.array(tok_lens)
print(f"Tokens with freq <= 300: {np.sum(tok_lens <= 300):,} ({np.mean(tok_lens <= 300)*100:.2f}%)")
print(f"Tokens with freq > 300 (SKIPPED): {np.sum(tok_lens > 300):,} ({np.mean(tok_lens > 300)*100:.2f}%)")

# Evaluate recall on the 1000 US entities
s1_us_records = {r['entity_id']: normalize_record(r['business_name'], r['business_address'], r['country']) 
                 for r in s1_eval_df.filter(pl.col('country') == 'US').to_dicts()}

hits_40 = 0
tot_us_true = 0
zero_hit_entities = 0
candidate_counts = []

for s1_id, r1 in s1_us_records.items():
    true_m = gt_map[s1_id]
    tot_us_true += len(true_m)
    cands = blocker_300.retrieve_candidates(r1, max_candidates=40)
    cands_set = set(cands)
    candidate_counts.append(len(cands))
    
    hits = len(cands_set & true_m)
    hits_40 += hits
    if hits == 0:
        zero_hit_entities += 1

print("\n--- RESULTS ON FULL US CORPUS (1,000 S1 Entities against 6M+ Candidates) ---")
print(f"True Match Pairs: {tot_us_true:,}")
print(f"Hits in Top-40 Candidates: {hits_40:,} / {tot_us_true:,}")
print(f"REAL Candidate Recall Ceiling (max 40): {hits_40 / tot_us_true * 100:.2f}%")
print(f"Average candidates retrieved: {np.mean(candidate_counts):.2f}")
print(f"Entities with ZERO true matches retrieved (recall=0.0): {zero_hit_entities} / {len(s1_us_records)} ({zero_hit_entities/len(s1_us_records)*100:.2f}%)")

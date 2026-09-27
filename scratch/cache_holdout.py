"""
Creates and caches the exact 10,000 stratified holdout dataset (5k US, 5k India, with singletons)
and the candidate universe (true matches + 80k noise records).
"""

import os
import time
from typing import Dict, Set
import polars as pl

print("Caching 10k holdout dataset...")
t0 = time.time()
os.makedirs("holdout_cache", exist_ok=True)

gt_path = "dataset/train/train_ground_truth.tsv"
s1_path = "dataset/train/train_source1.tsv"
s2_path = "dataset/train/train_source2.tsv"
s3_path = "dataset/train/train_source3.tsv"

gt_full = pl.read_csv(gt_path, separator='\t')
s1_full = pl.read_csv(s1_path, separator='\t')

N_EACH = 5000
N_SINGLE = int(N_EACH * 0.0558)
N_MATCH = N_EACH - N_SINGLE

s1_us_ids = set(s1_full.filter(pl.col('country') == 'US')['entity_id'].to_list())
s1_in_ids = set(s1_full.filter(pl.col('country') == 'India')['entity_id'].to_list())

gt_us = gt_full.filter(pl.col('source1_entity_id').is_in(s1_us_ids))
gt_in = gt_full.filter(pl.col('source1_entity_id').is_in(s1_in_ids))

us_m = gt_us.filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != '')).sample(N_MATCH, seed=42)
us_s = gt_us.filter(pl.col('matched_entity_ids').is_null() | (pl.col('matched_entity_ids') == '')).sample(N_SINGLE, seed=42)
us_sample = pl.concat([us_m, us_s])

in_m = gt_in.filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != '')).sample(N_MATCH, seed=42)
in_s = gt_in.filter(pl.col('matched_entity_ids').is_null() | (pl.col('matched_entity_ids') == '')).sample(N_SINGLE, seed=42)
in_sample = pl.concat([in_m, in_s])

holdout_gt = pl.concat([us_sample, in_sample]).sample(fraction=1.0, shuffle=True, seed=42)
holdout_s1_ids = set(holdout_gt['source1_entity_id'].to_list())

true_matches_s23: Set[str] = set()
for r in holdout_gt.to_dicts():
    m = r['matched_entity_ids']
    if m and str(m).strip():
        true_matches_s23.update(m.split(','))

s1_holdout_df = s1_full.filter(pl.col('entity_id').is_in(holdout_s1_ids))

print(f"Reading S2 and S3 for candidates...")
s2_full = pl.read_csv(s2_path, separator='\t')
s3_full = pl.read_csv(s3_path, separator='\t')

s2_pos = s2_full.filter(pl.col('entity_id').is_in(true_matches_s23))
s2_noise = s2_full.filter(~pl.col('entity_id').is_in(true_matches_s23)).head(40000)
s3_pos = s3_full.filter(pl.col('entity_id').is_in(true_matches_s23))
s3_noise = s3_full.filter(~pl.col('entity_id').is_in(true_matches_s23)).head(40000)

s23_df = pl.concat([s2_pos, s2_noise, s3_pos, s3_noise]).unique(subset=['entity_id'])

holdout_gt.write_parquet("holdout_cache/holdout_gt.parquet")
s1_holdout_df.write_parquet("holdout_cache/holdout_s1.parquet")
s23_df.write_parquet("holdout_cache/holdout_s23.parquet")

print(f"Holdout cached successfully in {time.time()-t0:.1f}s:")
print(f"  Holdout S1 entities: {len(s1_holdout_df):,}")
print(f"  Holdout S23 pool   : {len(s23_df):,}")
print(f"  Ground truth pairs : {len(true_matches_s23):,}")

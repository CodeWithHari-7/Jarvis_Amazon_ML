"""
Benchmark France pipeline using updated blocking.py and predict.py:
Verifies speed, memory, and match-count distribution.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
import os
import re
import time
from collections import defaultdict, Counter
import polars as pl
import numpy as np

sys.path.append('code/business_entity_resolution/src')
from blocking import normalize_record, BlockingIndex
from features import extract_pair_features
from predict import EntityMatcher

print("Testing France Pipeline on real test data...")
t0 = time.time()

# Load France cached parquet
s1_fr = pl.read_parquet("pipeline_cache/s1_france.parquet").head(10000)
s2_fr = pl.read_parquet("pipeline_cache/s2_france.parquet")
s3_fr = pl.read_parquet("pipeline_cache/s3_france.parquet")
s23_fr = pl.concat([s2_fr, s3_fr]).unique(subset=['entity_id'])

print(f"Loaded 10k S1 France, {len(s23_fr):,} S2/S3 France candidates.")

# Normalize records
t_norm = time.time()
s23_norm = {r['entity_id']: normalize_record(r['business_name'], r['business_address'], r['country']) for r in s23_fr.to_dicts()}
s1_norm = {r['entity_id']: normalize_record(r['business_name'], r['business_address'], r['country']) for r in s1_fr.to_dicts()}
print(f"Normalized records in {time.time()-t_norm:.1f}s.")

# Index
t_idx = time.time()
blocker = BlockingIndex()
for eid, rec in s23_norm.items():
    blocker.add_record(eid, rec)
print(f"Indexed in {time.time()-t_idx:.1f}s.")

# Matcher
matcher = EntityMatcher()
print(f"Matcher threshold: {matcher.threshold}")

# Candidate retrieval and scoring
t_eval = time.time()
match_counts = []
cand_counts = []

for r in s1_fr.to_dicts():
    eid = r['entity_id']
    n1 = s1_norm[eid]
    cands = blocker.retrieve_candidates(n1, max_candidates=40)
    cand_counts.append(len(cands))
    
    cand_tuples = [(cid, s23_norm[cid]) for cid in cands if cid in s23_norm]
    matches = matcher.predict_matches(n1, cand_tuples, override_threshold=0.85, max_matches=6)
    match_counts.append(len(matches))

mc = np.array(match_counts)
print(f"\nCompleted 10,000 France entities in {time.time()-t_eval:.1f}s.")
print(f"  Avg Matches/Entity: {np.mean(mc):.2f}")
print(f"  Singleton Rate: {np.mean(mc == 0)*100:.2f}%")
print(f"  Max Matches: {np.max(mc)}")
print(f"  Average Candidates: {np.mean(cand_counts):.2f}")
print("Frequency distribution (0 to 6 matches):", Counter(match_counts))

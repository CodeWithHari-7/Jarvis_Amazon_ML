import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import random
from collections import defaultdict
import polars as pl

print("=" * 80)
print("MANUAL AUDIT: SAMPLING 20 TEST ROWS ACROSS MATCH-COUNT BUCKETS")
print("=" * 80)

# Read matching_results.tsv
matching_path = "output/matching_results.tsv"
buckets = {0: [], 1: [], 3: [], 6: []}

with open(matching_path, 'r', encoding='utf-8') as f:
    next(f)
    for i, line in enumerate(f):
        parts = line.rstrip('\r\n').split('\t')
        eid = parts[0]
        matches = parts[1].split(',') if len(parts) > 1 and parts[1] else []
        n_m = len(matches)
        if n_m in buckets and len(buckets[n_m]) < 100:
            buckets[n_m].append((eid, matches))

# Set seed for reproducible sampling
random.seed(42)
sampled_per_bucket = {}
for k in [0, 1, 3, 6]:
    sampled_per_bucket[k] = random.sample(buckets[k], min(5, len(buckets[k])))

# Collect all entity IDs needed
all_s1_needed = set()
all_s23_needed = set()
for k, pairs in sampled_per_bucket.items():
    for s1_id, m_list in pairs:
        all_s1_needed.add(s1_id)
        for m_id in m_list:
            all_s23_needed.add(m_id)

# Fetch S1 records
s1_test = pl.read_csv("dataset/test/test_source1.tsv", separator='\t')
s1_lookup = {}
for r in s1_test.filter(pl.col('entity_id').is_in(all_s1_needed)).to_dicts():
    s1_lookup[r['entity_id']] = r

# Fetch S2/S3 candidate records from parquet cache
s23_lookup = {}
for c in ['us', 'india', 'france']:
    for s in ['s2', 's3']:
        p_path = f"pipeline_cache/{s}_{c}.parquet"
        if os.path.exists(p_path):
            df = pl.read_parquet(p_path)
            matching_rows = df.filter(pl.col('entity_id').is_in(all_s23_needed)).to_dicts()
            for r in matching_rows:
                s23_lookup[r['entity_id']] = r

print(f"Loaded {len(s1_lookup)} S1 records and {len(s23_lookup)} matched candidate records.\n")

for k in [0, 1, 3, 6]:
    print("=" * 80)
    print(f"BUCKET: MATCH COUNT = {k} (5 SAMPLE ROWS)")
    print("=" * 80)
    for idx, (s1_id, m_list) in enumerate(sampled_per_bucket[k], 1):
        r1 = s1_lookup.get(s1_id, {})
        print(f"\n[{k}-Match Sample #{idx}] S1 ID: {s1_id} | Country: {r1.get('country')}")
        print(f"  S1 Name   : {r1.get('business_name')}")
        print(f"  S1 Address: {r1.get('business_address')}")
        if not m_list:
            print("  --> No matches (Singleton).")
        else:
            print(f"  --> Matched {len(m_list)} entities:")
            for j, mid in enumerate(m_list, 1):
                r2 = s23_lookup.get(mid, {})
                print(f"      [{j}] ID: {mid} | Country: {r2.get('country')}")
                print(f"          Name   : {r2.get('business_name')}")
                print(f"          Address: {r2.get('business_address')}")

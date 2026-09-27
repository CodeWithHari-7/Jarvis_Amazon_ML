import sys
import polars as pl
import numpy as np
from collections import Counter

sys.path.append('code/business_entity_resolution/src')
from blocking import normalize_record, BlockingIndex

print("=" * 80)
print("INSPECTING WHY 41.7% OF TRUE MATCHES WERE MISSED IN FULL CORPUS")
print("=" * 80)

# Load the evaluated US entities and check their scores in blocker
gt = pl.read_csv("dataset/train/train_ground_truth.tsv", separator='\t')
s1 = pl.read_csv("dataset/train/train_source1.tsv", separator='\t')
s2 = pl.read_csv("dataset/train/train_source2.tsv", separator='\t')
s3 = pl.read_csv("dataset/train/train_source3.tsv", separator='\t')

# Sample 100 US entities
s1_us = s1.filter(pl.col('country') == 'US')
s1_us_ids = set(s1_us['entity_id'].to_list())
gt_us = gt.filter(pl.col('source1_entity_id').is_in(s1_us_ids) & pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != '')).sample(100, seed=42)

s1_lookup = {r['entity_id']: r for r in s1.filter(pl.col('entity_id').is_in(set(gt_us['source1_entity_id'].to_list()))).to_dicts()}

all_target_ids = set()
for r in gt_us.to_dicts():
    all_target_ids.update(r['matched_entity_ids'].split(','))

s23_lookup = {}
for r in s2.filter(pl.col('entity_id').is_in(all_target_ids)).to_dicts():
    s23_lookup[r['entity_id']] = r
for r in s3.filter(pl.col('entity_id').is_in(all_target_ids)).to_dicts():
    s23_lookup[r['entity_id']] = r

print(f"Loaded 100 sample S1 entities and {len(all_target_ids)} true target entities.")

# Now analyze token and prefix frequencies on full US S2+S3
s2_us = s2.filter(pl.col('country') == 'US')
s3_us = s3.filter(pl.col('country') == 'US')
s23_us = pl.concat([s2_us, s3_us]).unique(subset=['entity_id'])

# Build blocker on full US
blocker = BlockingIndex(token_frequency_cap=300, num_frequency_cap=150)
eids = s23_us['entity_id'].to_list()
names = s23_us['business_name'].to_list()
addrs = s23_us['business_address'].to_list()
countries = s23_us['country'].to_list()

for i in range(len(eids)):
    rec = normalize_record(names[i], addrs[i], countries[i])
    blocker.add_record(eids[i], rec)

# Check the 100 sample entities
missed_pairs = []
found_pairs = []

for r in gt_us.to_dicts():
    s1_id = r['source1_entity_id']
    r1_raw = s1_lookup[s1_id]
    r1 = normalize_record(r1_raw['business_name'], r1_raw['business_address'], r1_raw['country'])
    cands = set(blocker.retrieve_candidates(r1, max_candidates=40))
    
    true_m = set(r['matched_entity_ids'].split(','))
    for mid in true_m:
        if mid in s23_lookup:
            r2_raw = s23_lookup[mid]
            r2 = normalize_record(r2_raw['business_name'], r2_raw['business_address'], r2_raw['country'])
            if mid in cands:
                found_pairs.append((s1_id, mid, r1, r2))
            else:
                missed_pairs.append((s1_id, mid, r1, r2, r1_raw, r2_raw))

print(f"Sample 100: Found {len(found_pairs)}, Missed {len(missed_pairs)} ({len(missed_pairs)/(len(found_pairs)+len(missed_pairs))*100:.1f}%)")

print("\n--- Detailed Audit of 10 Missed True Match Pairs ---")
for idx, (s1_id, mid, r1, r2, r1_raw, r2_raw) in enumerate(missed_pairs[:10], 1):
    print(f"\n[Missed #{idx}] S1: {s1_id} <---> True Target: {mid}")
    print(f"  S1 Raw  : Name='{r1_raw['business_name']}', Addr='{r1_raw['business_address']}'")
    print(f"  Tgt Raw : Name='{r2_raw['business_name']}', Addr='{r2_raw['business_address']}'")
    print(f"  S1 Norm : prefix4='{r1['prefix4']}', sorted='{r1['sorted_tokens']}', distinct={r1['distinct_tokens']}, nums={r1['nums']}")
    print(f"  Tgt Norm: prefix4='{r2['prefix4']}', sorted='{r2['sorted_tokens']}', distinct={r2['distinct_tokens']}, nums={r2['nums']}")
    
    # Check why signals didn't connect:
    # 1. Prefix:
    p_len = len(blocker.idx_prefix.get(('us', r1['prefix4']), []))
    print(f"    idx_prefix for '{r1['prefix4']}': size = {p_len}")
    # 2. Sorted:
    st_len = len(blocker.idx_sorted_tok.get(('us', r1['sorted_tokens']), []))
    print(f"    idx_sorted_tok for '{r1['sorted_tokens']}': size = {st_len}")
    # 3. Shared distinct tokens:
    shared_toks = r1['distinct_tokens'] & r2['distinct_tokens']
    print(f"    Shared distinct tokens: {shared_toks}")
    for tok in shared_toks:
        t_len = len(blocker.idx_any_tok.get(('us', tok), []))
        print(f"      freq of '{tok}' in idx_any_tok: {t_len} (cap is 300!)")
    # 4. Shared nums:
    shared_nums = r1['nums'] & r2['nums']
    print(f"    Shared nums: {shared_nums}")
    for num in shared_nums:
        n_len = len(blocker.idx_addr_num.get(('us', num), []))
        print(f"      freq of '{num}' in idx_addr_num: {n_len} (cap is 60!)")

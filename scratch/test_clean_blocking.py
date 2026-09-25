import sys, io, re, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import polars as pl
from collections import defaultdict
from indic_transliteration import sanscript

sys.path.append(r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker")
from src.cleaning.data_cleaning import clean_dataset
from src.normalization.normalization import normalize_dataset

# Load 1M rows of S2
s2_df = pl.read_csv(r"dataset\test\test_source2.tsv", separator='\t', ignore_errors=True, schema_overrides={'entity_id': pl.Utf8}, n_rows=1000000)
s2_norm = normalize_dataset(clean_dataset(s2_df)).select([
    'entity_id', 'business_name_normalized', 'business_address_numbers', 'country_normalized'
])

prefix_idx = defaultdict(list)
addr_num_idx = defaultdict(list)

eids = s2_norm['entity_id'].to_list()
names = s2_norm['business_name_normalized'].to_list()
countries = s2_norm['country_normalized'].to_list()
nums_list = s2_norm['business_address_numbers'].to_list()

for i in range(len(eids)):
    c = countries[i]
    n = names[i]
    if c and n: prefix_idx[f"{c}_{n[:4]}"].append(eids[i])
    for num in nums_list[i]:
        if c and len(num) >= 3:
            b = addr_num_idx[f"{c}_{num}"]
            if len(b) < 50: b.append(eids[i])

print(f"Index built: Prefix keys={len(prefix_idx):,}, Addr keys={len(addr_num_idx):,}")

# Test 10,000 S1
s1_test_sample = pl.read_csv(r"dataset\test\test_source1.tsv", separator='\t', ignore_errors=True, n_rows=10000)
s1_norm = normalize_dataset(clean_dataset(s1_test_sample)).to_dicts()

t0 = time.time()
candidates = 0
for r in s1_norm:
    c = r['country_normalized']
    n = r['business_name_normalized']
    cands = set()
    if c and n:
        for m in prefix_idx.get(f"{c}_{n[:4]}", []): cands.add(m)
    for num in r['business_address_numbers']:
        if c and len(num) >= 3:
            for m in addr_num_idx.get(f"{c}_{num}", []): cands.add(m)
    candidates += len(cands)

print(f"Generated {candidates:,} candidates for 10,000 S1 in {time.time()-t0:.2f}s ({candidates/10000:.1f} cands/S1)")

import sys, io, time, os, gc, psutil, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append(os.path.abspath('.'))
import polars as pl
import pandas as pd
import numpy as np
from collections import defaultdict
from rapidfuzz import fuzz, distance

def get_mem():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

print("=== BENCHMARK 1 CHUNK FRANCE TEST ===")
print(f"Initial RAM: {get_mem():.1f} MB")

t0 = time.time()
s2_fr = pl.read_csv('dataset/test/test_source2.tsv', separator='\t', schema_overrides={'entity_id': pl.Utf8}).filter(pl.col('country').str.to_lowercase().str.strip_chars() == 'france')
s3_fr = pl.read_csv('dataset/test/test_source3.tsv', separator='\t', schema_overrides={'entity_id': pl.Utf8}).filter(pl.col('country').str.to_lowercase().str.strip_chars() == 'france')

from src.cleaning.data_cleaning import clean_dataset
from src.normalization.normalization import normalize_dataset

s23_fr = pl.concat([s2_fr, s3_fr])
del s2_fr, s3_fr
gc.collect()

s23_norm = normalize_dataset(clean_dataset(s23_fr)).select([
    'entity_id', 'business_name_normalized', 'business_address_normalized',
    'business_address_numbers', 'is_indic'
])
del s23_fr
gc.collect()

# Convert S23 to columnar arrays for blazing fast random access
s23_eids = s23_norm['entity_id'].to_list()
s23_names = s23_norm['business_name_normalized'].to_list()
s23_addrs = s23_norm['business_address_normalized'].to_list()
s23_nums = s23_norm['business_address_numbers'].to_list()
s23_indics = s23_norm['is_indic'].to_numpy()
del s23_norm
gc.collect()

print(f"Loaded and normalized {len(s23_eids):,} France S23 records in {time.time()-t0:.2f}s. RAM: {get_mem():.1f} MB")

# Build inverted index
t0 = time.time()
prefix_idx = defaultdict(list)
addr_num_idx = defaultdict(list)

for idx, (name, nums) in enumerate(zip(s23_names, s23_nums)):
    if name and len(name) >= 4:
        prefix_idx[name[:4]].append(idx)
    for num in nums:
        if len(num) >= 2:
            addr_num_idx[num].append(idx)

print(f"Built indices in {time.time()-t0:.2f}s. RAM: {get_mem():.1f} MB")

# Load model
with open('dataset/processed/lgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load S1 sample (5,000 France entities)
s1_fr = pl.read_csv('dataset/test/test_source1.tsv', separator='\t', schema_overrides={'entity_id': pl.Utf8}).filter(pl.col('country').str.to_lowercase().str.strip_chars() == 'france').head(5000)
s1_norm = normalize_dataset(clean_dataset(s1_fr)).select([
    'entity_id', 'business_name_normalized', 'business_address_normalized',
    'business_address_numbers', 'is_indic'
])

s1_eids = s1_norm['entity_id'].to_list()
s1_names = s1_norm['business_name_normalized'].to_list()
s1_addrs = s1_norm['business_address_normalized'].to_list()
s1_nums = s1_norm['business_address_numbers'].to_list()
s1_indics = s1_norm['is_indic'].to_numpy()

# Query candidates
t0 = time.time()
cand_s1_indices = []
cand_s23_indices = []

for i, (name, nums) in enumerate(zip(s1_names, s1_nums)):
    cands_set = set()
    if name and len(name) >= 4:
        cands_set.update(prefix_idx.get(name[:4], [])[:100])
    for num in nums:
        if len(num) >= 2:
            cands_set.update(addr_num_idx.get(num, [])[:50])
    for j in cands_set:
        cand_s1_indices.append(i)
        cand_s23_indices.append(j)

t_cands = time.time() - t0
print(f"Generated {len(cand_s1_indices):,} candidates for 5,000 S1 in {t_cands:.2f}s (Avg: {len(cand_s1_indices)/5000:.1f} per S1)")

# Feature extraction
t0 = time.time()
feat_name_ratio = []
feat_name_token = []
feat_name_jw = []
feat_addr_ratio = []
feat_addr_token = []
feat_num_overlap = []
feat_cross_script = []

for i, j in zip(cand_s1_indices, cand_s23_indices):
    n1, n2 = s1_names[i], s23_names[j]
    a1, a2 = s1_addrs[i], s23_addrs[j]
    num1, num2 = set(s1_nums[i]), set(s23_nums[j])
    
    # name features
    if n1 and n2:
        feat_name_ratio.append(fuzz.ratio(n1, n2))
        feat_name_token.append(fuzz.token_set_ratio(n1, n2))
        feat_name_jw.append(distance.JaroWinkler.normalized_similarity(n1, n2))
    else:
        feat_name_ratio.append(0.0)
        feat_name_token.append(0.0)
        feat_name_jw.append(0.0)
        
    # addr features
    if a1 and a2:
        feat_addr_ratio.append(fuzz.ratio(a1, a2))
        feat_addr_token.append(fuzz.token_set_ratio(a1, a2))
    else:
        feat_addr_ratio.append(0.0)
        feat_addr_token.append(0.0)
        
    # overlap
    if not num1 and not num2:
        feat_num_overlap.append(0.0)
    else:
        feat_num_overlap.append(len(num1 & num2) / max(len(num1 | num2), 1))
        
    # cross script
    feat_cross_script.append(1 if s1_indics[i] != s23_indics[j] else 0)

t_feat = time.time() - t0
print(f"Extracted features for {len(cand_s1_indices):,} candidates in {t_feat:.2f}s ({len(cand_s1_indices)/t_feat:,.0f} pairs/sec)")

# LightGBM predict
t0 = time.time()
X = np.column_stack([
    feat_name_ratio, feat_name_token, feat_name_jw,
    feat_addr_ratio, feat_addr_token,
    feat_num_overlap, feat_cross_script
])
probs = model.predict_proba(X)[:, 1]
t_pred = time.time() - t0
print(f"LightGBM prediction in {t_pred:.2f}s ({len(probs)/t_pred:,.0f} pairs/sec)")

# Thresholding
threshold = 0.89
matches_mask = probs >= threshold
matched_s1 = [cand_s1_indices[idx] for idx in np.where(matches_mask)[0]]
matched_s23 = [cand_s23_indices[idx] for idx in np.where(matches_mask)[0]]

# Group
s1_matches = defaultdict(list)
for i, j in zip(matched_s1, matched_s23):
    s1_id = s1_eids[i]
    s23_id = s23_eids[j]
    if s23_id not in s1_matches[s1_id]:
        s1_matches[s1_id].append(s23_id)

print(f"Found matches for {len(s1_matches):,} / 5,000 S1 entities ({len(s1_matches)/5000*100:.1f}%)")
total_matched_ids = sum(len(v) for v in s1_matches.values())
print(f"Total matched S2/S3 entity links: {total_matched_ids:,}")
print(f"Total chunk elapsed time: {t_cands + t_feat + t_pred:.2f}s")
print(f"Final RAM: {get_mem():.1f} MB")

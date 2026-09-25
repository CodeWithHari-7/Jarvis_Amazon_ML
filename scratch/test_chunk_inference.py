import os, sys, time, gc, psutil, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import polars as pl
from collections import defaultdict
from indic_transliteration import sanscript
import re
import pandas as pd

def get_ram():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

print(f"Start: RAM {get_ram():.1f} MB")
t0 = time.time()

sys.path.append(r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker")
from src.cleaning.data_cleaning import clean_dataset
from src.normalization.normalization import normalize_dataset
from src.features.feature_extraction import extract_features_for_pair
from src.models.inference import load_model, predict

test_s2_path = r"dataset\test\test_source2.tsv"
test_s3_path = r"dataset\test\test_source3.tsv"

# Read S2/S3
s2_df = pl.read_csv(test_s2_path, separator='\t', ignore_errors=True, schema_overrides={'entity_id': pl.Utf8})
s2_norm = normalize_dataset(clean_dataset(s2_df)).select([
    'entity_id', 'business_name_normalized', 'business_address_normalized',
    'business_address_numbers', 'country_normalized', 'is_indic'
])
del s2_df; gc.collect()

s3_df = pl.read_csv(test_s3_path, separator='\t', ignore_errors=True, schema_overrides={'entity_id': pl.Utf8})
s3_norm = normalize_dataset(clean_dataset(s3_df)).select([
    'entity_id', 'business_name_normalized', 'business_address_normalized',
    'business_address_numbers', 'country_normalized', 'is_indic'
])
del s3_df; gc.collect()

s23_norm = pl.concat([s2_norm, s3_norm], how="align")
del s2_norm, s3_norm; gc.collect()
print(f"Loaded S23 ({len(s23_norm):,} rows) in {time.time()-t0:.2f}s | RAM: {get_ram():.1f} MB")

# Transliteration helper
re_dev = re.compile(r'[\u0900-\u097F]')
re_guj = re.compile(r'[\u0A80-\u0AFF]')
re_tel = re.compile(r'[\u0C00-\u0C7F]')
re_clean = re.compile(r'[^a-z0-9\s]')

def translit_text(text: str) -> str:
    if not text: return ""
    s = str(text)
    try:
        if re_dev.search(s): s = sanscript.transliterate(s, sanscript.DEVANAGARI, sanscript.ITRANS)
        if re_guj.search(s): s = sanscript.transliterate(s, sanscript.GUJARATI, sanscript.ITRANS)
        if re_tel.search(s): s = sanscript.transliterate(s, sanscript.TELUGU, sanscript.ITRANS)
    except Exception: pass
    s = s.lower()
    s = re_clean.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

# Build indices using integer row indices
t_idx = time.time()
eids = s23_norm['entity_id'].to_list()
names = s23_norm['business_name_normalized'].to_list()
addrs = s23_norm['business_address_normalized'].to_list()
nums_list = s23_norm['business_address_numbers'].to_list()
countries = s23_norm['country_normalized'].to_list()
is_indics = s23_norm['is_indic'].to_list()

del s23_norm; gc.collect()

prefix_idx = defaultdict(list)
translit_idx = defaultdict(list)
addr_num_idx = defaultdict(list)

MAX_ADDR_BUCKET = 500

for i in range(len(eids)):
    c = countries[i]
    n = names[i]
    if c and n:
        prefix_idx[f"{c}_{n[:4]}"].append(i)
        if is_indics[i]:
            nt = translit_text(n)
            if nt: translit_idx[f"{c}_{nt[:4]}"].append(i)
        else:
            translit_idx[f"{c}_{n[:4]}"].append(i)
    for num in nums_list[i]:
        if c:
            addr_num_idx[f"{c}_{num}"].append(i)

# Filter out address keys that exceed MAX_ADDR_BUCKET
pruned_addr_keys = 0
for k in list(addr_num_idx.keys()):
    if len(addr_num_idx[k]) > MAX_ADDR_BUCKET:
        del addr_num_idx[k]
        pruned_addr_keys += 1

print(f"Indices built in {time.time()-t_idx:.2f}s | Pruned high-frequency address keys: {pruned_addr_keys:,} | RAM: {get_ram():.1f} MB")

# Test on 25,000 S1 records
t_chunk = time.time()
test_s1_path = r"dataset\test\test_source1.tsv"
s1_test_sample = pl.read_csv(test_s1_path, separator='\t', ignore_errors=True, n_rows=25000)
s1_norm = normalize_dataset(clean_dataset(s1_test_sample)).to_dicts()

s1_names_translit = [translit_text(r['business_name_normalized']) if r['is_indic'] else r['business_name_normalized'] for r in s1_norm]

candidates = []
for idx_s1, r in enumerate(s1_norm):
    s1_id = r['entity_id']
    c = r['country_normalized']
    n = r['business_name_normalized']
    nt = s1_names_translit[idx_s1]
    
    cand_row_ids = set()
    if c and n:
        for row_i in prefix_idx.get(f"{c}_{n[:4]}", []): cand_row_ids.add(row_i)
    if c and nt:
        for row_i in translit_idx.get(f"{c}_{nt[:4]}", []): cand_row_ids.add(row_i)
    for num in r['business_address_numbers']:
        if c:
            for row_i in addr_num_idx.get(f"{c}_{num}", []): cand_row_ids.add(row_i)
            
    for row_i in cand_row_ids:
        candidates.append((idx_s1, row_i))

print(f"Generated {len(candidates):,} candidates for 25,000 S1 ({len(candidates)/25000:.1f} per S1) in {time.time()-t_chunk:.2f}s")

# Test feature extraction & inference on this sample
t_feat = time.time()
model = load_model(r"dataset\processed\lgb_model.pkl")
feat_cols = ['name_fuzz_ratio', 'name_token_set', 'name_jw', 
             'addr_fuzz_ratio', 'addr_token_set', 'addr_num_overlap', 'cross_script']

features = []
for idx_s1, row_i in candidates:
    r1 = s1_norm[idx_s1]
    r2 = {
        'business_name_normalized': names[row_i],
        'business_address_normalized': addrs[row_i],
        'business_address_numbers': nums_list[row_i],
        'is_indic': is_indics[row_i]
    }
    feat = extract_features_for_pair(r1, r2)
    feat['source1_entity_id'] = r1['entity_id']
    feat['source_entity_id'] = eids[row_i]
    features.append(feat)

feat_df = pd.DataFrame(features)
probs = predict(model, feat_df, feat_cols)
feat_df['prob'] = probs

matches = feat_df[feat_df['prob'] >= 0.89]
print(f"Feature extraction + ML inference for {len(candidates):,} pairs in {time.time()-t_feat:.2f}s")
print(f"Matches found: {len(matches):,} ({(len(matches)/25000):.2f} matches per S1)")
print(f"Total time for 25k chunk: {time.time()-t_chunk:.2f}s | RAM: {get_ram():.1f} MB")

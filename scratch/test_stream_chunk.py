import os, sys, time, gc, psutil, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import polars as pl
from collections import defaultdict
from indic_transliteration import sanscript
import re
import pandas as pd
import pickle

def get_ram():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

print(f"Initial RAM: {get_ram():.1f} MB")
t_start = time.time()

sys.path.append(r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker")
from src.cleaning.data_cleaning import clean_dataset
from src.normalization.normalization import normalize_dataset
from src.features.feature_extraction import extract_features_for_pair
from src.models.inference import load_model, predict

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

# 1. Ingest S2 and S3 in streaming batches
test_s2_path = r"dataset\test\test_source2.tsv"
test_s3_path = r"dataset\test\test_source3.tsv"

print("\n[1] Reading S2 & S3 into compact columnar store...")
t0 = time.time()

# We only need entity_id, name, addr, nums, country, is_indic
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

total_s23 = len(s23_norm)
print(f"  Loaded {total_s23:,} S2/S3 records in {time.time()-t0:.2f}s | RAM: {get_ram():.1f} MB")

# Extract columns as flat arrays
t0 = time.time()
eids = s23_norm['entity_id'].to_list()
names = s23_norm['business_name_normalized'].to_list()
addrs = s23_norm['business_address_normalized'].to_list()
nums_list = s23_norm['business_address_numbers'].to_list()
countries = s23_norm['country_normalized'].to_list()
is_indics = s23_norm['is_indic'].to_list()

del s23_norm; gc.collect()
print(f"  Extracted arrays in {time.time()-t0:.2f}s | RAM: {get_ram():.1f} MB")

# 2. Build Inverted Indices
print("\n[2] Building Inverted Indices (Pass 1 Prefix, Pass 2 AddrNum, Pass 3 Translit)...")
t0 = time.time()
prefix_idx = defaultdict(list)
translit_idx = defaultdict(list)
addr_num_idx = defaultdict(list)

MAX_ADDR_BUCKET = 500

for i in range(total_s23):
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
        # Filter single digits or over-populated buckets
        if c and len(num) >= 2:
            bucket = addr_num_idx[f"{c}_{num}"]
            if len(bucket) < MAX_ADDR_BUCKET:
                bucket.append(i)

t_idx_done = time.time() - t0
print(f"  Indices built in {t_idx_done:.2f}s | Prefix keys: {len(prefix_idx):,} | Translit keys: {len(translit_idx):,} | Addr keys: {len(addr_num_idx):,} | RAM: {get_ram():.1f} MB")

# 3. Test on first 50,000 S1 records
print("\n[3] Testing chunk processing on first 50,000 S1 records...")
t0 = time.time()
test_s1_path = r"dataset\test\test_source1.tsv"
s1_test_sample = pl.read_csv(test_s1_path, separator='\t', ignore_errors=True, n_rows=50000)
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
        if c and len(num) >= 2:
            for row_i in addr_num_idx.get(f"{c}_{num}", []): cand_row_ids.add(row_i)
            
    for row_i in cand_row_ids:
        candidates.append((idx_s1, row_i))

t_cand = time.time() - t0
print(f"  Generated {len(candidates):,} candidates ({len(candidates)/50000:.1f}/S1) in {t_cand:.2f}s")

# 4. Feature Extraction & Scoring
t0 = time.time()
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
t_score = time.time() - t0
print(f"  Scored {len(candidates):,} pairs in {t_score:.2f}s | Matches (prob >= 0.89): {len(matches):,}")
print(f"  Total time for 50k chunk: {t_cand + t_score:.2f}s | Extrapolated 1.73M time: {(t_cand + t_score) * 35 / 60:.1f} minutes")
print(f"  Peak RAM: {get_ram():.1f} MB")

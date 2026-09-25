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

# Load 1M rows of S2
s2_df = pl.read_csv(r"dataset\test\test_source2.tsv", separator='\t', ignore_errors=True, schema_overrides={'entity_id': pl.Utf8}, n_rows=1000000)
s2_norm = normalize_dataset(clean_dataset(s2_df)).select([
    'entity_id', 'business_name_normalized', 'business_address_normalized',
    'business_address_numbers', 'country_normalized', 'is_indic'
])
del s2_df; gc.collect()
print(f"S2 (1M rows) loaded in {time.time()-t0:.2f}s | RAM: {get_ram():.1f} MB")

# Build indices
t0 = time.time()
prefix_idx = defaultdict(list)
translit_idx = defaultdict(list)
addr_num_idx = defaultdict(list)

MAX_ADDR_BUCKET = 500

eids = s2_norm['entity_id'].to_list()
names = s2_norm['business_name_normalized'].to_list()
countries = s2_norm['country_normalized'].to_list()
nums_list = s2_norm['business_address_numbers'].to_list()
is_indics = s2_norm['is_indic'].to_list()

for i in range(len(eids)):
    eid = eids[i]
    c = countries[i]
    n = names[i]
    if c and n:
        prefix_idx[f"{c}_{n[:4]}"].append(eid)
        if is_indics[i]:
            nt = translit_text(n)
            if nt: translit_idx[f"{c}_{nt[:4]}"].append(eid)
        else:
            translit_idx[f"{c}_{n[:4]}"].append(eid)
    for num in nums_list[i]:
        if c and len(num) >= 2:
            bucket = addr_num_idx[f"{c}_{num}"]
            if len(bucket) < MAX_ADDR_BUCKET:
                bucket.append(eid)

print(f"Built indices in {time.time()-t0:.2f}s | RAM: {get_ram():.1f} MB")

# Test on 10,000 S1 records
t0 = time.time()
s1_test_sample = pl.read_csv(r"dataset\test\test_source1.tsv", separator='\t', ignore_errors=True, n_rows=10000)
s1_norm = normalize_dataset(clean_dataset(s1_test_sample)).to_dicts()

candidates = []
all_src_ids = set()

for r in s1_norm:
    s1_id = r['entity_id']
    c = r['country_normalized']
    n = r['business_name_normalized']
    nt = translit_text(n) if r['is_indic'] else n
    
    cands_set = set()
    if c and n:
        for m in prefix_idx.get(f"{c}_{n[:4]}", []): cands_set.add(m)
    if c and nt:
        for m in translit_idx.get(f"{c}_{nt[:4]}", []): cands_set.add(m)
    for num in r['business_address_numbers']:
        if c and len(num) >= 2:
            for m in addr_num_idx.get(f"{c}_{num}", []): cands_set.add(m)
            
    for m in cands_set:
        candidates.append((s1_id, r, m))
        all_src_ids.add(m)

print(f"Generated {len(candidates):,} candidate pairs (unique src entities: {len(all_src_ids):,}) in {time.time()-t0:.2f}s")

# Extract only the needed 10k-20k records from s2_norm!
t0 = time.time()
needed_s2 = s2_norm.filter(pl.col('entity_id').is_in(all_src_ids)).to_dicts()
s2_lookup = {r['entity_id']: r for r in needed_s2}
print(f"Filtered {len(s2_lookup):,} candidate records in {time.time()-t0:.2f}s | RAM: {get_ram():.1f} MB")

# Score with model
t0 = time.time()
model = load_model(r"dataset\processed\lgb_model.pkl")
feat_cols = ['name_fuzz_ratio', 'name_token_set', 'name_jw', 
             'addr_fuzz_ratio', 'addr_token_set', 'addr_num_overlap', 'cross_script']

features = []
for s1_id, r1, src_id in candidates:
    r2 = s2_lookup.get(src_id)
    if not r2: continue
    feat = extract_features_for_pair(r1, r2)
    feat['source1_entity_id'] = s1_id
    feat['source_entity_id'] = src_id
    features.append(feat)

feat_df = pd.DataFrame(features)
probs = predict(model, feat_df, feat_cols)
feat_df['prob'] = probs
matches = feat_df[feat_df['prob'] >= 0.89]

print(f"Feature extraction + ML scoring in {time.time()-t0:.2f}s | Matches: {len(matches):,} | Peak RAM: {get_ram():.1f} MB")

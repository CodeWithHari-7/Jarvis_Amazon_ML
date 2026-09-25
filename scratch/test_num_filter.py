import sys, io, re, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import polars as pl
from collections import defaultdict

base_dir = r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker\dataset"
cache_dir = os.path.join(base_dir, "pipeline_cache")

s1_df = pd.read_csv(os.path.join(cache_dir, "subset_s1.tsv"), sep='\t', dtype=str).fillna("")
s2_df = pd.read_csv(os.path.join(cache_dir, "subset_s2.tsv"), sep='\t', dtype=str).fillna("")
s3_df = pd.read_csv(os.path.join(cache_dir, "subset_s3.tsv"), sep='\t', dtype=str).fillna("")
gt_df = pd.read_csv(os.path.join(cache_dir, "subset_gt.tsv"), sep='\t', dtype=str).fillna("")

sys.path.append(r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker")
from src.cleaning.data_cleaning import clean_dataset
from src.normalization.normalization import normalize_dataset

s1_norm = normalize_dataset(clean_dataset(pl.from_pandas(s1_df))).to_dicts()
s2_norm = normalize_dataset(clean_dataset(pl.from_pandas(s2_df))).to_dicts()
s3_norm = normalize_dataset(clean_dataset(pl.from_pandas(s3_df))).to_dicts()
s23_norm = s2_norm + s3_norm

true_pairs = set()
for _, row in gt_df.iterrows():
    s1 = row['source1_entity_id']
    ms = row['matched_entity_ids']
    if ms:
        for m in ms.split(','):
            true_pairs.add((s1, m))

# Test different num filtering
for min_len in [1, 2, 3]:
    addr_num_idx = defaultdict(list)
    prefix_idx = defaultdict(list)
    for r in s23_norm:
        eid = r['entity_id']
        c = r.get('country_normalized', '')
        n = r.get('business_name_normalized', '')
        if c and n: prefix_idx[f"{c}_{n[:4]}"].append(eid)
        for num in r.get('business_address_numbers', []):
            if c and len(num) >= min_len:
                addr_num_idx[f"{c}_{num}"].append(eid)
                
    cands = set()
    for r in s1_norm:
        s1_id = r['entity_id']
        c = r.get('country_normalized', '')
        n = r.get('business_name_normalized', '')
        if c and n:
            for m in prefix_idx.get(f"{c}_{n[:4]}", []): cands.add((s1_id, m))
        for num in r.get('business_address_numbers', []):
            if c and len(num) >= min_len:
                for m in addr_num_idx.get(f"{c}_{num}", []): cands.add((s1_id, m))
                
    tp = true_pairs & cands
    rec = len(tp) / len(true_pairs)
    print(f"Min Num Length {min_len}: Candidates = {len(cands):,} | Recall = {rec*100:.2f}% ({len(tp)}/{len(true_pairs)})")

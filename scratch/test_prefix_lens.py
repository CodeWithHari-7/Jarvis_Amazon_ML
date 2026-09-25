import sys, io, re, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import polars as pl
from collections import defaultdict
from indic_transliteration import sanscript

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

re_dev = re.compile(r'[\u0900-\u097F]')
re_guj = re.compile(r'[\u0A80-\u0AFF]')
re_tel = re.compile(r'[\u0C00-\u0C7F]')
re_clean = re.compile(r'[^a-z0-9\s]')

def translit_text(text: str, target_scheme=sanscript.ITRANS) -> str:
    if not text: return ""
    s = str(text)
    try:
        if re_dev.search(s): s = sanscript.transliterate(s, sanscript.DEVANAGARI, target_scheme)
        if re_guj.search(s): s = sanscript.transliterate(s, sanscript.GUJARATI, target_scheme)
        if re_tel.search(s): s = sanscript.transliterate(s, sanscript.TELUGU, target_scheme)
    except Exception: pass
    s = s.lower()
    s = re_clean.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

for r in s1_norm:
    r['norm_name_translit'] = translit_text(r['business_name_normalized'])
for r in s23_norm:
    r['norm_name_translit'] = translit_text(r['business_name_normalized'])

true_pairs = set()
for _, row in gt_df.iterrows():
    s1 = row['source1_entity_id']
    ms = row['matched_entity_ids']
    if ms:
        for m in ms.split(','):
            true_pairs.add((s1, m))

# Baseline passes
p1_idx = defaultdict(list)
p2_idx = defaultdict(list)
for r in s23_norm:
    eid = r['entity_id']
    c = r.get('country_normalized', '')
    n = r.get('business_name_normalized', '')
    nums = r.get('business_address_numbers', [])
    if c and n: p1_idx[f"{c}_{n[:4]}"].append(eid)
    for num in nums:
        if c: p2_idx[f"{c}_{num}"].append(eid)

base_cands = set()
for r in s1_norm:
    s1_id = r['entity_id']
    c = r.get('country_normalized', '')
    n = r.get('business_name_normalized', '')
    nums = r.get('business_address_numbers', [])
    if c and n:
        for m in p1_idx.get(f"{c}_{n[:4]}", []): base_cands.add((s1_id, m))
    for num in nums:
        if c:
            for m in p2_idx.get(f"{c}_{num}", []): base_cands.add((s1_id, m))

print(f"Baseline: Cands={len(base_cands):,} | Recall={len(true_pairs & base_cands)/len(true_pairs)*100:.2f}%")

# Test prefix lengths 3, 4, 5
for pfx_len in [3, 4, 5]:
    t_idx = defaultdict(list)
    for r in s23_norm:
        eid = r['entity_id']
        c = r.get('country_normalized', '')
        nt = r.get('norm_name_translit', '')
        if c and len(nt) >= pfx_len:
            t_idx[f"{c}_{nt[:pfx_len]}"].append(eid)
            
    t_cands = set()
    for r in s1_norm:
        s1_id = r['entity_id']
        c = r.get('country_normalized', '')
        nt = r.get('norm_name_translit', '')
        if c and len(nt) >= pfx_len:
            for m in t_idx.get(f"{c}_{nt[:pfx_len]}", []):
                t_cands.add((s1_id, m))
                
    union = base_cands.union(t_cands)
    tp = true_pairs.intersection(union)
    rec = len(tp) / len(true_pairs)
    recov = tp - (true_pairs & base_cands)
    added_cands = union - base_cands
    print(f"\nTranslit Prefix Length {pfx_len}:")
    print(f"  Standalone Translit Cands: {len(t_cands):,}")
    print(f"  New Cands Added to Base  : {len(added_cands):,} (+{len(added_cands)/len(base_cands)*100:.2f}%)")
    print(f"  Total Union Candidates   : {len(union):,}")
    print(f"  Union Recall             : {rec*100:.2f}% ({len(tp)}/{len(true_pairs)})")
    print(f"  Pairs Recovered          : {len(recov)}")

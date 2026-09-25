import sys, io, re, os
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

s1_map = {r['entity_id']: r for r in s1_norm}
s23_map = {r['entity_id']: r for r in s23_norm}

re_devanagari = re.compile(r'[\u0900-\u097F]')
re_gujarati = re.compile(r'[\u0A80-\u0AFF]')
re_telugu = re.compile(r'[\u0C00-\u0C7F]')
re_clean = re.compile(r'[^a-z0-9\s]')

def translit_text(text: str, target_scheme=sanscript.ITRANS) -> str:
    if not text: return ""
    s = str(text)
    try:
        if re_devanagari.search(s):
            s = sanscript.transliterate(s, sanscript.DEVANAGARI, target_scheme)
        if re_gujarati.search(s):
            s = sanscript.transliterate(s, sanscript.GUJARATI, target_scheme)
        if re_telugu.search(s):
            s = sanscript.transliterate(s, sanscript.TELUGU, target_scheme)
    except Exception:
        pass
    s = s.lower()
    s = re_clean.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

for r in s1_norm:
    r['norm_name_translit'] = translit_text(r['business_name_normalized'])
for r in s23_norm:
    r['norm_name_translit'] = translit_text(r['business_name_normalized'])

# Build indices
prefix_idx = defaultdict(list)
addr_num_idx = defaultdict(list)
translit_idx = defaultdict(list)

for r in s23_norm:
    eid = r['entity_id']
    c = r.get('country_normalized', '')
    n = r.get('business_name_normalized', '')
    nt = r.get('norm_name_translit', '')
    if c and n: prefix_idx[f"{c}_{n[:4]}"].append(eid)
    if c and nt: translit_idx[f"{c}_{nt[:4]}"].append(eid)
    nums = r.get('business_address_numbers', [])
    for num in nums:
        if c: addr_num_idx[f"{c}_{num}"].append(eid)

base_cands = set()
translit_cands = set()

for r in s1_norm:
    s1_id = r['entity_id']
    c = r.get('country_normalized', '')
    n = r.get('business_name_normalized', '')
    nt = r.get('norm_name_translit', '')
    if c and n:
        for m in prefix_idx.get(f"{c}_{n[:4]}", []): base_cands.add((s1_id, m))
    nums = r.get('business_address_numbers', [])
    for num in nums:
        if c:
            for m in addr_num_idx.get(f"{c}_{num}", []): base_cands.add((s1_id, m))
    if c and nt:
        for m in translit_idx.get(f"{c}_{nt[:4]}", []): translit_cands.add((s1_id, m))

union_cands = base_cands.union(translit_cands)

# True pairs
true_pairs = set()
for _, row in gt_df.iterrows():
    s1 = row['source1_entity_id']
    ms = row['matched_entity_ids']
    if ms:
        for m in ms.split(','):
            true_pairs.add((s1, m))

base_tp = true_pairs.intersection(base_cands)
union_tp = true_pairs.intersection(union_cands)
recovered = union_tp - base_tp

print(f"Base Candidates: {len(base_cands):,}")
print(f"Translit Candidates (standalone Pass 3): {len(translit_cands):,}")
print(f"New Candidates added by Pass 3: {len(union_cands - base_cands):,}")
print(f"Total Union Candidates: {len(union_cands):,}")
print(f"Base TP: {len(base_tp):,} / {len(true_pairs):,} ({len(base_tp)/len(true_pairs)*100:.2f}%)")
print(f"Union TP: {len(union_tp):,} / {len(true_pairs):,} ({len(union_tp)/len(true_pairs)*100:.2f}%)")
print(f"Recovered True Pairs: {len(recovered)}")

print("\n--- RECOVERED PAIRS DETAILS ---")
for s1, m in recovered:
    r1 = s1_map[s1]
    r2 = s23_map[m]
    print(f"S1: {s1} ({r1['business_name']}) <--> Match: {m} ({r2['business_name']})")
    print(f"  S1 Translit: '{r1['norm_name_translit']}' | S2 Translit: '{r2['norm_name_translit']}'")

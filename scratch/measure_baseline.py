import sys, io, re, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import polars as pl
from collections import defaultdict

base_dir = r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker\dataset"
cache_dir = os.path.join(base_dir, "pipeline_cache")

# Load subsets
s1_df = pd.read_csv(os.path.join(cache_dir, "subset_s1.tsv"), sep='\t', dtype=str).fillna("")
s2_df = pd.read_csv(os.path.join(cache_dir, "subset_s2.tsv"), sep='\t', dtype=str).fillna("")
s3_df = pd.read_csv(os.path.join(cache_dir, "subset_s3.tsv"), sep='\t', dtype=str).fillna("")
gt_df = pd.read_csv(os.path.join(cache_dir, "subset_gt.tsv"), sep='\t', dtype=str).fillna("")

# Load src normalization and blocking functions
sys.path.append(r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker")
from src.cleaning.data_cleaning import clean_dataset
from src.normalization.normalization import normalize_dataset

# Clean and normalize exactly as production pipeline does
s1_clean = clean_dataset(pl.from_pandas(s1_df))
s2_clean = clean_dataset(pl.from_pandas(s2_df))
s3_clean = clean_dataset(pl.from_pandas(s3_df))

s1_norm = normalize_dataset(s1_clean).to_dicts()
s2_norm = normalize_dataset(s2_clean).to_dicts()
s3_norm = normalize_dataset(s3_clean).to_dicts()
s23_norm = s2_norm + s3_norm

# True pairs
true_pairs = set()
for _, row in gt_df.iterrows():
    s1 = row['source1_entity_id']
    ms = row['matched_entity_ids']
    if ms:
        for m in ms.split(','):
            true_pairs.add((s1, m))

print(f"Total S1 entities: {len(s1_norm):,}")
print(f"Total S2/S3 entities: {len(s23_norm):,}")
print(f"Total Ground Truth pairs: {len(true_pairs):,}")

# Baseline Blocking
t0 = time.time()
prefix_idx = defaultdict(list)
addr_num_idx = defaultdict(list)

for r in s23_norm:
    eid = r['entity_id']
    c = r.get('country_normalized', '')
    n = r.get('business_name_normalized', '')
    if c and n:
        prefix_idx[f"{c}_{n[:4]}"].append(eid)
    nums = r.get('business_address_numbers', [])
    for num in nums:
        if c:
            addr_num_idx[f"{c}_{num}"].append(eid)

base_candidates = set()
base_cands_pass1 = set()
base_cands_pass2 = set()

for r in s1_norm:
    s1_id = r['entity_id']
    c = r.get('country_normalized', '')
    n = r.get('business_name_normalized', '')
    if c and n:
        for m in prefix_idx.get(f"{c}_{n[:4]}", []):
            base_cands_pass1.add((s1_id, m))
            base_candidates.add((s1_id, m))
    nums = r.get('business_address_numbers', [])
    for num in nums:
        if c:
            for m in addr_num_idx.get(f"{c}_{num}", []):
                base_cands_pass2.add((s1_id, m))
                base_candidates.add((s1_id, m))

t_base = time.time() - t0
base_tp = true_pairs.intersection(base_candidates)
base_recall = len(base_tp) / len(true_pairs)
base_fn = true_pairs - base_candidates

print(f"\n--- BASELINE BLOCKING RESULTS ---")
print(f"Runtime: {t_base:.3f}s")
print(f"Pass 1 Candidates (Country+Prefix4): {len(base_cands_pass1):,}")
print(f"Pass 2 Candidates (Country+AddrNum): {len(base_cands_pass2):,}")
print(f"Total Unique Candidates: {len(base_candidates):,}")
print(f"True Pairs Recovered: {len(base_tp):,} / {len(true_pairs):,} ({base_recall*100:.2f}%)")
print(f"Blocking Failures (FNs): {len(base_fn):,}")

# Inspect baseline blocking failures: how many are cross-script?
s1_map = {r['entity_id']: r for r in s1_norm}
s23_map = {r['entity_id']: r for r in s23_norm}

re_indic = re.compile(r'[\u0900-\u097F\u0A80-\u0AFF\u0C00-\u0C7F]')
fn_cross_script = 0
fn_indic_devanagari = 0
fn_indic_telugu = 0
fn_indic_gujarati = 0

re_dev = re.compile(r'[\u0900-\u097F]')
re_tel = re.compile(r'[\u0C00-\u0C7F]')
re_guj = re.compile(r'[\u0A80-\u0AFF]')

for s1_id, m in base_fn:
    r1 = s1_map.get(s1_id, {})
    r2 = s23_map.get(m, {})
    n1 = r1.get('business_name', '')
    n2 = r2.get('business_name', '')
    i1 = bool(re_indic.search(n1))
    i2 = bool(re_indic.search(n2))
    if i1 != i2:
        fn_cross_script += 1
        if re_dev.search(n1) or re_dev.search(n2): fn_indic_devanagari += 1
        if re_tel.search(n1) or re_tel.search(n2): fn_indic_telugu += 1
        if re_guj.search(n1) or re_guj.search(n2): fn_indic_gujarati += 1

print(f"\n--- BASELINE BLOCKING FAILURE ANALYSIS ---")
print(f"Total Blocking Failures: {len(base_fn)}")
print(f"Cross-Script Blocking Failures: {fn_cross_script} ({fn_cross_script/len(base_fn)*100:.1f}%)")
print(f"  - Devanagari/Hindi: {fn_indic_devanagari}")
print(f"  - Telugu: {fn_indic_telugu}")
print(f"  - Gujarati: {fn_indic_gujarati}")
print(f"Same-Script Blocking Failures: {len(base_fn) - fn_cross_script}")

import sys, io, re, os, time, psutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import polars as pl
from collections import defaultdict
from indic_transliteration import sanscript

def get_ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

base_dir = r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker\dataset"
cache_dir = os.path.join(base_dir, "pipeline_cache")

t_start = time.time()
ram_start = get_ram_mb()

# 1. Load Data
s1_df = pd.read_csv(os.path.join(cache_dir, "subset_s1.tsv"), sep='\t', dtype=str).fillna("")
s2_df = pd.read_csv(os.path.join(cache_dir, "subset_s2.tsv"), sep='\t', dtype=str).fillna("")
s3_df = pd.read_csv(os.path.join(cache_dir, "subset_s3.tsv"), sep='\t', dtype=str).fillna("")
gt_df = pd.read_csv(os.path.join(cache_dir, "subset_gt.tsv"), sep='\t', dtype=str).fillna("")

sys.path.append(r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker")
from src.cleaning.data_cleaning import clean_dataset
from src.normalization.normalization import normalize_dataset

t_clean_start = time.time()
s1_clean = clean_dataset(pl.from_pandas(s1_df))
s2_clean = clean_dataset(pl.from_pandas(s2_df))
s3_clean = clean_dataset(pl.from_pandas(s3_df))

s1_norm = normalize_dataset(s1_clean).to_dicts()
s2_norm = normalize_dataset(s2_clean).to_dicts()
s3_norm = normalize_dataset(s3_clean).to_dicts()
s23_norm = s2_norm + s3_norm
t_norm = time.time() - t_clean_start

# Ground Truth
true_pairs = set()
for _, row in gt_df.iterrows():
    s1 = row['source1_entity_id']
    ms = row['matched_entity_ids']
    if ms:
        for m in ms.split(','):
            true_pairs.add((s1, m))

# 2. Transliteration Preprocessing
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

t_translit_start = time.time()
for r in s1_norm:
    r['norm_name_translit'] = translit_text(r['business_name_normalized'])
for r in s23_norm:
    r['norm_name_translit'] = translit_text(r['business_name_normalized'])
t_translit = time.time() - t_translit_start

# 3. Blocking: Baseline vs Transliteration
t_block_start = time.time()

# Indices
prefix_idx = defaultdict(list)
addr_num_idx = defaultdict(list)
translit_prefix_idx = defaultdict(list)

for r in s23_norm:
    eid = r['entity_id']
    c = r.get('country_normalized', '')
    n = r.get('business_name_normalized', '')
    nt = r.get('norm_name_translit', '')
    nums = r.get('business_address_numbers', [])
    if c and n: prefix_idx[f"{c}_{n[:4]}"].append(eid)
    for num in nums:
        if c: addr_num_idx[f"{c}_{num}"].append(eid)
    if c and nt: translit_prefix_idx[f"{c}_{nt[:4]}"].append(eid)

base_candidates = set()
translit_candidates = set()

s1_base_counts = defaultdict(int)
s1_union_counts = defaultdict(int)

for r in s1_norm:
    s1_id = r['entity_id']
    c = r.get('country_normalized', '')
    n = r.get('business_name_normalized', '')
    nt = r.get('norm_name_translit', '')
    nums = r.get('business_address_numbers', [])
    
    b_cands = set()
    if c and n:
        for m in prefix_idx.get(f"{c}_{n[:4]}", []): b_cands.add(m)
    for num in nums:
        if c:
            for m in addr_num_idx.get(f"{c}_{num}", []): b_cands.add(m)
            
    t_cands = set()
    if c and nt:
        for m in translit_prefix_idx.get(f"{c}_{nt[:4]}", []): t_cands.add(m)
        
    for m in b_cands: base_candidates.add((s1_id, m))
    for m in t_cands: translit_candidates.add((s1_id, m))
    
    s1_base_counts[s1_id] = len(b_cands)
    s1_union_counts[s1_id] = len(b_cands | t_cands)

union_candidates = base_candidates.union(translit_candidates)
t_blocking = time.time() - t_block_start
total_runtime = time.time() - t_start
peak_ram = get_ram_mb()

# Metrics
base_tp = true_pairs & base_candidates
union_tp = true_pairs & union_candidates
recovered = union_tp - base_tp

base_fn = true_pairs - base_candidates
union_fn = true_pairs - union_candidates

base_recall = len(base_tp) / len(true_pairs)
union_recall = len(union_tp) / len(true_pairs)

added_cands = union_candidates - base_candidates

# Candidate distribution
base_dist = list(s1_base_counts.values())
union_dist = list(s1_union_counts.values())

print("=" * 70)
print("EXPERIMENT RESULTS: TRANSLITERATION-BASED BLOCKING")
print("=" * 70)

print("\n--- A. CANDIDATE COUNT ---")
print(f"Baseline Candidate Count       : {len(base_candidates):,}")
print(f"Transliteration Pass Alone     : {len(translit_candidates):,}")
print(f"Union Candidates (Total)       : {len(union_candidates):,}")
print(f"Absolute Increase              : +{len(added_cands):,}")
print(f"Percentage Increase            : +{len(added_cands)/len(base_candidates)*100:.3f}%")

print("\n--- B. CANDIDATE RECALL ---")
print(f"Total Ground Truth Pairs       : {len(true_pairs):,}")
print(f"Baseline Candidate Recall      : {base_recall*100:.2f}% ({len(base_tp):,}/{len(true_pairs):,})")
print(f"Transliteration Candidate Recall: {union_recall*100:.2f}% ({len(union_tp):,}/{len(true_pairs):,})")
print(f"Recall Delta                   : +{(union_recall - base_recall)*100:.3f}%")
print(f"Recovered True Pairs           : {len(recovered):,}")
print(f"Baseline Blocking Failures     : {len(base_fn):,}")
print(f"Remaining Blocking Failures    : {len(union_fn):,}")

print("\n--- C. CANDIDATE EXPLOSION ---")
print(f"Baseline - Avg: {pd.Series(base_dist).mean():.2f} | Med: {pd.Series(base_dist).median():.0f} | Max: {max(base_dist)} | P95: {pd.Series(base_dist).quantile(0.95):.0f}")
print(f"Union    - Avg: {pd.Series(union_dist).mean():.2f} | Med: {pd.Series(union_dist).median():.0f} | Max: {max(union_dist)} | P95: {pd.Series(union_dist).quantile(0.95):.0f}")

# Check largest transliterated buckets
bucket_sizes = [(k, len(v)) for k, v in translit_prefix_idx.items()]
bucket_sizes.sort(key=lambda x: x[1], reverse=True)
print("\nTop 10 Largest Transliterated Prefix Buckets in S2/S3:")
for k, cnt in bucket_sizes[:10]:
    print(f"  Key '{k}': {cnt} entities")

print("\n--- D. RUNTIME / RESOURCE COST ---")
print(f"Transliteration Preprocessing Time: {t_translit:.4f}s")
print(f"Blocking Runtime                  : {t_blocking:.4f}s")
print(f"Total Pipeline Runtime (Data->Cands): {total_runtime:.4f}s")
print(f"Peak RAM Usage                    : {peak_ram:.2f} MB")

# Breakdown of recovered pairs
s1_map = {r['entity_id']: r for r in s1_norm}
s23_map = {r['entity_id']: r for r in s23_norm}

rec_dev = 0
rec_tel = 0
rec_guj = 0
rec_s2 = 0
rec_s3 = 0

for s1_id, m in recovered:
    r1 = s1_map[s1_id]
    r2 = s23_map[m]
    n1 = r1.get('business_name', '')
    n2 = r2.get('business_name', '')
    txt = n1 + " " + n2
    if re_dev.search(txt): rec_dev += 1
    if re_tel.search(txt): rec_tel += 1
    if re_guj.search(txt): rec_guj += 1
    if m.startswith("S2"): rec_s2 += 1
    elif m.startswith("S3"): rec_s3 += 1

print("\n--- E. BREAKDOWN OF RECOVERED PAIRS ---")
print(f"Total Recovered: {len(recovered)}")
print(f"  - Devanagari/Hindi : {rec_dev}")
print(f"  - Telugu           : {rec_tel}")
print(f"  - Gujarati         : {rec_guj}")
print(f"  - S2 Source        : {rec_s2}")
print(f"  - S3 Source        : {rec_s3}")

print("\n--- F. ERROR ANALYSIS DETAILS ---")
print("\n1. RECOVERED TRUE MATCHES (All 9):")
for i, (s1_id, m) in enumerate(recovered, 1):
    r1 = s1_map[s1_id]
    r2 = s23_map[m]
    print(f"[{i}] S1: {s1_id} | {m} ({r2.get('source')})")
    print(f"    S1: '{r1.get('business_name')}' -> '{r1.get('norm_name_translit')}' | Addr: '{r1.get('business_address')}'")
    print(f"    S2: '{r2.get('business_name')}' -> '{r2.get('norm_name_translit')}' | Addr: '{r2.get('business_address')}'")

# Transliteration collisions (New false positive candidates generated by transliteration)
collisions = added_cands - true_pairs
print(f"\n2. TRANSLITERATION COLLISIONS (Total: {len(collisions)}):")
collision_list = list(collisions)[:10]
for i, (s1_id, m) in enumerate(collision_list, 1):
    r1 = s1_map[s1_id]
    r2 = s23_map[m]
    print(f"[{i}] S1: {s1_id} ('{r1.get('business_name')}') <--> Match: {m} ('{r2.get('business_name')}')")
    print(f"    S1 Prefix: '{r1.get('norm_name_translit')[:4]}' | S2 Prefix: '{r2.get('norm_name_translit')[:4]}'")
    print(f"    S1 Addr: '{r1.get('business_address')}'")
    print(f"    S2 Addr: '{r2.get('business_address')}'")

# Entities with largest candidate buckets
print("\n3. S1 ENTITIES WITH LARGEST CANDIDATE GROUPS UNDER TRANSLITERATION:")
s1_translit_cands = defaultdict(set)
for s1_id, m in translit_candidates:
    s1_translit_cands[s1_id].add(m)

top_s1 = sorted(s1_translit_cands.items(), key=lambda x: len(x[1]), reverse=True)[:5]
for s1_id, cset in top_s1:
    r1 = s1_map[s1_id]
    print(f"S1: {s1_id} ('{r1.get('business_name')}') -> Translit: '{r1.get('norm_name_translit')}' (Prefix: '{r1.get('norm_name_translit')[:4]}') -> {len(cset)} translit cands")

# Remaining missed true matches
print("\n4. REMAINING MISSED TRUE MATCHES (Top 10 of 74):")
for i, (s1_id, m) in enumerate(list(union_fn)[:10], 1):
    r1 = s1_map[s1_id]
    r2 = s23_map[m]
    n1 = r1.get('business_name', '')
    n2 = r2.get('business_name', '')
    is_cross = bool(re_dev.search(n1+n2) or re_tel.search(n1+n2) or re_guj.search(n1+n2))
    print(f"[{i}] S1: {s1_id} | {m} (Cross-Script: {is_cross})")
    print(f"    S1: '{n1}' (Translit: '{r1.get('norm_name_translit')}') | Addr: '{r1.get('business_address')}'")
    print(f"    S2: '{n2}' (Translit: '{r2.get('norm_name_translit')}') | Addr: '{r2.get('business_address')}'")

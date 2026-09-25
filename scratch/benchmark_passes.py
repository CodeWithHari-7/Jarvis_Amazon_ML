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

# Build indices
p1_idx = defaultdict(list)
p2_idx = defaultdict(list)
p3_idx = defaultdict(list)

for r in s23_norm:
    eid = r['entity_id']
    c = r.get('country_normalized', '')
    n = r.get('business_name_normalized', '')
    nt = r.get('norm_name_translit', '')
    nums = r.get('business_address_numbers', [])
    if c and n: p1_idx[f"{c}_{n[:4]}"].append(eid)
    for num in nums:
        if c: p2_idx[f"{c}_{num}"].append(eid)
    if c and nt: p3_idx[f"{c}_{nt[:4]}"].append(eid)

def evaluate_passes(name, use_p1=False, use_p2=False, use_p3=False):
    t0 = time.time()
    cands = set()
    s1_counts = defaultdict(int)
    for r in s1_norm:
        s1_id = r['entity_id']
        c = r.get('country_normalized', '')
        n = r.get('business_name_normalized', '')
        nt = r.get('norm_name_translit', '')
        nums = r.get('business_address_numbers', [])
        
        cur = set()
        if use_p1 and c and n:
            for m in p1_idx.get(f"{c}_{n[:4]}", []): cur.add(m)
        if use_p2 and c:
            for num in nums:
                for m in p2_idx.get(f"{c}_{num}", []): cur.add(m)
        if use_p3 and c and nt:
            for m in p3_idx.get(f"{c}_{nt[:4]}", []): cur.add(m)
            
        s1_counts[s1_id] = len(cur)
        for m in cur:
            cands.add((s1_id, m))
            
    runtime = time.time() - t0
    tp = true_pairs.intersection(cands)
    recall = len(tp) / len(true_pairs)
    c_counts = list(s1_counts.values())
    avg_c = sum(c_counts)/len(c_counts)
    med_c = pd.Series(c_counts).median()
    max_c = max(c_counts) if c_counts else 0
    p95_c = pd.Series(c_counts).quantile(0.95)
    
    print(f"\n{name}:")
    print(f"  Candidates: {len(cands):,}")
    print(f"  Recall: {recall*100:.2f}% ({len(tp):,}/{len(true_pairs):,})")
    print(f"  Missed (FNs): {len(true_pairs) - len(tp)}")
    print(f"  Avg Cands/S1: {avg_c:.1f}, Median: {med_c:.0f}, Max: {max_c}, P95: {p95_c:.0f}")
    print(f"  Runtime: {runtime:.3f}s")
    return cands, tp

print("=" * 60)
print("PASS COMPARISON BENCHMARK")
print("=" * 60)

c_p1, tp_p1 = evaluate_passes("1. Pass 1 alone (Country + Name Prefix 4)", use_p1=True)
c_p3, tp_p3 = evaluate_passes("2. Pass 3 alone (Country + Translit Prefix 4)", use_p3=True)
c_p13, tp_p13 = evaluate_passes("3. Pass 1 + Pass 3 (Name Prefix + Translit Prefix)", use_p1=True, use_p3=True)
c_p2, tp_p2 = evaluate_passes("4. Pass 2 alone (Country + Address Numbers)", use_p2=True)
c_base, tp_base = evaluate_passes("5. Baseline Multi-Pass (Pass 1 UNION Pass 2)", use_p1=True, use_p2=True)
c_full, tp_full = evaluate_passes("6. Proposed Multi-Pass (Pass 1 UNION Pass 2 UNION Pass 3)", use_p1=True, use_p2=True, use_p3=True)

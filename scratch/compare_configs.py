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

# Regex definitions
scripts = [
    (re.compile(r'[\u0900-\u097F]'), sanscript.DEVANAGARI),
    (re.compile(r'[\u0A80-\u0AFF]'), sanscript.GUJARATI),
    (re.compile(r'[\u0C00-\u0C7F]'), sanscript.TELUGU),
    (re.compile(r'[\u0980-\u09FF]'), sanscript.BENGALI),
    (re.compile(r'[\u0A00-\u0A7F]'), sanscript.GURMUKHI),
    (re.compile(r'[\u0B80-\u0BFF]'), sanscript.TAMIL),
    (re.compile(r'[\u0C80-\u0CFF]'), sanscript.KANNADA),
    (re.compile(r'[\u0D00-\u0D7F]'), sanscript.MALAYALAM),
]
re_clean = re.compile(r'[^a-z0-9\s]')

def translit_all_indic(text):
    if not text: return ""
    s = str(text)
    for pattern, scheme in scripts:
        if pattern.search(s):
            try:
                s = sanscript.transliterate(s, scheme, sanscript.ITRANS)
            except Exception:
                pass
    s = s.lower()
    s = re_clean.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

def translit_min_indic(text):
    if not text: return ""
    s = str(text)
    for pattern, scheme in scripts[:3]: # Devanagari, Gujarati, Telugu
        if pattern.search(s):
            try:
                s = sanscript.transliterate(s, scheme, sanscript.ITRANS)
            except Exception:
                pass
    s = s.lower()
    s = re_clean.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

true_pairs = set()
for _, row in gt_df.iterrows():
    s1 = row['source1_entity_id']
    ms = row['matched_entity_ids']
    if ms:
        for m in ms.split(','):
            true_pairs.add((s1, m))

def test_config(translit_fn, name):
    t0 = time.time()
    for r in s1_norm: r['norm_name_translit'] = translit_fn(r['business_name_normalized'])
    for r in s23_norm: r['norm_name_translit'] = translit_fn(r['business_name_normalized'])
    t_trans = time.time() - t0
    
    t_idx = defaultdict(list)
    b_idx = defaultdict(list)
    a_idx = defaultdict(list)
    for r in s23_norm:
        eid = r['entity_id']
        c = r.get('country_normalized', '')
        n = r.get('business_name_normalized', '')
        nt = r.get('norm_name_translit', '')
        nums = r.get('business_address_numbers', [])
        if c and n: b_idx[f"{c}_{n[:4]}"].append(eid)
        for num in nums:
            if c: a_idx[f"{c}_{num}"].append(eid)
        if c and nt: t_idx[f"{c}_{nt[:4]}"].append(eid)
        
    base_cands = set()
    translit_cands = set()
    for r in s1_norm:
        s1_id = r['entity_id']
        c = r.get('country_normalized', '')
        n = r.get('business_name_normalized', '')
        nt = r.get('norm_name_translit', '')
        nums = r.get('business_address_numbers', [])
        if c and n:
            for m in b_idx.get(f"{c}_{n[:4]}", []): base_cands.add((s1_id, m))
        for num in nums:
            if c:
                for m in a_idx.get(f"{c}_{num}", []): base_cands.add((s1_id, m))
        if c and nt:
            for m in t_idx.get(f"{c}_{nt[:4]}", []): translit_cands.add((s1_id, m))
            
    union = base_cands | translit_cands
    tp = true_pairs & union
    rec = len(tp) / len(true_pairs)
    recov = tp - (true_pairs & base_cands)
    added = union - base_cands
    print(f"\n{name}:")
    print(f"  Translit Preproc Time: {t_trans:.3f}s")
    print(f"  New Candidates Added : {len(added):,} (+{len(added)/len(base_cands)*100:.3f}%)")
    print(f"  Total Candidates     : {len(union):,}")
    print(f"  Recall               : {rec*100:.2f}% ({len(tp)}/{len(true_pairs)})")
    print(f"  True Pairs Recovered : {len(recov)}")

test_config(translit_min_indic, "Configuration 1: Minimum (Devanagari, Gujarati, Telugu)")
test_config(translit_all_indic, "Configuration 2: Extended (All Indic Scripts)")

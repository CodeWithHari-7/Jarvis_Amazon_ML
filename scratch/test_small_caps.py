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

for r in s1_norm: r['norm_name_translit'] = translit_text(r['business_name_normalized'])
for r in s23_norm: r['norm_name_translit'] = translit_text(r['business_name_normalized'])

true_pairs = set()
for _, row in gt_df.iterrows():
    s1 = row['source1_entity_id']
    ms = row['matched_entity_ids']
    if ms:
        for m in ms.split(','):
            true_pairs.add((s1, m))

for cap in [5, 10, 20, 50]:
    p1_idx = defaultdict(list)
    p3_idx = defaultdict(list)
    p2_idx = defaultdict(list)
    for r in s23_norm:
        eid = r['entity_id']
        c = r.get('country_normalized', '')
        n = r.get('business_name_normalized', '')
        nt = r.get('norm_name_translit', '')
        if c and n: p1_idx[f"{c}_{n[:4]}"].append(eid)
        if c and nt: p3_idx[f"{c}_{nt[:4]}"].append(eid)
        for num in r.get('business_address_numbers', []):
            if c and len(num) >= 2:
                b = p2_idx[f"{c}_{num}"]
                if len(b) < cap: b.append(eid)
                
    cands = set()
    for r in s1_norm:
        s1_id = r['entity_id']
        c = r.get('country_normalized', '')
        n = r.get('business_name_normalized', '')
        nt = r.get('norm_name_translit', '')
        if c and n:
            for m in p1_idx.get(f"{c}_{n[:4]}", []): cands.add((s1_id, m))
        if c and nt:
            for m in p3_idx.get(f"{c}_{nt[:4]}", []): cands.add((s1_id, m))
        for num in r.get('business_address_numbers', []):
            if c and len(num) >= 2:
                for m in p2_idx.get(f"{c}_{num}", []): cands.add((s1_id, m))
                
    tp = true_pairs & cands
    rec = len(tp) / len(true_pairs)
    print(f"Cap {cap:2d}: Cands = {len(cands):6,} ({len(cands)/1500:5.1f}/S1) | Recall = {rec*100:.2f}% ({len(tp)}/{len(true_pairs)})")

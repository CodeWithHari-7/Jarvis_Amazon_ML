import sys, io, re, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import polars as pl
from collections import defaultdict
from indic_transliteration import sanscript

base_dir = r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker\dataset"
cache_dir = os.path.join(base_dir, "pipeline_cache")

# Load subsets
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

prefix_idx = defaultdict(list)
addr_num_idx = defaultdict(list)

for r in s23_norm:
    eid = r['entity_id']
    c = r.get('country_normalized', '')
    n = r.get('business_name_normalized', '')
    if c and n: prefix_idx[f"{c}_{n[:4]}"].append(eid)
    nums = r.get('business_address_numbers', [])
    for num in nums:
        if c: addr_num_idx[f"{c}_{num}"].append(eid)

base_candidates = set()
for r in s1_norm:
    s1_id = r['entity_id']
    c = r.get('country_normalized', '')
    n = r.get('business_name_normalized', '')
    if c and n:
        for m in prefix_idx.get(f"{c}_{n[:4]}", []): base_candidates.add((s1_id, m))
    nums = r.get('business_address_numbers', [])
    for num in nums:
        if c:
            for m in addr_num_idx.get(f"{c}_{num}", []): base_candidates.add((s1_id, m))

base_fn = true_pairs - base_candidates

s1_map = {r['entity_id']: r for r in s1_norm}
s23_map = {r['entity_id']: r for r in s23_norm}

re_devanagari = re.compile(r'[\u0900-\u097F]')
re_gujarati = re.compile(r'[\u0A80-\u0AFF]')
re_telugu = re.compile(r'[\u0C00-\u0C7F]')
re_indic = re.compile(r'[\u0900-\u097F\u0A80-\u0AFF\u0C00-\u0C7F]')
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

print(f"=== ALL {len(base_fn)} BASELINE BLOCKING FAILURES ===")
cross_fns = []
for s1_id, m in base_fn:
    r1 = s1_map.get(s1_id, {})
    r2 = s23_map.get(m, {})
    n1 = r1.get('business_name', '')
    n2 = r2.get('business_name', '')
    a1 = r1.get('business_address', '')
    a2 = r2.get('business_address', '')
    c1 = r1.get('country', '')
    c2 = r2.get('country', '')
    i1 = bool(re_indic.search(n1))
    i2 = bool(re_indic.search(n2))
    if i1 != i2:
        t1 = translit_text(n1)
        t2 = translit_text(n2)
        pfx4_match = (t1[:4] == t2[:4])
        pfx3_match = (t1[:3] == t2[:3])
        cross_fns.append({
            's1_id': s1_id, 'm_id': m,
            'n1': n1, 'n2': n2,
            't1': t1, 't2': t2,
            't1_4': t1[:4], 't2_4': t2[:4],
            'match4': pfx4_match,
            'match3': pfx3_match,
            'a1': a1, 'a2': a2
        })

print(f"\nCross-script blocking failures: {len(cross_fns)}")
for i, x in enumerate(cross_fns, 1):
    print(f"\n[{i}] S1: {x['s1_id']} | Match: {x['m_id']}")
    print(f"    S1 Name: {x['n1']} -> translit: '{x['t1']}' (prefix4: '{x['t1_4']}')")
    print(f"    S2 Name: {x['n2']} -> translit: '{x['t2']}' (prefix4: '{x['t2_4']}')")
    print(f"    Prefix4 Match: {x['match4']} | Prefix3 Match: {x['match3']}")
    print(f"    S1 Addr: {x['a1']}")
    print(f"    S2 Addr: {x['a2']}")

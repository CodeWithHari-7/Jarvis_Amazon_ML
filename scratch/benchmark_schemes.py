import sys, io, re, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
from indic_transliteration import sanscript

base_dir = r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker\dataset"
gt_sub = pd.read_csv(os.path.join(base_dir, "pipeline_cache", "subset_gt.tsv"), sep='\t', dtype=str).fillna("")
s1_sub = pd.read_csv(os.path.join(base_dir, "pipeline_cache", "subset_s1.tsv"), sep='\t', dtype=str).fillna("")
s2_sub = pd.read_csv(os.path.join(base_dir, "pipeline_cache", "subset_s2.tsv"), sep='\t', dtype=str).fillna("")
s3_sub = pd.read_csv(os.path.join(base_dir, "pipeline_cache", "subset_s3.tsv"), sep='\t', dtype=str).fillna("")

s1_lookup = s1_sub.set_index('entity_id').to_dict('index')
s23_lookup = pd.concat([s2_sub, s3_sub]).drop_duplicates('entity_id').set_index('entity_id').to_dict('index')

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
    except Exception as e:
        pass
    s = s.lower()
    s = re_clean.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

schemes = {
    'ITRANS': sanscript.ITRANS,
    'IAST': sanscript.IAST,
    'HK': sanscript.HK,
    'SLP1': sanscript.SLP1,
    'VELTHUIS': sanscript.VELTHUIS
}

true_pairs = []
for _, row in gt_sub.iterrows():
    s1 = row['source1_entity_id']
    ms = row['matched_entity_ids']
    if ms:
        for m in ms.split(','):
            if s1 in s1_lookup and m in s23_lookup:
                true_pairs.append((s1, m))

cross_pairs = []
for s1, m in true_pairs:
    r1 = s1_lookup[s1]
    r2 = s23_lookup[m]
    n1 = r1.get('business_name', '')
    n2 = r2.get('business_name', '')
    has_indic = bool(re_devanagari.search(n1) or re_gujarati.search(n1) or re_telugu.search(n1) or
                     re_devanagari.search(n2) or re_gujarati.search(n2) or re_telugu.search(n2))
    if has_indic:
        cross_pairs.append((s1, m, n1, n2))

print(f"Total True Pairs: {len(true_pairs)}")
print(f"Cross-script true pairs: {len(cross_pairs)}")

for sname, target_scheme in schemes.items():
    prefix_matches_4 = 0
    prefix_matches_3 = 0
    for s1, m, n1, n2 in cross_pairs:
        t1 = translit_text(n1, target_scheme)
        t2 = translit_text(n2, target_scheme)
        if t1[:4] == t2[:4] and len(t1) >= 4 and len(t2) >= 4:
            prefix_matches_4 += 1
        if t1[:3] == t2[:3] and len(t1) >= 3 and len(t2) >= 3:
            prefix_matches_3 += 1
    print(f"Scheme {sname:10}: prefix_4 matches = {prefix_matches_4}/{len(cross_pairs)} ({prefix_matches_4/len(cross_pairs)*100:.1f}%), prefix_3 = {prefix_matches_3}/{len(cross_pairs)} ({prefix_matches_3/len(cross_pairs)*100:.1f}%)")

print("\n--- SAMPLE TRANSLITERATIONS (ITRANS) ---")
for s1, m, n1, n2 in cross_pairs[:15]:
    t1 = translit_text(n1, sanscript.ITRANS)
    t2 = translit_text(n2, sanscript.ITRANS)
    match_4 = (t1[:4] == t2[:4])
    print(f"S1: {n1:<35} -> {t1:<35} | pfx: '{t1[:4]}'")
    print(f"S2: {n2:<35} -> {t2:<35} | pfx: '{t2[:4]}' | Match4: {match_4}")
    print("-" * 80)

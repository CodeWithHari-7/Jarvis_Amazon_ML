import sys, io, re, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

base_dir = r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker\dataset"

# Check subset_gt.tsv
gt_sub = pd.read_csv(os.path.join(base_dir, "pipeline_cache", "subset_gt.tsv"), sep='\t', dtype=str).fillna("")
s1_sub = pd.read_csv(os.path.join(base_dir, "pipeline_cache", "subset_s1.tsv"), sep='\t', dtype=str).fillna("")
s2_sub = pd.read_csv(os.path.join(base_dir, "pipeline_cache", "subset_s2.tsv"), sep='\t', dtype=str).fillna("")
s3_sub = pd.read_csv(os.path.join(base_dir, "pipeline_cache", "subset_s3.tsv"), sep='\t', dtype=str).fillna("")

print("Subset S1 rows:", len(s1_sub))
print("Subset S2 rows:", len(s2_sub))
print("Subset S3 rows:", len(s3_sub))
print("Subset GT rows:", len(gt_sub))

s1_names = dict(zip(s1_sub['entity_id'], s1_sub['business_name']))
s23_names = dict(zip(s2_sub['entity_id'], s2_sub['business_name']))
s23_names.update(dict(zip(s3_sub['entity_id'], s3_sub['business_name'])))

re_indic = re.compile(r'[\u0900-\u097F\u0A80-\u0AFF\u0C00-\u0C7F]')
re_dev = re.compile(r'[\u0900-\u097F]')
re_tel = re.compile(r'[\u0C00-\u0C7F]')
re_guj = re.compile(r'[\u0A80-\u0AFF]')

sub_true_pairs = []
for _, row in gt_sub.iterrows():
    s1 = row['source1_entity_id']
    ms = row['matched_entity_ids']
    if ms:
        for m in ms.split(','):
            sub_true_pairs.append((s1, m))

print(f"Total True Pairs in subset_gt: {len(sub_true_pairs)}")

cross_script_pairs = []
indic_s1_pairs = []
indic_src_pairs = []

for s1, m in sub_true_pairs:
    n1 = s1_names.get(s1, "")
    n2 = s23_names.get(m, "")
    is_indic1 = bool(re_indic.search(n1))
    is_indic2 = bool(re_indic.search(n2))
    if is_indic1: indic_s1_pairs.append((s1, m, n1, n2))
    if is_indic2: indic_src_pairs.append((s1, m, n1, n2))
    if is_indic1 != is_indic2:
        cross_script_pairs.append((s1, m, n1, n2))

print(f"Indic S1 pairs: {len(indic_s1_pairs)}")
print(f"Indic S2/S3 pairs: {len(indic_src_pairs)}")
print(f"Cross-script true pairs in subset_gt: {len(cross_script_pairs)}")

for s1, m, n1, n2 in cross_script_pairs[:10]:
    print(f"  S1: {s1} ({n1}) <--> Match: {m} ({n2})")

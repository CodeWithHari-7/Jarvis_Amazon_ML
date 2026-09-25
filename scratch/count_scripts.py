import sys, io, re, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

base_dir = r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker\dataset"
cache_dir = os.path.join(base_dir, "pipeline_cache")

s1_df = pd.read_csv(os.path.join(cache_dir, "subset_s1.tsv"), sep='\t', dtype=str).fillna("")
s2_df = pd.read_csv(os.path.join(cache_dir, "subset_s2.tsv"), sep='\t', dtype=str).fillna("")
s3_df = pd.read_csv(os.path.join(cache_dir, "subset_s3.tsv"), sep='\t', dtype=str).fillna("")
gt_df = pd.read_csv(os.path.join(cache_dir, "subset_gt.tsv"), sep='\t', dtype=str).fillna("")

s1_names = dict(zip(s1_df['entity_id'], s1_df['business_name']))
s23_names = dict(zip(s2_df['entity_id'], s2_df['business_name']))
s23_names.update(dict(zip(s3_df['entity_id'], s3_df['business_name'])))

re_dev = re.compile(r'[\u0900-\u097F]')
re_tel = re.compile(r'[\u0C00-\u0C7F]')
re_guj = re.compile(r'[\u0A80-\u0AFF]')

dev_count = 0
tel_count = 0
guj_count = 0
other_count = 0

for _, row in gt_df.iterrows():
    s1 = row['source1_entity_id']
    ms = row['matched_entity_ids']
    if ms:
        for m in ms.split(','):
            n1 = s1_names.get(s1, "")
            n2 = s23_names.get(m, "")
            text = n1 + " " + n2
            d = bool(re_dev.search(text))
            t = bool(re_tel.search(text))
            g = bool(re_guj.search(text))
            if d: dev_count += 1
            if t: tel_count += 1
            if g: guj_count += 1
            if not (d or t or g) and any(ord(c) > 127 for c in text):
                other_count += 1

print(f"Total True Pairs in subset_gt: {len(gt_df)}")
print(f"Devanagari true pairs: {dev_count}")
print(f"Telugu true pairs: {tel_count}")
print(f"Gujarati true pairs: {guj_count}")
print(f"Other non-ASCII: {other_count}")

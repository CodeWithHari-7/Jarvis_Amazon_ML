import pandas as pd
import numpy as np
import re
import random
from rapidfuzz import fuzz, distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
import os

base_dir = r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker\dataset"
gt_path = os.path.join(base_dir, "train", "train_ground_truth.tsv")

print("--- 1. BUILDING RESEARCH CANDIDATE SET ---")
gt_df = pd.read_csv(gt_path, sep='\t', dtype=str)
# Take random positive pairs
sample_gt = gt_df[gt_df['matched_entity_ids'].notna()].sample(2000, random_state=42)
s1_ids = set(sample_gt['source1_entity_id'])

target_s23 = set()
positive_pairs = set()
for _, row in sample_gt.iterrows():
    s1 = row['source1_entity_id']
    for m in str(row['matched_entity_ids']).split(','):
        target_s23.add(m)
        positive_pairs.add((s1, m))

print("Loading data...")
s1_data = []
try:
    for chunk in pd.read_csv(os.path.join(base_dir, "train", "train_source1.tsv"), sep='\t', dtype=str, chunksize=200000, on_bad_lines='skip'):
        s1_data.append(chunk[chunk['entity_id'].isin(s1_ids)])
except Exception as e:
    print(e)
s1_df = pd.concat(s1_data) if s1_data else pd.DataFrame()

s2_df = pd.read_csv(os.path.join(base_dir, "train", "train_source2.tsv"), sep='\t', dtype=str, nrows=50000, on_bad_lines='skip')
s3_df = pd.read_csv(os.path.join(base_dir, "train", "train_source3.tsv"), sep='\t', dtype=str, nrows=50000, on_bad_lines='skip')

s2_targets = set([t for t in target_s23 if t.startswith('S2-')])
s3_targets = set([t for t in target_s23 if t.startswith('S3-')])

missing_s2 = s2_targets - set(s2_df['entity_id'])
if missing_s2:
    s2_extra = []
    for chunk in pd.read_csv(os.path.join(base_dir, "train", "train_source2.tsv"), sep='\t', dtype=str, chunksize=200000, on_bad_lines='skip'):
        found = chunk[chunk['entity_id'].isin(missing_s2)]
        s2_extra.append(found)
    s2_df = pd.concat([s2_df] + s2_extra)

missing_s3 = s3_targets - set(s3_df['entity_id'])
if missing_s3:
    s3_extra = []
    for chunk in pd.read_csv(os.path.join(base_dir, "train", "train_source3.tsv"), sep='\t', dtype=str, chunksize=200000, on_bad_lines='skip'):
        found = chunk[chunk['entity_id'].isin(missing_s3)]
        s3_extra.append(found)
    s3_df = pd.concat([s3_df] + s3_extra)

s23_df = pd.concat([s2_df, s3_df])
s1_lookup = s1_df.set_index('entity_id').to_dict('index')
s23_lookup = s23_df.set_index('entity_id').to_dict('index')

print("Generating negative pairs...")
negative_pairs = set()

def norm_str(s): return re.sub(r'[^a-z0-9]', '', str(s).lower()) if not pd.isna(s) else ""
def get_prefix(s): return norm_str(s)[:3]

s23_blocks = defaultdict(list) if 'defaultdict' in globals() else {}
from collections import defaultdict
s23_blocks = defaultdict(list)
for eid, r in s23_lookup.items():
    s23_blocks[f"{str(r['country']).lower()}_{get_prefix(r['business_name'])}"].append(eid)

for s1_id, s1_r in s1_lookup.items():
    block_key = f"{str(s1_r['country']).lower()}_{get_prefix(s1_r['business_name'])}"
    cands = s23_blocks.get(block_key, [])
    for c in cands:
        if (s1_id, c) not in positive_pairs:
            negative_pairs.add((s1_id, c))
            if len(negative_pairs) > 10000: break
    if len(negative_pairs) > 10000: break

pos_list = list(positive_pairs)
neg_list = list(negative_pairs)[:len(pos_list)*3]

all_pairs = [(s1, s23, 1) for s1, s23 in pos_list] + [(s1, s23, 0) for s1, s23 in neg_list]
print(f"Created {len(pos_list)} positive and {len(neg_list)} negative pairs.")

print("\n--- 2 & 3. FEATURE EXTRACTION ---")
def extract_nums(s): return set(re.findall(r'\d+', str(s)))
def is_indic(s): return bool(re.search(r'[\u0900-\u097F\u0C00-\u0C7F]', str(s)))

features = []
for s1_id, s23_id, label in all_pairs:
    if s1_id not in s1_lookup or s23_id not in s23_lookup: continue
    r1, r2 = s1_lookup[s1_id], s23_lookup[s23_id]
    
    n1, n2 = str(r1['business_name']), str(r2['business_name'])
    a1, a2 = str(r1['business_address']), str(r2['business_address'])
    c1, c2 = str(r1['country']).strip().lower(), str(r2['country']).strip().lower()
    nn1, nn2 = norm_str(n1), norm_str(n2)
    
    num1, num2 = extract_nums(a1), extract_nums(a2)
    num_overlap = len(num1.intersection(num2)) / max(len(num1.union(num2)), 1)
    
    features.append({
        'label': label,
        'name_exact': 1 if n1.lower().strip() == n2.lower().strip() else 0,
        'name_fuzz_ratio': fuzz.ratio(n1.lower(), n2.lower()),
        'name_token_set': fuzz.token_set_ratio(n1.lower(), n2.lower()),
        'name_jw': distance.JaroWinkler.normalized_similarity(n1.lower(), n2.lower()),
        'addr_fuzz_ratio': fuzz.ratio(a1.lower(), a2.lower()),
        'addr_token_set': fuzz.token_set_ratio(a1.lower(), a2.lower()),
        'addr_num_overlap': num_overlap,
        'country_match': 1 if c1 == c2 else 0,
        'cross_script': 1 if is_indic(n1) != is_indic(n2) else 0
    })

df = pd.DataFrame(features)
print("\nPositive Pair Medians:")
print(df[df['label']==1].median().to_string())
print("\nNegative Pair Medians:")
print(df[df['label']==0].median().to_string())

print("\n--- 10. FEATURE ABLATION EXPERIMENT ---")
X = df.drop('label', axis=1).fillna(0)
y = df['label']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate(cols, name):
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train[cols], y_train)
    preds = clf.predict(X_test[cols])
    p, r = precision_score(y_test, preds, zero_division=0), recall_score(y_test, preds, zero_division=0)
    f05 = (1.25 * p * r) / (0.25 * p + r) if (p+r)>0 else 0
    print(f"{name:25s} -> Prec: {p:.3f} | Rec: {r:.3f} | F0.5: {f05:.3f}")

evaluate([c for c in X.columns if c.startswith('name_')], "Group A: Name Only")
evaluate([c for c in X.columns if c.startswith('addr_')], "Group B: Address Only")
evaluate(['country_match'], "Group C: Country Only")
evaluate([c for c in X.columns if c.startswith('name_') or c.startswith('addr_')], "Group D: Name + Address")
evaluate(list(X.columns), "Group I: All Features")

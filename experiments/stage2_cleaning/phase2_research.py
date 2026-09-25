import pandas as pd
import numpy as np
import os
import time
import re
from collections import defaultdict, Counter

base_dir = r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker\dataset"
gt_path = os.path.join(base_dir, "train", "train_ground_truth.tsv")

print("--- 1. GROUND TRUTH FORENSICS ---")
gt_df = pd.read_csv(gt_path, sep='\t', dtype=str)
total_s1 = len(gt_df)

s1_match_counts = []
total_positive_pairs = 0
s1_to_s2 = 0
s1_to_s3 = 0

s2_in_gt = []
s3_in_gt = []

for idx, row in gt_df.iterrows():
    matches = str(row['matched_entity_ids'])
    if pd.isna(row['matched_entity_ids']) or matches.strip() == "" or matches == "nan":
        s1_match_counts.append(0)
    else:
        m_list = matches.split(',')
        s1_match_counts.append(len(m_list))
        total_positive_pairs += len(m_list)
        for m in m_list:
            if m.startswith('S2-'):
                s1_to_s2 += 1
                s2_in_gt.append(m)
            elif m.startswith('S3-'):
                s1_to_s3 += 1
                s3_in_gt.append(m)

counts = Counter(s1_match_counts)
print(f"Total S1 entities: {total_s1}")
print(f"Total positive pairs: {total_positive_pairs}")
print(f"S1->S2 pairs: {s1_to_s2}")
print(f"S1->S3 pairs: {s1_to_s3}")

print(f"0 matches: {counts[0]} ({counts[0]/total_s1*100:.2f}%)")
print(f"1 match: {counts[1]} ({counts[1]/total_s1*100:.2f}%)")
print(f"2 matches: {counts[2]} ({counts[2]/total_s1*100:.2f}%)")
print(f"3+ matches: {sum(v for k,v in counts.items() if k>=3)} ({sum(v for k,v in counts.items() if k>=3)/total_s1*100:.2f}%)")
print(f"Max matches for one S1: {max(s1_match_counts)}")
print(f"Fraction of S1 with >= 1 match: {(total_s1 - counts[0])/total_s1*100:.2f}%")

s2_counts = Counter(s2_in_gt)
s3_counts = Counter(s3_in_gt)
print(f"Max occurrences of a single S2 ID in ground truth: {max(s2_counts.values() if s2_counts else [0])}")
print(f"Max occurrences of a single S3 ID in ground truth: {max(s3_counts.values() if s3_counts else [0])}")

print("\n--- 2. SAMPLING FOR ANALYSIS ---")
# Build a small universe to test blocking & match stats
sample_gt = gt_df[gt_df['matched_entity_ids'].notna()].sample(5000, random_state=42)
s1_ids = set(sample_gt['source1_entity_id'])
target_s2 = set()
target_s3 = set()
for _, row in sample_gt.iterrows():
    for m in str(row['matched_entity_ids']).split(','):
        if m.startswith('S2-'): target_s2.add(m)
        elif m.startswith('S3-'): target_s3.add(m)

print("Loading S1 sample...")
s1_data = []
for chunk in pd.read_csv(os.path.join(base_dir, "train", "train_source1.tsv"), sep='\t', dtype=str, chunksize=200000, on_bad_lines='skip'):
    s1_data.append(chunk[chunk['entity_id'].isin(s1_ids)])
s1_df = pd.concat(s1_data)

print("Loading S2/S3 sample (targets + random distractors)...")
s2_data = []
s3_data = []
# We take targets + first 50000 rows of each as distractors to form a realistic pool
distractor_limit = 50000
s2_distractors = 0
for chunk in pd.read_csv(os.path.join(base_dir, "train", "train_source2.tsv"), sep='\t', dtype=str, chunksize=100000, on_bad_lines='skip'):
    targets = chunk[chunk['entity_id'].isin(target_s2)]
    s2_data.append(targets)
    if s2_distractors < distractor_limit:
        distr = chunk[~chunk['entity_id'].isin(target_s2)].head(distractor_limit - s2_distractors)
        s2_data.append(distr)
        s2_distractors += len(distr)

s3_distractors = 0
for chunk in pd.read_csv(os.path.join(base_dir, "train", "train_source3.tsv"), sep='\t', dtype=str, chunksize=100000, on_bad_lines='skip'):
    targets = chunk[chunk['entity_id'].isin(target_s3)]
    s3_data.append(targets)
    if s3_distractors < distractor_limit:
        distr = chunk[~chunk['entity_id'].isin(target_s3)].head(distractor_limit - s3_distractors)
        s3_data.append(distr)
        s3_distractors += len(distr)

s2_df = pd.concat(s2_data)
s3_df = pd.concat(s3_data)
s23_df = pd.concat([s2_df, s3_df])

print(f"Sample universe: {len(s1_df)} S1, {len(s23_df)} S2/S3 candidates.")

print("\n--- 3. POSITIVE MATCH ANALYSIS ---")
def norm_str(s):
    if pd.isna(s): return ""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def is_indic(s):
    if pd.isna(s): return False
    # Check for Devanagari (Hindi) or Telugu ranges
    return bool(re.search(r'[\u0900-\u097F\u0C00-\u0C7F]', str(s)))

exact_name, norm_name, exact_addr, norm_addr = 0, 0, 0, 0
cross_lingual = 0
total_pairs = 0

s1_lookup = s1_df.set_index('entity_id').to_dict('index')
s23_lookup = s23_df.set_index('entity_id').to_dict('index')

for _, row in sample_gt.iterrows():
    s1_id = row['source1_entity_id']
    if s1_id not in s1_lookup: continue
    s1_r = s1_lookup[s1_id]
    matches = str(row['matched_entity_ids']).split(',')
    
    for m in matches:
        if m in s23_lookup:
            s23_r = s23_lookup[m]
            total_pairs += 1
            if str(s1_r['business_name']).strip().lower() == str(s23_r['business_name']).strip().lower():
                exact_name += 1
            if norm_str(s1_r['business_name']) == norm_str(s23_r['business_name']):
                norm_name += 1
            if str(s1_r['business_address']).strip().lower() == str(s23_r['business_address']).strip().lower():
                exact_addr += 1
            if norm_str(s1_r['business_address']) == norm_str(s23_r['business_address']):
                norm_addr += 1
            if is_indic(s1_r['business_name']) != is_indic(s23_r['business_name']):
                cross_lingual += 1

print(f"Total positive pairs analyzed: {total_pairs}")
print(f"Exact name match: {exact_name/total_pairs*100:.2f}%")
print(f"Normalized name match: {norm_name/total_pairs*100:.2f}%")
print(f"Exact address match: {exact_addr/total_pairs*100:.2f}%")
print(f"Normalized address match: {norm_addr/total_pairs*100:.2f}%")
print(f"Cross-lingual script difference: {cross_lingual/total_pairs*100:.2f}%")

print("\n--- 4. BLOCKING EXPERIMENT ---")
def run_blocking(name, block_func):
    start = time.time()
    # Create blocks for S2/S3
    s23_blocks = defaultdict(list)
    for eid, r in s23_lookup.items():
        keys = block_func(r)
        for k in keys:
            if k: s23_blocks[k].append(eid)
    
    # Generate candidates for S1
    generated_pairs = 0
    found_true_positives = 0
    max_cands = 0
    cands_per_s1 = []

    for s1_id, s1_r in s1_lookup.items():
        s1_keys = block_func(s1_r)
        cands = set()
        for k in s1_keys:
            if k in s23_blocks:
                cands.update(s23_blocks[k])
        
        cands_per_s1.append(len(cands))
        generated_pairs += len(cands)
        max_cands = max(max_cands, len(cands))
        
        # Check against ground truth
        gt_row = sample_gt[sample_gt['source1_entity_id'] == s1_id]
        if not gt_row.empty:
            true_matches = set(str(gt_row.iloc[0]['matched_entity_ids']).split(','))
            found_true_positives += len(cands.intersection(true_matches))

    recall = found_true_positives / total_pairs * 100 if total_pairs > 0 else 0
    brute_force = len(s1_lookup) * len(s23_lookup)
    reduction = (1 - (generated_pairs / brute_force)) * 100 if brute_force > 0 else 0
    
    print(f"Strategy: {name}")
    print(f"  Candidate Pairs generated: {generated_pairs}")
    print(f"  Reduction Ratio: {reduction:.4f}%")
    print(f"  Candidate Recall: {recall:.2f}% ({found_true_positives}/{total_pairs})")
    print(f"  Avg candidates per S1: {np.mean(cands_per_s1):.1f}")
    print(f"  Max candidates per S1: {max_cands}")
    print(f"  Runtime: {time.time() - start:.2f}s\n")

# A: Country exact match
run_blocking("A: Country Exact Match", lambda r: [str(r.get('country')).lower().strip()])

# B: Country + Norm Name Prefix (first 3 chars)
def block_b(r):
    c = str(r.get('country')).lower().strip()
    n = norm_str(r.get('business_name'))[:3]
    return [f"{c}_{n}"] if n else []
run_blocking("B: Country + Norm Name Prefix (3 chars)", block_b)

# C: Country + Any Name Token
def block_c(r):
    c = str(r.get('country')).lower().strip()
    tokens = str(r.get('business_name')).lower().split()
    return [f"{c}_{norm_str(t)}" for t in tokens if len(norm_str(t)) > 2]
run_blocking("C: Country + Any Name Token (>2 chars)", block_c)

# F: Multi-pass (Country+NamePrefix UNION Country+AddressNumber)
def block_f(r):
    keys = block_b(r)
    addr_nums = re.findall(r'\d+', str(r.get('business_address')))
    c = str(r.get('country')).lower().strip()
    keys.extend([f"{c}_addr_{num}" for num in addr_nums if len(num) >= 2])
    return keys
run_blocking("F: Multi-pass (Name Prefix UNION Address Number)", block_f)


import os
import pandas as pd
import numpy as np

print("=" * 70)
print("Creating Curated Pipeline Validation Subset")
print("=" * 70)

base_dir = r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker\dataset\train"
cache_dir = r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker\dataset\pipeline_cache"
os.makedirs(cache_dir, exist_ok=True)

# 1. Load Ground Truth
gt_df = pd.read_csv(os.path.join(base_dir, "train_ground_truth.tsv"), sep='\t', dtype=str)

# 2. Select S1 IDs representing different cases
# - 500 with 0 matches
# - 500 with 1 match
# - 500 with multiple matches

gt_df['match_count'] = gt_df['matched_entity_ids'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) and x != "" else 0)

np.random.seed(42)
s1_zero = gt_df[gt_df['match_count'] == 0].sample(500, random_state=42)
s1_single = gt_df[gt_df['match_count'] == 1].sample(500, random_state=42)
s1_multi = gt_df[gt_df['match_count'] > 1].sample(500, random_state=42)

curated_gt = pd.concat([s1_zero, s1_single, s1_multi])
s1_target_ids = set(curated_gt['source1_entity_id'])

# Extract all required S2/S3 IDs to ensure we can find the matches
required_src_ids = set()
for m in curated_gt['matched_entity_ids'].dropna():
    required_src_ids.update(str(m).split(','))

print(f"Curated S1 Entities: {len(s1_target_ids)}")
print(f"Required S2/S3 Matches to retain: {len(required_src_ids)}")

# 3. Read raw data and filter
print("Extracting from raw S1...")
s1_chunks = []
for chunk in pd.read_csv(os.path.join(base_dir, "train_source1.tsv"), sep='\t', dtype=str, chunksize=100000, on_bad_lines='skip'):
    s1_chunks.append(chunk[chunk['entity_id'].isin(s1_target_ids)])
s1_subset = pd.concat(s1_chunks)
s1_subset.to_csv(os.path.join(cache_dir, "subset_s1.tsv"), sep='\t', index=False)

print("Extracting from raw S2...")
s2_chunks = []
noise_s2 = []
for i, chunk in enumerate(pd.read_csv(os.path.join(base_dir, "train_source2.tsv"), sep='\t', dtype=str, chunksize=100000, on_bad_lines='skip')):
    s2_chunks.append(chunk[chunk['entity_id'].isin(required_src_ids)])
    # Add random noise
    if i < 5: noise_s2.append(chunk.sample(min(2000, len(chunk)), random_state=42))
s2_subset = pd.concat(s2_chunks + noise_s2).drop_duplicates(subset=['entity_id'])
s2_subset.to_csv(os.path.join(cache_dir, "subset_s2.tsv"), sep='\t', index=False)

print("Extracting from raw S3...")
s3_chunks = []
noise_s3 = []
for i, chunk in enumerate(pd.read_csv(os.path.join(base_dir, "train_source3.tsv"), sep='\t', dtype=str, chunksize=100000, on_bad_lines='skip')):
    s3_chunks.append(chunk[chunk['entity_id'].isin(required_src_ids)])
    # Add random noise
    if i < 5: noise_s3.append(chunk.sample(min(2000, len(chunk)), random_state=42))
s3_subset = pd.concat(s3_chunks + noise_s3).drop_duplicates(subset=['entity_id'])
s3_subset.to_csv(os.path.join(cache_dir, "subset_s3.tsv"), sep='\t', index=False)

# Save the curated GT
curated_gt.to_csv(os.path.join(cache_dir, "subset_gt.tsv"), sep='\t', index=False)

print("\nSubset Creation Complete:")
print(f"S1 subset size: {len(s1_subset)}")
print(f"S2 subset size: {len(s2_subset)}")
print(f"S3 subset size: {len(s3_subset)}")

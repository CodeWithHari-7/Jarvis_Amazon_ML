import os
import re
import sys
import time
from collections import Counter, defaultdict
import numpy as np
import polars as pl

print("=" * 80)
print("DEEP DIAGNOSIS OF HOLDOUT VS LEADERBOARD DISCREPANCY")
print("=" * 80)

# -------------------------------------------------------------
# 1. GROUND TRUTH MATCH-COUNT DISTRIBUTION & INDIA PARTITION LOSS
# -------------------------------------------------------------
print("\n--- 1. Ground Truth Analysis on Training Set ---")
t0 = time.time()
gt = pl.read_csv("dataset/train/train_ground_truth.tsv", separator='\t')
s1_tr = pl.read_csv("dataset/train/train_source1.tsv", separator='\t')
s2_tr = pl.read_csv("dataset/train/train_source2.tsv", separator='\t')
s3_tr = pl.read_csv("dataset/train/train_source3.tsv", separator='\t')

print(f"Train ground truth rows: {len(gt):,}")
print(f"Train S1 rows: {len(s1_tr):,}, S2: {len(s2_tr):,}, S3: {len(s3_tr):,}")

# Match count distribution in Ground Truth
gt_match_counts = []
gt_over_6 = 0
for r in gt.to_dicts():
    m = r['matched_entity_ids']
    if m and str(m).strip():
        n = len(str(m).split(','))
    else:
        n = 0
    gt_match_counts.append(n)
    if n > 6:
        gt_over_6 += 1

gt_match_counts = np.array(gt_match_counts)
print(f"GT Mean matches/entity: {np.mean(gt_match_counts):.2f}")
print(f"GT Singletons (0 matches): {np.sum(gt_match_counts == 0):,} ({np.mean(gt_match_counts == 0)*100:.2f}%)")
print(f"GT Entities with > 6 matches: {gt_over_6:,} ({gt_over_6/len(gt_match_counts)*100:.2f}%)")
print(f"GT Max matches for an entity: {np.max(gt_match_counts)}")
gt_hist = Counter(gt_match_counts)
print("GT Match Count Histogram (0 to 10):")
for k in range(11):
    print(f"  Matches = {k}: {gt_hist[k]:>8,} ({gt_hist[k]/len(gt_match_counts)*100:5.2f}%)")

# -------------------------------------------------------------
# 2. CHECK INDIA FIRST-LETTER PARTITION LOSS
# -------------------------------------------------------------
print("\n--- 2. Checking India First-Letter Partition Loss ---")
# Build lookup for S2 and S3 business names
s23_names = {}
for r in s2_tr.select(['entity_id', 'business_name']).to_dicts():
    s23_names[r['entity_id']] = r['business_name'] or ''
for r in s3_tr.select(['entity_id', 'business_name']).to_dicts():
    s23_names[r['entity_id']] = r['business_name'] or ''

p1_chars = set('0123456789abcdefghijklm')

# Check India ground truth pairs
s1_in_ids = set(s1_tr.filter(pl.col('country') == 'India')['entity_id'].to_list())
gt_in = gt.filter(pl.col('source1_entity_id').is_in(s1_in_ids))

s1_names = {r['entity_id']: r['business_name'] or '' for r in s1_tr.to_dicts()}

total_in_pairs = 0
cross_partition_pairs = 0
cross_entities_affected = 0

for r in gt_in.to_dicts():
    s1_id = r['source1_entity_id']
    m_str = r['matched_entity_ids']
    if not m_str or not str(m_str).strip():
        continue
    m_ids = str(m_str).split(',')
    s1_name = s1_names.get(s1_id, '')
    s1_char = s1_name[:1].lower() if s1_name else ''
    s1_is_p1 = s1_char in p1_chars

    entity_has_cross = False
    for cid in m_ids:
        total_in_pairs += 1
        c_name = s23_names.get(cid, '')
        c_char = c_name[:1].lower() if c_name else ''
        c_is_p1 = c_char in p1_chars
        if s1_is_p1 != c_is_p1:
            cross_partition_pairs += 1
            entity_has_cross = True
    if entity_has_cross:
        cross_entities_affected += 1

print(f"Total true India match pairs: {total_in_pairs:,}")
print(f"True India match pairs crossing A-M vs N-Z: {cross_partition_pairs:,} ({cross_partition_pairs/total_in_pairs*100:.2f}%)")
print(f"India entities with at least one cross-partition match lost: {cross_entities_affected:,} ({cross_entities_affected/len(gt_in)*100:.2f}%)")

# -------------------------------------------------------------
# 3. TEST CORPUS DISTRIBUTION & SIZES VS TRAIN
# -------------------------------------------------------------
print("\n--- 3. Dataset Sizes & Country Distribution: Train vs Test ---")
test_s1 = pl.read_csv("dataset/test/test_source1.tsv", separator='\t')
print(f"Test S1 entities: {len(test_s1):,}")
print("Test S1 country distribution:")
test_s1_counts = test_s1['country'].value_counts().to_dicts()
for row in test_s1_counts:
    c = row['country']
    cnt = row['count']
    print(f"  {c}: {cnt:,} ({cnt/len(test_s1)*100:.2f}%)")

print("Train S1 country distribution:")
train_s1_counts = s1_tr['country'].value_counts().to_dicts()
for row in train_s1_counts:
    c = row['country']
    cnt = row['count']
    print(f"  {c}: {cnt:,} ({cnt/len(s1_tr)*100:.2f}%)")

# Check test_source2 and test_source3 sizes and country distributions
print("\nScanning Test Source2 and Source3...")
for s_name, s_path in [("test_source2", "dataset/test/test_source2.tsv"), ("test_source3", "dataset/test/test_source3.tsv")]:
    size_mb = os.path.getsize(s_path) / (1024 * 1024)
    print(f"  {s_name} file size: {size_mb:.1f} MB")

# Check France specifically in matching_results.tsv
print("\n--- 4. Inspection of France in Current matching_results.tsv ---")
fr_s1_ids = set(test_s1.filter(pl.col('country').str.to_lowercase() == 'france')['entity_id'].to_list())
print(f"France test S1 count: {len(fr_s1_ids):,}")

fr_matches = []
with open("output/matching_results.tsv", 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.rstrip('\r\n').split('\t')
        eid = parts[0]
        if eid in fr_s1_ids:
            m = parts[1].split(',') if len(parts) > 1 and parts[1] else []
            fr_matches.append(len(m))

fr_matches = np.array(fr_matches)
print(f"France Mean matches/entity: {np.mean(fr_matches):.2f}")
print(f"France Singletons (0 matches): {np.sum(fr_matches == 0):,} ({np.mean(fr_matches == 0)*100:.2f}%)")
print(f"France 6 matches (capped): {np.sum(fr_matches == 6):,} ({np.mean(fr_matches == 6)*100:.2f}%)")

print(f"\nCompleted in {time.time()-t0:.1f}s.")

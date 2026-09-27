"""
Step 1: Diagnostic script for Business Entity Resolution over-matching analysis.
Computes ground truth vs test distributions and inspects top 20 highest-match rows.
"""

import polars as pl
from collections import Counter
import numpy as np

print("=" * 80)
print("STEP 1: DIAGNOSTIC COMPARISON (GROUND TRUTH VS CURRENT TEST OUTPUT)")
print("=" * 80)

# 1. Ground Truth Distribution
gt = pl.read_csv("dataset/train/train_ground_truth.tsv", separator='\t')
tot_gt = len(gt)

gt_match_counts = []
for m in gt['matched_entity_ids'].to_list():
    if not m or str(m).strip() == '':
        gt_match_counts.append(0)
    else:
        gt_match_counts.append(len(str(m).strip().split(',')))

gt_match_counts = np.array(gt_match_counts)
gt_singletons = np.sum(gt_match_counts == 0)
gt_matched = np.sum(gt_match_counts > 0)
gt_avg_all = np.mean(gt_match_counts)
gt_avg_nonzero = np.mean(gt_match_counts[gt_match_counts > 0])

# 2. Test Output Distribution
test_match_counts = []
test_eids = []
test_matches_dict = {}

with open("output/matching_results.tsv", 'r', encoding='utf-8') as f:
    hdr = f.readline()
    for line in f:
        parts = line.rstrip('\r\n').split('\t')
        eid = parts[0]
        test_eids.append(eid)
        if len(parts) < 2 or not parts[1].strip():
            test_match_counts.append(0)
            test_matches_dict[eid] = []
        else:
            m_list = [x.strip() for x in parts[1].split(',') if x.strip()]
            test_match_counts.append(len(m_list))
            test_matches_dict[eid] = m_list

test_match_counts = np.array(test_match_counts)
tot_test = len(test_match_counts)
test_singletons = np.sum(test_match_counts == 0)
test_matched = np.sum(test_match_counts > 0)
test_avg_all = np.mean(test_match_counts)
test_avg_nonzero = np.mean(test_match_counts[test_match_counts > 0])

print(f"\n[Summary Metrics Comparison]:")
print(f"{'Metric':<35} | {'Train Ground Truth':<20} | {'Current Test Output':<20}")
print("-" * 80)
print(f"{'Total S1 Entities':<35} | {tot_gt:>18,} | {tot_test:>18,}")
print(f"{'Singleton Count (0 matches)':<35} | {gt_singletons:>18,} | {test_singletons:>18,}")
print(f"{'Singleton Rate':<35} | {gt_singletons/tot_gt*100:>17.2f}% | {test_singletons/tot_test*100:>17.2f}%")
print(f"{'Matched Entities Count':<35} | {gt_matched:>18,} | {test_matched:>18,}")
print(f"{'Matched Entities Rate':<35} | {gt_matched/tot_gt*100:>17.2f}% | {test_matched/tot_test*100:>17.2f}%")
print(f"{'Avg Matches / Entity (All S1)':<35} | {gt_avg_all:>18.2f} | {test_avg_all:>18.2f}")
print(f"{'Avg Matches / Non-Zero Entity':<35} | {gt_avg_nonzero:>18.2f} | {test_avg_nonzero:>18.2f}")
print(f"{'Max Matches for an Entity':<35} | {np.max(gt_match_counts):>18} | {np.max(test_match_counts):>18}")

# 3. Match Count Distribution Buckets
print(f"\n[Match Count Detailed Frequency Distribution]:")
print(f"{'Matches / Entity':<18} | {'Train Ground Truth Count':<26} | {'Current Test Output Count':<26}")
print("-" * 80)

buckets = list(range(12)) + [999]  # 0 to 11, then 12+
for i in range(len(buckets) - 1):
    b = buckets[i]
    if b < 11:
        lbl = f"{b}"
        gt_c = np.sum(gt_match_counts == b)
        ts_c = np.sum(test_match_counts == b)
    else:
        lbl = ">= 11"
        gt_c = np.sum(gt_match_counts >= 11)
        ts_c = np.sum(test_match_counts >= 11)
    
    gt_pct = gt_c / tot_gt * 100
    ts_pct = ts_c / tot_test * 100
    print(f"{lbl:<18} | {gt_c:>10,} ({gt_pct:5.2f}%)          | {ts_c:>10,} ({ts_pct:5.2f}%)")

# 4. Pull 20 Highest-Match Rows from Current Output
print("\n" + "=" * 80)
print("INSPECTING 20 HIGHEST-MATCH-COUNT ROWS FROM CURRENT OUTPUT")
print("=" * 80)

# Sort test entities by match count descending
high_match_indices = np.argsort(-test_match_counts)[:20]
high_match_eids = [test_eids[idx] for idx in high_match_indices]

# Load S1 test info
s1_test_all = pl.read_csv("dataset/test/test_source1.tsv", separator='\t')
s1_high = s1_test_all.filter(pl.col('entity_id').is_in(high_match_eids)).to_dicts()
s1_high_map = {r['entity_id']: r for r in s1_high}

# Collect target S2/S3 IDs to load
needed_s23 = set()
for eid in high_match_eids:
    needed_s23.update(test_matches_dict[eid])

s2_test_all = pl.read_csv("dataset/test/test_source2.tsv", separator='\t').filter(pl.col('entity_id').is_in(needed_s23))
s3_test_all = pl.read_csv("dataset/test/test_source3.tsv", separator='\t').filter(pl.col('entity_id').is_in(needed_s23))
s23_map = {r['entity_id']: r for r in s2_test_all.to_dicts()}
s23_map.update({r['entity_id']: r for r in s3_test_all.to_dicts()})

for rank, s1_id in enumerate(high_match_eids, 1):
    s1_info = s1_high_map.get(s1_id, {})
    m_list = test_matches_dict[s1_id]
    print(f"\n--- [Rank {rank}] {s1_id} ({len(m_list)} matches) | Country: {s1_info.get('country')} ---")
    print(f"  S1 Business Name: {s1_info.get('business_name')}")
    print(f"  S1 Address      : {s1_info.get('business_address')}")
    print("  Matched S2/S3 Entities:")
    for mid in m_list[:6]:  # Show first 6 matches
        m_info = s23_map.get(mid, {})
        print(f"    * [{mid}] Name: {m_info.get('business_name')} | Addr: {m_info.get('business_address')}")
    if len(m_list) > 6:
        print(f"    ... and {len(m_list) - 6} more matches.")

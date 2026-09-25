import pandas as pd
import numpy as np
import os
import time

print("=" * 70)
print("FINAL SUBMISSION AUDIT: matching_results.tsv")
print("=" * 70)

res_file = "matching_results.tsv"
s1_test_file = r"dataset\test\test_source1.tsv"

# 1. Basic properties
print("\n[1] SCHEMA VALIDATION & DATA INTEGRITY")
try:
    # Read as pure strings to check for truncation or scientific notation
    df = pd.read_csv(res_file, sep='\t', dtype=str, keep_default_na=False)
    print("  PASS: File successfully parsed as TSV.")
except Exception as e:
    print(f"  FAIL: Could not parse TSV - {e}")
    exit(1)

cols = df.columns.tolist()
if cols == ['source1_entity_id', 'matched_entity_ids']:
    print("  PASS: Exact required column names found.")
else:
    print(f"  FAIL: Wrong columns: {cols}")

# Check data integrity
if df.isnull().any().any() or (df == "nan").any().any() or (df == "None").any().any():
    print("  FAIL: Found NaN/None literal strings or nulls.")
else:
    print("  PASS: No NaN/None strings found.")

# Sci notation / truncation check
if df['source1_entity_id'].str.contains(r'e\+').any():
    print("  FAIL: Scientific notation detected in S1 IDs.")
else:
    print("  PASS: No scientific notation in S1 IDs.")

# 2. S1 Coverage
print("\n[2] S1 COVERAGE")
t0 = time.time()
s1_test_df = pd.read_csv(s1_test_file, sep='\t', usecols=['entity_id'], dtype=str)
s1_test_ids = set(s1_test_df['entity_id'])
res_s1_ids = set(df['source1_entity_id'])
print(f"  (Loaded S1 ground truth in {time.time()-t0:.2f}s)")

if len(df) == len(df['source1_entity_id'].unique()):
    print("  PASS: No duplicate S1 IDs in submission.")
else:
    print(f"  FAIL: Found {len(df) - len(df['source1_entity_id'].unique())} duplicate S1 IDs.")

missing_s1 = s1_test_ids - res_s1_ids
extra_s1 = res_s1_ids - s1_test_ids

if not missing_s1 and not extra_s1:
    print(f"  PASS: Exact S1 coverage ({len(s1_test_ids):,} entities).")
else:
    print(f"  FAIL: Missing {len(missing_s1):,} S1 IDs. Extra {len(extra_s1):,} S1 IDs.")

# 3 & 4 & 6. Match ID Validation & Stats
print("\n[3] MATCH ID VALIDATION & STATISTICS")

zero_match = 0
single_match = 0
multi_match = 0
max_match = 0
total_matches = 0
s2_count = 0
s3_count = 0
invalid_ids = 0
dupes_in_pred = 0
s1_in_pred = 0

# Fast iteration over numpy/python arrays
s1_arr = df['source1_entity_id'].values
m_arr = df['matched_entity_ids'].values

for i in range(len(df)):
    s1 = s1_arr[i]
    m_str = m_arr[i]
    
    if not m_str:
        zero_match += 1
        continue
        
    m_list = m_str.split(',')
    m_len = len(m_list)
    
    if m_len == 1: single_match += 1
    else: multi_match += 1
    
    if m_len > max_match: max_match = m_len
    total_matches += m_len
    
    # Validation checks
    m_set = set(m_list)
    if len(m_set) < m_len: dupes_in_pred += 1
    
    for m in m_list:
        if m.startswith('S2'): s2_count += 1
        elif m.startswith('S3'): s3_count += 1
        else: invalid_ids += 1
        
        if m.startswith('S1'): s1_in_pred += 1

if dupes_in_pred == 0: print("  PASS: No duplicated IDs inside individual predictions.")
else: print(f"  FAIL: {dupes_in_pred} predictions contain duplicated IDs.")

if invalid_ids == 0: print("  PASS: All matched IDs start with S2 or S3.")
else: print(f"  FAIL: {invalid_ids} matched IDs are invalid.")

if s1_in_pred == 0: print("  PASS: No S1 IDs found in matched_entity_ids.")
else: print(f"  FAIL: {s1_in_pred} S1 IDs found in matched_entity_ids.")

print("\n[6] FINAL PIPELINE STATISTICS")
print(f"  Total S1 test entities     : {len(df):,}")
print(f"  Total predicted matches    : {total_matches:,}")
print(f"  S1 entities with 0 matches : {zero_match:,} ({(zero_match/len(df))*100:.1f}%)")
print(f"  S1 entities with 1 match   : {single_match:,} ({(single_match/len(df))*100:.1f}%)")
print(f"  S1 entities with >1 matches: {multi_match:,} ({(multi_match/len(df))*100:.1f}%)")
print(f"  Average matches per S1     : {total_matches/len(df):.4f}")
print(f"  Maximum matches per S1     : {max_match:,}")
print(f"  Matches to S2              : {s2_count:,}")
print(f"  Matches to S3              : {s3_count:,}")
print(f"  Output file size           : {os.path.getsize(res_file)/1024**2:.2f} MB")
print("\n=== AUDIT COMPLETE ===")

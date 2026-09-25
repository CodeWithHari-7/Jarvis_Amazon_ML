"""
JARVIS_CHECKER — Complete Test Inference Pipeline
Generates matching_results.tsv and candidate_pairs.tsv for submission.

This script:
1. Loads all test data (S1, S2, S3)
2. Normalizes all records
3. Builds blocking indexes from S2/S3
4. Processes S1 in chunks (memory efficient)
5. Generates candidates and features for each chunk
6. Runs ML inference
7. Applies optimized threshold
8. Writes matching_results.tsv (final matches)
9. Writes candidate_pairs.tsv (all candidates that went through model)
10. Performs integrity audit

CRITICAL: Every S1 entity MUST appear in matching_results.tsv exactly once.

Usage:
    python infer.py [--model dataset/processed/lgb_model.pkl]
                    [--threshold 0.89]
                    [--output-dir output/]
                    [--chunksize 50000]
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import time
import gc
import argparse
import numpy as np
import pandas as pd
import psutil
from collections import defaultdict

# ── Parse arguments ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='dataset/processed/lgb_model.pkl')
parser.add_argument('--threshold', type=float, default=None,
                    help='Decision threshold (default: read from dataset/processed/best_threshold.txt)')
parser.add_argument('--output-dir', type=str, default='output')
parser.add_argument('--chunksize', type=int, default=50000)
parser.add_argument('--test-dir', type=str, default='dataset/test')
args = parser.parse_args()

# ── Import pipeline modules ───────────────────────────────────────────────────
from src.cleaning.data_cleaning import load_and_clean_tsv
from src.normalization.normalization import normalize_record
from src.blocking.blocking import build_s23_index, generate_candidates_for_record
from src.features.feature_extraction import extract_features_for_pair, FEATURE_COLS
from src.models.inference import load_model

def get_mem():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

print("=" * 70)
print("JARVIS_CHECKER — TEST INFERENCE")
print("=" * 70)

# ── Paths ────────────────────────────────────────────────────────────────────
TEST_S1 = os.path.join(args.test_dir, 'test_source1.tsv')
TEST_S2 = os.path.join(args.test_dir, 'test_source2.tsv')
TEST_S3 = os.path.join(args.test_dir, 'test_source3.tsv')
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

MATCHING_OUT = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
CANDIDATES_OUT = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')

for path in [TEST_S1, TEST_S2, TEST_S3, args.model]:
    if not os.path.exists(path):
        print(f"ERROR: Required file not found: {path}")
        sys.exit(1)

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ── Threshold ────────────────────────────────────────────────────────────────
if args.threshold is not None:
    THRESHOLD = args.threshold
else:
    model_dir = os.path.dirname(args.model) if os.path.dirname(args.model) else '.'
    threshold_file_candidates = [
        os.path.join(model_dir, 'best_threshold.txt'),
        'dataset/processed/best_threshold.txt',
    ]
    THRESHOLD = None
    for cand_path in threshold_file_candidates:
        if os.path.exists(cand_path):
            try:
                with open(cand_path) as f:
                    THRESHOLD = float(f.read().strip())
                print(f"  Loaded threshold from {cand_path}: {THRESHOLD}")
                break
            except Exception:
                pass
    if THRESHOLD is None:
        THRESHOLD = 0.89  # fallback from DECISIONS.md
        print(f"  Using fallback threshold: {THRESHOLD}")
print(f"  Decision threshold: {THRESHOLD}")

# ── [1] Load model ────────────────────────────────────────────────────────────
print(f"\n[1] Loading model: {args.model} | Mem: {get_mem():.0f}MB")
model = load_model(args.model)
print(f"  Model loaded. Features expected: {len(FEATURE_COLS)}")

# ── [2] Load and normalize S2/S3 ─────────────────────────────────────────────
print(f"\n[2] Loading and normalizing S2/S3 | Mem: {get_mem():.0f}MB")
t0 = time.time()

s2_raw = load_and_clean_tsv(TEST_S2)
s3_raw = load_and_clean_tsv(TEST_S3)
print(f"  Raw: S2={len(s2_raw):,} | S3={len(s3_raw):,}")

s2_norm = [normalize_record(r) for r in s2_raw]
s3_norm = [normalize_record(r) for r in s3_raw]
del s2_raw, s3_raw; gc.collect()

s23_records = s2_norm + s3_norm
s23_ids = {r['entity_id'] for r in s23_records}
s23_lookup = {r['entity_id']: r for r in s23_records}
del s2_norm, s3_norm; gc.collect()

print(f"  Total S2/S3: {len(s23_records):,} | Time: {time.time()-t0:.1f}s | Mem: {get_mem():.0f}MB")

# ── [3] Build blocking indexes ────────────────────────────────────────────────
print(f"\n[3] Building blocking indexes | Mem: {get_mem():.0f}MB")
t0 = time.time()
indexes = build_s23_index(s23_records)
print(f"  Indexes built in {time.time()-t0:.1f}s | Mem: {get_mem():.0f}MB")

# ── [4] Load ALL S1 IDs first (for coverage guarantee) ───────────────────────
print(f"\n[4] Loading all S1 entity IDs | Mem: {get_mem():.0f}MB")
t0 = time.time()
s1_id_df = pd.read_csv(TEST_S1, sep='\t', dtype=str, usecols=['entity_id'],
                       keep_default_na=False, on_bad_lines='skip')
all_s1_ids = s1_id_df['entity_id'].tolist()
total_s1 = len(all_s1_ids)
print(f"  Total S1 entities: {total_s1:,}")

# ── [5] Process S1 in chunks ──────────────────────────────────────────────────
print(f"\n[5] Processing S1 in chunks (chunksize={args.chunksize:,}) | Mem: {get_mem():.0f}MB")

# Write output file headers
with open(MATCHING_OUT, 'w', encoding='utf-8') as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
with open(CANDIDATES_OUT, 'w', encoding='utf-8') as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")

# Track which S1 IDs have been written (to detect any missed ones)
written_s1_ids = set()

# Statistics
total_cands = 0
total_matches = 0
stats = {'0_match': 0, '1_match': 0, 'multi_match': 0, 'max_match': 0}
start_time = time.time()

chunk_reader = pd.read_csv(
    TEST_S1, sep='\t', dtype=str,
    keep_default_na=False, on_bad_lines='skip',
    chunksize=args.chunksize
)

for chunk_num, chunk_df in enumerate(chunk_reader):
    t_chunk = time.time()
    
    # Ensure entity_id column exists
    if 'entity_id' not in chunk_df.columns:
        print(f"  WARNING: chunk {chunk_num} missing entity_id column, skipping")
        continue
    
    # Clean and normalize chunk
    raw_records = [
        {col: (str(val) if val is not None else '') 
         for col, val in row.items()}
        for row in chunk_df.fillna('').to_dict('records')
    ]
    s1_norm_chunk = [normalize_record(r) for r in raw_records]
    chunk_s1_ids = [r['entity_id'] for r in s1_norm_chunk]
    
    # Per-S1 candidate and match collection
    s1_candidates = {}   # s1_id -> set of candidate S23 IDs
    s1_matches = {}      # s1_id -> list of matched S23 IDs
    
    # Blocking: generate candidates for each S1 in chunk
    all_pairs = []
    for r in s1_norm_chunk:
        s1_id = r['entity_id']
        cands = generate_candidates_for_record(r, indexes)
        s1_candidates[s1_id] = cands
        for cid in cands:
            all_pairs.append((s1_id, r, cid))
    
    chunk_cand_count = len(all_pairs)
    total_cands += chunk_cand_count
    
    # Feature extraction + inference (only if there are candidates)
    if all_pairs:
        features = []
        for s1_id, r1, cid in all_pairs:
            r2 = s23_lookup.get(cid)
            if not r2:
                continue
            feat = extract_features_for_pair(r1, r2)
            feat['source1_entity_id'] = s1_id
            feat['source_entity_id'] = cid
            features.append(feat)
        
        if features:
            feat_df = pd.DataFrame(features)
            # Ensure all feature columns present
            for col in FEATURE_COLS:
                if col not in feat_df.columns:
                    feat_df[col] = 0.0
            
            X = feat_df[FEATURE_COLS].fillna(0).values
            probs = model.predict_proba(X)[:, 1]
            feat_df['prob'] = probs
            
            # Apply threshold
            matched_df = feat_df[feat_df['prob'] >= THRESHOLD]
            grouped = matched_df.groupby('source1_entity_id')['source_entity_id'].apply(list).to_dict()
        else:
            grouped = {}
    else:
        grouped = {}
    
    # Write results for every S1 in this chunk
    match_lines = []
    cand_lines = []
    
    for r in s1_norm_chunk:
        s1_id = r['entity_id']
        written_s1_ids.add(s1_id)
        
        # Matched entities
        m_list = grouped.get(s1_id, [])
        # Deduplicate and validate
        m_set = []
        seen = set()
        for m in m_list:
            if m not in seen and not str(m).startswith('S1'):
                m_set.append(m)
                seen.add(m)
        
        match_count = len(m_set)
        total_matches += match_count
        
        if match_count == 0:
            stats['0_match'] += 1
        elif match_count == 1:
            stats['1_match'] += 1
        else:
            stats['multi_match'] += 1
        if match_count > stats['max_match']:
            stats['max_match'] = match_count
        
        match_str = ','.join(m_set)
        match_lines.append(f"{s1_id}\t{match_str}\n")
        
        # Candidate entities (ALL candidates that went through the model)
        cand_set_for_s1 = s1_candidates.get(s1_id, set())
        cand_str = ','.join(sorted(cand_set_for_s1))
        cand_lines.append(f"{s1_id}\t{cand_str}\n")
    
    # Flush to files
    with open(MATCHING_OUT, 'a', encoding='utf-8') as f:
        f.writelines(match_lines)
    with open(CANDIDATES_OUT, 'a', encoding='utf-8') as f:
        f.writelines(cand_lines)
    
    elapsed = time.time() - t_chunk
    total_processed = len(written_s1_ids)
    print(f"  Chunk {chunk_num+1}: S1={len(chunk_s1_ids):,} "
          f"| Cands={chunk_cand_count:,} "
          f"| Total processed={total_processed:,}/{total_s1:,} "
          f"| Time={elapsed:.1f}s | Mem={get_mem():.0f}MB")
    
    del raw_records, s1_norm_chunk, all_pairs
    if 'features' in dir():
        del features, feat_df
    gc.collect()

# ── [6] Coverage check and fix ────────────────────────────────────────────────
print(f"\n[6] Coverage check...")
missing_s1 = set(all_s1_ids) - written_s1_ids
print(f"  Written: {len(written_s1_ids):,} | Expected: {total_s1:,} | Missing: {len(missing_s1):,}")

if missing_s1:
    print(f"  WARNING: {len(missing_s1):,} S1 IDs were not written! Adding empty rows...")
    with open(MATCHING_OUT, 'a', encoding='utf-8') as f_m, \
         open(CANDIDATES_OUT, 'a', encoding='utf-8') as f_c:
        for s1_id in sorted(missing_s1):
            f_m.write(f"{s1_id}\t\n")
            f_c.write(f"{s1_id}\t\n")
            stats['0_match'] += 1
    print(f"  Fixed: added {len(missing_s1):,} empty rows")

# ── [7] Final statistics ──────────────────────────────────────────────────────
total_time = time.time() - start_time
print("\n" + "=" * 70)
print("POST-RUN STATISTICS")
print("=" * 70)
print(f"Total S1 entities            : {total_s1:,}")
print(f"Total candidate pairs        : {total_cands:,}")
print(f"Avg candidates per S1        : {total_cands/total_s1:.1f}")
print(f"Total predicted matches      : {total_matches:,}")
print(f"Avg matches per S1           : {total_matches/total_s1:.4f}")
print(f"Max matches for one S1       : {stats['max_match']:,}")
print("-" * 30)
print(f"S1 with 0 matches            : {stats['0_match']:,} ({100*stats['0_match']/total_s1:.1f}%)")
print(f"S1 with 1 match              : {stats['1_match']:,} ({100*stats['1_match']/total_s1:.1f}%)")
print(f"S1 with >1 matches           : {stats['multi_match']:,} ({100*stats['multi_match']/total_s1:.1f}%)")
print("-" * 30)
print(f"Output: {MATCHING_OUT}")
print(f"        {CANDIDATES_OUT}")
m_size = os.path.getsize(MATCHING_OUT)/1024**2
c_size = os.path.getsize(CANDIDATES_OUT)/1024**2
print(f"File sizes: matching={m_size:.2f}MB  candidates={c_size:.2f}MB")
print(f"Total runtime                : {total_time:.0f}s ({total_time/60:.1f}min)")
print("=" * 70)

# ── [8] Quick integrity audit ─────────────────────────────────────────────────
print("\n[8] Running quick integrity audit...")

match_df = pd.read_csv(MATCHING_OUT, sep='\t', dtype=str, keep_default_na=False)
cand_df_audit = pd.read_csv(CANDIDATES_OUT, sep='\t', dtype=str, keep_default_na=False)

issues = []

# Check column names
if list(match_df.columns) != ['source1_entity_id', 'matched_entity_ids']:
    issues.append(f"FAIL: Wrong column names in matching_results.tsv: {list(match_df.columns)}")
else:
    print("  PASS: matching_results.tsv columns correct")

if list(cand_df_audit.columns) != ['source1_entity_id', 'candidate_entity_ids']:
    issues.append(f"FAIL: Wrong column names in candidate_pairs.tsv: {list(cand_df_audit.columns)}")
else:
    print("  PASS: candidate_pairs.tsv columns correct")

# Row count
if len(match_df) == total_s1:
    print(f"  PASS: matching_results.tsv has {len(match_df):,} rows (= total S1)")
else:
    issues.append(f"FAIL: matching_results.tsv has {len(match_df):,} rows, expected {total_s1:,}")

# Duplicate S1 IDs
n_unique = match_df['source1_entity_id'].nunique()
if n_unique == len(match_df):
    print(f"  PASS: No duplicate S1 IDs in matching_results.tsv")
else:
    issues.append(f"FAIL: {len(match_df)-n_unique} duplicate S1 IDs in matching_results.tsv")

# Check all test S1 IDs present
s1_set = set(all_s1_ids)
result_s1_set = set(match_df['source1_entity_id'])
missing = s1_set - result_s1_set
extra = result_s1_set - s1_set
if not missing and not extra:
    print(f"  PASS: Exact S1 coverage ({total_s1:,} entities)")
else:
    if missing:
        issues.append(f"FAIL: {len(missing):,} test S1 IDs missing from submission")
    if extra:
        issues.append(f"FAIL: {len(extra):,} extra S1 IDs in submission not in test set")

# Check matched IDs validity
invalid_matches = 0
s1_in_matches = 0
for _, row in match_df.iterrows():
    m_str = row['matched_entity_ids']
    if not m_str:
        continue
    for m in m_str.split(','):
        if m.startswith('S1'):
            s1_in_matches += 1
        if not m.startswith('S2') and not m.startswith('S3'):
            invalid_matches += 1

if invalid_matches == 0:
    print("  PASS: All matched IDs start with S2 or S3")
else:
    issues.append(f"FAIL: {invalid_matches} matched IDs are invalid (not S2/S3)")
if s1_in_matches == 0:
    print("  PASS: No S1 IDs in matched_entity_ids")
else:
    issues.append(f"FAIL: {s1_in_matches} S1 IDs found in matched_entity_ids")

# NaN check
if match_df.isnull().any().any():
    issues.append("FAIL: NaN values found in matching_results.tsv")
else:
    print("  PASS: No NaN values in matching_results.tsv")

if issues:
    print("\n" + "=" * 70)
    print("INTEGRITY ISSUES FOUND:")
    for issue in issues:
        print(f"  {issue}")
    print("=" * 70)
    sys.exit(1)
else:
    print("\n" + "=" * 70)
    print("INTEGRITY AUDIT PASSED")
    print(f"  matching_results.tsv: {len(match_df):,} rows | {m_size:.2f}MB")
    print(f"  candidate_pairs.tsv:  {len(cand_df_audit):,} rows | {c_size:.2f}MB")
    print("  All checks passed. Ready for submission validation.")
    print("=" * 70)
    print("\nNext step: Run the official validator:")
    print(f"  python utils/validate_submission.py "
          f"--matching {MATCHING_OUT} "
          f"--candidate {CANDIDATES_OUT} "
          f"--test-dir {args.test_dir}")

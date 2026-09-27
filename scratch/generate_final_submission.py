"""
Production Test Submission Generator
ML Challenge 2026 — Business Entity Resolution

Generates output/candidate_pairs.tsv and output/matching_results.tsv from scratch
using the enhanced blocking (honorific stripping, any-token indexing, address numbers)
and the calibrated GPU matcher (threshold 0.85, hard vetoes, max 6 matches).
Maintains exact sequential order of test_source1.tsv and guarantees PASS on validate_submission.py.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
import os
import gc
import re
import time
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any

import polars as pl
import numpy as np

sys.path.append('code/business_entity_resolution/src')
from blocking import normalize_record, BlockingIndex, LEGAL_SUFFIXES, ABBREVIATIONS, STOPWORDS
from features import extract_pair_features
from predict import EntityMatcher

print("=" * 80)
print("ML CHALLENGE 2026: GENERATING PRODUCTION TEST SUBMISSION")
print("=" * 80)
t0_total = time.time()

cache_dir = "pipeline_cache"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

candidate_tsv = os.path.join(output_dir, "candidate_pairs.tsv")
matching_tsv = os.path.join(output_dir, "matching_results.tsv")

# Remove previous output files to ensure generating from scratch
for p in [candidate_tsv, matching_tsv]:
    if os.path.isfile(p):
        os.remove(p)
        print(f"Removed previous {p}.")

matcher = EntityMatcher("code/business_entity_resolution/src/model.pkl", threshold=0.85)
print(f"Loaded GPU EntityMatcher (Threshold: {matcher.threshold:.2f})")

# In-memory candidate & match dictionaries: {s1_id: cand_str}, {s1_id: match_str}
final_candidates: Dict[str, str] = {}
final_matches: Dict[str, str] = {}
country_match_stats: Dict[str, List[int]] = defaultdict(list)

def process_partition(
    part_name: str,
    s1_df: pl.DataFrame,
    s23_df: pl.DataFrame,
    batch_size: int = 15000,
    max_cands: int = 40
):
    t_start = time.time()
    n_s1 = len(s1_df)
    n_s23 = len(s23_df)
    print(f"\n[{part_name}] Processing {n_s1:,} S1 entities against {n_s23:,} candidate pool...")

    # 1. Build Index using columnar lists
    t_idx = time.time()
    s23_eids = s23_df['entity_id'].to_list()
    raw_names = s23_df['business_name'].to_list()
    raw_addrs = s23_df['business_address'].to_list()
    raw_countries = s23_df['country'].to_list()
    del s23_df
    gc.collect()

    blocker = BlockingIndex(token_frequency_cap=300, num_frequency_cap=150)
    s23_records = {}

    for i in range(n_s23):
        eid = s23_eids[i]
        rec = normalize_record(raw_names[i], raw_addrs[i], raw_countries[i])
        blocker.add_record(eid, rec)
        s23_records[eid] = rec

    del raw_names, raw_addrs, raw_countries
    gc.collect()
    print(f"  Indexed {n_s23:,} candidate records in {time.time()-t_idx:.1f}s.")

    # 2. Process S1 Entities in Batches
    t_score = time.time()
    s1_rows = s1_df.to_dicts()
    del s1_df
    gc.collect()

    part_matches_total = 0
    part_singletons_total = 0

    for batch_start in range(0, n_s1, batch_size):
        batch = s1_rows[batch_start:batch_start + batch_size]
        batch_pairs: List[Tuple[str, str]] = []
        batch_feats: List[List[float]] = []
        batch_s1_matches = defaultdict(list)
        batch_cands = {}

        for r1 in batch:
            s1_id = r1['entity_id']
            norm1 = normalize_record(r1['business_name'], r1['business_address'], r1['country'])

            # Retrieve top candidates
            cands = blocker.retrieve_candidates(norm1, max_candidates=max_cands)
            batch_cands[s1_id] = ",".join(cands)

            for cid in cands:
                rec2 = s23_records[cid]
                # Fast pre-filtering veto check (country mismatch)
                if norm1['country'] != rec2['country']:
                    continue
                batch_pairs.append((s1_id, cid))
                batch_feats.append(extract_pair_features(norm1, rec2))

        # Vectorized GPU Inference
        if batch_feats:
            X_batch = np.array(batch_feats, dtype=np.float32)
            probs = matcher.predict_probs(X_batch)

            for (s1_id, cid), prob in zip(batch_pairs, probs):
                p_val = float(prob)
                if p_val >= matcher.threshold:
                    batch_s1_matches[s1_id].append((cid, p_val))

        # Format Final Matches for Batch
        for r1 in batch:
            s1_id = r1['entity_id']
            c_name = str(r1.get('country', '')).strip()
            norm1 = normalize_record(r1['business_name'], r1['business_address'], r1['country'])

            scored_matches = batch_s1_matches.get(s1_id, [])
            
            # Apply hard vetoes (street number mismatch, core name mismatch)
            surviving = []
            for cid, p_val in scored_matches:
                rec2 = s23_records[cid]
                if not matcher.check_veto(norm1, rec2):
                    surviving.append((cid, p_val))

            # Sort by confidence descending and cap at max 6 matches
            surviving.sort(key=lambda x: x[1], reverse=True)
            top_m = [cid for cid, _ in surviving[:6]]

            final_candidates[s1_id] = batch_cands.get(s1_id, "")
            final_matches[s1_id] = ",".join(top_m)
            
            n_m = len(top_m)
            country_match_stats[c_name].append(n_m)
            part_matches_total += n_m
            if n_m == 0:
                part_singletons_total += 1

        done_so_far = min(batch_start + batch_size, n_s1)
        if done_so_far % 50000 < batch_size or done_so_far >= n_s1:
            print(f"    Progress: {done_so_far:,}/{n_s1:,} S1 entities scored ({time.time()-t_score:.1f}s)")

    del s23_eids, s23_records, s1_rows
    gc.collect()

    s_rate = (part_singletons_total / n_s1) * 100 if n_s1 else 0
    avg_m = part_matches_total / n_s1 if n_s1 else 0
    print(f"  [{part_name}] Complete in {time.time()-t_start:.1f}s: {part_matches_total:,} matches (avg {avg_m:.2f}/entity), {part_singletons_total:,} singletons ({s_rate:.2f}%)")


# =====================================================================
# 1. PROCESS FRANCE (259,452 entities)
# =====================================================================
print("\n" + "=" * 80)
print("STAGE 1: PROCESSING FRANCE PARTITION (259,452 ENTITIES)")
print("=" * 80)
s1_fr = pl.read_parquet(os.path.join(cache_dir, "s1_france.parquet"))
s2_fr = pl.read_parquet(os.path.join(cache_dir, "s2_france.parquet"))
s3_fr = pl.read_parquet(os.path.join(cache_dir, "s3_france.parquet"))
s23_fr = pl.concat([s2_fr, s3_fr]).unique(subset=['entity_id'])
del s2_fr, s3_fr
gc.collect()

process_partition("France", s1_fr, s23_fr, batch_size=15000, max_cands=40)
del s1_fr, s23_fr
gc.collect()

# =====================================================================
# 2. PROCESS US (663,106 entities)
# =====================================================================
print("\n" + "=" * 80)
print("STAGE 2: PROCESSING US PARTITION (663,106 ENTITIES)")
print("=" * 80)
s1_us = pl.read_parquet(os.path.join(cache_dir, "s1_us.parquet"))
s2_us = pl.read_parquet(os.path.join(cache_dir, "s2_us.parquet"))
s3_us = pl.read_parquet(os.path.join(cache_dir, "s3_us.parquet"))
s23_us = pl.concat([s2_us, s3_us]).unique(subset=['entity_id'])
del s2_us, s3_us
gc.collect()

process_partition("US", s1_us, s23_us, batch_size=15000, max_cands=40)
del s1_us, s23_us
gc.collect()

# =====================================================================
# 3. PROCESS INDIA (809,986 entities in 2 sub-partitions A-M and N-Z)
# =====================================================================
print("\n" + "=" * 80)
print("STAGE 3: PROCESSING INDIA PARTITION (809,986 ENTITIES)")
print("=" * 80)
p1_chars = list('0123456789abcdefghijklm')

s1_in = pl.read_parquet(os.path.join(cache_dir, "s1_india.parquet"))
s1_in_p1 = s1_in.filter(pl.col('business_name').str.slice(0, 1).str.to_lowercase().is_in(p1_chars))
s1_in_p2 = s1_in.filter(~pl.col('business_name').str.slice(0, 1).str.to_lowercase().is_in(p1_chars))
del s1_in
gc.collect()

# India Part 1 (A-M + 0-9)
s2_in_p1 = pl.read_parquet(os.path.join(cache_dir, "s2_india.parquet")).filter(
    pl.col('business_name').str.slice(0, 1).str.to_lowercase().is_in(p1_chars)
)
s3_in_p1 = pl.read_parquet(os.path.join(cache_dir, "s3_india.parquet")).filter(
    pl.col('business_name').str.slice(0, 1).str.to_lowercase().is_in(p1_chars)
)
s23_in_p1 = pl.concat([s2_in_p1, s3_in_p1]).unique(subset=['entity_id'])
del s2_in_p1, s3_in_p1
gc.collect()

process_partition("India Partition 1 (A-M)", s1_in_p1, s23_in_p1, batch_size=15000, max_cands=40)
del s1_in_p1, s23_in_p1
gc.collect()

# India Part 2 (N-Z + other)
s2_in_p2 = pl.read_parquet(os.path.join(cache_dir, "s2_india.parquet")).filter(
    ~pl.col('business_name').str.slice(0, 1).str.to_lowercase().is_in(p1_chars)
)
s3_in_p2 = pl.read_parquet(os.path.join(cache_dir, "s3_india.parquet")).filter(
    ~pl.col('business_name').str.slice(0, 1).str.to_lowercase().is_in(p1_chars)
)
s23_in_p2 = pl.concat([s2_in_p2, s3_in_p2]).unique(subset=['entity_id'])
del s2_in_p2, s3_in_p2
gc.collect()

process_partition("India Partition 2 (N-Z)", s1_in_p2, s23_in_p2, batch_size=15000, max_cands=40)
del s1_in_p2, s23_in_p2
gc.collect()

# =====================================================================
# 4. ASSEMBLE OUTPUT IN EXACT TEST_SOURCE1 ORDER
# =====================================================================
print("\n" + "=" * 80)
print("STAGE 4: ASSEMBLING OUTPUT FILES IN EXACT TEST_SOURCE1 ORDER")
print("=" * 80)
t_asm = time.time()

s1_test_path = "dataset/test/test_source1.tsv"
print(f"Streaming {s1_test_path} to write matching_results.tsv and candidate_pairs.tsv...")

total_written = 0
buffer_cand = []
buffer_match = []

with open(s1_test_path, 'r', encoding='utf-8') as f_in:
    with open(candidate_tsv, 'w', encoding='utf-8') as f_cand:
        with open(matching_tsv, 'w', encoding='utf-8') as f_match:
            f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
            f_match.write("source1_entity_id\tmatched_entity_ids\n")

            next(f_in, None)  # Skip header
            for line in f_in:
                parts = line.split('\t', 1)
                eid = parts[0].strip()

                c_str = final_candidates.get(eid, "")
                m_str = final_matches.get(eid, "")

                buffer_cand.append(f"{eid}\t{c_str}\n")
                buffer_match.append(f"{eid}\t{m_str}\n")
                total_written += 1

                if len(buffer_cand) >= 50000:
                    f_cand.writelines(buffer_cand)
                    f_match.writelines(buffer_match)
                    buffer_cand.clear()
                    buffer_match.clear()

            if buffer_cand:
                f_cand.writelines(buffer_cand)
                f_match.writelines(buffer_match)
                buffer_cand.clear()
                buffer_match.clear()

print(f"Successfully wrote {total_written:,} entity rows to output files in {time.time()-t_asm:.1f}s.")

# =====================================================================
# 5. SANITY-CHECK OUTPUT DISTRIBUTION
# =====================================================================
print("\n" + "=" * 80)
print("FINAL TEST-SET DISTRIBUTION SUMMARY (OVERALL & COUNTRY BREAKDOWN)")
print("=" * 80)
print(f"{'Country':<10} | {'Entities':<12} | {'Avg Matches/Entity':<22} | {'Singleton Rate':<18} | {'Max Matches':<12}")
print("-" * 80)

all_match_counts = []
for c_name in ['US', 'India', 'France']:
    counts = np.array(country_match_stats[c_name])
    all_match_counts.extend(counts)
    s_rate = np.mean(counts == 0) * 100
    avg_m = np.mean(counts)
    max_m = np.max(counts) if len(counts) else 0
    print(f"{c_name:<10} | {len(counts):<12,} | {avg_m:>20.2f} | {s_rate:>16.2f}% | {max_m:>12}")

print("-" * 80)
all_arr = np.array(all_match_counts)
overall_avg = np.mean(all_arr)
overall_single = np.mean(all_arr == 0) * 100
print(f"{'OVERALL':<10} | {len(all_arr):<12,} | {overall_avg:>20.2f} | {overall_single:>16.2f}% | {np.max(all_arr):>12}")

# France-only Match Frequency Distribution
print("\n[France Match-Count Frequency Distribution]:")
fr_counts = Counter(country_match_stats['France'])
for m_val in range(7):
    c_m = fr_counts.get(m_val, 0)
    pct = c_m / len(country_match_stats['France']) * 100 if country_match_stats['France'] else 0
    print(f"  Matches = {m_val}: {c_m:>8,} entities ({pct:5.2f}%)")

print("\n" + "=" * 80)
print(f"ALL STAGES COMPLETE IN {time.time()-t0_total:.1f}s.")
print("=" * 80)

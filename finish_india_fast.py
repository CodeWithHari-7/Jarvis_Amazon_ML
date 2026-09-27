"""
finish_india_fast.py

High-performance, memory-safe execution of India partition (809,986 entities)
using two partition blocks (A-M and N-Z) to maintain peak RAM < 1.8 GB and zero paging.
Uses GPU-accelerated XGBoost for candidate pair scoring.
"""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import gc
import re
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import polars as pl
import numpy as np

sys.path.append('code/business_entity_resolution/src')
from blocking import normalize_record, LEGAL_SUFFIXES, ABBREVIATIONS, STOPWORDS
from features import extract_pair_features
from predict import EntityMatcher


def process_sub_partition(
    part_name: str,
    s1_df_sub: pl.DataFrame,
    s23_sub: pl.DataFrame,
    matcher: EntityMatcher,
    batch_size: int = 15000,
    max_cands: int = 40
) -> Tuple[Dict[str, str], Dict[str, str], int, int]:
    """
    Indexes s23_sub, streams s1_df_sub, scores candidate pairs on GPU,
    and returns {s1_id: cand_str} and {s1_id: match_str}.
    """
    t_start = time.time()
    n_s1 = len(s1_df_sub)
    n_s23 = len(s23_sub)
    print(f"\n[{part_name}] Processing {n_s1:,} S1 entities against {n_s23:,} candidate pool...")

    # 1. Build compact integer index
    t_idx = time.time()
    s23_eids = s23_sub['entity_id'].to_list()
    raw_names = s23_sub['business_name'].to_list()
    raw_addrs = s23_sub['business_address'].to_list()
    del s23_sub
    gc.collect()

    compiled_abbrevs = [(re.compile(p), r) for p, r in ABBREVIATIONS]
    clean_re = re.compile(r'[^a-z0-9\s]')
    num_re = re.compile(r'\d+')

    s23_norm_names = []
    s23_norm_addrs = []

    idx_prefix = defaultdict(list)
    idx_prefix5 = defaultdict(list)
    idx_sorted_tok = defaultdict(list)
    idx_addr_num = defaultdict(list)

    for i in range(n_s23):
        # Name
        s = str(raw_names[i] or '').lower()
        for cp, rep in compiled_abbrevs: s = cp.sub(rep, s)
        clean_n = clean_re.sub(' ', s)
        toks = [w for w in clean_n.split() if w]
        core = [w for w in toks if w not in LEGAL_SUFFIXES]
        core_name = ' '.join(core) if core else ' '.join(toks)
        norm_name = ' '.join(toks)

        # Address
        sa = str(raw_addrs[i] or '').lower()
        for cp, rep in compiled_abbrevs: sa = cp.sub(rep, sa)
        clean_a = clean_re.sub(' ', sa)
        clean_a = ' '.join(clean_a.split())
        nums = [n for n in num_re.findall(clean_a) if len(n) <= 8]

        s23_norm_names.append(norm_name)
        s23_norm_addrs.append(clean_a)

        pfx4 = core_name[:4] if len(core_name) >= 3 else core_name
        pfx5 = core_name[:5] if len(core_name) >= 4 else core_name
        if pfx4: idx_prefix[pfx4].append(i)
        if pfx5: idx_prefix5[pfx5].append(i)

        clean_core = [w for w in core if w not in STOPWORDS]
        if clean_core:
            st = ' '.join(sorted(clean_core[:4]))
            idx_sorted_tok[st].append(i)

        for num in nums: idx_addr_num[num].append(i)

    del raw_names, raw_addrs
    gc.collect()
    print(f"  Index built in {time.time()-t_idx:.1f}s.")

    # 2. Stream S1 Entities in Batches and Score with GPU Model
    t_score = time.time()
    s1_rows = s1_df_sub.to_dicts()
    del s1_df_sub
    gc.collect()

    cand_map = {}
    match_map = {}
    total_cands = 0
    total_matches = 0
    total_blanks = 0

    for batch_start in range(0, n_s1, batch_size):
        batch = s1_rows[batch_start:batch_start + batch_size]
        batch_pairs = []
        batch_feats = []
        batch_s1_matches = defaultdict(list)
        batch_top_prob = defaultdict(float)
        batch_top_cand = defaultdict(str)

        for r1 in batch:
            s1_id = r1['entity_id']
            norm1 = normalize_record(r1['business_name'], r1['business_address'], r1['country'])

            cand_scores = defaultdict(int)
            p4 = norm1.get('prefix4', '')
            if p4:
                p_m = idx_prefix.get(p4, [])
                if len(p_m) <= 300:
                    for idx in p_m: cand_scores[idx] += 10
                else:
                    p5_m = idx_prefix5.get(norm1.get('prefix5', ''), [])
                    if p5_m and len(p5_m) <= 300:
                        for idx in p5_m: cand_scores[idx] += 10
                    else:
                        for idx in p_m[:100]: cand_scores[idx] += 8

            st = norm1.get('sorted_tokens', '')
            if st and st in idx_sorted_tok:
                st_m = idx_sorted_tok[st]
                if len(st_m) <= 300:
                    for idx in st_m: cand_scores[idx] += 10

            for num in norm1.get('nums', []):
                n_m = idx_addr_num.get(num, [])
                if len(n_m) <= 150:
                    for idx in n_m: cand_scores[idx] += 5

            top_idx = sorted(cand_scores.keys(), key=lambda x: cand_scores[x], reverse=True)[:max_cands]
            c_eids = [s23_eids[idx] for idx in top_idx]
            cand_map[s1_id] = ",".join(c_eids)
            total_cands += len(c_eids)

            for idx, c_eid in zip(top_idx, c_eids):
                rec2 = {
                    'norm_name': s23_norm_names[idx],
                    'norm_addr': s23_norm_addrs[idx],
                    'nums': set(n for n in num_re.findall(s23_norm_addrs[idx]) if len(n) <= 8),
                    'suffix_tokens': set(w for w in s23_norm_names[idx].split() if w in LEGAL_SUFFIXES)
                }
                batch_pairs.append((s1_id, c_eid))
                batch_feats.append(extract_pair_features(norm1, rec2))

        # Vectorized GPU inference
        if batch_feats:
            X_batch = np.array(batch_feats, dtype=np.float32)
            probs = matcher.predict_probs(X_batch)
            for (s1_id, c_eid), prob in zip(batch_pairs, probs):
                p_val = float(prob)
                if p_val >= matcher.threshold:
                    batch_s1_matches[s1_id].append((c_eid, p_val))

        # Format matches: confidence-sorted top 6 matches (NO 0.50 fallback!)
        for r1 in batch:
            s1_id = r1['entity_id']
            scored_matches = batch_s1_matches.get(s1_id, [])
            scored_matches.sort(key=lambda x: x[1], reverse=True)
            m_list = [c_eid for c_eid, _ in scored_matches[:6]]
            total_matches += len(m_list)
            if not m_list:
                total_blanks += 1
            match_map[s1_id] = ",".join(m_list)

        cur_done = min(batch_start + batch_size, n_s1)
        if cur_done % 50000 < batch_size or cur_done >= n_s1:
            print(f"    Progress: {cur_done:,}/{n_s1:,} entities ({time.time()-t_score:.1f}s)")

    # Cleanup memory
    del s23_eids, s23_norm_names, s23_norm_addrs, idx_prefix, idx_prefix5, idx_addr_num, idx_sorted_tok, s1_rows
    gc.collect()

    print(f"  [{part_name}] Finished in {time.time()-t_start:.1f}s: {total_matches:,} matches, {total_blanks:,} blanks ({total_blanks/n_s1*100:.2f}%)")
    return cand_map, match_map, total_cands, total_matches


def run_pipeline_india_fast(
    cache_dir: str = "pipeline_cache",
    output_dir: str = "output",
    model_path: str = "code/business_entity_resolution/src/model.pkl"
):
    print("=" * 75)
    print("HIGH-THROUGHPUT PIPELINE: COMPLETING FULL INDIA PARTITION (809,986 ROWS)")
    print("=" * 75)
    t0_all = time.time()

    cand_tsv = os.path.join(output_dir, "candidate_pairs.tsv")
    match_tsv = os.path.join(output_dir, "matching_results.tsv")

    # Verify baseline France + US state
    with open(match_tsv, 'r', encoding='utf-8') as f:
        curr_lines = sum(1 for _ in f)
    print(f"Current output state: {curr_lines:,} lines in matching_results.tsv (expected 922,559).")
    assert curr_lines == 922559, f"Unexpected line count in {match_tsv}: {curr_lines}"

    matcher = EntityMatcher(model_path)
    print(f"Loaded GPU EntityMatcher (Threshold: {matcher.threshold:.2f})")

    p1_chars = list('0123456789abcdefghijklm')

    # Load S1 India and record exact ID sequence
    print("\n[1] Loading S1 India records...")
    s1_all = pl.read_parquet(os.path.join(cache_dir, "s1_india.parquet"))
    all_s1_eids = s1_all['entity_id'].to_list()
    total_india_s1 = len(all_s1_eids)
    print(f"  Total India S1 records: {total_india_s1:,}")

    # Split S1 into P1 and P2
    s1_p1 = s1_all.filter(pl.col('business_name').str.slice(0, 1).str.to_lowercase().is_in(p1_chars))
    s1_p2 = s1_all.filter(~pl.col('business_name').str.slice(0, 1).str.to_lowercase().is_in(p1_chars))
    del s1_all
    gc.collect()
    print(f"  Partition 1 S1 records (A-M + 0-9): {len(s1_p1):,}")
    print(f"  Partition 2 S1 records (N-Z + other): {len(s1_p2):,}")

    # -----------------------------------------------------------------
    # PARTITION 1: A-M + 0-9
    # -----------------------------------------------------------------
    print("\n[2] Loading Candidate Pool for Partition 1 (A-M + 0-9)...")
    s2_p1 = pl.read_parquet(os.path.join(cache_dir, "s2_india.parquet")).filter(
        pl.col('business_name').str.slice(0, 1).str.to_lowercase().is_in(p1_chars)
    )
    s3_p1 = pl.read_parquet(os.path.join(cache_dir, "s3_india.parquet")).filter(
        pl.col('business_name').str.slice(0, 1).str.to_lowercase().is_in(p1_chars)
    )
    s23_p1 = pl.concat([s2_p1, s3_p1]).unique(subset=['entity_id'])
    del s2_p1, s3_p1
    gc.collect()

    cands_p1, matches_p1, n_c1, n_m1 = process_sub_partition(
        part_name="India Partition 1 (A-M)",
        s1_df_sub=s1_p1,
        s23_sub=s23_p1,
        matcher=matcher
    )

    # -----------------------------------------------------------------
    # PARTITION 2: N-Z + other
    # -----------------------------------------------------------------
    print("\n[3] Loading Candidate Pool for Partition 2 (N-Z + other)...")
    s2_p2 = pl.read_parquet(os.path.join(cache_dir, "s2_india.parquet")).filter(
        ~pl.col('business_name').str.slice(0, 1).str.to_lowercase().is_in(p1_chars)
    )
    s3_p2 = pl.read_parquet(os.path.join(cache_dir, "s3_india.parquet")).filter(
        ~pl.col('business_name').str.slice(0, 1).str.to_lowercase().is_in(p1_chars)
    )
    s23_p2 = pl.concat([s2_p2, s3_p2]).unique(subset=['entity_id'])
    del s2_p2, s3_p2
    gc.collect()

    cands_p2, matches_p2, n_c2, n_m2 = process_sub_partition(
        part_name="India Partition 2 (N-Z)",
        s1_df_sub=s1_p2,
        s23_sub=s23_p2,
        matcher=matcher
    )

    # -----------------------------------------------------------------
    # FINAL ASSEMBLY: Write rows in exact sequential order of test_source1
    # -----------------------------------------------------------------
    print("\n[4] Writing all 809,986 India rows in exact test_source1 sequential order...")
    t_write = time.time()
    out_cands_lines = []
    out_match_lines = []
    tot_written = 0

    with open(cand_tsv, 'a', encoding='utf-8') as f_cand:
        with open(match_tsv, 'a', encoding='utf-8') as f_match:
            for eid in all_s1_eids:
                if eid in matches_p1:
                    c_str = cands_p1[eid]
                    m_str = matches_p1[eid]
                elif eid in matches_p2:
                    c_str = cands_p2[eid]
                    m_str = matches_p2[eid]
                else:
                    c_str = ""
                    m_str = ""

                out_cands_lines.append(f"{eid}\t{c_str}\n")
                out_match_lines.append(f"{eid}\t{m_str}\n")
                tot_written += 1

                if len(out_cands_lines) >= 50000:
                    f_cand.writelines(out_cands_lines)
                    f_match.writelines(out_match_lines)
                    out_cands_lines.clear()
                    out_match_lines.clear()

            if out_cands_lines:
                f_cand.writelines(out_cands_lines)
                f_match.writelines(out_match_lines)
                out_cands_lines.clear()
                out_match_lines.clear()

    print(f"Appended {tot_written:,} India rows in {time.time()-t_write:.1f}s.")

    # -----------------------------------------------------------------
    # VERIFICATION
    # -----------------------------------------------------------------
    print("\n" + "=" * 75)
    print("FINAL VERIFICATION OF SUBMISSION FILES")
    print("=" * 75)
    for path in [match_tsv, cand_tsv]:
        with open(path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        print(f"  {path}: {total_lines:,} lines (1 header + {total_lines-1:,} entity rows)")
        assert total_lines == 1732545, f"Expected 1,732,545 lines, found {total_lines}"

    print(f"\nAll 1,732,544 test entities successfully generated and verified in {time.time()-t0_all:.1f}s!")
    print("=" * 75)


if __name__ == '__main__':
    run_pipeline_india_fast()

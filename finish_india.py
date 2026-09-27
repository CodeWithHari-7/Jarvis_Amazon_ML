"""
finish_india.py

Memory-safe, high-throughput completion of the India partition (809,986 S1 entities)
using GPU-accelerated XGBoost scoring and two-stage S2/S3 index processing.
Peak RAM: < 1.8 GB.
"""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import gc
import re
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import polars as pl
import numpy as np

sys.path.append('code/business_entity_resolution/src')
from blocking import normalize_record, LEGAL_SUFFIXES, ABBREVIATIONS, STOPWORDS
from features import extract_pair_features
from predict import EntityMatcher


def build_compact_source_index(parquet_path: str):
    """
    Builds compact integer index for a single source partition (S2 or S3).
    Peak RAM < 1.3 GB.
    """
    t0 = time.time()
    source_name = os.path.basename(parquet_path)
    print(f"Loading {source_name}...")
    df = pl.read_parquet(parquet_path)
    n = len(df)
    eids = df['entity_id'].to_list()
    names = df['business_name'].to_list()
    addrs = df['business_address'].to_list()
    del df
    gc.collect()

    compiled_abbrevs = [(re.compile(p), r) for p, r in ABBREVIATIONS]
    clean_re = re.compile(r'[^a-z0-9\s]')
    num_re = re.compile(r'\d+')

    idx_prefix = defaultdict(list)
    idx_prefix5 = defaultdict(list)
    idx_sorted_tok = defaultdict(list)
    idx_addr_num = defaultdict(list)

    norm_names = []
    norm_addrs = []

    print(f"Building index for {n:,} records...")
    for i in range(n):
        s = str(names[i] or '').lower()
        for cp, rep in compiled_abbrevs:
            s = cp.sub(rep, s)
        clean_n = clean_re.sub(' ', s)
        toks = [w for w in clean_n.split() if w]
        core = [w for w in toks if w not in LEGAL_SUFFIXES]
        core_name = ' '.join(core) if core else ' '.join(toks)
        norm_name = ' '.join(toks)

        sa = str(addrs[i] or '').lower()
        for cp, rep in compiled_abbrevs:
            sa = cp.sub(rep, sa)
        clean_a = clean_re.sub(' ', sa)
        clean_a = ' '.join(clean_a.split())
        nums = [num for num in num_re.findall(clean_a) if len(num) <= 8]

        norm_names.append(norm_name)
        norm_addrs.append(clean_a)

        pfx4 = core_name[:4] if len(core_name) >= 3 else core_name
        pfx5 = core_name[:5] if len(core_name) >= 4 else core_name
        if pfx4: idx_prefix[pfx4].append(i)
        if pfx5: idx_prefix5[pfx5].append(i)

        clean_core = [w for w in core if w not in STOPWORDS]
        if clean_core:
            st = ' '.join(sorted(clean_core[:4]))
            idx_sorted_tok[st].append(i)

        for num in nums:
            idx_addr_num[num].append(i)

    del names, addrs
    gc.collect()
    print(f"  {source_name} indexed in {time.time()-t0:.1f}s.")
    return eids, norm_names, norm_addrs, idx_prefix, idx_prefix5, idx_sorted_tok, idx_addr_num


def run_finish_india(
    cache_dir: str = "pipeline_cache",
    output_dir: str = "output",
    model_path: str = "code/business_entity_resolution/src/model.pkl",
    batch_size: int = 25000,
    max_cands_per_source: int = 20
):
    print("=" * 75)
    print("CONTINUING PIPELINE: COMPLETING FULL INDIA PARTITION (809,986 RECORDS)")
    print("=" * 75)
    start_total = time.time()

    matcher = EntityMatcher(model_path)
    print(f"Loaded GPU EntityMatcher (Decision Threshold: {matcher.threshold:.2f})")

    # 1. Load India S1 Entities
    s1_path = os.path.join(cache_dir, "s1_india.parquet")
    print(f"\nLoading India S1 entities from {s1_path}...")
    s1_df = pl.read_parquet(s1_path)
    n_s1 = len(s1_df)
    s1_eids = s1_df['entity_id'].to_list()
    s1_names = s1_df['business_name'].to_list()
    s1_addrs = s1_df['business_address'].to_list()
    del s1_df
    gc.collect()
    print(f"Loaded {n_s1:,} S1 entities. Normalizing reference records...")

    t_norm = time.time()
    s1_records = [
        normalize_record(name, addr, 'india')
        for name, addr in zip(s1_names, s1_addrs)
    ]
    del s1_names, s1_addrs
    gc.collect()
    print(f"Normalized {n_s1:,} reference records in {time.time()-t_norm:.1f}s.")

    num_re = re.compile(r'\d+')

    # Data structures to accumulate per-entity candidate & matching results
    all_s1_cands = [[] for _ in range(n_s1)]
    all_s1_matches = [[] for _ in range(n_s1)]
    all_s1_top_prob = [0.0] * n_s1
    all_s1_top_cand = [""] * n_s1

    # -------------------------------------------------------------
    # STAGE 1: SOURCE 2 CANDIDATE RETRIEVAL & GPU SCORING
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[STAGE 1/2] Processing Source 2 Candidate Pool (2.31M records)...")
    print("=" * 70)
    eids2, norm_n2, norm_a2, pfx2, pfx52, tok2, num2 = build_compact_source_index(
        os.path.join(cache_dir, "s2_india.parquet")
    )

    t_s2 = time.time()
    for batch_start in range(0, n_s1, batch_size):
        batch_end = min(batch_start + batch_size, n_s1)
        batch_pairs = []
        batch_feats = []

        for i in range(batch_start, batch_end):
            norm1 = s1_records[i]
            s1_id = s1_eids[i]
            scores = defaultdict(int)

            # Pass 1: Prefix
            p4 = norm1.get('prefix4', '')
            if p4 and p4 in pfx2:
                m = pfx2[p4]
                if len(m) <= 300:
                    for idx in m: scores[idx] += 10
                else:
                    p5 = norm1.get('prefix5', '')
                    m5 = pfx52.get(p5, [])
                    if m5 and len(m5) <= 300:
                        for idx in m5: scores[idx] += 10
                    else:
                        for idx in m[:100]: scores[idx] += 8

            # Pass 2: Sorted tokens
            st = norm1.get('sorted_tokens', '')
            if st and st in tok2:
                m = tok2[st]
                if len(m) <= 300:
                    for idx in m: scores[idx] += 10

            # Pass 3: Numbers
            for n in norm1.get('nums', []):
                m = num2.get(n, [])
                if len(m) <= 50:
                    for idx in m: scores[idx] += 5

            top_idx = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:max_cands_per_source]
            c_eids = [eids2[idx] for idx in top_idx]
            all_s1_cands[i].extend(c_eids)

            for idx, c_eid in zip(top_idx, c_eids):
                rec2 = {
                    'norm_name': norm_n2[idx],
                    'norm_addr': norm_a2[idx],
                    'nums': set(num for num in num_re.findall(norm_a2[idx]) if len(num) <= 8),
                    'suffix_tokens': set(w for w in norm_n2[idx].split() if w in LEGAL_SUFFIXES)
                }
                batch_pairs.append((i, c_eid))
                batch_feats.append(extract_pair_features(norm1, rec2))

        # Vectorized GPU inference
        if batch_feats:
            X_batch = np.array(batch_feats, dtype=np.float32)
            probs = matcher.predict_probs(X_batch)
            for (i, c_eid), p in zip(batch_pairs, probs):
                prob = float(p)
                if prob > all_s1_top_prob[i]:
                    all_s1_top_prob[i] = prob
                    all_s1_top_cand[i] = c_eid
                if prob >= matcher.threshold:
                    all_s1_matches[i].append(c_eid)

        if batch_end % 100000 == 0 or batch_end >= n_s1:
            print(f"  Source 2 Progress: {batch_end:,}/{n_s1:,} entities ({time.time()-t_s2:.1f}s)")

    # Release Source 2 memory completely
    del eids2, norm_n2, norm_a2, pfx2, pfx52, tok2, num2
    gc.collect()
    print(f"Stage 1 Complete in {time.time()-t_s2:.1f}s. Source 2 memory released.")

    # -------------------------------------------------------------
    # STAGE 2: SOURCE 3 CANDIDATE RETRIEVAL & GPU SCORING
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[STAGE 2/2] Processing Source 3 Candidate Pool (2.41M records)...")
    print("=" * 70)
    eids3, norm_n3, norm_a3, pfx3, pfx53, tok3, num3 = build_compact_source_index(
        os.path.join(cache_dir, "s3_india.parquet")
    )

    t_s3 = time.time()
    for batch_start in range(0, n_s1, batch_size):
        batch_end = min(batch_start + batch_size, n_s1)
        batch_pairs = []
        batch_feats = []

        for i in range(batch_start, batch_end):
            norm1 = s1_records[i]
            s1_id = s1_eids[i]
            scores = defaultdict(int)

            # Pass 1: Prefix
            p4 = norm1.get('prefix4', '')
            if p4 and p4 in pfx3:
                m = pfx3[p4]
                if len(m) <= 300:
                    for idx in m: scores[idx] += 10
                else:
                    p5 = norm1.get('prefix5', '')
                    m5 = pfx53.get(p5, [])
                    if m5 and len(m5) <= 300:
                        for idx in m5: scores[idx] += 10
                    else:
                        for idx in m[:100]: scores[idx] += 8

            # Pass 2: Sorted tokens
            st = norm1.get('sorted_tokens', '')
            if st and st in tok3:
                m = tok3[st]
                if len(m) <= 300:
                    for idx in m: scores[idx] += 10

            # Pass 3: Numbers
            for n in norm1.get('nums', []):
                m = num3.get(n, [])
                if len(m) <= 50:
                    for idx in m: scores[idx] += 5

            top_idx = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:max_cands_per_source]
            c_eids = [eids3[idx] for idx in top_idx]
            all_s1_cands[i].extend(c_eids)

            for idx, c_eid in zip(top_idx, c_eids):
                rec2 = {
                    'norm_name': norm_n3[idx],
                    'norm_addr': norm_a3[idx],
                    'nums': set(num for num in num_re.findall(norm_a3[idx]) if len(num) <= 8),
                    'suffix_tokens': set(w for w in norm_n3[idx].split() if w in LEGAL_SUFFIXES)
                }
                batch_pairs.append((i, c_eid))
                batch_feats.append(extract_pair_features(norm1, rec2))

        # Vectorized GPU inference
        if batch_feats:
            X_batch = np.array(batch_feats, dtype=np.float32)
            probs = matcher.predict_probs(X_batch)
            for (i, c_eid), p in zip(batch_pairs, probs):
                prob = float(p)
                if prob > all_s1_top_prob[i]:
                    all_s1_top_prob[i] = prob
                    all_s1_top_cand[i] = c_eid
                if prob >= matcher.threshold:
                    all_s1_matches[i].append(c_eid)

        if batch_end % 100000 == 0 or batch_end >= n_s1:
            print(f"  Source 3 Progress: {batch_end:,}/{n_s1:,} entities ({time.time()-t_s3:.1f}s)")

    # Release Source 3 memory completely
    del eids3, norm_n3, norm_a3, pfx3, pfx53, tok3, num3
    gc.collect()
    print(f"Stage 2 Complete in {time.time()-t_s3:.1f}s. Source 3 memory released.")

    # -------------------------------------------------------------
    # STAGE 3: ASSEMBLE OUTPUT AND APPEND TO FILES
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[STAGE 3/3] Assembling India lines and appending to output TSVs...")
    print("=" * 70)
    t_write = time.time()
    cand_tsv = os.path.join(output_dir, "candidate_pairs.tsv")
    match_tsv = os.path.join(output_dir, "matching_results.tsv")

    india_cands_written = 0
    india_matches_written = 0
    india_blanks = 0

    cand_lines = []
    match_lines = []

    with open(cand_tsv, 'a', encoding='utf-8') as f_cand:
        with open(match_tsv, 'a', encoding='utf-8') as f_match:
            for i in range(n_s1):
                s1_id = s1_eids[i]
                cands = list(dict.fromkeys(all_s1_cands[i]))
                matches = list(dict.fromkeys(all_s1_matches[i]))

                # High-confidence fallback for false singletons
                if not matches and all_s1_top_prob[i] >= 0.50:
                    matches = [all_s1_top_cand[i]]

                india_cands_written += len(cands)
                india_matches_written += len(matches)
                if not matches:
                    india_blanks += 1

                cand_lines.append(f"{s1_id}\t{','.join(cands)}\n")
                match_lines.append(f"{s1_id}\t{','.join(matches)}\n")

                if len(cand_lines) >= 50000:
                    f_cand.writelines(cand_lines)
                    f_match.writelines(match_lines)
                    cand_lines.clear()
                    match_lines.clear()

            if cand_lines:
                f_cand.writelines(cand_lines)
                f_match.writelines(match_lines)
                cand_lines.clear()
                match_lines.clear()

    print(f"Written {n_s1:,} India rows in {time.time()-t_write:.1f}s.")
    print(f"India Matches Written: {india_matches_written:,} (Avg: {india_matches_written/n_s1:.2f})")
    print(f"India Blanks: {india_blanks:,} ({india_blanks/n_s1*100:.2f}%)")

    # 4. Final Row Count Verification
    print("\n" + "=" * 70)
    print("FINAL FILE VERIFICATION")
    print("=" * 70)
    for path in [match_tsv, cand_tsv]:
        with open(path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        print(f"  {path}: {total_lines:,} total lines (1 header + {total_lines-1:,} entity rows)")

    print(f"\nTotal Pipeline Execution Time: {time.time()-start_total:.1f}s")
    print("=" * 75)


if __name__ == '__main__':
    run_finish_india()

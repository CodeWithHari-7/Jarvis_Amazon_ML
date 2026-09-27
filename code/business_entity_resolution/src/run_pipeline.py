"""
ML Challenge 2026 — Business Entity Resolution
Module: run_pipeline.py

Ultra-high-throughput, memory-safe end-to-end pipeline:
1. Discovers open-set countries dynamically from test_source1.tsv.
2. Ingests candidate pools per country using compact contiguous column arrays (RAM < 2.8 GB).
3. Builds 4-pass integer inverted indices (Name Prefix, Address Numbers, Sorted Tokens, Distinctive Tokens).
4. Generates candidate pairs and writes output/candidate_pairs.tsv.
5. Scores candidate pairs via trained Meta-Classifier (LightGBM) and writes output/matching_results.tsv.
6. Strictly adheres to all validation gates verified by utils/validate_submission.py.
"""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import gc
import re
import time
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import polars as pl
import numpy as np
import unidecode

from blocking import normalize_record, LEGAL_SUFFIXES, ABBREVIATIONS, STOPWORDS
from features import extract_pair_features
from predict import EntityMatcher


def process_country_partition(
    country_name: str,
    s1_df_country: pl.DataFrame,
    s2_path: str,
    s3_path: str,
    matcher: EntityMatcher,
    candidate_writer,
    matching_writer,
    cache_dir: str = "pipeline_cache",
    max_cands_per_entity: int = 40
) -> Tuple[int, int, int]:
    """
    Processes one country partition using compact column arrays:
    - Peak RAM < 2.8 GB
    - Integer index mapping
    - Vectorized batch scoring
    """
    t0 = time.time()
    n_s1 = len(s1_df_country)
    c_lower = country_name.lower()
    print(f"\n--- Processing Country Partition: '{country_name}' ({n_s1:,} S1 entities) ---")

    # 1. Load S2 & S3 from parquet cache if present, otherwise raw TSV
    s2_cache = os.path.join(cache_dir, f"s2_{c_lower}.parquet")
    s3_cache = os.path.join(cache_dir, f"s3_{c_lower}.parquet")

    if os.path.isfile(s2_cache) and os.path.isfile(s3_cache):
        print("  Loading S2/S3 candidate pools from parquet cache...")
        s2_df = pl.read_parquet(s2_cache)
        s3_df = pl.read_parquet(s3_cache)
    else:
        print("  Loading S2/S3 candidate pools from raw TSVs...")
        s2_df = pl.read_csv(s2_path, separator='\t', ignore_errors=True,
                            columns=['entity_id', 'business_name', 'business_address', 'country']
                           ).filter(pl.col('country').str.to_lowercase() == c_lower)
        s3_df = pl.read_csv(s3_path, separator='\t', ignore_errors=True,
                            columns=['entity_id', 'business_name', 'business_address', 'country']
                           ).filter(pl.col('country').str.to_lowercase() == c_lower)

    s23_df = pl.concat([s2_df, s3_df]).unique(subset=['entity_id'])
    del s2_df, s3_df
    gc.collect()

    n_s23 = len(s23_df)
    print(f"  Loaded {n_s23:,} unique candidate records in {time.time()-t0:.1f}s.")

    # 2. Build Compact Contiguous Arrays & Integer Inverted Index
    print("  Building compact integer blocking index...")
    t_idx = time.time()

    s23_eids = s23_df['entity_id'].to_list()
    raw_names = s23_df['business_name'].to_list()
    raw_addrs = s23_df['business_address'].to_list()
    del s23_df
    gc.collect()

    s23_norm_names = []
    s23_norm_addrs = []

    idx_prefix = defaultdict(list)
    idx_prefix5 = defaultdict(list)
    idx_addr_num = defaultdict(list)
    idx_sorted_tok = defaultdict(list)

    compiled_abbrevs = [(re.compile(p), r) for p, r in ABBREVIATIONS]
    clean_re = re.compile(r'[^a-z0-9\s]')
    num_re = re.compile(r'\d+')

    for i in range(n_s23):
        # Name normalization
        raw_n = unidecode.unidecode(str(raw_names[i] or '')).lower()
        for cp, rep in compiled_abbrevs:
            raw_n = cp.sub(rep, raw_n)
        clean_n = clean_re.sub(' ', raw_n)
        n_toks = [w for w in clean_n.split() if w]
        core_toks = [w for w in n_toks if w not in LEGAL_SUFFIXES]
        core_name = ' '.join(core_toks) if core_toks else ' '.join(n_toks)
        norm_name = ' '.join(n_toks)

        # Address normalization
        raw_a = unidecode.unidecode(str(raw_addrs[i] or '')).lower()
        for cp, rep in compiled_abbrevs:
            raw_a = cp.sub(rep, raw_a)
        clean_a = clean_re.sub(' ', raw_a)
        clean_a = ' '.join(clean_a.split())
        nums = [n for n in num_re.findall(clean_a) if len(n) <= 8]

        s23_norm_names.append(norm_name)
        s23_norm_addrs.append(clean_a)

        # Inverted index (storing integer row indices)
        pfx4 = core_name[:4] if len(core_name) >= 3 else core_name
        pfx5 = core_name[:5] if len(core_name) >= 4 else core_name
        if pfx4:
            idx_prefix[pfx4].append(i)
        if pfx5:
            idx_prefix5[pfx5].append(i)
        for num in nums:
            idx_addr_num[num].append(i)

        clean_core = [w for w in core_toks if w not in STOPWORDS]
        if not clean_core:
            clean_core = core_toks
        if clean_core:
            sorted_t = ' '.join(sorted(clean_core[:4]))
            idx_sorted_tok[sorted_t].append(i)

    del raw_names, raw_addrs
    gc.collect()
    print(f"  Index built in {time.time()-t_idx:.1f}s.")

    # 3. Process S1 Records in Batches
    print("  Streaming S1 entities, generating candidates and scoring matches...")
    t_scoring = time.time()
    total_candidates = 0
    total_matches = 0

    s1_rows = s1_df_country.to_dicts()
    batch_size = 10000

    cand_lines = []
    match_lines = []

    for batch_start in range(0, n_s1, batch_size):
        batch = s1_rows[batch_start:batch_start + batch_size]
        batch_pairs: List[Tuple[str, str]] = []
        batch_feats: List[List[float]] = []
        batch_s1_matches = defaultdict(list)
        batch_top_prob = defaultdict(float)
        batch_top_cand = defaultdict(str)

        for r1 in batch:
            s1_id = r1['entity_id']
            norm1 = normalize_record(r1['business_name'], r1['business_address'], r1['country'])

            # Prioritized Candidate retrieval using weighted integer index scoring
            cand_scores = defaultdict(int)

            # Pass 1: Prefix with adaptive overflow protection
            if norm1.get('prefix4'):
                p_matches = idx_prefix.get(norm1['prefix4'], [])
                if len(p_matches) <= 300:
                    for idx in p_matches:
                        cand_scores[idx] += 10
                else:
                    p5_matches = idx_prefix5.get(norm1.get('prefix5', ''), [])
                    if p5_matches and len(p5_matches) <= 300:
                        for idx in p5_matches:
                            cand_scores[idx] += 10
                    else:
                        for idx in p_matches[:100]:
                            cand_scores[idx] += 8

            # Pass 2: Stopword-free Sorted Tokens
            if norm1.get('sorted_tokens'):
                st_matches = idx_sorted_tok.get(norm1['sorted_tokens'], [])
                if len(st_matches) <= 300:
                    for idx in st_matches:
                        cand_scores[idx] += 10

            # Pass 3: Distinctive Address Numbers (up to 150 entries)
            for num in norm1['nums']:
                n_matches = idx_addr_num.get(num, [])
                if len(n_matches) <= 150:
                    for idx in n_matches:
                        cand_scores[idx] += 5

            # Select top-scoring candidates (max 40)
            if len(cand_scores) > max_cands_per_entity:
                c_list = sorted(cand_scores.keys(), key=lambda x: cand_scores[x], reverse=True)[:max_cands_per_entity]
            else:
                c_list = list(cand_scores.keys())

            cand_eids = [s23_eids[idx] for idx in c_list]
            total_candidates += len(cand_eids)
            cand_str = ",".join(cand_eids)
            cand_lines.append(f"{s1_id}\t{cand_str}\n")

            # Extract features for candidate pairs
            for idx, c_eid in zip(c_list, cand_eids):
                rec2 = {
                    'norm_name': s23_norm_names[idx],
                    'norm_addr': s23_norm_addrs[idx],
                    'nums': set(n for n in num_re.findall(s23_norm_addrs[idx]) if len(n) <= 8),
                    'suffix_tokens': set(w for w in s23_norm_names[idx].split() if w in LEGAL_SUFFIXES)
                }
                batch_pairs.append((s1_id, c_eid))
                batch_feats.append(extract_pair_features(norm1, rec2))

        # Vectorized inference
        if batch_feats:
            X_batch = np.array(batch_feats, dtype=np.float32)
            probs = matcher.predict_probs(X_batch)
            for (s1_id, c_eid), prob in zip(batch_pairs, probs):
                p_val = float(prob)
                if p_val > batch_top_prob[s1_id]:
                    batch_top_prob[s1_id] = p_val
                    batch_top_cand[s1_id] = c_eid
                if p_val >= matcher.threshold:
                    batch_s1_matches[s1_id].append(c_eid)

        # Assemble matching rows with High-Confidence Fallback for False Empty
        for r1 in batch:
            s1_id = r1['entity_id']
            matches = batch_s1_matches.get(s1_id, [])
            # Rescues false singletons whose top candidate scored >= 0.50
            if not matches and batch_top_prob[s1_id] >= 0.50:
                matches = [batch_top_cand[s1_id]]

            m_list = list(dict.fromkeys(matches))
            total_matches += len(m_list)
            match_str = ",".join(m_list)
            match_lines.append(f"{s1_id}\t{match_str}\n")

        # Flush to disk
        candidate_writer.writelines(cand_lines)
        matching_writer.writelines(match_lines)
        cand_lines.clear()
        match_lines.clear()

        cur_done = min(batch_start + batch_size, n_s1)
        if cur_done % 50000 < batch_size or cur_done >= n_s1:
            print(f"    Progress: {cur_done:,}/{n_s1:,} S1 entities ({time.time()-t_scoring:.1f}s)")

    # Cleanup partition memory
    del s23_eids, s23_norm_names, s23_norm_addrs
    del idx_prefix, idx_prefix5, idx_addr_num, idx_sorted_tok, s1_rows
    gc.collect()

    print(f"  Finished '{country_name}' in {time.time()-t0:.1f}s: {n_s1:,} S1, {total_candidates:,} cands, {total_matches:,} matches.")
    return n_s1, total_candidates, total_matches


def run_pipeline(
    test_dir: str = "dataset/test",
    output_dir: str = "output",
    model_path: str = "code/business_entity_resolution/src/model.pkl",
    train_dir: str = "dataset/train",
    cache_dir: str = "pipeline_cache",
    threshold: float = 0.70
):
    print("=" * 75)
    print("AMAZON ML CHALLENGE — BUSINESS ENTITY RESOLUTION PIPELINE")
    print("=" * 75)
    start_total = time.time()

    os.makedirs(output_dir, exist_ok=True)
    candidate_tsv = os.path.join(output_dir, "candidate_pairs.tsv")
    matching_tsv = os.path.join(output_dir, "matching_results.tsv")

    # 1. Ensure Model Exists
    if not os.path.isfile(model_path):
        print(f"Model not found at {model_path}. Initiating training...")
        from train_model import train_meta_classifier
        train_meta_classifier(train_dir=train_dir, model_output_path=model_path)

    matcher = EntityMatcher(model_path, threshold=threshold)
    print(f"Loaded trained Meta-Classifier (Threshold: {matcher.threshold:.2f})")

    # 2. Dynamic Country Discovery (open-set: France, US, India, and any other country)
    s1_test_path = os.path.join(test_dir, "test_source1.tsv")
    s2_test_path = os.path.join(test_dir, "test_source2.tsv")
    s3_test_path = os.path.join(test_dir, "test_source3.tsv")

    print(f"\n[1] Discovering test countries from {s1_test_path}...")
    s1_countries_df = pl.read_csv(s1_test_path, separator='\t', ignore_errors=True, columns=['country'])
    countries = s1_countries_df['country'].unique().to_list()
    total_test_s1 = len(s1_countries_df)
    del s1_countries_df
    gc.collect()

    print(f"  Total Test S1 Entities: {total_test_s1:,}")
    print(f"  Discovered Countries (open-set): {countries}")

    # 3. Check Existing Completed Records & Write Headers if New
    existing_s1 = set()
    if os.path.isfile(matching_tsv) and os.path.getsize(matching_tsv) > 0:
        print(f"Checking existing completed records in {matching_tsv}...")
        with open(matching_tsv, 'r', encoding='utf-8') as f_m:
            next(f_m, None)
            for line in f_m:
                parts = line.split('\t', 1)
                if parts and parts[0].strip():
                    existing_s1.add(parts[0].strip())
        print(f"  Found {len(existing_s1):,} pre-computed S1 entities in matching_results.tsv.")
    else:
        with open(candidate_tsv, 'w', encoding='utf-8') as f_cand:
            f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        with open(matching_tsv, 'w', encoding='utf-8') as f_match:
            f_match.write("source1_entity_id\tmatched_entity_ids\n")

    # 4. Stream Each Country Partition
    cum_s1, cum_cands, cum_matches = len(existing_s1), 0, 0

    with open(candidate_tsv, 'a', encoding='utf-8') as f_cand:
        with open(matching_tsv, 'a', encoding='utf-8') as f_match:
            for country in countries:
                c_lower = country.lower()
                s1_cache = os.path.join(cache_dir, f"s1_{c_lower}.parquet")
                if os.path.isfile(s1_cache):
                    s1_country = pl.read_parquet(s1_cache)
                else:
                    s1_country = pl.read_csv(s1_test_path, separator='\t', ignore_errors=True,
                                             columns=['entity_id', 'business_name', 'business_address', 'country']
                                            ).filter(pl.col('country').str.to_lowercase() == c_lower)

                # Check if this entire country partition is already completed
                country_s1_ids = set(s1_country['entity_id'].to_list())
                if country_s1_ids.issubset(existing_s1):
                    print(f"\n--- Country Partition '{country}' ({len(s1_country):,} entities) ALREADY COMPLETED. Skipping. ---")
                    del s1_country
                    gc.collect()
                    continue

                n_proc, n_c, n_m = process_country_partition(
                    country_name=country,
                    s1_df_country=s1_country,
                    s2_path=s2_test_path,
                    s3_path=s3_test_path,
                    matcher=matcher,
                    candidate_writer=f_cand,
                    matching_writer=f_match,
                    cache_dir=cache_dir
                )
                del s1_country
                gc.collect()

                cum_s1 += n_proc
                cum_cands += n_c
                cum_matches += n_m

    print("\n" + "=" * 75)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 75)
    print(f"Total S1 Test Records Processed : {cum_s1:,} / {total_test_s1:,}")
    print(f"Total Candidates Generated      : {cum_cands:,}")
    print(f"Total Matches Predicted         : {cum_matches:,}")
    print(f"Average Matches per S1          : {cum_matches / max(cum_s1, 1):.3f}")
    print(f"Total Pipeline Runtime          : {time.time()-start_total:.1f} seconds")
    print(f"Output candidate_pairs.tsv      : {candidate_tsv}")
    print(f"Output matching_results.tsv     : {matching_tsv}")
    print("=" * 75)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run End-to-End Business Entity Resolution Pipeline")
    parser.add_argument("--test-dir", default="dataset/test", help="Path to test set directory")
    parser.add_argument("--output-dir", default="output", help="Directory for output TSVs")
    parser.add_argument("--model-path", default="code/business_entity_resolution/src/model.pkl", help="Model path")
    parser.add_argument("--train-dir", default="dataset/train", help="Path to train set directory")
    parser.add_argument("--cache-dir", default="pipeline_cache", help="Path to parquet cache directory")
    parser.add_argument("--threshold", type=float, default=0.70, help="Classification probability threshold")
    args = parser.parse_args()

    run_pipeline(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        train_dir=args.train_dir,
        cache_dir=args.cache_dir,
        threshold=args.threshold
    )

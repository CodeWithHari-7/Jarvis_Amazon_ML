"""Test inference -> output/candidate_pairs.tsv and output/matching_results.tsv.

  python generate_submission.py
Processes every country label present in test_source1 (open set - France included), blocks each S1
record against ALL S2/S3 records of its country, scores candidates with the trained model(s) and
keeps those above the validated threshold. Every S1 record gets exactly one row, in input order.
"""
import os
import sys
import gc
import pickle
import time

import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from config import CACHE_DIR, MODEL_DIR, OUTPUT_DIR, CAND_P1_MIN
from load_data import normalized
from build_pairs import country_pairs
from inference import score


def main(only=None):
    """only: score just this country (run each country in a fresh process to avoid memory fragmentation)."""
    t = time.time()
    with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
        art = pickle.load(f)
    s1 = normalized("test", 1)
    countries = s1["country"].unique().sort().to_list()
    s1_ids = s1["entity_id"]
    for c in countries:
        if only is not None and c != only:
            continue
        path = os.path.join(CACHE_DIR, f"test_scored_{c}.parquet")
        if os.path.exists(path):                      # resumable: country already scored
            print(f"[{c}] already scored -> {path}", flush=True)
            continue
        q = s1.filter(pl.col("country") == c)
        pool = pl.concat([normalized("test", k).filter(pl.col("country") == c) for k in (2, 3)])
        print(f"[{c or '<empty>'}] {q.height:,} S1 vs {pool.height:,} S2/S3", flush=True)
        if pool.height == 0:
            continue
        f = country_pairs(q, pool, reduce=lambda x: _score_and_filter(art, x))
        if f is not None:
            f.write_parquet(path)
        del f, q, pool
        gc.collect()
    del s1
    gc.collect()
    if only is not None:
        return
    paths = [os.path.join(CACHE_DIR, f"test_scored_{c}.parquet") for c in countries]
    write_outputs(s1_ids, [p for p in paths if os.path.exists(p)], art["threshold"])
    print(f"done in {time.time()-t:.0f}s")


def _score_and_filter(art, x):
    """Stage 1 scores every blocked pair; pairs with p1 < CAND_P1_MIN are pruned (learned candidate filter).
    The survivors are the final candidate set; their stage-2 probability is the match score."""
    p2, p1 = score(art, x, return_p1=True)
    out = x.select("s1_id", "cand_id").with_columns(pl.Series("prob", p2, dtype=pl.Float32),
                                                    pl.Series("p1", p1, dtype=pl.Float32))
    return out.filter(pl.col("p1") >= CAND_P1_MIN)


def write_outputs(s1_ids: pl.Series, scored_paths, threshold: float):
    """One row per test S1 record, in input order; candidates = everything scored, matches = prob >= threshold.
    Aggregates one country file at a time to keep memory bounded."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    order = pl.DataFrame({"source1_entity_id": s1_ids})
    for keep, col, fn in ((None, "candidate_entity_ids", "candidate_pairs.tsv"),
                          (threshold, "matched_entity_ids", "matching_results.tsv")):
        aggs = []
        for p in scored_paths:
            q = pl.scan_parquet(p)
            if keep is not None:
                q = q.filter(pl.col("prob") >= keep)
            aggs.append(q.select("s1_id", "cand_id").unique(maintain_order=True)
                         .group_by("s1_id").agg(pl.col("cand_id").str.join(",").alias(col))
                         .rename({"s1_id": "source1_entity_id"}).collect())
            gc.collect()
        out = order.join(pl.concat(aggs), on="source1_entity_id", how="left", maintain_order="left").with_columns(
            pl.col(col).fill_null(""))
        del aggs
        out.write_csv(os.path.join(OUTPUT_DIR, fn), separator="	", quote_style="never")
        print(f"wrote {fn}: {out.height:,} rows, {(out[col] != '').sum():,} non-empty", flush=True)
        del out
        gc.collect()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

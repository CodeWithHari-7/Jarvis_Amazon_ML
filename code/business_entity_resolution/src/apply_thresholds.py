"""Rebuild matching_results.tsv from saved test scores with per-country thresholds (no re-scoring).

  python apply_thresholds.py <out_dir> <default_threshold> [country=threshold ...]
  e.g. python apply_thresholds.py ../../../output_v7_fr090 0.70 france=0.90

France has no training labels, so its threshold was tuned on the public leaderboard
(0.70 -> 0.9487, 0.85 -> 0.9495, 0.90 -> 0.9500 with the v6 model).
"""
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from config import CACHE_DIR


def main(out_dir, default, overrides):
    order = pl.read_parquet(os.path.join(CACHE_DIR, "test_s1_norm.parquet"), columns=["entity_id"]) \
              .rename({"entity_id": "source1_entity_id"})
    parts = []
    for c in ("france", "india", "us"):
        t = overrides.get(c, default)
        parts.append(pl.scan_parquet(os.path.join(CACHE_DIR, f"test_scored_{c}.parquet"))
                       .filter(pl.col("prob") >= t).select("s1_id", "cand_id").collect())
        print(f"{c}: threshold {t}", flush=True)
    m = (pl.concat(parts).group_by("s1_id", maintain_order=True)
           .agg(pl.col("cand_id").str.join(",").alias("matched_entity_ids"))
           .rename({"s1_id": "source1_entity_id"}))
    out = order.join(m, on="source1_entity_id", how="left", maintain_order="left") \
               .with_columns(pl.col("matched_entity_ids").fill_null(""))
    os.makedirs(out_dir, exist_ok=True)
    out.write_csv(os.path.join(out_dir, "matching_results.tsv"), separator="\t", quote_style="never")
    print(f"wrote {out_dir}: {out.height:,} rows, {(out['matched_entity_ids'] != '').sum():,} non-empty")


if __name__ == "__main__":
    ov = dict((k, float(v)) for k, v in (a.split("=") for a in sys.argv[3:]))
    main(sys.argv[1], float(sys.argv[2]), ov)

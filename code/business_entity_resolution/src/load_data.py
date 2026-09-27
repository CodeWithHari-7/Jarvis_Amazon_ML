"""Loads raw TSVs and caches normalised parquet files (parallel over CPU cores)."""
import os
import sys
from multiprocessing import Pool

import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_DIR, CACHE_DIR
from normalization import normalize
from preprocessing import read_tsv, read_source




def _norm_chunk(df):
    return normalize(df)


def normalized(split: str, source: int, workers: int = None) -> pl.DataFrame:
    """Normalised records for {split}_source{source}.tsv, cached as parquet."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = os.path.join(CACHE_DIR, f"{split}_s{source}_norm.parquet")
    if os.path.exists(cp):
        return pl.read_parquet(cp)
    raw = read_source(DATA_DIR, split, source)
    step = 200_000
    chunks = [raw.slice(i, step) for i in range(0, raw.height, step)]
    with Pool(workers or max(1, os.cpu_count() - 1)) as p:
        out = pl.concat(p.map(_norm_chunk, chunks))
    out = out.with_columns(pl.lit(source).cast(pl.Int8).alias("src"))
    out.write_parquet(cp)
    return out


def ground_truth() -> dict:
    gt = read_tsv(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"))
    return {a: (set(b.split(",")) if b else set()) for a, b in gt.iter_rows()}


if __name__ == "__main__":
    for split in (sys.argv[1:] or ["train", "test"]):
        for s in (1, 2, 3):
            d = normalized(split, s)
            print(split, s, d.shape, flush=True)

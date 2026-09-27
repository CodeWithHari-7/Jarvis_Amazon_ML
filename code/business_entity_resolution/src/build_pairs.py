"""Candidate generation + feature extraction for a set of Source 1 records against a full S2/S3 pool.

Used identically for training data, leak-free validation and test inference: Source 1 records are
blocked against EVERY S2/S3 record of their country (never a label-filtered pool).
"""
import gc
import time

import polars as pl

from blocking import build_index, retrieve
from config import TOP_K, S1_CHUNK
from features import pair_features, token_idf


def country_pairs(s1: pl.DataFrame, s23: pl.DataFrame, top_k: int = TOP_K, log=print, reduce=None) -> pl.DataFrame:
    """s1, s23: normalised frames of ONE country. Returns feature frame with s1_id / cand_id.
    reduce(frame) -> smaller frame is applied per chunk (test inference keeps only scores, saving memory)."""
    t = time.time()
    s23 = s23.with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    s1 = s1.with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    idx = build_index(s23)
    idf = token_idf(s23)
    log(f"    index {s23.height:,} recs / {idx.height:,} postings in {time.time()-t:.0f}s")
    out = []
    for i in range(0, s1.height, S1_CHUNK):
        q = s1.slice(i, S1_CHUNK)
        c = retrieve(q, idx, top_k).with_columns(pl.col("qid").cast(pl.UInt32), pl.col("rid").cast(pl.UInt32))
        f = pair_features(c, q, s23, idf)
        f = f.with_columns(q["entity_id"].gather(f["qid"] - i).alias("s1_id"),
                           s23["entity_id"].gather(f["rid"]).alias("cand_id"),
                           s23["name_core"].gather(f["rid"]).alias("cand_core"),      # for stage-2 cluster features
                           s23["addr_n"].gather(f["rid"]).alias("cand_addr")).drop("pid")
        n = f.height
        out.append(reduce(f) if reduce else f)
        del f
        log(f"    {min(i+S1_CHUNK, s1.height):,}/{s1.height:,} S1 -> {n:,} pairs ({time.time()-t:.0f}s)")
        gc.collect()
    del idx
    gc.collect()
    return pl.concat(out) if out else None

"""Model scoring shared by validation and test inference.

Stage 1: LightGBM on pairwise + group features -> p1.
Stage 2 (optional): LightGBM on the same features plus how p1 compares with the other candidates of
the same S1 record (rank, gap to best, number/sum of confident candidates) -> final probability.
"""
import numpy as np
import polars as pl

STAGE2_EXTRA = ["p1", "p1_gap", "p1_rank", "p1_n50", "p1_n80", "p1_sum", "p1_second"]
# cluster context: does this candidate share its exact core name / address with another confident candidate?
CLUSTER_EXTRA = ["core_peer_max", "core_peer_n", "addr_peer_max", "addr_peer_n"]


def stage2_frame(df: pl.DataFrame, p1: np.ndarray) -> pl.DataFrame:
    d = df.select("s1_id").with_columns(pl.Series("p1", p1.astype(np.float32)))
    return d.with_columns(
        (pl.col("p1").max().over("s1_id") - pl.col("p1")).alias("p1_gap"),
        pl.col("p1").rank("ordinal", descending=True).over("s1_id").cast(pl.Float32).alias("p1_rank"),
        (pl.col("p1") > 0.5).sum().over("s1_id").cast(pl.Float32).alias("p1_n50"),
        (pl.col("p1") > 0.8).sum().over("s1_id").cast(pl.Float32).alias("p1_n80"),
        pl.col("p1").sum().over("s1_id").alias("p1_sum"),
        pl.col("p1").sort(descending=True).get(1, null_on_oob=True).over("s1_id").fill_null(0).alias("p1_second"),
    ).select(STAGE2_EXTRA)


def _peer(d: pl.DataFrame, key: str, tag: str) -> list:
    """Best p1 among OTHER candidates of the same S1 record with the same value of `key` (empty key -> none)."""
    g = ["s1_id", key]
    first = pl.col("p1").max().over(g)
    second = pl.col("p1").sort(descending=True).get(1, null_on_oob=True).over(g).fill_null(0)
    valid = pl.col(key) != ""
    return [pl.when(valid).then(pl.when(pl.col("p1") < first).then(first).otherwise(second)).otherwise(0).alias(f"{tag}_peer_max"),
            pl.when(valid).then(pl.len().over(g) - 1).otherwise(0).cast(pl.Float32).alias(f"{tag}_peer_n")]


def cluster_frame(df: pl.DataFrame, p1: np.ndarray) -> pl.DataFrame:
    d = df.select("s1_id", "cand_core", "cand_addr").with_columns(pl.Series("p1", p1.astype(np.float32)))
    return d.with_columns(_peer(d, "cand_core", "core") + _peer(d, "cand_addr", "addr")).select(CLUSTER_EXTRA)


def score(art: dict, df: pl.DataFrame, return_p1: bool = False):
    """Final match probability for every row of a feature frame (needs column s1_id).
    return_p1=True also returns the stage-1 probability (used as the learned candidate filter)."""
    p1 = art["model"].predict(df.select(art["features"]).to_numpy())
    if art.get("model2") is None:
        return (p1, p1) if return_p1 else p1
    parts = [df.select(art["features"]).to_numpy(), stage2_frame(df, p1).to_numpy()]
    if art.get("cluster"):
        parts.append(cluster_frame(df, p1).to_numpy())
    x2 = np.hstack(parts)
    p2 = art["model2"].predict(x2)
    return (p2, p1) if return_p1 else p2

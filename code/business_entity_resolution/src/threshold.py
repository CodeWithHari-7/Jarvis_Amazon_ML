"""Entity-level decision layer and macro-F0.5 threshold search.

A candidate is a match when its final probability >= threshold. Zero, one or many matches per S1 record are
allowed; an empty list is a valid (singleton) prediction. The threshold is chosen on leak-free validation
S1 records, reported overall and per country.
"""
import numpy as np
import polars as pl

from evaluate import macro_f05, blocking_recall

GRID = np.arange(0.2, 0.96, 0.025)


def decide(df: pl.DataFrame, prob: np.ndarray, thr: float) -> dict:
    """df has s1_id, cand_id aligned with prob -> {s1_id: set(matched ids)}."""
    keep = df.select("s1_id", "cand_id").filter(pl.Series(prob >= thr))
    pred = {}
    for s, c in keep.iter_rows():
        pred.setdefault(s, set()).add(c)
    return pred


def sweep(va: pl.DataFrame, p: np.ndarray, gt: dict, splits: pl.DataFrame, thresholds=GRID) -> dict:
    """Best threshold by macro F0.5 on validation S1 records (all of them, incl. those without candidates)."""
    val = splits.filter(pl.col("split") == "val")
    res = {}
    for c in [None] + val["country"].unique().sort().to_list():
        ids = (val.filter(pl.col("country") == c) if c else val)["s1_id"].to_list()
        mask = va["s1_id"].is_in(ids).to_numpy()
        sub = va.filter(pl.Series(mask))
        cands = {}
        for s, x in sub.select("s1_id", "cand_id").iter_rows():
            cands.setdefault(s, set()).add(x)
        best = max(((macro_f05(ids, gt, decide(sub, p[mask], th)), th) for th in thresholds), key=lambda z: z[0][0])
        res[c or "ALL"] = best
        print(f"  {c or 'ALL':6s} blockRecall={blocking_recall(ids, gt, cands):.4f}  thr={best[1]:.3f}  "
              f"F0.5={best[0][0]:.4f} P={best[0][1]:.4f} R={best[0][2]:.4f} singletonAcc={best[0][3]:.4f}", flush=True)
    return res

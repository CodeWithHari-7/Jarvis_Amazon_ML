"""Training-data construction (leak-free) and pairwise model training.

  python train.py data  [n_train_per_country] [n_val_per_country]
      Samples disjoint S1 train / validation records per country, blocks them against the FULL
      S2/S3 pool of that country, extracts features and labels -> pipeline_cache/pairs_train.parquet
  python train.py fit
      Trains LightGBM on train S1 records, reports leak-free macro F0.5 on validation S1 records.
"""
import os
import sys
import pickle
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from config import CACHE_DIR, MODEL_DIR, SEED
from load_data import normalized, ground_truth
from build_pairs import country_pairs
from features import feature_names
from threshold import sweep

PAIRS = os.path.join(CACHE_DIR, "pairs_train.parquet")


def make_data(n_train=50_000, n_val=10_000, country=None):
    """One subprocess per country keeps peak memory within 16 GB; pairs are concatenated at the end."""
    import subprocess
    if country is None:
        cs = normalized("train", 1)["country"].unique().sort().to_list()
        for c in cs:
            subprocess.run([sys.executable, __file__, "data", str(n_train), str(n_val), c], check=True)
        df = pl.concat([pl.read_parquet(os.path.join(CACHE_DIR, f"pairs_train_{c}.parquet")) for c in cs])
        df.write_parquet(PAIRS)
        print(df.shape, df["label"].mean())
        return
    gt = ground_truth()
    c = country
    sc = normalized("train", 1).filter(pl.col("country") == c).sample(shuffle=True, fraction=1.0, seed=SEED)
    sel = sc.head(n_train + n_val).with_columns(
        pl.when(pl.int_range(pl.len()) < n_train).then(pl.lit("train")).otherwise(pl.lit("val")).alias("split"))
    del sc
    pool = pl.concat([pl.scan_parquet(os.path.join(CACHE_DIR, f"train_s{s}_norm.parquet")).filter(pl.col("country") == c)
                      for s in (2, 3)]).collect()
    print(f"[{c}] {sel.height:,} S1 records", flush=True)
    f = country_pairs(sel.drop("split"), pool)
    del pool
    f = f.join(sel.select(pl.col("entity_id").alias("s1_id"), "split"), on="s1_id")
    sel.select(pl.col("entity_id").alias("s1_id"), "split", pl.lit(c).alias("country")).write_parquet(
        os.path.join(CACHE_DIR, f"s1_split_{c}.parquet"))
    lab = [x in gt[s] for s, x in zip(f["s1_id"].to_list(), f["cand_id"].to_list())]
    f.with_columns(pl.Series("label", lab, dtype=pl.Int8)).write_parquet(os.path.join(CACHE_DIR, f"pairs_train_{c}.parquet"))


def load_splits():
    return pl.concat([pl.read_parquet(os.path.join(CACHE_DIR, f)) for f in os.listdir(CACHE_DIR) if f.startswith("s1_split_")])


PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=SEED, num_threads=os.cpu_count())


def fit_two_stage(rounds1=800, rounds2=400, folds=4, save=True):
    """Stage 1 with out-of-fold predictions on train S1 records, stage 2 on top; leak-free val report."""
    import lightgbm as lgb
    from inference import stage2_frame, cluster_frame
    gt, sp = ground_truth(), load_splits()
    df = pl.read_parquet(PAIRS)
    if "cand_core" not in df.columns:           # pairs built before cluster features existed
        s23 = pl.concat([normalized("train", k).select("entity_id", "name_core", "addr_n") for k in (2, 3)])
        df = df.join(s23.rename({"entity_id": "cand_id", "name_core": "cand_core", "addr_n": "cand_addr"}),
                     on="cand_id", how="left", maintain_order="left")
        del s23
    feats = feature_names()
    tr, va = df.filter(pl.col("split") == "train"), df.filter(pl.col("split") == "val")
    Xtr, ytr, Xva = tr.select(feats).cast(pl.Float32).to_numpy(), tr["label"].to_numpy(), va.select(feats).cast(pl.Float32).to_numpy()
    fold = (tr["s1_id"].hash(SEED) % folds).to_numpy()          # folds grouped by S1 entity
    oof = np.zeros(len(tr))
    for k in range(folds):
        m = lgb.train(PARAMS, lgb.Dataset(Xtr[fold != k], ytr[fold != k]), rounds1)
        oof[fold == k] = m.predict(Xtr[fold == k])
    m1 = lgb.train(PARAMS, lgb.Dataset(Xtr, ytr), rounds1)
    p1 = m1.predict(Xva)
    print("stage 1:")
    r1 = sweep(va, p1, gt, sp)
    X2tr = np.hstack([Xtr, stage2_frame(tr, oof).to_numpy()])
    X2va = np.hstack([Xva, stage2_frame(va, p1).to_numpy()])
    m2 = lgb.train(PARAMS, lgb.Dataset(X2tr, ytr), rounds2)
    p2 = m2.predict(X2va)
    print("stage 2:")
    r2 = sweep(va, p2, gt, sp)
    X2tr = np.hstack([X2tr, cluster_frame(tr, oof).to_numpy()])
    X2va = np.hstack([X2va, cluster_frame(va, p1).to_numpy()])
    m2c = lgb.train(PARAMS, lgb.Dataset(X2tr, ytr), rounds2)
    p2c = m2c.predict(X2va)
    print("stage 2 + cluster context:")
    r2c = sweep(va, p2c, gt, sp)
    best = max([("stage1", r1, None, False), ("stage2", r2, m2, False), ("stage2+cluster", r2c, m2c, True)],
               key=lambda z: z[1]["ALL"][0][0])
    print("selected:", best[0])
    art = {"model": m1, "model2": best[2], "cluster": best[3], "features": feats,
           "threshold": float(best[1]["ALL"][1]), "val": {"stage1": r1, "stage2": r2, "stage2_cluster": r2c}}
    use2 = best[0]
    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
            pickle.dump(art, f)
        print(f"saved model.pkl (stage2={use2}, threshold={art['threshold']:.3f})")
    return art, va, p1, p2


if __name__ == "__main__":
    if sys.argv[1] == "data":
        make_data(*(int(a) for a in sys.argv[2:4]), *(sys.argv[4:5]))
    elif sys.argv[1] == "fit":
        fit_two_stage()

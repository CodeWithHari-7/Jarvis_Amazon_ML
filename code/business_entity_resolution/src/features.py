"""Pairwise + group-level features for (S1, candidate) pairs.

Vectorised with polars and rapidfuzz.process.cpdist (element-wise, multi-threaded C++).
Feature groups
  name   : Levenshtein / token-set / token-sort / partial / Jaro-Winkler on full and core names,
           compact (space-free) ratio, IDF-weighted token overlap, max IDF of unmatched tokens
           (distinguishes "Chavez and Devine" from sibling "Chavez and Devine SOUTHSIDE Group")
  address: string similarities, house-number agreement / conflict / numeric distance
           (siblings sit at 700 vs 709 Dupont Ave), street-token overlap
  flags  : missing address, non-Latin script, legal-form agreement, source (S2/S3)
  block  : blocking score / rank / number of shared keys
  group  : each feature's rank and gap to the best candidate of the same S1 record
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist


def _sim(a, b, scorer):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _col_idf(s23: pl.DataFrame, col: str) -> pl.DataFrame:
    n = s23.group_by("country").agg(pl.len().alias("N"))
    df = (s23.select("country", pl.col(col).str.split(" ").list.unique().alias("t")).explode("t")
             .group_by("country", "t").agg(pl.len().alias("df")))
    return df.join(n, on="country").select("country", "t", (pl.col("N") / pl.col("df")).log().cast(pl.Float32).alias("idf"))


def token_idf(s23: pl.DataFrame) -> dict:
    """Per-country IDF of core-name tokens and of address word tokens, from the S2/S3 pool (no labels).
    Address IDF down-weights city / region / "rue" / "road" so that a different STREET name stands out."""
    return {"name": _col_idf(s23, "name_core"), "addr": _col_idf(s23, "addr_alpha")}


def _idf_overlap(p: pl.DataFrame, idf: pl.DataFrame, col: str = "name_core", tag: str = "name") -> pl.DataFrame:
    """IDF-weighted jaccard and max IDF of tokens present on only one side."""
    def ex(c, side):
        return (p.select("pid", "country", pl.col(c).str.split(" ").list.unique().alias("t")).explode("t")
                 .filter(pl.col("t") != "").with_columns(pl.lit(side).alias("side")))
    t = pl.concat([ex(col + "_1", 1), ex(col + "_2", 2)])
    t = t.join(idf, on=["country", "t"], how="left").with_columns(pl.col("idf").fill_null(12.0))
    g = t.group_by("pid", "t").agg(pl.col("idf").first(), pl.col("side").n_unique().alias("both"), pl.col("side").first())
    return g.group_by("pid").agg(
        (pl.col("idf").filter(pl.col("both") == 2).sum() / pl.col("idf").sum()).alias(f"{tag}_idf_jacc"),
        pl.col("idf").filter((pl.col("both") == 1) & (pl.col("side") == 1)).max().fill_null(0).alias(f"{tag}_idf_only1"),
        pl.col("idf").filter((pl.col("both") == 1) & (pl.col("side") == 2)).max().fill_null(0).alias(f"{tag}_idf_only2"),
        pl.col("idf").filter(pl.col("both") == 2).max().fill_null(0).alias(f"{tag}_idf_shared_max"),
    )


def _nums(p: pl.DataFrame) -> pl.DataFrame:
    """House-number agreement features."""
    a = p["addr_nums_1"].to_list()
    b = p["addr_nums_2"].to_list()
    n = len(a)
    shared = np.zeros(n, np.float32); jac = np.zeros(n, np.float32); first_eq = np.zeros(n, np.float32)
    conflict = np.zeros(n, np.float32); dist = np.full(n, -1.0, np.float32); has = np.zeros(n, np.float32)
    for i in range(n):
        x = a[i].split() if a[i] else []
        y = b[i].split() if b[i] else []
        has[i] = (len(x) > 0) * 1 + (len(y) > 0) * 2
        if not x or not y:
            continue
        sx, sy = set(x), set(y)
        k = len(sx & sy)
        shared[i] = k
        jac[i] = k / len(sx | sy)
        first_eq[i] = x[0] == y[0]
        conflict[i] = k == 0
        if k == 0:
            ix = [int(v) for v in x if v.isdigit() and len(v) < 9]
            iy = [int(v) for v in y if v.isdigit() and len(v) < 9]
            if ix and iy:
                dist[i] = min(abs(u - v) for u in ix for v in iy)
            # prefix/containment (411 vs 4110, 10617 vs 10617a) counts as soft agreement
            if any(u.startswith(v) or v.startswith(u) for u in x for v in y):
                conflict[i] = 0.5
    return pl.DataFrame({"num_shared": shared, "num_jacc": jac, "num_first_eq": first_eq,
                         "num_conflict": conflict, "num_dist": dist, "num_has": has})


PAIR_FEATURES = [
    "name_ratio", "name_tset", "name_tsort", "name_partial", "name_jw", "core_ratio", "core_tset", "compact_ratio",
    "name_idf_jacc", "name_idf_only1", "name_idf_only2", "name_idf_shared_max",
    "addr_ratio", "addr_tset", "alpha_tset", "alpha_ratio",
    "num_shared", "num_jacc", "num_first_eq", "num_conflict", "num_dist", "num_has",
    "legal_eq", "nonlatin_1", "nonlatin_2", "addr_missing_1", "addr_missing_2", "src",
    "len_name_1", "len_name_2", "len_addr_1", "len_addr_2",
    "bscore", "bkeys", "brank",
    "addr_idf_jacc", "addr_idf_only1", "addr_idf_only2", "addr_idf_shared_max",
]
GROUP_BASE = ["name_ratio", "core_tset", "name_idf_jacc", "addr_tset", "num_jacc", "bscore", "addr_idf_jacc"]


def pair_features(pairs: pl.DataFrame, s1: pl.DataFrame, s23: pl.DataFrame, idf: pl.DataFrame) -> pl.DataFrame:
    """pairs: qid, rid, bscore, bkeys, brank.  s1/s23 have rid and normalised columns."""
    cols = ["rid", "name_n", "name_core", "legal", "addr_n", "addr_nums", "addr_alpha", "nonlatin", "addr_missing"]
    p = (pairs.with_row_index("pid")
              .join(s1.select([pl.col("rid").alias("qid"), "country"] + [pl.col(c).alias(c + "_1") for c in cols[1:]]), on="qid")
              .join(s23.select(["rid", "src"] + [pl.col(c).alias(c + "_2") for c in cols[1:]]), on="rid")
              .sort("pid"))
    n1, n2 = p["name_n_1"].to_list(), p["name_n_2"].to_list()
    c1, c2 = p["name_core_1"].to_list(), p["name_core_2"].to_list()
    a1, a2 = p["addr_n_1"].to_list(), p["addr_n_2"].to_list()
    w1, w2 = p["addr_alpha_1"].to_list(), p["addr_alpha_2"].to_list()
    k1 = [s.replace(" ", "") for s in c1]; k2 = [s.replace(" ", "") for s in c2]
    f = {
        "name_ratio": _sim(n1, n2, fuzz.ratio), "name_tset": _sim(n1, n2, fuzz.token_set_ratio),
        "name_tsort": _sim(n1, n2, fuzz.token_sort_ratio), "name_partial": _sim(n1, n2, fuzz.partial_ratio),
        "name_jw": _sim(n1, n2, JaroWinkler.normalized_similarity) * 100, "core_ratio": _sim(c1, c2, fuzz.ratio),
        "core_tset": _sim(c1, c2, fuzz.token_set_ratio), "compact_ratio": _sim(k1, k2, fuzz.ratio),
        "addr_ratio": _sim(a1, a2, fuzz.ratio), "addr_tset": _sim(a1, a2, fuzz.token_set_ratio),
        "alpha_tset": _sim(w1, w2, fuzz.token_set_ratio), "alpha_ratio": _sim(w1, w2, fuzz.ratio),
    }
    out = p.select("pid", "qid", "rid", "country", "bscore", "bkeys", "brank",
                   (pl.col("legal_1") == pl.col("legal_2")).cast(pl.Float32).alias("legal_eq"),
                   pl.col("nonlatin_1").cast(pl.Float32), pl.col("nonlatin_2").cast(pl.Float32),
                   pl.col("addr_missing_1").cast(pl.Float32), pl.col("addr_missing_2").cast(pl.Float32),
                   pl.col("src").cast(pl.Float32),
                   pl.col("name_n_1").str.len_chars().cast(pl.Float32).alias("len_name_1"),
                   pl.col("name_n_2").str.len_chars().cast(pl.Float32).alias("len_name_2"),
                   pl.col("addr_n_1").str.len_chars().cast(pl.Float32).alias("len_addr_1"),
                   pl.col("addr_n_2").str.len_chars().cast(pl.Float32).alias("len_addr_2"))
    out = out.with_columns([pl.Series(k, v) for k, v in f.items()])
    out = pl.concat([out, _nums(p)], how="horizontal")
    for col, tag in (("name_core", "name"), ("addr_alpha", "addr")):
        ov = _idf_overlap(p.select("pid", "country", col + "_1", col + "_2"), idf[tag], col, tag)
        out = out.join(ov, on="pid", how="left").with_columns(
            [pl.col(f"{tag}_idf_{k}").fill_null(0) for k in ("jacc", "only1", "only2", "shared_max")])
    return add_group_features(out.sort("pid"), GROUP_BASE, "g")


def add_group_features(df: pl.DataFrame, base, tag) -> pl.DataFrame:
    """Rank / gap-to-best of each base feature among candidates of the same S1 record."""
    ex = []
    for c in base:
        ex.append((pl.col(c).max().over("qid") - pl.col(c)).alias(f"{tag}_gap_{c}"))
        ex.append(pl.col(c).rank("dense", descending=True).over("qid").cast(pl.Float32).alias(f"{tag}_rank_{c}"))
    ex.append(pl.len().over("qid").cast(pl.Float32).alias(f"{tag}_ncand"))
    return df.with_columns(ex)


def feature_names(tag="g"):
    return PAIR_FEATURES + [f"{tag}_{k}_{c}" for c in GROUP_BASE for k in ("gap", "rank")] + [f"{tag}_ncand"]

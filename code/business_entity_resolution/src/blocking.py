"""Multi-key, IDF-weighted blocking implemented as vectorised polars joins.

Key families (all prefixed by country, so any country label works - France included):
  n  single core-name token            (rare tokens dominate through IDF weight)
  p  pair of core-name tokens          (handles common words like "global", "services")
  a  house number + address token      (finds matches whose NAME is unusable: DBA names, Tamil/Devanagari script)
  x  name prefix(4) + house number
  q  consecutive address token pair    (street names)
  f  exact core name
Each shared key adds weight w_type / log2(1 + df(key)); keys with df above a per-family cap are ignored.
The TOP_K highest scoring S2/S3 records per S1 record are the candidate set.
"""
import polars as pl

CAPS = {"n": 600, "p": 300, "a": 150, "x": 200, "q": 150, "f": 400, "s": 300, "c": 300}
WEIGHTS = {"n": 1.0, "p": 2.0, "a": 2.0, "x": 2.0, "q": 1.0, "f": 3.0, "s": 1.5, "c": 1.5}


def _skeleton(col):
    """Consonant skeleton of a token: transliteration variants collapse (krssnn / krishna -> krsn).
    Repeated letters are collapsed letter by letter (the regex engine has no back-references)."""
    e = pl.col(col).str.slice(0, 1) + pl.col(col).str.slice(1).str.replace_all("[aeiouyh]", "")
    for ch in "bcdfgjklmnpqrstvwxz":
        e = e.str.replace_all(ch + "{2,}", ch)
    return e


def _tok(df, col, name, min_len, max_n):
    return (df.select("rid", "country", pl.col(col).str.split(" ").list.head(max_n).alias(name))
              .explode(name).with_columns(pl.int_range(pl.len()).over("rid").alias(name + "_pos"))
              .filter(pl.col(name).str.len_chars() >= min_len))


class _HashedParts:
    """Hashes each key family as soon as it is built so raw key strings never pile up in memory."""
    def __init__(self):
        self.items = []

    def append(self, k):
        self.items.append(k.select(pl.col("rid"), pl.col("k").str.slice(0, 1).cast(pl.Categorical).alias("kt"),
                                   pl.concat_str("country", pl.lit(""), "k").hash(7).alias("key")))


def make_keys(df: pl.DataFrame) -> pl.DataFrame:
    """df: rid(int), country, name_core, addr_nums, addr_alpha -> (rid, key:u64, kt:str)."""
    base = df.select("rid", "country", "name_core", "addr_nums", "addr_alpha")
    nt = _tok(base, "name_core", "t", 3, 6).unique(["rid", "t"])
    nums = _tok(base, "addr_nums", "u", 1, 3)
    at = _tok(base, "addr_alpha", "w", 4, 8)
    parts = _HashedParts()
    parts.append(nt.select("rid", "country", pl.concat_str(pl.lit("n"), "t").alias("k")))
    pr = nt.join(nt.select("rid", pl.col("t").alias("t2")), on="rid").filter(pl.col("t") < pl.col("t2"))
    parts.append(pr.select("rid", "country", pl.concat_str(pl.lit("p"), "t", pl.lit("|"), "t2").alias("k")))
    au = nums.join(at.filter(pl.col("w_pos") < 6).select("rid", "w"), on="rid")
    parts.append(au.select("rid", "country", pl.concat_str(pl.lit("a"), "u", pl.lit("|"), "w").alias("k")))
    px = nums.join(base.select("rid", pl.col("name_core").str.replace_all(" ", "").str.slice(0, 4).alias("p4")), on="rid")
    parts.append(px.filter(pl.col("p4").str.len_chars() >= 3)
                   .select("rid", "country", pl.concat_str(pl.lit("x"), "p4", pl.lit("|"), "u").alias("k")))
    at_all = (base.select("rid", "country", pl.col("addr_alpha").str.split(" ").list.head(10).alias("w"))
                  .explode("w").with_columns(pl.int_range(pl.len()).over("rid").alias("pos")))
    qq = at_all.join(at_all.select("rid", pl.col("w").alias("w2"), (pl.col("pos") - 1).alias("pos")), on=["rid", "pos"])
    parts.append(qq.filter((pl.col("w").str.len_chars() >= 3) & (pl.col("w2").str.len_chars() >= 3))
                   .select("rid", "country", pl.concat_str(pl.lit("q"), "w", pl.lit("|"), "w2").alias("k")))
    parts.append(base.filter(pl.col("name_core").str.len_chars() >= 3)
                     .select("rid", "country", pl.concat_str(pl.lit("f"), "name_core").alias("k")))
    sk = nt.filter(pl.col("t_pos") < 4).with_columns(_skeleton("t").alias("sk")).filter(pl.col("sk").str.len_chars() >= 2).unique(["rid", "sk"])
    sp_ = sk.join(sk.select("rid", pl.col("sk").alias("sk2")), on="rid").filter(pl.col("sk") < pl.col("sk2"))
    parts.append(sp_.select("rid", "country", pl.concat_str(pl.lit("s"), "sk", pl.lit("|"), "sk2").alias("k")))
    cp = base.select("rid", "country", pl.col("name_core").str.replace_all(" ", "").alias("cc"))
    parts.append(cp.filter(pl.col("cc").str.len_chars() >= 7)
                   .select("rid", "country", pl.concat_str(pl.lit("c"), pl.col("cc").str.slice(0, 7)).alias("k")))
    return pl.concat(parts.items).unique(["rid", "key"])


def build_index(s23: pl.DataFrame) -> pl.DataFrame:
    """Posting table (key, rid, w) for S2/S3 records; frequent keys dropped per family cap."""
    # keys are built in slices so the intermediate token strings never exist for all records at once
    k = pl.concat([make_keys(s23.slice(i, 750_000)) for i in range(0, s23.height, 750_000)])
    df = k.group_by("key").agg(pl.len().alias("df"), pl.first("kt").cast(pl.Utf8))
    caps = pl.DataFrame({"kt": list(CAPS), "cap": list(CAPS.values()), "wt": [WEIGHTS[t] for t in CAPS]})
    df = df.join(caps, on="kt").filter(pl.col("df") <= pl.col("cap"))
    df = df.with_columns((pl.col("wt") / (1 + pl.col("df")).log(2)).cast(pl.Float32).alias("w"))
    return k.join(df.select("key", "w"), on="key").select("key", "rid", "w")


def retrieve(s1: pl.DataFrame, index: pl.DataFrame, top_k: int) -> pl.DataFrame:
    """(qid, rid, bscore, brank) top_k candidates per S1 record (s1 has rid = query id)."""
    q = make_keys(s1).select(pl.col("rid").alias("qid"), "key")
    hits = q.join(index, on="key").group_by("qid", "rid").agg(pl.col("w").sum().alias("bscore"),
                                                               pl.len().cast(pl.Int16).alias("bkeys"))
    return (hits.sort(["qid", "bscore"], descending=[False, True])
                .group_by("qid", maintain_order=True).head(top_k)
                .with_columns(pl.int_range(pl.len()).over("qid").cast(pl.Int16).alias("brank")))

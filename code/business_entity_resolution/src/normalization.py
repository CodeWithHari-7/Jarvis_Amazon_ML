"""Vectorised (polars) normalisation of business names and addresses.

Produces multiple views per record instead of one cleaned string:
  name_n   : transliterated, lower-cased, punctuation-free name with canonical legal forms
  name_core: name_n without legal forms / honorifics (used for blocking and core similarity)
  addr_n   : transliterated address with canonical street abbreviations (EN + FR + IN)
  addr_nums: list of house / plot / door numbers ("010/45" -> "10/45")
  addr_alpha: alphabetic address tokens
  nonlatin : original name contained a non-Latin script (Devanagari, Tamil, ...)
No country is hard-coded: every rule is language-level and applies to any country label.
"""
import re
import polars as pl
from unidecode import unidecode

LEGAL = {
    "private": "pvt", "pvt": "pvt", "limited": "ltd", "ltd": "ltd", "corporation": "corp", "corp": "corp",
    "incorporated": "inc", "inc": "inc", "company": "co", "co": "co", "llc": "llc", "llp": "llp", "pllc": "pllc",
    "pc": "pc", "lp": "lp", "plc": "plc", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl",
    "sci": "sci", "snc": "snc", "cie": "cie", "gmbh": "gmbh",
}
HONORIFIC = {"m s", "ms", "sri", "shri", "shree", "smt", "dr", "mr", "mrs", "the", "messrs"}
ADDR_MAP = {
    "road": "rd", "street": "st", "avenue": "ave", "av": "ave", "drive": "dr", "boulevard": "blvd", "bd": "blvd",
    "highway": "hwy", "lane": "ln", "court": "ct", "place": "pl", "parkway": "pkwy", "circle": "cir",
    "north": "n", "south": "s", "east": "e", "west": "w", "nrth": "n", "suite": "ste", "apartment": "apt",
    "r": "rue", "ch": "chemin", "imp": "impasse", "fbg": "faubourg", "rte": "route", "sq": "square",
    "mainroad": "main rd", "opp": "opposite", "nr": "near", "bldg": "building", "flr": "floor",
}
ADDR_DROP = {"no", "door", "d", "h", "plot", "flat", "unit", "n a", "na", "ste", "apt", "floor", "number"}

_non_latin = re.compile(r"[^\u0000-ɏ\s]")
_lead_digit = re.compile(r"\b8(?=[a-z]{2})")        # OCR noise: "8lack" -> "black"
_lead_zero = re.compile(r"\b0(?=[a-z]{2})")
_num = re.compile(r"\d+[a-z]?(?:[/-]\d+[a-z]?)*")


def _translit(s):
    return unidecode(s).lower() if s else ""


def _norm_num(t):
    return "/".join(p.lstrip("0") or "0" for p in re.split(r"[/-]", t))


def _name_tokens(s):
    s = _lead_zero.sub("o", _lead_digit.sub("b", s))
    s = s.replace("&", " and ").replace("m/s", " ")
    s = re.sub(r"(?<=\b[a-z])\.(?=[a-z]\b)", "", s)          # l.l.p -> llp
    s = re.sub(r"\.com\b", " ", s)
    toks = re.sub(r"[^a-z0-9]+", " ", s).split()
    return [LEGAL.get(t, t) for t in toks]


def _addr_tokens(s):
    s = re.sub(r"([a-z]{3,})(?=\d)", r" ", s.replace("#", " "))       # plot367 -> plot 367
    s = re.sub(r"(?<=[a-z])-|-(?=[a-z]{2})", " ", s)                      # nestglory-5 -> nestglory 5
    toks = re.sub(r"[^a-z0-9/\-]+", " ", s).split()
    out = []
    for t in toks:
        t = t.strip("-/")
        if not t:
            continue
        t = ADDR_MAP.get(t, t)
        out.extend(t.split())
    return out


def _row(name, addr):
    n_raw = name or ""
    nonlatin = bool(_non_latin.search(n_raw))
    nt = _name_tokens(_translit(n_raw))
    core = [t for t in nt if t not in LEGAL.values() and t not in HONORIFIC and t != "and"]
    if not core:
        core = nt
    legal = sorted({t for t in nt if t in LEGAL.values()})
    at = _addr_tokens(_translit((addr or "").replace("°", " ").replace("º", " ")))   # "N° 2" -> "n 2", not "n deg 2"
    nums = []
    for t in at:
        if any(c.isdigit() for c in t):
            v = _norm_num(t) if _num.fullmatch(t) else t
            if v not in nums:
                nums.append(v)
            lead = re.match(r"\d+", v)                      # 10401/c, 102a, 67/69 -> also 10401, 102, 67
            lead = lead.group(0).lstrip("0") if lead else ""
            if lead and lead != v and lead not in nums:
                nums.append(lead)
    alpha = [t for t in at if t.isalpha() and t not in ADDR_DROP and len(t) > 1]
    return (" ".join(nt), " ".join(core), " ".join(legal), " ".join(at), " ".join(nums), " ".join(alpha), nonlatin)


def normalize(df: pl.DataFrame) -> pl.DataFrame:
    """df has entity_id, business_name, business_address, country -> normalised frame."""
    rows = [_row(n, a) for n, a in zip(df["business_name"].to_list(), df["business_address"].to_list())]
    cols = list(zip(*rows)) if rows else [[] for _ in range(7)]
    out = pl.DataFrame({
        "entity_id": df["entity_id"],
        "country": df["country"].fill_null("").str.strip_chars().str.to_lowercase(),
        "name_n": pl.Series(cols[0], dtype=pl.Utf8),
        "name_core": pl.Series(cols[1], dtype=pl.Utf8),
        "legal": pl.Series(cols[2], dtype=pl.Utf8),
        "addr_n": pl.Series(cols[3], dtype=pl.Utf8),
        "addr_nums": pl.Series(cols[4], dtype=pl.Utf8),
        "addr_alpha": pl.Series(cols[5], dtype=pl.Utf8),
        "nonlatin": pl.Series(cols[6], dtype=pl.Boolean),
        "addr_missing": df["business_address"].is_null(),
    })
    return out

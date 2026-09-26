"""
Feature extraction for JARVIS_CHECKER Business Entity Resolution.

Generates a rich 34-feature vector for each (S1, S2/S3) candidate pair.

Feature groups:
  NAME:    name_exact, name_ratio, name_partial, name_token_set, name_token_sort,
           name_jw, name_token_jaccard, name_token_containment, name_is_abbrev,
           name_len_diff, name_prefix4_match, name_prefix3_match, name_both_present,
           name_levenshtein_norm, name_char3gram_overlap, name_mono_token
  ADDRESS: addr_exact, addr_ratio, addr_partial, addr_token_set, addr_token_sort,
           addr_jw, addr_token_jaccard, addr_num_overlap, addr_num_exact,
           addr_len_diff, addr_both_present, both_have_numbers, numbers_exactly_match
  COUNTRY: country_match, country_either_empty
  STRUCT:  source_is_s2, cross_script, name_suffix_match
"""
from rapidfuzz import fuzz, distance as rfz_dist
from rapidfuzz.distance import Levenshtein


def _token_jaccard(s1: str, s2: str) -> float:
    """Token-level Jaccard similarity."""
    t1 = set(s1.split())
    t2 = set(s2.split())
    if not t1 and not t2:
        return 0.0
    union = t1 | t2
    if not union:
        return 0.0
    return len(t1 & t2) / len(union)


def _token_containment(s1: str, s2: str) -> float:
    """max(|A∩B|/|A|, |A∩B|/|B|) — one is substring-like of the other."""
    t1 = set(s1.split())
    t2 = set(s2.split())
    if not t1 or not t2:
        return 0.0
    inter = len(t1 & t2)
    return max(inter / len(t1), inter / len(t2))


def _numeric_overlap(nums1: list, nums2: list) -> float:
    """Jaccard over numeric sequences."""
    s1, s2 = set(nums1), set(nums2)
    if not s1 and not s2:
        return 0.0
    union = s1 | s2
    return len(s1 & s2) / len(union)


def _numeric_exact(nums1: list, nums2: list) -> float:
    """1.0 if BOTH have numbers AND they share at least one; else 0."""
    s1, s2 = set(nums1), set(nums2)
    if not s1 or not s2:
        return 0.0
    return 1.0 if s1 & s2 else 0.0


def _is_abbreviation(short: str, long_: str) -> float:
    """Heuristic: check if short could be an initialism of long_."""
    if not short or not long_:
        return 0.0
    s_tokens = short.split()
    l_tokens = long_.split()
    if len(s_tokens) == 1 and len(l_tokens) >= 2:
        # Check if short is the initials of long
        initials = ''.join(t[0] for t in l_tokens if t)
        if short == initials or short.replace(' ', '') == initials:
            return 1.0
    return 0.0


def _char_ngram_overlap(s1: str, s2: str, n: int = 3) -> float:
    """Character n-gram Jaccard overlap between two strings."""
    if not s1 or not s2:
        return 0.0
    ng1 = {s1[i:i+n] for i in range(len(s1) - n + 1)}
    ng2 = {s2[i:i+n] for i in range(len(s2) - n + 1)}
    if not ng1 or not ng2:
        return 0.0
    union = ng1 | ng2
    return len(ng1 & ng2) / len(union)


def _suffix_match(s1: str, s2: str, n: int = 4) -> float:
    """Check if last n chars match — catches common 'mart', 'corp' suffixes."""
    if len(s1) < n or len(s2) < n:
        return 0.0
    return 1.0 if s1[-n:] == s2[-n:] else 0.0


def extract_features_for_pair(r1: dict, r2: dict) -> dict:
    """
    Generates a 34-feature vector for a (S1, S2/S3) candidate pair.
    r1 = S1 normalized record dict
    r2 = S2/S3 normalized record dict
    All features are in [0, 1] or {0, 1}.
    """
    n1 = r1.get('business_name_normalized', '') or ''
    n2 = r2.get('business_name_normalized', '') or ''
    a1 = r1.get('business_address_normalized', '') or ''
    a2 = r2.get('business_address_normalized', '') or ''
    c1 = r1.get('country_normalized', '') or ''
    c2 = r2.get('country_normalized', '') or ''
    nums1 = r1.get('business_address_numbers', []) or []
    nums2 = r2.get('business_address_numbers', []) or []

    # ── NAME FEATURES ────────────────────────────────────────────────────
    name_exact          = 1.0 if n1 and n2 and n1 == n2 else 0.0
    name_ratio          = fuzz.ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    name_partial        = fuzz.partial_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    name_token_set      = fuzz.token_set_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    name_token_sort     = fuzz.token_sort_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    name_jw             = rfz_dist.JaroWinkler.normalized_similarity(n1, n2) if n1 and n2 else 0.0
    name_token_jaccard  = _token_jaccard(n1, n2) if n1 and n2 else 0.0
    name_token_containment = _token_containment(n1, n2) if n1 and n2 else 0.0
    name_is_abbrev      = max(_is_abbreviation(n1, n2), _is_abbreviation(n2, n1))
    name_len_diff       = abs(len(n1) - len(n2)) / (max(len(n1), len(n2)) + 1)
    name_prefix4_match  = 1.0 if n1[:4] == n2[:4] and len(n1) >= 4 and len(n2) >= 4 else 0.0
    name_prefix3_match  = 1.0 if n1[:3] == n2[:3] and len(n1) >= 3 and len(n2) >= 3 else 0.0
    name_both_present   = 1.0 if n1 and n2 else 0.0
    # Normalized Levenshtein distance (0=identical, 1=completely different)
    name_levenshtein_norm = 1.0 - (Levenshtein.normalized_similarity(n1, n2) if n1 and n2 else 0.0)
    # Character 3-gram Jaccard (robust to OCR / transliteration errors)
    name_char3gram_overlap = _char_ngram_overlap(n1, n2, n=3) if n1 and n2 else 0.0
    # Suffix match (last 4 chars) — catches shared "mart", "shop", etc.
    name_suffix_match   = _suffix_match(n1, n2, n=4) if n1 and n2 else 0.0
    # Single-token name flag (abbreviations / short names)
    name_mono_token     = 1.0 if (n1 and len(n1.split()) == 1) or (n2 and len(n2.split()) == 1) else 0.0

    # ── ADDRESS FEATURES ─────────────────────────────────────────────────
    addr_exact          = 1.0 if a1 and a2 and a1 == a2 else 0.0
    addr_ratio          = fuzz.ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
    addr_partial        = fuzz.partial_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
    addr_token_set      = fuzz.token_set_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
    addr_token_sort     = fuzz.token_sort_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
    addr_jw             = rfz_dist.JaroWinkler.normalized_similarity(a1, a2) if a1 and a2 else 0.0
    addr_token_jaccard  = _token_jaccard(a1, a2) if a1 and a2 else 0.0
    addr_num_overlap    = _numeric_overlap(nums1, nums2)
    addr_num_exact      = _numeric_exact(nums1, nums2)
    addr_len_diff       = abs(len(a1) - len(a2)) / (max(len(a1), len(a2)) + 1) if (a1 or a2) else 0.0
    addr_both_present   = 1.0 if a1 and a2 else 0.0
    both_have_numbers   = 1.0 if nums1 and nums2 else 0.0
    numbers_exactly_match = 1.0 if nums1 and nums2 and set(nums1) == set(nums2) else 0.0

    # ── COUNTRY FEATURES ─────────────────────────────────────────────────
    country_match        = 1.0 if c1 and c2 and c1 == c2 else 0.0
    country_either_empty = 1.0 if not c1 or not c2 else 0.0

    # ── STRUCTURAL FEATURES ──────────────────────────────────────────────
    src_id       = r2.get('entity_id', '')
    source_is_s2 = 1.0 if str(src_id).startswith('S2') else 0.0
    cross_script = 1.0 if r1.get('is_indic', False) != r2.get('is_indic', False) else 0.0

    return {
        # Name (17)
        'name_exact':              name_exact,
        'name_ratio':              name_ratio,
        'name_partial':            name_partial,
        'name_token_set':          name_token_set,
        'name_token_sort':         name_token_sort,
        'name_jw':                 name_jw,
        'name_token_jaccard':      name_token_jaccard,
        'name_token_containment':  name_token_containment,
        'name_is_abbrev':          name_is_abbrev,
        'name_len_diff':           name_len_diff,
        'name_prefix4_match':      name_prefix4_match,
        'name_prefix3_match':      name_prefix3_match,
        'name_both_present':       name_both_present,
        'name_levenshtein_norm':   name_levenshtein_norm,
        'name_char3gram_overlap':  name_char3gram_overlap,
        'name_suffix_match':       name_suffix_match,
        'name_mono_token':         name_mono_token,
        # Address (13)
        'addr_exact':              addr_exact,
        'addr_ratio':              addr_ratio,
        'addr_partial':            addr_partial,
        'addr_token_set':          addr_token_set,
        'addr_token_sort':         addr_token_sort,
        'addr_jw':                 addr_jw,
        'addr_token_jaccard':      addr_token_jaccard,
        'addr_num_overlap':        addr_num_overlap,
        'addr_num_exact':          addr_num_exact,
        'addr_len_diff':           addr_len_diff,
        'addr_both_present':       addr_both_present,
        'both_have_numbers':       both_have_numbers,
        'numbers_exactly_match':   numbers_exactly_match,
        # Country (2)
        'country_match':           country_match,
        'country_either_empty':    country_either_empty,
        # Structural (3)
        'source_is_s2':            source_is_s2,
        'cross_script':            cross_script,
        'name_suffix_match_struct': name_suffix_match,
    }


# Canonical feature column order — must match training
FEATURE_COLS = [
    # Name
    'name_exact', 'name_ratio', 'name_partial', 'name_token_set', 'name_token_sort',
    'name_jw', 'name_token_jaccard', 'name_token_containment', 'name_is_abbrev',
    'name_len_diff', 'name_prefix4_match', 'name_prefix3_match', 'name_both_present',
    'name_levenshtein_norm', 'name_char3gram_overlap', 'name_suffix_match', 'name_mono_token',
    # Address
    'addr_exact', 'addr_ratio', 'addr_partial', 'addr_token_set', 'addr_token_sort',
    'addr_jw', 'addr_token_jaccard', 'addr_num_overlap', 'addr_num_exact',
    'addr_len_diff', 'addr_both_present', 'both_have_numbers', 'numbers_exactly_match',
    # Country
    'country_match', 'country_either_empty',
    # Structural
    'source_is_s2', 'cross_script', 'name_suffix_match_struct',
]

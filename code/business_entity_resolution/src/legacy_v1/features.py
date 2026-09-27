"""
ML Challenge 2026 — Business Entity Resolution
Module: features.py

Extracts string similarity, token overlap, address numerical alignment,
and character n-gram cosine similarities for candidate entity pairs.
Optimized via C-accelerated RapidFuzz routines.
"""

from typing import Dict, Any, List
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

FEATURE_NAMES = [
    'name_fuzz_ratio',
    'name_token_set',
    'name_jw',
    'addr_fuzz_ratio',
    'addr_token_set',
    'addr_num_overlap',
    'legal_suffix_match',
    'name_qgram_sim'
]


def compute_char_qgram_sim(s1: str, s2: str, q: int = 3) -> float:
    """Computes fast character q-gram Dice/Cosine similarity."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    
    n1 = len(s1) - q + 1
    n2 = len(s2) - q + 1
    if n1 <= 0 or n2 <= 0:
        return 1.0 if s1 == s2 else 0.0
        
    grams1 = set(s1[i:i+q] for i in range(n1))
    grams2 = set(s2[i:i+q] for i in range(n2))
    
    intersection = len(grams1 & grams2)
    return (2.0 * intersection) / (len(grams1) + len(grams2))


def extract_pair_features(
    rec1: Dict[str, Any],
    rec2: Dict[str, Any]
) -> List[float]:
    """
    Extracts dense numerical feature vector for an (S1, S2/S3) candidate pair:
    1. name_fuzz_ratio: Levenshtein ratio on normalized business name
    2. name_token_set: Token set ratio (invariant to word order / transpositions)
    3. name_jw: Jaro-Winkler prefix-weighted string distance
    4. addr_fuzz_ratio: Levenshtein ratio on normalized address
    5. addr_token_set: Token set ratio on address components
    6. addr_num_overlap: Jaccard overlap on numerical tokens (street/plot/PIN)
    7. legal_suffix_match: Concordance of legal forms (LLC, Pvt Ltd, Inc, SARL)
    8. name_qgram_sim: Character 3-gram Dice similarity (TF-IDF surrogate)
    """
    n1, n2 = rec1['norm_name'], rec2['norm_name']
    a1, a2 = rec1['norm_addr'], rec2['norm_addr']

    # Name String Metrics
    name_fuzz = fuzz.ratio(n1, n2) / 100.0
    name_tok = fuzz.token_set_ratio(n1, n2) / 100.0
    name_jw = JaroWinkler.similarity(n1, n2)

    # Address String Metrics
    addr_fuzz = fuzz.ratio(a1, a2) / 100.0 if (a1 and a2) else 0.0
    addr_tok = fuzz.token_set_ratio(a1, a2) / 100.0 if (a1 and a2) else 0.0

    # Address Numbers Overlap
    num1, num2 = rec1['nums'], rec2['nums']
    if num1 and num2:
        num_overlap = len(num1 & num2) / len(num1 | num2)
    elif not num1 and not num2:
        num_overlap = 0.8  # neutral indicator when neither address has numbers
    else:
        num_overlap = 0.0  # discrepancy: one has numbers, the other doesn't

    # Legal Suffix Concordance
    suf1, suf2 = rec1.get('suffix_tokens', set()), rec2.get('suffix_tokens', set())
    if suf1 and suf2:
        suffix_match = 1.0 if (suf1 & suf2) else 0.0
    elif not suf1 and not suf2:
        suffix_match = 0.5
    else:
        suffix_match = 0.3  # one explicitly mentions legal suffix, one doesn't

    # Character Q-gram Cosine/Dice
    qgram_sim = compute_char_qgram_sim(n1, n2, q=3)

    return [
        name_fuzz,
        name_tok,
        name_jw,
        addr_fuzz,
        addr_tok,
        num_overlap,
        suffix_match,
        qgram_sim
    ]

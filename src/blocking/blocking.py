"""
Blocking / Candidate Generation module for JARVIS_CHECKER.

Multi-pass blocking strategy using complementary inverted indexes:
    Pass 1: country_norm + name_prefix_4
    Pass 2: country_norm + name_prefix_3  (catches short names)
    Pass 3: country_norm + name_prefix_2  (catches 2-char abbreviations)
    Pass 4: country_norm + address_numbers (each number separately)
    Pass 5: country_norm + individual name_tokens (>= 4 chars)
    Pass 6: country_norm + individual addr_tokens (city/locality matching)
    Pass 7: country_norm + name bigrams (high recall for scrambled names)

All passes are UNIONed. Recall is the primary objective.
Bucket size cap prevents O(N) explosion on very common tokens.
"""
import time
from collections import defaultdict
from typing import List, Tuple, Set

# Maximum number of candidates retrieved per bucket — raised for higher recall
MAX_PREFIX_BUCKET   = 150
MAX_NUM_BUCKET      = 100
MAX_NAME_TOK_BUCKET = 80
MAX_ADDR_TOK_BUCKET = 60
MAX_BIGRAM_BUCKET   = 40

MIN_NUMBER_LEN_FOR_BLOCKING = 2

# Generic tokens to skip in name blocking (too common, low precision)
_GENERIC_NAME_TOKENS = {
    'road', 'street', 'near', 'shop', 'opp', 'cross', 'lane', 'west', 'east',
    'north', 'south', 'main', 'new', 'old', 'india', 'united', 'states', 'france',
    'pvt', 'ltd', 'llc', 'inc', 'corp', 'limited', 'company', 'services',
    'trading', 'enterprises', 'group', 'international', 'national',
}

# Generic address tokens to skip
_GENERIC_ADDR_TOKENS = {
    'road', 'street', 'avenue', 'lane', 'near', 'opp', 'opposite', 'plot',
    'house', 'floor', 'building', 'sector', 'phase', 'area', 'nagar', 'marg',
    'drive', 'place', 'court', 'boulevard', 'extension',
}


def _name_bigrams(name: str) -> List[str]:
    """Generate character bigrams from the first 8 chars of a name."""
    s = name[:8]
    return [s[i:i+2] for i in range(len(s)-1)] if len(s) >= 2 else []


def build_s23_index(s23_records: List[dict]) -> dict:
    """
    Builds complementary inverted indexes from S2/S3 records.
    Returns a dict of indexes.
    """
    prefix4_idx  = defaultdict(list)  # country+prefix4 -> [eids]
    prefix3_idx  = defaultdict(list)  # country+prefix3 -> [eids]
    prefix2_idx  = defaultdict(list)  # country+prefix2 -> [eids]
    addr_num_idx = defaultdict(list)  # country+num -> [eids]
    name_tok_idx = defaultdict(list)  # country+name_token -> [eids]
    addr_tok_idx = defaultdict(list)  # country+addr_token -> [eids]
    bigram_idx   = defaultdict(list)  # country+bigram -> [eids]

    for r in s23_records:
        eid = r.get('entity_id')
        if not eid:
            continue
        c = r.get('country_normalized', '')
        if not c:
            continue

        name   = r.get('business_name_normalized', '')
        prefix4 = r.get('name_prefix4', name[:4] if len(name) >= 4 else name)
        prefix3 = r.get('name_prefix3', name[:3] if len(name) >= 3 else name)
        prefix2 = name[:2] if len(name) >= 2 else name

        # Pass 1: country + 4-char prefix
        if prefix4:
            b = prefix4_idx[f"{c}\x00{prefix4}"]
            if len(b) < MAX_PREFIX_BUCKET:
                b.append(eid)

        # Pass 2: country + 3-char prefix
        if prefix3 and prefix3 != prefix4:
            b = prefix3_idx[f"{c}\x00{prefix3}"]
            if len(b) < MAX_PREFIX_BUCKET:
                b.append(eid)

        # Pass 3: country + 2-char prefix (for very short / abbreviated names)
        if prefix2 and len(prefix2) == 2:
            b = prefix2_idx[f"{c}\x00{prefix2}"]
            if len(b) < MAX_PREFIX_BUCKET:
                b.append(eid)

        # Pass 4: country + address numbers (high discriminative power)
        nums = r.get('business_address_numbers', [])
        for num in nums:
            if len(num) >= MIN_NUMBER_LEN_FOR_BLOCKING:
                b = addr_num_idx[f"{c}\x00{num}"]
                if len(b) < MAX_NUM_BUCKET:
                    b.append(eid)

        # Pass 5: country + name tokens (>= 4 chars, non-generic)
        name_tokens = r.get('name_tokens', set())
        for tok in name_tokens:
            if len(tok) >= 4 and tok not in _GENERIC_NAME_TOKENS:
                b = name_tok_idx[f"{c}\x00{tok}"]
                if len(b) < MAX_NAME_TOK_BUCKET:
                    b.append(eid)

        # Pass 6: country + addr tokens (>= 4 chars, non-generic)
        addr_tokens = r.get('addr_tokens', set())
        for tok in addr_tokens:
            if len(tok) >= 4 and tok not in _GENERIC_ADDR_TOKENS:
                b = addr_tok_idx[f"{c}\x00{tok}"]
                if len(b) < MAX_ADDR_TOK_BUCKET:
                    b.append(eid)

        # Pass 7: name character bigrams (catches transpositions / OCR errors)
        for bg in _name_bigrams(name):
            b = bigram_idx[f"{c}\x00{bg}"]
            if len(b) < MAX_BIGRAM_BUCKET:
                b.append(eid)

    return {
        'prefix4':  prefix4_idx,
        'prefix3':  prefix3_idx,
        'prefix2':  prefix2_idx,
        'addr_num': addr_num_idx,
        'name_tok': name_tok_idx,
        'addr_tok': addr_tok_idx,
        'bigram':   bigram_idx,
    }


def generate_candidates_for_record(r: dict, indexes: dict) -> Set[str]:
    """
    Returns the set of S2/S3 candidate IDs for a single S1 record.
    Uses all 7 index passes for maximum recall.
    """
    cands = set()
    c = r.get('country_normalized', '')
    if not c:
        return cands

    name    = r.get('business_name_normalized', '')
    prefix4 = r.get('name_prefix4', name[:4] if len(name) >= 4 else name)
    prefix3 = r.get('name_prefix3', name[:3] if len(name) >= 3 else name)
    prefix2 = name[:2] if len(name) >= 2 else name

    # Pass 1
    if prefix4:
        for cid in indexes['prefix4'].get(f"{c}\x00{prefix4}", []):
            cands.add(cid)

    # Pass 2
    if prefix3 and prefix3 != prefix4:
        for cid in indexes['prefix3'].get(f"{c}\x00{prefix3}", []):
            cands.add(cid)

    # Pass 3
    if prefix2 and len(prefix2) == 2:
        for cid in indexes.get('prefix2', {}).get(f"{c}\x00{prefix2}", []):
            cands.add(cid)

    # Pass 4
    nums = r.get('business_address_numbers', [])
    for num in nums:
        if len(num) >= MIN_NUMBER_LEN_FOR_BLOCKING:
            for cid in indexes['addr_num'].get(f"{c}\x00{num}", []):
                cands.add(cid)

    # Pass 5
    name_tokens = r.get('name_tokens', set())
    for tok in name_tokens:
        if len(tok) >= 4:
            for cid in indexes['name_tok'].get(f"{c}\x00{tok}", []):
                cands.add(cid)

    # Pass 6
    addr_tokens = r.get('addr_tokens', set())
    for tok in addr_tokens:
        if len(tok) >= 4 and tok not in _GENERIC_ADDR_TOKENS:
            for cid in indexes.get('addr_tok', {}).get(f"{c}\x00{tok}", []):
                cands.add(cid)

    # Pass 7: character bigrams
    for bg in _name_bigrams(name):
        for cid in indexes.get('bigram', {}).get(f"{c}\x00{bg}", []):
            cands.add(cid)

    return cands


def generate_candidates(s1_records: List[dict], s23_records: List[dict]) -> List[Tuple[str, str]]:
    """
    Full candidate generation pipeline.
    Returns list of (s1_entity_id, s23_entity_id) tuples.
    """
    print("  Building inverted indexes for S2/S3...")
    t0 = time.time()
    indexes = build_s23_index(s23_records)
    print(f"  Index built in {time.time()-t0:.2f}s")

    print("  Querying indexes for each S1...")
    t0 = time.time()
    candidates = []
    for r in s1_records:
        s1_id = r['entity_id']
        cands = generate_candidates_for_record(r, indexes)
        for cid in cands:
            candidates.append((s1_id, cid))

    print(f"  Candidate generation done in {time.time()-t0:.2f}s")
    print(f"  Total candidate pairs: {len(candidates):,}")
    return candidates

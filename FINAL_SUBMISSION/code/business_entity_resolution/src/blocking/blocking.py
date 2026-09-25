"""
Blocking / Candidate Generation module for JARVIS_CHECKER.

Multi-pass blocking strategy using complementary inverted indexes:
    Pass 1: country_norm + name_prefix_4
    Pass 2: country_norm + name_prefix_3  (catches short names / different spelling)
    Pass 3: country_norm + address_numbers (each number separately)
    Pass 4: country_norm + individual name_tokens  (token-level, capped per bucket)
    Pass 5: country_norm + first addr_token  (city/locality matching)

All passes are UNIONed. Recall is the primary objective.
Bucket size cap prevents O(N) explosion on very common tokens.
"""
import time
from collections import defaultdict
from typing import List, Tuple, Set

# Maximum number of S2/S3 entities in a single blocking bucket.
# Prevents memory/time explosion on very common tokens like "pvt ltd" or "123".
MAX_BUCKET_SIZE = 3000

# For address number pass, only use numbers with >= 2 digits
# (single-digit "1", "2" etc. are too noisy)
MIN_NUMBER_LEN_FOR_BLOCKING = 2


def build_s23_index(s23_records: List[dict]) -> dict:
    """
    Builds all inverted indexes from S2/S3 records.
    Returns a dict of indexes.
    """
    prefix4_idx = defaultdict(list)   # country+prefix4 -> [eids]
    prefix3_idx = defaultdict(list)   # country+prefix3 -> [eids]
    addr_num_idx = defaultdict(list)  # country+num -> [eids]
    name_tok_idx = defaultdict(list)  # country+token -> [eids]
    addr_tok_idx = defaultdict(list)  # country+first_addr_tok -> [eids]

    for r in s23_records:
        eid = r['entity_id']
        c = r.get('country_normalized', '')

        name = r.get('business_name_normalized', '')
        prefix4 = r.get('name_prefix4', name[:4] if len(name) >= 4 else name)
        prefix3 = r.get('name_prefix3', name[:3] if len(name) >= 3 else name)

        # Pass 1: country + 4-char prefix
        if c and prefix4:
            prefix4_idx[f"{c}\x00{prefix4}"].append(eid)

        # Pass 2: country + 3-char prefix (for short names)
        if c and prefix3 and prefix3 != prefix4:
            prefix3_idx[f"{c}\x00{prefix3}"].append(eid)

        # Pass 3: country + address numbers
        nums = r.get('business_address_numbers', [])
        for num in nums:
            if len(num) >= MIN_NUMBER_LEN_FOR_BLOCKING and c:
                addr_num_idx[f"{c}\x00{num}"].append(eid)

        # Pass 4: country + individual name tokens (cap bucket)
        tokens = r.get('name_tokens', set())
        if isinstance(tokens, list):
            tokens = set(tokens)
        for tok in tokens:
            if len(tok) >= 3:  # skip very short tokens
                key = f"{c}\x00{tok}"
                name_tok_idx[key].append(eid)

        # Pass 5: country + first address token (city/locality)
        addr_norm = r.get('business_address_normalized', '')
        if addr_norm and c:
            addr_parts = addr_norm.split()
            # Use the LAST token (often city/state) AND first token
            for tok in [addr_parts[0], addr_parts[-1]] if len(addr_parts) > 1 else addr_parts[:1]:
                if len(tok) >= 3:
                    addr_tok_idx[f"{c}\x00{tok}"].append(eid)

    return {
        'prefix4': prefix4_idx,
        'prefix3': prefix3_idx,
        'addr_num': addr_num_idx,
        'name_tok': name_tok_idx,
        'addr_tok': addr_tok_idx,
    }


def generate_candidates_for_record(r: dict, indexes: dict) -> Set[str]:
    """
    Returns the set of S2/S3 candidate IDs for a single S1 record.
    """
    cands = set()
    c = r.get('country_normalized', '')

    name = r.get('business_name_normalized', '')
    prefix4 = r.get('name_prefix4', name[:4] if len(name) >= 4 else name)
    prefix3 = r.get('name_prefix3', name[:3] if len(name) >= 3 else name)

    prefix4_idx = indexes['prefix4']
    prefix3_idx = indexes['prefix3']
    addr_num_idx = indexes['addr_num']
    name_tok_idx = indexes['name_tok']
    addr_tok_idx = indexes['addr_tok']

    # Pass 1: country + 4-char prefix
    if c and prefix4:
        bucket = prefix4_idx.get(f"{c}\x00{prefix4}", [])
        if len(bucket) <= MAX_BUCKET_SIZE:
            cands.update(bucket)
        else:
            # Too generic, use pass 2 instead
            pass

    # Pass 2: country + 3-char prefix
    if c and prefix3 and prefix3 != prefix4:
        bucket = prefix3_idx.get(f"{c}\x00{prefix3}", [])
        if len(bucket) <= MAX_BUCKET_SIZE:
            cands.update(bucket)

    # Pass 3: country + address numbers
    nums = r.get('business_address_numbers', [])
    for num in nums:
        if len(num) >= MIN_NUMBER_LEN_FOR_BLOCKING and c:
            bucket = addr_num_idx.get(f"{c}\x00{num}", [])
            if len(bucket) <= MAX_BUCKET_SIZE:
                cands.update(bucket)

    # Pass 4: individual name tokens
    tokens = r.get('name_tokens', set())
    if isinstance(tokens, list):
        tokens = set(tokens)
    for tok in tokens:
        if len(tok) >= 3:
            bucket = name_tok_idx.get(f"{c}\x00{tok}", [])
            if len(bucket) <= MAX_BUCKET_SIZE:
                cands.update(bucket)

    # Pass 5: first/last address token (city/locality)
    addr_norm = r.get('business_address_normalized', '')
    if addr_norm and c:
        addr_parts = addr_norm.split()
        for tok in ([addr_parts[0], addr_parts[-1]] if len(addr_parts) > 1 else addr_parts[:1]):
            if len(tok) >= 3:
                bucket = addr_tok_idx.get(f"{c}\x00{tok}", [])
                if len(bucket) <= MAX_BUCKET_SIZE:
                    cands.update(bucket)

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

import os
import sys
import time
from collections import defaultdict, Counter
import itertools
from typing import Dict, List, Set, Tuple, Any, Optional
import unidecode
import re
import polars as pl
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

print("=" * 80)
print("STEP 1: FIX BLOCKING RECALL AT SCALE (2-TOKEN PAIRS + COMPOSITE KEYS)")
print("=" * 80)

LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'limited', 'ltd',
    'private', 'pvt', 'co', 'company', 'llp', 'pllc', 'sa', 'sarl', 'sas',
    'sasu', 'eurl', 'gmbh', 'bv', 'nv', 'spa', 'srl', 'cie', 'snc'
}

HONORIFICS_LEADING = {
    'sri', 'shri', 'm/s', 'ms', 'mr', 'mrs', 'dr', 'the', 'incorporated', 'inc',
    'corp', 'corporation', 'llc', 'limited', 'ltd', 'private', 'pvt', 'co', 'company',
    'sa', 'sarl', 'sas', 'sasu', 'eurl', 'gmbh', 'bv', 'nv', 'spa', 'srl', 'cie', 'snc',
    'dr.', 'mr.', 'mrs.', 'm/s.'
}

ABBREVIATIONS = [
    (r'\bcorp\b', 'corporation'),
    (r'\bltd\b', 'limited'),
    (r'\bpvt\b', 'private'),
    (r'\brd\b', 'road'),
    (r'\bst\b', 'street'),
    (r'\bave\b', 'avenue'),
    (r'\bdr\b', 'drive'),
    (r'\bblvd\b', 'boulevard'),
    (r'\bhwy\b', 'highway'),
    (r'\bco\b', 'company'),
    (r'\+', ' and '),
    (r'&', ' and '),
]

STOPWORDS = {'and', 'the', 'of', 'in', 'at', 'on', 'de', 'la', 'le', 'les', 'des', 'du', 'en', 'et', 'und'}

clean_re = re.compile(r'[^a-z0-9\s]')
num_re = re.compile(r'\b\d+[a-z]?\b')
addr_unit_patterns = [
    re.compile(r'#\s*(\d+[a-z]?)'),
    re.compile(r'\bunit\s*#?\s*(\d+[a-z]?)'),
    re.compile(r'\bplot\s*#?\s*(?:no\.?)?\s*(\d+[a-z]?)'),
    re.compile(r'\bdoor\s*(?:no\.?)?\s*(\d+[a-z]?)'),
    re.compile(r'\bflat\s*(?:no\.?)?\s*(\d+[a-z]?)'),
    re.compile(r'\broom\s*(?:no\.?)?\s*(\d+[a-z]?)'),
    re.compile(r'\bh\.?\s*no\.?\s*(\d+[a-z]?)'),
    re.compile(r'^\s*(\d+[a-z]?)\b'),
]


def normalize_record_fast(name: Optional[str], addr: Optional[str], country: Optional[str]) -> Dict[str, Any]:
    country_clean = str(country or '').strip().lower()

    # Name normalization
    raw_name_str = unidecode.unidecode(str(name or '')).lower()
    for pattern, repl in ABBREVIATIONS:
        raw_name_str = re.sub(pattern, repl, raw_name_str)
    clean_name = clean_re.sub(' ', raw_name_str)
    raw_tokens = [tok for tok in clean_name.split() if tok]

    name_tokens = list(raw_tokens)
    while name_tokens and name_tokens[0] in HONORIFICS_LEADING:
        name_tokens = name_tokens[1:]
    if not name_tokens:
        name_tokens = list(raw_tokens)

    core_tokens = [tok for tok in name_tokens if tok not in LEGAL_SUFFIXES]
    if not core_tokens:
        core_tokens = name_tokens
    core_name = ' '.join(core_tokens)

    prefix4 = core_name[:4] if len(core_name) >= 3 else core_name
    prefix5 = core_name[:5] if len(core_name) >= 4 else core_name
    prefix6 = core_name[:6] if len(core_name) >= 5 else core_name

    clean_core_tokens = [tok for tok in core_tokens if tok not in STOPWORDS]
    if not clean_core_tokens:
        clean_core_tokens = core_tokens
    sorted_tokens = ' '.join(sorted(clean_core_tokens[:4]))

    # Distinctive tokens (len >= 3, not stopword, not legal suffix)
    distinct_tokens = [w for w in core_tokens if len(w) >= 3 and w not in STOPWORDS and w not in LEGAL_SUFFIXES]

    # Generate 2-token combination pairs (sorted tuple)
    token_pairs = set()
    if len(distinct_tokens) >= 2:
        for t1, t2 in itertools.combinations(sorted(set(distinct_tokens)), 2):
            token_pairs.add((t1, t2))
    elif len(distinct_tokens) == 1:
        token_pairs.add((distinct_tokens[0], '_SINGLE_'))

    # Address normalization & Numbers
    raw_addr_str = unidecode.unidecode(str(addr or '')).lower()
    for pattern, repl in ABBREVIATIONS:
        raw_addr_str = re.sub(pattern, repl, raw_addr_str)
    clean_addr = clean_re.sub(' ', raw_addr_str)
    clean_addr = ' '.join(clean_addr.split()).strip()

    raw_nums = num_re.findall(clean_addr)
    addr_nums = [n for n in raw_nums if len(n) <= 8]

    key_nums = set()
    for pat in addr_unit_patterns:
        for m in pat.findall(raw_addr_str):
            if len(m) <= 8:
                key_nums.add(m.lower())

    # Composite prefix + num keys
    prefix_num_keys = set()
    if prefix4 and addr_nums:
        for num in addr_nums[:2]:
            prefix_num_keys.add((prefix4, num))

    return {
        'country': country_clean,
        'core_name': core_name,
        'prefix4': prefix4,
        'prefix5': prefix5,
        'prefix6': prefix6,
        'sorted_tokens': sorted_tokens,
        'distinct_tokens': set(distinct_tokens),
        'token_pairs': token_pairs,
        'norm_addr': clean_addr,
        'nums': set(addr_nums),
        'key_nums': key_nums,
        'prefix_num_keys': prefix_num_keys
    }


class ScaledBlockingIndex:
    """
    Scale-Resilient Inverted Index:
    1. 2-Token Combination Pairs (solves 6M scale without candidate explosion)
    2. Prefix-4 with Prefix-5/6 fallbacks
    3. Prefix + Address Number composite keys
    4. Sorted core tokens
    5. Key unit/door numbers
    6. Rare single distinctive tokens (freq <= 800)
    """
    def __init__(self, pair_cap: int = 500, single_cap: int = 800, num_cap: int = 200):
        self.idx_prefix: Dict[str, List[str]] = defaultdict(list)
        self.idx_prefix5: Dict[str, List[str]] = defaultdict(list)
        self.idx_prefix6: Dict[str, List[str]] = defaultdict(list)
        self.idx_sorted_tok: Dict[str, List[str]] = defaultdict(list)
        self.idx_tok_pairs: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_single_tok: Dict[str, List[str]] = defaultdict(list)
        self.idx_prefix_num: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_key_num: Dict[str, List[str]] = defaultdict(list)
        self.pair_cap = pair_cap
        self.single_cap = single_cap
        self.num_cap = num_cap

    def add_record(self, entity_id: str, norm_rec: Dict[str, Any]):
        p4 = norm_rec.get('prefix4')
        if p4:
            self.idx_prefix[p4].append(entity_id)
        p5 = norm_rec.get('prefix5')
        if p5:
            self.idx_prefix5[p5].append(entity_id)
        p6 = norm_rec.get('prefix6')
        if p6:
            self.idx_prefix6[p6].append(entity_id)

        st = norm_rec.get('sorted_tokens')
        if st:
            self.idx_sorted_tok[st].append(entity_id)

        # 2-Token Pairs
        for pair in norm_rec.get('token_pairs', set()):
            self.idx_tok_pairs[pair].append(entity_id)

        # Single tokens for rare words
        for tok in norm_rec.get('distinct_tokens', set()):
            self.idx_single_tok[tok].append(entity_id)

        # Prefix + Num
        for pn in norm_rec.get('prefix_num_keys', set()):
            self.idx_prefix_num[pn].append(entity_id)

        # Key unit numbers
        for kn in norm_rec.get('key_nums', set()):
            self.idx_key_num[kn].append(entity_id)

    def retrieve_candidates(self, norm_rec: Dict[str, Any], max_candidates: int = 40) -> List[str]:
        cand_scores = defaultdict(int)

        # Signal 1: Prefix-Num Composite Key (Very high precision + recall for businesses with addresses)
        for pn in norm_rec.get('prefix_num_keys', set()):
            pn_m = self.idx_prefix_num.get(pn, [])
            if pn_m and len(pn_m) <= 200:
                for eid in pn_m:
                    cand_scores[eid] += 16

        # Signal 2: 2-Token Pairs (Solves high-frequency business tokens when combined with 2nd token)
        for pair in norm_rec.get('token_pairs', set()):
            pair_m = self.idx_tok_pairs.get(pair, [])
            if pair_m and len(pair_m) <= self.pair_cap:
                for eid in pair_m:
                    cand_scores[eid] += 14

        # Signal 3: Sorted tokens
        st = norm_rec.get('sorted_tokens')
        if st:
            st_m = self.idx_sorted_tok.get(st, [])
            if st_m and len(st_m) <= 400:
                for eid in st_m:
                    cand_scores[eid] += 12

        # Signal 4: Hierarchical Prefix (4 -> 5 -> 6)
        p4 = norm_rec.get('prefix4', '')
        if p4:
            p4_m = self.idx_prefix.get(p4, [])
            if len(p4_m) <= 300:
                for eid in p4_m:
                    cand_scores[eid] += 11
            else:
                p5 = norm_rec.get('prefix5', '')
                p5_m = self.idx_prefix5.get(p5, [])
                if p5_m and len(p5_m) <= 300:
                    for eid in p5_m:
                        cand_scores[eid] += 11
                else:
                    p6 = norm_rec.get('prefix6', '')
                    p6_m = self.idx_prefix6.get(p6, [])
                    if p6_m and len(p6_m) <= 300:
                        for eid in p6_m:
                            cand_scores[eid] += 10
                    elif p5_m:
                        for eid in p5_m[:100]:
                            cand_scores[eid] += 6

        # Signal 5: Rare single tokens (freq <= 800)
        for tok in norm_rec.get('distinct_tokens', set()):
            tok_m = self.idx_single_tok.get(tok, [])
            if tok_m and len(tok_m) <= self.single_cap:
                weight = 10 if len(tok_m) <= 200 else 7
                for eid in tok_m:
                    cand_scores[eid] += weight

        # Signal 6: Specific Key Unit / Door numbers
        for kn in norm_rec.get('key_nums', set()):
            kn_m = self.idx_key_num.get(kn, [])
            if kn_m and len(kn_m) <= self.num_cap:
                for eid in kn_m:
                    cand_scores[eid] += 6

        # Rank and return top candidates
        top_candidates = sorted(cand_scores.keys(), key=lambda x: cand_scores[x], reverse=True)[:max_candidates]
        return top_candidates


def main():
    t0_start = time.time()
    
    # 1. Load Ground Truth and evaluation sample (1,000 US entities)
    print("Loading Ground Truth and sampling 1,000 holdout US entities...")
    gt = pl.read_csv("dataset/train/train_ground_truth.tsv", separator='\t')
    s1 = pl.read_csv("dataset/train/train_source1.tsv", separator='\t')

    gt_matched = gt.filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != ''))
    s1_us_ids = set(s1.filter(pl.col('country') == 'US')['entity_id'].to_list())

    sample_us_gt = gt_matched.filter(pl.col('source1_entity_id').is_in(s1_us_ids)).sample(1000, seed=42)
    s1_eval_ids = set(sample_us_gt['source1_entity_id'].to_list())
    s1_eval_df = s1.filter(pl.col('entity_id').is_in(s1_eval_ids))

    gt_map = {}
    for r in sample_us_gt.to_dicts():
        gt_map[r['source1_entity_id']] = set(r['matched_entity_ids'].split(','))

    tot_true_matches = sum(len(v) for v in gt_map.values())
    print(f"Sampled {len(s1_eval_ids)} S1 entities. Total true matches: {tot_true_matches:,}")

    # 2. Ingest Full US candidate pool from raw TSVs (6,186,873 records)
    print("\nIngesting FULL Train S2 and S3 for US (6.18M+ candidate pool)...")
    t_load = time.time()
    s2 = pl.read_csv("dataset/train/train_source2.tsv", separator='\t', columns=['entity_id', 'business_name', 'business_address', 'country']).filter(pl.col('country') == 'US')
    s3 = pl.read_csv("dataset/train/train_source3.tsv", separator='\t', columns=['entity_id', 'business_name', 'business_address', 'country']).filter(pl.col('country') == 'US')
    s23 = pl.concat([s2, s3]).unique(subset=['entity_id'])
    del s2, s3
    n_pool = len(s23)
    print(f"Loaded {n_pool:,} unique candidate records in {time.time()-t_load:.1f}s.")

    # 3. Build ScaledBlockingIndex
    print("\nBuilding ScaledBlockingIndex (2-token pairs + prefix-num composite + hierarchical prefix)...")
    t_idx = time.time()
    blocker = ScaledBlockingIndex(pair_cap=500, single_cap=800, num_cap=200)

    eids = s23['entity_id'].to_list()
    names = s23['business_name'].to_list()
    addrs = s23['business_address'].to_list()
    countries = s23['country'].to_list()
    del s23

    # Batch normalization & addition
    for i in range(n_pool):
        rec = normalize_record_fast(names[i], addrs[i], countries[i])
        blocker.add_record(eids[i], rec)
        if (i + 1) % 1500000 == 0:
            print(f"  Indexed {i+1:,}/{n_pool:,} records ({time.time()-t_idx:.1f}s)...")

    print(f"Full ScaledBlockingIndex built in {time.time()-t_idx:.1f}s.")
    print(f"  2-token pair keys : {len(blocker.idx_tok_pairs):,}")
    print(f"  Prefix-num keys   : {len(blocker.idx_prefix_num):,}")
    print(f"  Single token keys : {len(blocker.idx_single_tok):,}")

    # 4. Evaluate Candidate Recall on Held-Out Sample against FULL 6.18M CORPUS
    print("\nEvaluating Candidate Recall against FULL 6.18M CORPUS (NO POSITIVE INJECTION)...")
    s1_records = {r['entity_id']: normalize_record_fast(r['business_name'], r['business_address'], r['country']) 
                  for r in s1_eval_df.to_dicts()}

    t_eval = time.time()
    hits_top40 = 0
    zero_hit_entities = 0
    cands_per_entity = []

    for s1_id, r1 in s1_records.items():
        true_m = gt_map[s1_id]
        cands = blocker.retrieve_candidates(r1, max_candidates=40)
        cands_set = set(cands)
        cands_per_entity.append(len(cands))

        hits = len(cands_set & true_m)
        hits_top40 += hits
        if hits == 0:
            zero_hit_entities += 1

    recall_ceiling = (hits_top40 / tot_true_matches) * 100
    zero_hit_rate = (zero_hit_entities / len(s1_records)) * 100
    avg_cands = np.mean(cands_per_entity)

    print("\n" + "=" * 80)
    print("STEP 1 RESULTS ON REAL FULL-SCALE 6.18M CORPUS:")
    print("=" * 80)
    print(f"Previous Baseline Candidate Recall : 58.26%")
    print(f"New Scaled Candidate Recall (Top-40): {recall_ceiling:.2f}%")
    print(f"Previous Zero-Hit Entity Rate      : 16.00%")
    print(f"New Zero-Hit Entity Rate           : {zero_hit_rate:.2f}% ({zero_hit_entities}/{len(s1_records)})")
    print(f"Average Candidates Retrieved       : {avg_cands:.2f} / entity")
    print(f"Evaluation Time (1,000 entities)   : {time.time()-t_eval:.2f}s")
    print(f"Total Test Runtime                 : {time.time()-t0_start:.1f}s")
    print("=" * 80)


if __name__ == '__main__':
    main()

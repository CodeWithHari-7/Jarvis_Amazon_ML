"""
Test new blocking implementation on the 10,000 holdout set:
Measures new blocking recall ceiling (overall, US, India)
Measures candidate set size growth and reduction ratio.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
import os
import re
import time
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any, Optional
import unidecode

import polars as pl
import numpy as np

# Open-set legal entity forms
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


def normalize_record_v2(name: Optional[str], addr: Optional[str], country: Optional[str]) -> Dict[str, Any]:
    country_clean = str(country or '').strip().lower()

    # 1. Name Transliteration & Normalization
    raw_name_str = unidecode.unidecode(str(name or '')).lower()
    for pattern, repl in ABBREVIATIONS:
        raw_name_str = re.sub(pattern, repl, raw_name_str)
        
    clean_name = clean_re.sub(' ', raw_name_str)
    raw_tokens = [tok for tok in clean_name.split() if tok]

    # a. Honorific / Leading noise stripping
    name_tokens = list(raw_tokens)
    while name_tokens and name_tokens[0] in HONORIFICS_LEADING:
        name_tokens = name_tokens[1:]
    if not name_tokens:
        name_tokens = list(raw_tokens)

    # Core tokens
    core_tokens = [tok for tok in name_tokens if tok not in LEGAL_SUFFIXES]
    if not core_tokens:
        core_tokens = name_tokens
    core_name = ' '.join(core_tokens)

    # Collapsed repeated characters for Indic Devanagari transliteration (haaii investtmeNtts -> hai investments)
    collapsed_tokens = set()
    for tok in core_tokens:
        col = re.sub(r'(.)\1+', r'\1', tok)
        if len(col) >= 4 and col != tok:
            collapsed_tokens.add(col)

    # Prefixes
    prefix4 = core_name[:4] if len(core_name) >= 3 else core_name
    prefix5 = core_name[:5] if len(core_name) >= 4 else core_name

    # Sorted core tokens
    clean_core = [w for w in core_tokens if w not in STOPWORDS]
    if not clean_core:
        clean_core = core_tokens
    sorted_tokens = ' '.join(sorted(clean_core[:4]))

    # Distinctive tokens (len >= 4, not stopword, not legal suffix)
    distinct_tokens = set(w for w in core_tokens if len(w) >= 4 and w not in STOPWORDS and w not in LEGAL_SUFFIXES)

    # 2. Address Normalization & Units
    raw_addr_str = unidecode.unidecode(str(addr or '')).lower()
    for pattern, repl in ABBREVIATIONS:
        raw_addr_str = re.sub(pattern, repl, raw_addr_str)
    clean_addr = clean_re.sub(' ', raw_addr_str)
    clean_addr = ' '.join(clean_addr.split())

    # All numbers
    nums = set(n for n in num_re.findall(clean_addr) if len(n) <= 8)

    # Specific unit/door/plot/street numbers
    key_nums = set()
    for pat in addr_unit_patterns:
        for m in pat.findall(raw_addr_str):
            if len(m) <= 8:
                key_nums.add(m.lower())

    return {
        'country': country_clean,
        'norm_name': ' '.join(name_tokens),
        'core_name': core_name,
        'prefix4': prefix4,
        'prefix5': prefix5,
        'sorted_tokens': sorted_tokens,
        'distinct_tokens': distinct_tokens,
        'collapsed_tokens': collapsed_tokens,
        'norm_addr': clean_addr,
        'nums': nums,
        'key_nums': key_nums
    }


class EnhancedBlockingIndex:
    def __init__(self, token_frequency_cap: int = 300, num_frequency_cap: int = 150):
        self.idx_prefix: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_prefix5: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_sorted_tok: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_any_tok: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_addr_num: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_key_num: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.token_freq_cap = token_frequency_cap
        self.num_freq_cap = num_frequency_cap

    def add_record(self, entity_id: str, norm_rec: Dict[str, Any]):
        c = norm_rec['country']
        
        # 1. Prefix
        if norm_rec.get('prefix4'):
            self.idx_prefix[(c, norm_rec['prefix4'])].append(entity_id)
        if norm_rec.get('prefix5'):
            self.idx_prefix5[(c, norm_rec['prefix5'])].append(entity_id)

        # 2. Sorted tokens
        if norm_rec.get('sorted_tokens'):
            self.idx_sorted_tok[(c, norm_rec['sorted_tokens'])].append(entity_id)

        # 3. Any distinctive token (len >= 4) + Collapsed transliterated tokens
        for tok in norm_rec.get('distinct_tokens', set()):
            self.idx_any_tok[(c, tok)].append(entity_id)
        for c_tok in norm_rec.get('collapsed_tokens', set()):
            self.idx_any_tok[(c, c_tok)].append(entity_id)

        # 4. Address numbers
        for num in norm_rec.get('nums', set()):
            self.idx_addr_num[(c, num)].append(entity_id)
        for knum in norm_rec.get('key_nums', set()):
            self.idx_key_num[(c, knum)].append(entity_id)

    def retrieve_candidates(self, norm_rec: Dict[str, Any], max_candidates: int = 40) -> List[str]:
        c = norm_rec['country']
        cand_scores = defaultdict(int)

        # Signal 1: Prefix-4 match
        p4 = norm_rec.get('prefix4', '')
        if p4:
            p_m = self.idx_prefix.get((c, p4), [])
            if len(p_m) <= 300:
                for eid in p_m: cand_scores[eid] += 12
            else:
                p5_m = self.idx_prefix5.get((c, norm_rec.get('prefix5', '')), [])
                if p5_m and len(p5_m) <= 300:
                    for eid in p5_m: cand_scores[eid] += 12
                else:
                    for eid in p_m[:100]: cand_scores[eid] += 8

        # Signal 2: Sorted tokens match
        st = norm_rec.get('sorted_tokens', '')
        if st:
            st_m = self.idx_sorted_tok.get((c, st), [])
            if len(st_m) <= 300:
                for eid in st_m: cand_scores[eid] += 10

        # Signal 3: Any distinctive token match (Fixes word-order inversion & prefix noise)
        for tok in norm_rec.get('distinct_tokens', set()):
            tok_m = self.idx_any_tok.get((c, tok), [])
            if len(tok_m) <= self.token_freq_cap:
                for eid in tok_m: cand_scores[eid] += 9

        # Signal 3b: Collapsed transliterated token match
        for c_tok in norm_rec.get('collapsed_tokens', set()):
            c_m = self.idx_any_tok.get((c, c_tok), [])
            if len(c_m) <= self.token_freq_cap:
                for eid in c_m: cand_scores[eid] += 8

        # Signal 4: Specific Key Unit / Door / Plot numbers
        for knum in norm_rec.get('key_nums', set()):
            kn_m = self.idx_key_num.get((c, knum), [])
            if len(kn_m) <= self.num_freq_cap:
                for eid in kn_m: cand_scores[eid] += 6

        # Signal 5: General address numbers (only distinctive buckets <= 60)
        for num in norm_rec.get('nums', set()):
            n_m = self.idx_addr_num.get((c, num), [])
            if len(n_m) <= 60:
                for eid in n_m: cand_scores[eid] += 4

        # Rank candidates by total signal score
        top_candidates = sorted(cand_scores.keys(), key=lambda x: cand_scores[x], reverse=True)[:max_candidates]
        return top_candidates


# Run Evaluation on 10k holdout
print("=" * 80)
print("EVALUATING ENHANCED BLOCKING ON 10,000 TRAIN HOLDOUT ENTITIES")
print("=" * 80)

# Load holdout sample generated earlier
from scratch.root_cause_diagnosis import holdout_s1_ids, holdout_s1_set, gt_map, s1_holdout_df, s23_df, true_matches_s23, s1_us_ids, s1_in_ids

print(f"Loaded {len(holdout_s1_ids):,} holdout S1 entities.")
tot_gt = sum(len(m) for m in gt_map.values())
print(f"Total True Positive Pairs: {tot_gt:,}")

# Ingest with new normalization
t0 = time.time()
s1_norm2 = {r['entity_id']: normalize_record_v2(r['business_name'], r['business_address'], r['country']) for r in s1_holdout_df.to_dicts()}
s23_norm2 = {r['entity_id']: normalize_record_v2(r['business_name'], r['business_address'], r['country']) for r in s23_df.to_dicts()}

# Build Enhanced Index
enhanced_blocker = EnhancedBlockingIndex(token_frequency_cap=300, num_frequency_cap=150)
for eid, rec in s23_norm2.items():
    enhanced_blocker.add_record(eid, rec)
print(f"Enhanced Index built in {time.time()-t0:.1f}s.")

# Measure Recall Ceiling
hits_all, tot_all = 0, 0
hits_us, tot_us = 0, 0
hits_in, tot_in = 0, 0
total_cands_retrieved = 0

for s1_id in holdout_s1_ids:
    r1 = s1_norm2[s1_id]
    c_name = r1['country']
    true_m = gt_map[s1_id]
    
    cands = set(enhanced_blocker.retrieve_candidates(r1, max_candidates=40))
    total_cands_retrieved += len(cands)
    
    if true_m:
        cap = len(cands & true_m)
        hits_all += cap
        tot_all += len(true_m)
        if c_name == 'us':
            hits_us += cap
            tot_us += len(true_m)
        else:
            hits_in += cap
            tot_in += len(true_m)

rec_all = hits_all / tot_all
rec_us = hits_us / tot_us
rec_in = hits_in / tot_in
avg_cands = total_cands_retrieved / len(holdout_s1_ids)

print("\n" + "=" * 80)
print("BLOCKING RECALL CEILING: BEFORE VS AFTER ENHANCEMENT")
print("=" * 80)
print(f"{'Region':<12} | {'Before Recall':<15} | {'After Recall':<15} | {'Delta Recall':<15}")
print("-" * 80)
print(f"{'Overall':<12} | {'51.55%':<15} | {rec_all*100:>14.2f}% | {rec_all*100 - 51.55:>+14.2f}%")
print(f"{'US':<12} | {'51.06%':<15} | {rec_us*100:>14.2f}% | {rec_us*100 - 51.06:>+14.2f}%")
print(f"{'India':<12} | {'52.02%':<15} | {rec_in*100:>14.2f}% | {rec_in*100 - 52.02:>+14.2f}%")

print(f"\nCandidate Pool Statistics:")
print(f"  Average candidates retrieved per S1 entity: {avg_cands:.2f} / 40.0")
reduction_ratio = 1.0 - (total_cands_retrieved / (len(holdout_s1_ids) * len(s23_norm2)))
print(f"  Reduction Ratio: {reduction_ratio*100:.4f}% (No uncontrolled candidate explosion!)")
print("=" * 80)

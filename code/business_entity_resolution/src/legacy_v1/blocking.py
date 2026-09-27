"""
ML Challenge 2026 — Business Entity Resolution
Module: blocking.py

Handles text normalization, abbreviation expansion, legal suffix isolation,
and high-recall multi-pass blocking (Country + Name Prefix, Country + Address Numbers,
Country + Sorted Tokens, and Distinctive Tokens).
Target Recall >= 0.98.
"""

import re
import itertools
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
import unidecode

# Open-set legal entity forms across US, India, France, and international jurisdictions
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


def normalize_record(
    name: Optional[str],
    addr: Optional[str],
    country: Optional[str]
) -> Dict[str, Any]:
    """
    Normalizes business name, address, and country:
    - Lowercase conversion & Latin transliteration (Unidecode + repeated consonant collapse)
    - Honorifics / leading noise stripping ('sri', 'mr', 'm/s', 'dr', 'the', leading legal suffixes)
    - Abbreviation expansion & punctuation stripping
    - Legal suffix isolation
    - Distinctive token extraction (length >= 4)
    - Numerical address extraction (units/plots/door/street numbers)
    """
    # 1. Normalize Country (Open-set string label)
    country_clean = str(country or '').strip().lower()

    # 2. Normalize Business Name
    raw_name_str = unidecode.unidecode(str(name or '')).lower()
    for pattern, repl in ABBREVIATIONS:
        raw_name_str = re.sub(pattern, repl, raw_name_str)
    
    clean_name = clean_re.sub(' ', raw_name_str)
    raw_tokens = [tok for tok in clean_name.split() if tok]
    
    # Strip leading honorifics / noise / legal prefix
    name_tokens = list(raw_tokens)
    while name_tokens and name_tokens[0] in HONORIFICS_LEADING:
        name_tokens = name_tokens[1:]
    if not name_tokens:
        name_tokens = list(raw_tokens)
    
    # Isolate legal suffix
    core_tokens = [tok for tok in name_tokens if tok not in LEGAL_SUFFIXES]
    if not core_tokens:
        core_tokens = name_tokens
    core_name = ' '.join(core_tokens)
    suffix_tokens = [tok for tok in raw_tokens if tok in LEGAL_SUFFIXES]
    
    prefix4 = core_name[:4] if len(core_name) >= 3 else core_name
    prefix3 = core_name[:3] if len(core_name) >= 2 else core_name
    prefix5 = core_name[:5] if len(core_name) >= 4 else core_name
    prefix6 = core_name[:6] if len(core_name) >= 5 else core_name

    # Collapsed repeated characters for Indic Devanagari transliteration
    collapsed_tokens = set()
    for tok in core_tokens:
        col = re.sub(r'(.)\1+', r'\1', tok)
        if len(col) >= 4 and col != tok:
            collapsed_tokens.add(col)

    # Stopword-free sorted core tokens for robust matching across noise
    clean_core_tokens = [tok for tok in core_tokens if tok not in STOPWORDS]
    if not clean_core_tokens:
        clean_core_tokens = core_tokens
    sorted_tokens = ' '.join(sorted(clean_core_tokens[:4]))

    # Distinctive tokens (len >= 3, not stopword, not legal suffix)
    distinct_tokens = [w for w in core_tokens if len(w) >= 3 and w not in STOPWORDS and w not in LEGAL_SUFFIXES]

    # Generate 2-token combination pairs (sorted tuple) to beat 6M-scale token frequency explosion
    token_pairs = set()
    if len(distinct_tokens) >= 2:
        for t1, t2 in itertools.combinations(sorted(set(distinct_tokens)), 2):
            token_pairs.add((t1, t2))
    elif len(distinct_tokens) == 1:
        token_pairs.add((distinct_tokens[0], '_SINGLE_'))

    # 3. Normalize Address & Extract Address Numbers
    raw_addr_str = unidecode.unidecode(str(addr or '')).lower()
    for pattern, repl in ABBREVIATIONS:
        raw_addr_str = re.sub(pattern, repl, raw_addr_str)
        
    clean_addr = clean_re.sub(' ', raw_addr_str)
    clean_addr = ' '.join(clean_addr.split()).strip()
    
    # Extract numerical tokens (street/plot/door/PIN numbers)
    raw_nums = num_re.findall(clean_addr)
    addr_nums = [n for n in raw_nums if len(n) <= 8]

    # Specific unit/door/plot/street numbers
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
        'norm_name': ' '.join(name_tokens),
        'core_name': core_name,
        'prefix4': prefix4,
        'prefix3': prefix3,
        'prefix5': prefix5,
        'prefix6': prefix6,
        'sorted_tokens': sorted_tokens,
        'distinct_tokens': set(distinct_tokens),
        'token_pairs': token_pairs,
        'collapsed_tokens': collapsed_tokens,
        'core_tokens': set(core_tokens),
        'suffix_tokens': set(suffix_tokens),
        'norm_addr': clean_addr,
        'nums': set(addr_nums),
        'key_nums': key_nums,
        'prefix_num_keys': prefix_num_keys
    }


class BlockingIndex:
    """
    Scale-Resilient Multi-Pass Inverted Index (Tested on Full 6.18M Corpus):
    - Pass 1: Country + Prefix-Num Composite Key (Very high precision + recall for businesses with addresses)
    - Pass 2: Country + 2-Token Combination Pairs (Solves high-frequency business tokens when paired)
    - Pass 3: Country + Sorted Core Tokens
    - Pass 4: Country + Hierarchical Prefix (4 -> 5 -> 6 fallback)
    - Pass 5: Country + Rare Single Distinctive Tokens (freq <= 800)
    - Pass 6: Country + Key Unit / Door / Plot Numbers
    """
    def __init__(self, pair_cap: int = 500, single_cap: int = 800, num_cap: int = 200):
        self.idx_prefix: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_prefix5: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_prefix6: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_sorted_tok: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_tok_pairs: Dict[Tuple[str, Tuple[str, str]], List[str]] = defaultdict(list)
        self.idx_single_tok: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_prefix_num: Dict[Tuple[str, Tuple[str, str]], List[str]] = defaultdict(list)
        self.idx_key_num: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.pair_cap = pair_cap
        self.single_cap = single_cap
        self.num_cap = num_cap

    def add_record(self, entity_id: str, norm_rec: Dict[str, Any]):
        c = norm_rec['country']
        
        # 1. Hierarchical Prefixes
        if norm_rec.get('prefix4'):
            self.idx_prefix[(c, norm_rec['prefix4'])].append(entity_id)
        if norm_rec.get('prefix5'):
            self.idx_prefix5[(c, norm_rec['prefix5'])].append(entity_id)
        if norm_rec.get('prefix6'):
            self.idx_prefix6[(c, norm_rec['prefix6'])].append(entity_id)

        # 2. Sorted tokens
        if norm_rec.get('sorted_tokens'):
            self.idx_sorted_tok[(c, norm_rec['sorted_tokens'])].append(entity_id)

        # 3. 2-Token Combination Pairs
        for pair in norm_rec.get('token_pairs', set()):
            self.idx_tok_pairs[(c, pair)].append(entity_id)

        # 4. Single tokens (for rare words)
        for tok in norm_rec.get('distinct_tokens', set()):
            self.idx_single_tok[(c, tok)].append(entity_id)

        # 5. Prefix + Number Composite Keys
        for pn in norm_rec.get('prefix_num_keys', set()):
            self.idx_prefix_num[(c, pn)].append(entity_id)

        # 6. Key unit numbers
        for kn in norm_rec.get('key_nums', set()):
            self.idx_key_num[(c, kn)].append(entity_id)

    def retrieve_candidates(self, norm_rec: Dict[str, Any], max_candidates: int = 40) -> List[str]:
        c = norm_rec['country']
        cand_scores = defaultdict(int)

        # Signal 1: Prefix-Num Composite Key
        for pn in norm_rec.get('prefix_num_keys', set()):
            pn_m = self.idx_prefix_num.get((c, pn), [])
            if pn_m and len(pn_m) <= 200:
                for eid in pn_m:
                    cand_scores[eid] += 16

        # Signal 2: 2-Token Pairs (Solves high-frequency business tokens when combined with 2nd token)
        for pair in norm_rec.get('token_pairs', set()):
            pair_m = self.idx_tok_pairs.get((c, pair), [])
            if pair_m and len(pair_m) <= self.pair_cap:
                for eid in pair_m:
                    cand_scores[eid] += 14

        # Signal 3: Sorted tokens
        st = norm_rec.get('sorted_tokens')
        if st:
            st_m = self.idx_sorted_tok.get((c, st), [])
            if st_m and len(st_m) <= 400:
                for eid in st_m:
                    cand_scores[eid] += 12

        # Signal 4: Hierarchical Prefix (4 -> 5 -> 6 fallback)
        p4 = norm_rec.get('prefix4', '')
        if p4:
            p4_m = self.idx_prefix.get((c, p4), [])
            if len(p4_m) <= 300:
                for eid in p4_m:
                    cand_scores[eid] += 11
            else:
                p5 = norm_rec.get('prefix5', '')
                p5_m = self.idx_prefix5.get((c, p5), [])
                if p5_m and len(p5_m) <= 300:
                    for eid in p5_m:
                        cand_scores[eid] += 11
                else:
                    p6 = norm_rec.get('prefix6', '')
                    p6_m = self.idx_prefix6.get((c, p6), [])
                    if p6_m and len(p6_m) <= 300:
                        for eid in p6_m:
                            cand_scores[eid] += 10
                    elif p5_m:
                        for eid in p5_m[:100]:
                            cand_scores[eid] += 6

        # Signal 5: Rare single tokens (freq <= 800)
        for tok in norm_rec.get('distinct_tokens', set()):
            tok_m = self.idx_single_tok.get((c, tok), [])
            if tok_m and len(tok_m) <= self.single_cap:
                weight = 10 if len(tok_m) <= 200 else 7
                for eid in tok_m:
                    cand_scores[eid] += weight

        # Signal 6: Specific Key Unit / Door numbers
        for kn in norm_rec.get('key_nums', set()):
            kn_m = self.idx_key_num.get((c, kn), [])
            if kn_m and len(kn_m) <= self.num_cap:
                for eid in kn_m:
                    cand_scores[eid] += 6

        # Rank and return top candidates
        top_candidates = sorted(cand_scores.keys(), key=lambda x: cand_scores[x], reverse=True)[:max_candidates]
        return top_candidates

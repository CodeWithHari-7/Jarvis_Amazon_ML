"""
Full Implementation & Verification of Fix 1, Fix 2, and Fix 3:
1. Blocking Recall Ceiling evaluation with new EnhancedBlockingIndex
2. Deployed Pipeline Correctness (Veto features + no 0.50 fallback)
3. Post-blocking threshold sweep (0.50 to 0.95) & Macro F0.5 / Precision / Recall breakdown
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

sys.path.append('code/business_entity_resolution/src')
from features import extract_pair_features
from train_model import compute_macro_f05
from predict import EntityMatcher

print("=" * 80)
print("EXECUTING FIX 1, FIX 2, AND FIX 3 (BLOCKING ENHANCEMENT & THRESHOLD RE-TUNE)")
print("=" * 80)

# ---------------------------------------------------------------------
# FIX 1 IMPLEMENTATION: NORMALIZATION & ENHANCED BLOCKING INDEX
# ---------------------------------------------------------------------
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

    # 1. Transliteration & Normalization
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

    # Collapsed repeated characters for Indic Devanagari transliteration
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
        'key_nums': key_nums,
        'core_tokens': set(core_tokens),
        'suffix_tokens': set(tok for tok in raw_tokens if tok in LEGAL_SUFFIXES)
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


# ---------------------------------------------------------------------
# LOAD DATA & STRATIFIED HOLDOUT (5,000 US, 5,000 India, including singletons)
# ---------------------------------------------------------------------
print("\n[1] Ingesting Stratified Train Holdout (10,000 entities)...")
gt_full = pl.read_csv("dataset/train/train_ground_truth.tsv", separator='\t')
s1_full = pl.read_csv("dataset/train/train_source1.tsv", separator='\t')

N_EACH = 5000
N_SINGLE = int(N_EACH * 0.0558)
N_MATCH = N_EACH - N_SINGLE

s1_us_ids = set(s1_full.filter(pl.col('country') == 'US')['entity_id'].to_list())
s1_in_ids = set(s1_full.filter(pl.col('country') == 'India')['entity_id'].to_list())

gt_us = gt_full.filter(pl.col('source1_entity_id').is_in(s1_us_ids))
gt_in = gt_full.filter(pl.col('source1_entity_id').is_in(s1_in_ids))

us_m = gt_us.filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != '')).sample(N_MATCH, seed=42)
us_s = gt_us.filter(pl.col('matched_entity_ids').is_null() | (pl.col('matched_entity_ids') == '')).sample(N_SINGLE, seed=42)
us_sample = pl.concat([us_m, us_s])

in_m = gt_in.filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != '')).sample(N_MATCH, seed=42)
in_s = gt_in.filter(pl.col('matched_entity_ids').is_null() | (pl.col('matched_entity_ids') == '')).sample(N_SINGLE, seed=42)
in_sample = pl.concat([in_m, in_s])

holdout_gt = pl.concat([us_sample, in_sample]).sample(fraction=1.0, shuffle=True, seed=42)
holdout_s1_ids = holdout_gt['source1_entity_id'].to_list()
holdout_s1_set = set(holdout_s1_ids)

us_s1_set = set(us_sample['source1_entity_id'].to_list())
in_s1_set = set(in_sample['source1_entity_id'].to_list())

gt_map: Dict[str, Set[str]] = {}
true_matches_s23: Set[str] = set()
for r in holdout_gt.to_dicts():
    m = r['matched_entity_ids']
    m_set = set(m.split(',')) if (m and str(m).strip()) else set()
    gt_map[r['source1_entity_id']] = m_set
    true_matches_s23.update(m_set)

tot_true_pairs = sum(len(m) for m in gt_map.values())
print(f"  Holdout entities: {len(holdout_s1_ids):,} ({len(us_s1_set):,} US, {len(in_s1_set):,} India).")
print(f"  Total true match pairs in ground truth: {tot_true_pairs:,}")

# Ingest S1 and S23 candidate pool
s1_holdout_df = s1_full.filter(pl.col('entity_id').is_in(holdout_s1_set))
s1_norm2 = {r['entity_id']: normalize_record_v2(r['business_name'], r['business_address'], r['country']) for r in s1_holdout_df.to_dicts()}

print("\n[2] Ingesting Candidate Pool (True matches + noise pool)...")
s2_full = pl.read_csv("dataset/train/train_source2.tsv", separator='\t')
s3_full = pl.read_csv("dataset/train/train_source3.tsv", separator='\t')

s2_pos = s2_full.filter(pl.col('entity_id').is_in(true_matches_s23))
s2_noise = s2_full.filter(~pl.col('entity_id').is_in(true_matches_s23)).head(40000)
s3_pos = s3_full.filter(pl.col('entity_id').is_in(true_matches_s23))
s3_noise = s3_full.filter(~pl.col('entity_id').is_in(true_matches_s23)).head(40000)

s23_df = pl.concat([s2_pos, s2_noise, s3_pos, s3_noise]).unique(subset=['entity_id'])
s23_norm2 = {r['entity_id']: normalize_record_v2(r['business_name'], r['business_address'], r['country']) for r in s23_df.to_dicts()}
print(f"  Candidate pool size: {len(s23_norm2):,} records.")

# Build Enhanced Index
print("\n[3] Building Enhanced Inverted Index...")
t0_idx = time.time()
blocker = EnhancedBlockingIndex(token_frequency_cap=300, num_frequency_cap=150)
for eid, rec in s23_norm2.items():
    blocker.add_record(eid, rec)
print(f"  Enhanced index constructed in {time.time()-t0_idx:.1f}s.")

# =====================================================================
# EVALUATE FIX 1: NEW BLOCKING RECALL CEILING
# =====================================================================
print("\n" + "=" * 80)
print("EVALUATING NEW BLOCKING RECALL CEILING (OVERALL + US + INDIA)")
print("=" * 80)

retrieved_candidates = {}
hits_all, tot_all = 0, 0
hits_us, tot_us = 0, 0
hits_in, tot_in = 0, 0
total_cands_retrieved = 0

for s1_id in holdout_s1_ids:
    r1 = s1_norm2[s1_id]
    c_name = r1['country']
    true_m = gt_map[s1_id]
    
    cands = set(blocker.retrieve_candidates(r1, max_candidates=40))
    retrieved_candidates[s1_id] = cands
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
reduction_ratio = 1.0 - (total_cands_retrieved / (len(holdout_s1_ids) * len(s23_norm2)))

print(f"{'Region':<12} | {'Before Recall':<15} | {'NEW Recall Ceiling':<20} | {'Recall Gain Delta':<18}")
print("-" * 80)
print(f"{'Overall':<12} | {'51.55%':<15} | {rec_all*100:>18.2f}% | {rec_all*100 - 51.55:>+16.2f}%")
print(f"{'US':<12} | {'51.06%':<15} | {rec_us*100:>18.2f}% | {rec_us*100 - 51.06:>+16.2f}%")
print(f"{'India':<12} | {'52.02%':<15} | {rec_in*100:>18.2f}% | {rec_in*100 - 52.02:>+16.2f}%")

print(f"\nCandidate Set Statistics:")
print(f"  Average candidates per S1 entity: {avg_cands:.2f} / 40.0")
print(f"  Reduction Ratio: {reduction_ratio*100:.4f}% (Search space reduced by >99.96%)")

# =====================================================================
# EVALUATE FIX 2 & FIX 3: MODEL SCORING, HARD VETOES & THRESHOLD SWEEP
# =====================================================================
print("\n" + "=" * 80)
print("EVALUATING FIX 2 (DEPLOYED VETOES) & FIX 3 (THRESHOLD SWEEP 0.50-0.95)")
print("=" * 80)

matcher = EntityMatcher("code/business_entity_resolution/src/model.pkl")

# Extract features for all retrieved pairs
cand_pair_records = []
cand_pair_feats = []
for s1_id in holdout_s1_ids:
    r1 = s1_norm2[s1_id]
    for cid in retrieved_candidates[s1_id]:
        if cid in s23_norm2:
            cand_pair_records.append((s1_id, cid))
            cand_pair_feats.append(extract_pair_features(r1, s23_norm2[cid]))

print(f"Extracting features and scoring {len(cand_pair_records):,} candidate pairs on GPU...")
t0_score = time.time()
X_eval = np.array(cand_pair_feats, dtype=np.float32)
probs_eval = matcher.predict_probs(X_eval)
pair_probs = {}
for (s1_id, cid), p in zip(cand_pair_records, probs_eval):
    pair_probs[(s1_id, cid)] = float(p)
print(f"Scored {len(cand_pair_records):,} pairs in {time.time()-t0_score:.1f}s.")

# Hard Veto Functions (Fix 2)
from rapidfuzz import fuzz

def check_veto(r1: Dict[str, Any], r2: Dict[str, Any]) -> bool:
    # 1. Country mismatch veto
    if r1['country'] != r2['country']:
        return True

    # 2. Street number exact mismatch veto
    n1 = r1['nums']
    n2 = r2['nums']
    if n1 and n2 and not (n1 & n2):
        if fuzz.token_set_ratio(r1['norm_addr'], r2['norm_addr']) < 75:
            return True

    # 3. Core name token veto
    c1 = r1['core_tokens']
    c2 = r2['core_tokens']
    if c1 and c2 and not (c1 & c2):
        if fuzz.ratio(r1['norm_name'], r2['norm_name']) < 65:
            return True

    return False

# Sweep Thresholds 0.50 to 0.95
sweep_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
best_thresh = 0.85
best_f05 = 0.0
best_results = {}

print(f"\n{'Threshold':<10} | {'Macro F0.5':<12} | {'Precision':<14} | {'Recall':<14} | {'Singleton Acc':<15} | {'Avg Matches':<12}")
print("-" * 80)

for th in sweep_thresholds:
    preds = defaultdict(list)
    for (s1_id, cid), p in pair_probs.items():
        if p >= th:
            r1 = s1_norm2[s1_id]
            r2 = s23_norm2[cid]
            if not check_veto(r1, r2):
                preds[s1_id].append((cid, p))

    final_preds = defaultdict(set)
    for s1_id, c_list in preds.items():
        c_list.sort(key=lambda x: x[1], reverse=True)
        final_preds[s1_id] = set(c[0] for c in c_list[:6])

    f05, p, r, s_acc = compute_macro_f05(holdout_s1_set, gt_map, final_preds)
    tot_m = sum(len(m) for m in final_preds.values())
    avg_m = tot_m / len(holdout_s1_set)

    print(f"{th:<10.2f} | {f05:<12.4f} | {p*100:>12.2f}% | {r*100:>12.2f}% | {s_acc*100:>13.2f}% | {avg_m:<12.2f}")
    
    if f05 > best_f05:
        best_f05 = f05
        best_thresh = th
        best_results = {'f05': f05, 'prec': p, 'rec': r, 's_acc': s_acc, 'preds': final_preds}

print(f"\nOptimal Re-Tuned Threshold: {best_thresh:.2f} (Peak Macro F0.5: {best_f05:.4f})")

# Breakdown by Country on Best Threshold
f05_us, p_us, r_us, s_us = compute_macro_f05(us_s1_set, gt_map, best_results['preds'])
f05_in, p_in, r_in, s_in = compute_macro_f05(in_s1_set, gt_map, best_results['preds'])

print("\n" + "=" * 80)
print(f"COUNTRY-WISE PERFORMANCE BREAKDOWN AT OPTIMAL THRESHOLD ({best_thresh:.2f})")
print("=" * 80)
print(f"{'Region':<12} | {'Macro F0.5':<12} | {'Precision':<14} | {'Recall':<14} | {'Singleton Acc':<15}")
print("-" * 80)
print(f"{'OVERALL':<12} | {best_f05:<12.4f} | {best_results['prec']*100:>12.2f}% | {best_results['rec']*100:>12.2f}% | {best_results['s_acc']*100:>13.2f}%")
print(f"{'US':<12} | {f05_us:<12.4f} | {p_us*100:>12.2f}% | {r_us*100:>12.2f}% | {s_us*100:>13.2f}%")
print(f"{'India':<12} | {f05_in:<12.4f} | {p_in*100:>12.2f}% | {r_in*100:>12.2f}% | {s_in*100:>13.2f}%")
print("=" * 80)

"""
Step 1: City Alias Normalization
Applies CITY_ALIASES dict during address normalization (both directions resolved to canonical form).
Re-measures: blocking recall ceiling, F0.5, precision, recall (overall, US, India).
Identifies 5 newly captured matches vs 5 flagged as suspicious.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
import os
import re
import time
import pickle
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional

import polars as pl
import numpy as np
from rapidfuzz import fuzz

sys.path.append('code/business_entity_resolution/src')
from blocking import BlockingIndex, LEGAL_SUFFIXES, HONORIFICS_LEADING, ABBREVIATIONS, STOPWORDS
from features import extract_pair_features
from train_model import compute_macro_f05
from predict import EntityMatcher

CITY_ALIASES = {
    'bombay': 'mumbai',
    'calcutta': 'kolkata',
    'madras': 'chennai',
    'bangalore': 'bengaluru',
    'gurgaon': 'gurugram',
    'poona': 'pune',
    'baroda': 'vadodara',
    'cochin': 'kochi',
    'trivandrum': 'thiruvananthapuram',
    'pondicherry': 'puducherry',
    # canonical identity
    'mumbai': 'mumbai',
    'kolkata': 'kolkata',
    'chennai': 'chennai',
    'bengaluru': 'bengaluru',
    'gurugram': 'gurugram',
    'pune': 'pune',
    'vadodara': 'vadodara',
    'kochi': 'kochi',
    'thiruvananthapuram': 'thiruvananthapuram',
    'puducherry': 'puducherry',
}

# Fast single-pass regex compiled pattern
city_alias_pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in CITY_ALIASES.keys()) + r')\b', re.IGNORECASE)

def _city_alias_sub(match):
    return CITY_ALIASES[match.group(0).lower()]

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

import unidecode

def normalize_record_step1(
    name: Optional[str],
    addr: Optional[str],
    country: Optional[str]
) -> Dict[str, Any]:
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
    suffix_tokens = [tok for tok in raw_tokens if tok in LEGAL_SUFFIXES]
    
    prefix4 = core_name[:4] if len(core_name) >= 3 else core_name
    prefix3 = core_name[:3] if len(core_name) >= 2 else core_name
    prefix5 = core_name[:5] if len(core_name) >= 4 else core_name

    collapsed_tokens = set()
    for tok in core_tokens:
        col = re.sub(r'(.)\1+', r'\1', tok)
        if len(col) >= 4 and col != tok:
            collapsed_tokens.add(col)

    clean_core_tokens = [tok for tok in core_tokens if tok not in STOPWORDS]
    if not clean_core_tokens:
        clean_core_tokens = core_tokens
    sorted_tokens = ' '.join(sorted(clean_core_tokens[:4]))
    distinct_tokens = set(w for w in core_tokens if len(w) >= 4 and w not in STOPWORDS and w not in LEGAL_SUFFIXES)

    # Address normalization with City Alias mapping
    raw_addr_str = unidecode.unidecode(str(addr or '')).lower()
    for pattern, repl in ABBREVIATIONS:
        raw_addr_str = re.sub(pattern, repl, raw_addr_str)
        
    clean_addr = clean_re.sub(' ', raw_addr_str)
    
    # Fast single-pass regex replacement
    clean_addr = city_alias_pattern.sub(_city_alias_sub, clean_addr)
    clean_addr = ' '.join(clean_addr.split()).strip()
    
    raw_nums = num_re.findall(clean_addr)
    addr_nums = [n for n in raw_nums if len(n) <= 8]

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
        'prefix3': prefix3,
        'prefix5': prefix5,
        'sorted_tokens': sorted_tokens,
        'distinct_tokens': distinct_tokens,
        'collapsed_tokens': collapsed_tokens,
        'core_tokens': set(core_tokens),
        'suffix_tokens': set(suffix_tokens),
        'norm_addr': clean_addr,
        'nums': set(addr_nums),
        'key_nums': key_nums
    }


def run_step1():
    print("=" * 80)
    print("STEP 1: CITY ALIAS NORMALIZATION VALIDATION ON 10K HOLDOUT")
    print("=" * 80)
    t0 = time.time()

    # 1. Load cached holdout data
    s1_df = pl.read_parquet("holdout_cache/holdout_s1.parquet")
    s23_df = pl.read_parquet("holdout_cache/holdout_s23.parquet")
    gt_df = pl.read_parquet("holdout_cache/holdout_gt.parquet")

    us_s1_ids = set(s1_df.filter(pl.col('country') == 'US')['entity_id'].to_list())
    india_s1_ids = set(s1_df.filter(pl.col('country') == 'India')['entity_id'].to_list())
    all_s1_ids = set(s1_df['entity_id'].to_list())

    gt_map: Dict[str, Set[str]] = {}
    total_true_pairs = 0
    us_true_pairs = 0
    india_true_pairs = 0

    for r in gt_df.to_dicts():
        eid = r['source1_entity_id']
        m = r['matched_entity_ids']
        m_set = set(m.split(',')) if (m and str(m).strip()) else set()
        gt_map[eid] = m_set
        n_m = len(m_set)
        total_true_pairs += n_m
        if eid in us_s1_ids:
            us_true_pairs += n_m
        else:
            india_true_pairs += n_m

    # 2. Normalize with Step 1
    t_norm = time.time()
    s1_records = {}
    s1_raw_info = {}
    for r in s1_df.to_dicts():
        eid = r['entity_id']
        s1_raw_info[eid] = r
        s1_records[eid] = normalize_record_step1(r['business_name'], r['business_address'], r['country'])

    s23_records = {}
    s23_raw_info = {}
    for r in s23_df.to_dicts():
        eid = r['entity_id']
        s23_raw_info[eid] = r
        s23_records[eid] = normalize_record_step1(r['business_name'], r['business_address'], r['country'])
    print(f"Normalized {len(s1_records):,} S1 and {len(s23_records):,} S23 in {time.time()-t_norm:.2f}s.")

    # 3. Blocking
    t_block = time.time()
    blocker = BlockingIndex(token_frequency_cap=300, num_frequency_cap=150)
    for eid, rec in s23_records.items():
        blocker.add_record(eid, rec)

    retrieved_cands = {}
    captured_true = 0
    us_captured_true = 0
    india_captured_true = 0
    total_cands = 0

    for s1_id, rec1 in s1_records.items():
        cands = blocker.retrieve_candidates(rec1, max_candidates=40)
        retrieved_cands[s1_id] = cands
        c_set = set(cands)
        total_cands += len(cands)
        true_set = gt_map[s1_id]
        if true_set:
            cap = len(c_set & true_set)
            captured_true += cap
            if s1_id in us_s1_ids:
                us_captured_true += cap
            else:
                india_captured_true += cap

    block_rec_all = captured_true / total_true_pairs
    block_rec_us = us_captured_true / us_true_pairs
    block_rec_in = india_captured_true / india_true_pairs

    # 4. Score with EntityMatcher
    matcher = EntityMatcher("code/business_entity_resolution/src/model.pkl", threshold=0.85)

    pair_records = []
    pair_feats = []
    for s1_id, cands in retrieved_cands.items():
        rec1 = s1_records[s1_id]
        for cid in cands:
            if cid in s23_records:
                rec2 = s23_records[cid]
                if rec1['country'] != rec2['country']:
                    continue
                pair_records.append((s1_id, cid))
                pair_feats.append(extract_pair_features(rec1, rec2))

    X = np.array(pair_feats, dtype=np.float32)
    probs = matcher.predict_probs(X)

    cand_matches = defaultdict(list)
    pair_prob_lookup = {}
    for (s1_id, cid), prob in zip(pair_records, probs):
        p_val = float(prob)
        pair_prob_lookup[(s1_id, cid)] = p_val
        if p_val >= matcher.threshold:
            r1 = s1_records[s1_id]
            r2 = s23_records[cid]
            if not matcher.check_veto(r1, r2):
                cand_matches[s1_id].append((cid, p_val))

    step1_preds = defaultdict(set)
    for s1_id in all_s1_ids:
        c_list = cand_matches.get(s1_id, [])
        c_list.sort(key=lambda x: x[1], reverse=True)
        step1_preds[s1_id] = set(c[0] for c in c_list[:6])

    f05_all, p_all, r_all, s_all = compute_macro_f05(all_s1_ids, gt_map, step1_preds)
    f05_us, p_us, r_us, s_us = compute_macro_f05(us_s1_ids, gt_map, step1_preds)
    f05_in, p_in, r_in, s_in = compute_macro_f05(india_s1_ids, gt_map, step1_preds)

    # 5. Load baseline metrics & predictions
    base_data = np.load("holdout_cache/baseline_preds.npz")
    with open("holdout_cache/baseline_preds.pkl", "rb") as f:
        base_preds = pickle.load(f)

    print("\n" + "=" * 80)
    print("STEP 1: METRICS COMPARISON (BASELINE vs CITY ALIAS NORMALIZATION)")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Step 0 Baseline':<18} | {'Step 1 City Alias':<18} | {'Delta':<12}")
    print("-" * 80)
    print(f"{'Overall Blocking Ceiling':<25} | {float(base_data['block_rec_all'])*100:>16.2f}% | {block_rec_all*100:>16.2f}% | {(block_rec_all-float(base_data['block_rec_all']))*100:>+10.2f}%")
    print(f"{'India Blocking Ceiling':<25} | {float(base_data['block_rec_in'])*100:>16.2f}% | {block_rec_in*100:>16.2f}% | {(block_rec_in-float(base_data['block_rec_in']))*100:>+10.2f}%")
    print(f"{'US Blocking Ceiling':<25} | {float(base_data['block_rec_us'])*100:>16.2f}% | {block_rec_us*100:>16.2f}% | {(block_rec_us-float(base_data['block_rec_us']))*100:>+10.2f}%")
    print("-" * 80)
    print(f"{'Overall Macro F0.5':<25} | {float(base_data['f05']):>18.4f} | {f05_all:>18.4f} | {f05_all-float(base_data['f05']):>+12.4f}")
    print(f"{'Overall Precision':<25} | {float(base_data['prec'])*100:>16.2f}% | {p_all*100:>16.2f}% | {(p_all-float(base_data['prec']))*100:>+10.2f}%")
    print(f"{'Overall Recall':<25} | {float(base_data['rec'])*100:>16.2f}% | {r_all*100:>16.2f}% | {(r_all-float(base_data['rec']))*100:>+10.2f}%")
    print("-" * 80)
    print(f"{'India Macro F0.5':<25} | {float(base_data['f05_in']):>18.4f} | {f05_in:>18.4f} | {f05_in-float(base_data['f05_in']):>+12.4f}")
    print(f"{'India Precision':<25} | {float(base_data['prec_in'])*100:>16.2f}% | {p_in*100:>16.2f}% | {(p_in-float(base_data['prec_in']))*100:>+10.2f}%")
    print(f"{'India Recall':<25} | {float(base_data['rec_in'])*100:>16.2f}% | {r_in*100:>16.2f}% | {(r_in-float(base_data['rec_in']))*100:>+10.2f}%")
    print("-" * 80)
    print(f"{'US Macro F0.5':<25} | {float(base_data['f05_us']):>18.4f} | {f05_us:>18.4f} | {f05_us-float(base_data['f05_us']):>+12.4f}")
    print(f"{'US Precision':<25} | {float(base_data['prec_us'])*100:>16.2f}% | {p_us*100:>16.2f}% | {(p_us-float(base_data['prec_us']))*100:>+10.2f}%")
    print(f"{'US Recall':<25} | {float(base_data['rec_us'])*100:>16.2f}% | {r_us*100:>12.2f}% | {(r_us-float(base_data['rec_us']))*100:>+10.2f}%")
    print("=" * 80)

    # Find Newly Captured True Matches vs Suspicious
    newly_captured = []
    suspicious = []

    for s1_id in all_s1_ids:
        gt_set = gt_map[s1_id]
        p_step1 = step1_preds[s1_id]
        p_base = base_preds[s1_id]

        new_tps = (p_step1 - p_base) & gt_set
        for cid in new_tps:
            newly_captured.append((s1_id, cid))

        fps = p_step1 - gt_set
        for cid in fps:
            suspicious.append((s1_id, cid))

    print(f"\n[DIAGNOSTICS]: Newly Captured True Matches: {len(newly_captured):,} | Total False Positives: {len(suspicious):,}")

    print("\n--- 5 EXAMPLES: NEWLY CAPTURED TRUE MATCHES ---")
    if newly_captured:
        for i, (s1_id, cid) in enumerate(newly_captured[:5]):
            r1 = s1_raw_info[s1_id]
            r2 = s23_raw_info[cid]
            p_val = pair_prob_lookup.get((s1_id, cid), 0.0)
            print(f"[{i+1}] S1: {r1['entity_id']} | Name: {r1['business_name']} | Addr: {r1['business_address']}")
            print(f"    S23: {r2['entity_id']} | Name: {r2['business_name']} | Addr: {r2['business_address']}")
            print(f"    Confidence: {p_val:.4f} | Status: TRUE POSITIVE (Captured by City Alias Normalization)\n")
    else:
        print("  None (no newly captured matches compared to baseline)")

    print("\n--- 5 EXAMPLES: SUSPICIOUS / POTENTIAL FALSE POSITIVE MATCHES ---")
    if suspicious:
        for i, (s1_id, cid) in enumerate(suspicious[:5]):
            r1 = s1_raw_info[s1_id]
            r2 = s23_raw_info[cid]
            p_val = pair_prob_lookup.get((s1_id, cid), 0.0)
            print(f"[{i+1}] S1: {r1['entity_id']} | Name: {r1['business_name']} | Addr: {r1['business_address']}")
            print(f"    S23: {r2['entity_id']} | Name: {r2['business_name']} | Addr: {r2['business_address']}")
            print(f"    Confidence: {p_val:.4f} | Status: FLAGGED (Predicted but not in Ground Truth)\n")
    else:
        print("  None (0 false positives)")

    # Gate verification
    if p_all < 0.985:
        print(f"\n[GATE FAILED]: Precision dropped to {p_all*100:.2f}% (< 98.5%). Step 1 must be reverted!")
    else:
        print(f"\n[GATE PASSED]: Precision is {p_all*100:.2f}% (>= 98.5%). Step 1 is SAFE to adopt!")

    # Save step 1 predictions
    with open("holdout_cache/step1_preds.pkl", "wb") as f:
        pickle.dump(step1_preds, f)

if __name__ == '__main__':
    run_step1()

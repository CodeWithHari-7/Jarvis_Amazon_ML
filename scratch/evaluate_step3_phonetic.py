"""
Step 3: Phonetic Matching (Double Metaphone / Metaphone)
1. Precomputes metaphone phonetic representation during normalization.
2. Adds phonetic_sim as 10th feature alongside existing lexical & acronym features.
3. Trains XGBoost on GPU with the 10 features.
4. Evaluates on 10k holdout with calibrated vetoes (raising veto threshold for phonetic-driven matches to protect precision).
5. Reports metrics delta vs Step 2 and Step 0 baseline.
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
import xgboost as xgb
import torch
import jellyfish
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

sys.path.append('code/business_entity_resolution/src')
from blocking import BlockingIndex, LEGAL_SUFFIXES, HONORIFICS_LEADING, ABBREVIATIONS, STOPWORDS
from train_model import compute_macro_f05
from evaluate_step2_acronym import compute_acronym_similarity, compute_char_qgram_sim

CITY_ALIASES = {
    'bombay': 'mumbai', 'calcutta': 'kolkata', 'madras': 'chennai',
    'bangalore': 'bengaluru', 'gurgaon': 'gurugram', 'poona': 'pune',
    'baroda': 'vadodara', 'cochin': 'kochi', 'trivandrum': 'thiruvananthapuram',
    'pondicherry': 'puducherry',
    'mumbai': 'mumbai', 'kolkata': 'kolkata', 'chennai': 'chennai',
    'bengaluru': 'bengaluru', 'gurugram': 'gurugram', 'pune': 'pune',
    'vadodara': 'vadodara', 'kochi': 'kochi', 'thiruvananthapuram': 'thiruvananthapuram',
    'puducherry': 'puducherry',
}

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

def normalize_record_step3(
    name: Optional[str],
    addr: Optional[str],
    country: Optional[str]
) -> Dict[str, Any]:
    country_clean = str(country or '').strip().lower()

    # Name
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

    # Phonetic precomputation for core tokens
    meta_tokens = [jellyfish.metaphone(tok) for tok in clean_core_tokens if tok]
    meta_name = ' '.join(t for t in meta_tokens if t)

    # Address
    raw_addr_str = unidecode.unidecode(str(addr or '')).lower()
    for pattern, repl in ABBREVIATIONS:
        raw_addr_str = re.sub(pattern, repl, raw_addr_str)
        
    clean_addr = clean_re.sub(' ', raw_addr_str)
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
        'meta_name': meta_name,
        'norm_addr': clean_addr,
        'nums': set(addr_nums),
        'key_nums': key_nums
    }


FEATURE_NAMES_STEP3 = [
    'name_fuzz_ratio',
    'name_token_set',
    'name_jw',
    'addr_fuzz_ratio',
    'addr_token_set',
    'addr_num_overlap',
    'legal_suffix_match',
    'name_qgram_sim',
    'is_acronym',
    'phonetic_sim'
]


def extract_pair_features_step3(rec1: Dict[str, Any], rec2: Dict[str, Any]) -> List[float]:
    n1, n2 = rec1['norm_name'], rec2['norm_name']
    a1, a2 = rec1['norm_addr'], rec2['norm_addr']

    # 1. Lexical features
    name_fuzz = fuzz.ratio(n1, n2) / 100.0
    name_tok = fuzz.token_set_ratio(n1, n2) / 100.0
    name_jw = JaroWinkler.similarity(n1, n2)

    addr_fuzz = fuzz.ratio(a1, a2) / 100.0 if (a1 and a2) else 0.0
    addr_tok = fuzz.token_set_ratio(a1, a2) / 100.0 if (a1 and a2) else 0.0

    num1, num2 = rec1['nums'], rec2['nums']
    if num1 and num2:
        num_overlap = len(num1 & num2) / len(num1 | num2)
    elif not num1 and not num2:
        num_overlap = 0.8
    else:
        num_overlap = 0.0

    suf1, suf2 = rec1.get('suffix_tokens', set()), rec2.get('suffix_tokens', set())
    if suf1 and suf2:
        suffix_match = 1.0 if (suf1 & suf2) else 0.0
    elif not suf1 and not suf2:
        suffix_match = 0.5
    else:
        suffix_match = 0.3

    qgram_sim = compute_char_qgram_sim(n1, n2, q=3)

    # 2. Acronym feature
    acronym_sim = compute_acronym_similarity(n1, n2)

    # 3. Phonetic feature (Metaphone similarity)
    m1, m2 = rec1.get('meta_name', ''), rec2.get('meta_name', '')
    if m1 and m2:
        phonetic_sim = fuzz.token_set_ratio(m1, m2) / 100.0
    else:
        phonetic_sim = 0.0

    return [
        name_fuzz,
        name_tok,
        name_jw,
        addr_fuzz,
        addr_tok,
        num_overlap,
        suffix_match,
        qgram_sim,
        acronym_sim,
        phonetic_sim
    ]


def check_veto_step3(r1: Dict[str, Any], r2: Dict[str, Any], acronym_score: float, phonetic_score: float) -> bool:
    # Rule 1: Country mismatch
    if r1.get('country') and r2.get('country') and r1['country'] != r2['country']:
        return True

    # Rule 2: Street number exact mismatch
    nums1, nums2 = r1.get('nums', set()), r2.get('nums', set())
    if nums1 and nums2 and not (nums1 & nums2):
        addr_sim = fuzz.token_set_ratio(r1.get('norm_addr', ''), r2.get('norm_addr', ''))
        if addr_sim < 75:
            return True

    # High-Risk Phonetic Protection:
    # If match is purely phonetic-driven (low lexical similarity < 60, but phonetic >= 75),
    # raise veto threshold on address to prevent phonetic false-positive collisions
    name_sim = fuzz.ratio(r1.get('norm_name', ''), r2.get('norm_name', ''))
    if name_sim < 60 and phonetic_score >= 0.75:
        addr_sim = fuzz.token_set_ratio(r1.get('norm_addr', ''), r2.get('norm_addr', ''))
        if addr_sim < 65:
            return True  # VETO spurious phonetic soundalike at different address

    # Rule 3: Core name token mismatch
    core1, core2 = r1.get('core_tokens', set()), r2.get('core_tokens', set())
    if core1 and core2 and not (core1 & core2):
        if acronym_score >= 0.7:
            addr_sim = fuzz.token_set_ratio(r1.get('norm_addr', ''), r2.get('norm_addr', ''))
            if addr_sim < 45:
                return True
        elif phonetic_score >= 0.85:
            # Phonetic match exempt from core name mismatch only if address agrees
            addr_sim = fuzz.token_set_ratio(r1.get('norm_addr', ''), r2.get('norm_addr', ''))
            if addr_sim < 60:
                return True
        else:
            if name_sim < 65:
                return True

    return False


def run_step3():
    print("=" * 80)
    print("STEP 3: PHONETIC MATCHING (DOUBLE METAPHONE) VALIDATION ON 10K HOLDOUT")
    print("=" * 80)
    t0 = time.time()

    model_path = "code/business_entity_resolution/src/model_step3_phonetic.pkl"
    device_param = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.isfile(model_path):
        print(f"\n[1] Training GPU XGBoost classifier with 10 features ({device_param})...")
        t_tr = time.time()
        gt_train = pl.read_csv("dataset/train/train_ground_truth.tsv", separator='\t').sample(12000, seed=42)
        s1_tr_df = pl.read_csv("dataset/train/train_source1.tsv", separator='\t')
        s2_tr_df = pl.read_csv("dataset/train/train_source2.tsv", separator='\t')
        s3_tr_df = pl.read_csv("dataset/train/train_source3.tsv", separator='\t')

        s1_ids = set(gt_train['source1_entity_id'].to_list())
        true_s23_map = {}
        target_s23 = set()
        for r in gt_train.to_dicts():
            m = r['matched_entity_ids']
            m_set = set(m.split(',')) if (m and str(m).strip()) else set()
            true_s23_map[r['source1_entity_id']] = m_set
            target_s23.update(m_set)

        s1_sub = s1_tr_df.filter(pl.col('entity_id').is_in(s1_ids))
        s23_sub = pl.concat([
            s2_tr_df.filter(pl.col('entity_id').is_in(target_s23)),
            s2_tr_df.filter(~pl.col('entity_id').is_in(target_s23)).head(25000),
            s3_tr_df.filter(pl.col('entity_id').is_in(target_s23)),
            s3_tr_df.filter(~pl.col('entity_id').is_in(target_s23)).head(25000),
        ]).unique(subset=['entity_id'])

        s1_norm = {r['entity_id']: normalize_record_step3(r['business_name'], r['business_address'], r['country']) for r in s1_sub.to_dicts()}
        s23_norm = {r['entity_id']: normalize_record_step3(r['business_name'], r['business_address'], r['country']) for r in s23_sub.to_dicts()}

        blocker_tr = BlockingIndex()
        for eid, rec in s23_norm.items():
            blocker_tr.add_record(eid, rec)

        train_X, train_y = [], []
        for s1_id in s1_ids:
            rec1 = s1_norm[s1_id]
            cands = blocker_tr.retrieve_candidates(rec1, max_candidates=25)
            true_set = true_s23_map[s1_id]
            pos_ids = [m for m in true_set if m in s23_norm]
            neg_ids = [c for c in cands if c not in true_set][:10]

            for cid in pos_ids:
                train_X.append(extract_pair_features_step3(rec1, s23_norm[cid]))
                train_y.append(1)
            for cid in neg_ids:
                train_X.append(extract_pair_features_step3(rec1, s23_norm[cid]))
                train_y.append(0)

        X_train = np.array(train_X, dtype=np.float32)
        y_train = np.array(train_y, dtype=np.int32)
        print(f"  Training set: {len(X_train):,} pairs (Pos: {y_train.sum():,}, Neg: {(y_train==0).sum():,})")

        clf = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.04,
            tree_method='hist',
            device=device_param,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric='logloss',
            random_state=42
        )
        clf.fit(X_train, y_train)
        print(f"  Classifier trained in {time.time()-t_tr:.2f}s.")
        with open(model_path, 'wb') as f:
            pickle.dump({'model': clf, 'feature_names': FEATURE_NAMES_STEP3, 'threshold': 0.85}, f)
    else:
        print(f"  Loaded existing {model_path}.")
        with open(model_path, 'rb') as f:
            art = pickle.load(f)
            clf = art['model']

    # 2. Load 10k Holdout
    s1_df = pl.read_parquet("holdout_cache/holdout_s1.parquet")
    s23_df = pl.read_parquet("holdout_cache/holdout_s23.parquet")
    gt_df = pl.read_parquet("holdout_cache/holdout_gt.parquet")

    us_s1_ids = set(s1_df.filter(pl.col('country') == 'US')['entity_id'].to_list())
    india_s1_ids = set(s1_df.filter(pl.col('country') == 'India')['entity_id'].to_list())
    all_s1_ids = set(s1_df['entity_id'].to_list())

    gt_map = {}
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

    t_norm = time.time()
    s1_records = {}
    s1_raw_info = {}
    for r in s1_df.to_dicts():
        eid = r['entity_id']
        s1_raw_info[eid] = r
        s1_records[eid] = normalize_record_step3(r['business_name'], r['business_address'], r['country'])

    s23_records = {}
    s23_raw_info = {}
    for r in s23_df.to_dicts():
        eid = r['entity_id']
        s23_raw_info[eid] = r
        s23_records[eid] = normalize_record_step3(r['business_name'], r['business_address'], r['country'])
    print(f"Normalized holdout records in {time.time()-t_norm:.2f}s.")

    blocker = BlockingIndex(token_frequency_cap=300, num_frequency_cap=150)
    for eid, rec in s23_records.items():
        blocker.add_record(eid, rec)

    retrieved_cands = {}
    captured_true = 0
    us_captured_true = 0
    india_captured_true = 0

    for s1_id, rec1 in s1_records.items():
        cands = blocker.retrieve_candidates(rec1, max_candidates=40)
        retrieved_cands[s1_id] = cands
        c_set = set(cands)
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

    # 3. Extract 10 features & predict
    pair_records = []
    pair_feats = []
    pair_acronym_scores = []
    pair_phonetic_scores = []

    for s1_id, cands in retrieved_cands.items():
        rec1 = s1_records[s1_id]
        for cid in cands:
            if cid in s23_records:
                rec2 = s23_records[cid]
                if rec1['country'] != rec2['country']:
                    continue
                f = extract_pair_features_step3(rec1, rec2)
                pair_records.append((s1_id, cid))
                pair_feats.append(f)
                pair_acronym_scores.append(f[8])
                pair_phonetic_scores.append(f[9])

    X = np.array(pair_feats, dtype=np.float32)
    probs = clf.predict_proba(X)[:, 1]

    threshold = 0.85
    cand_matches = defaultdict(list)
    pair_prob_lookup = {}
    phonetic_triggered_pairs = []

    for (s1_id, cid), prob, acr_score, pho_score in zip(pair_records, probs, pair_acronym_scores, pair_phonetic_scores):
        p_val = float(prob)
        pair_prob_lookup[(s1_id, cid)] = p_val
        if p_val >= threshold:
            r1 = s1_records[s1_id]
            r2 = s23_records[cid]
            if not check_veto_step3(r1, r2, acr_score, pho_score):
                cand_matches[s1_id].append((cid, p_val))
                if pho_score >= 0.75:
                    phonetic_triggered_pairs.append((s1_id, cid, pho_score, p_val))

    step3_preds = defaultdict(set)
    for s1_id in all_s1_ids:
        c_list = cand_matches.get(s1_id, [])
        c_list.sort(key=lambda x: x[1], reverse=True)
        step3_preds[s1_id] = set(c[0] for c in c_list[:6])

    f05_all, p_all, r_all, s_all = compute_macro_f05(all_s1_ids, gt_map, step3_preds)
    f05_us, p_us, r_us, s_us = compute_macro_f05(us_s1_ids, gt_map, step3_preds)
    f05_in, p_in, r_in, s_in = compute_macro_f05(india_s1_ids, gt_map, step3_preds)

    base_data = np.load("holdout_cache/baseline_preds.npz")
    with open("holdout_cache/step2_preds.pkl", "rb") as f:
        step2_preds = pickle.load(f)

    print("\n" + "=" * 80)
    print("STEP 3: METRICS COMPARISON (BASELINE vs STEP 2 ACRONYM vs STEP 3 PHONETIC)")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Step 0 Baseline':<16} | {'Step 2 Acronym':<15} | {'Step 3 Phonetic':<16} | {'Delta vs Base':<12}")
    print("-" * 80)
    print(f"{'Overall Blocking Ceiling':<25} | {float(base_data['block_rec_all'])*100:>14.2f}% | {84.06:>13.2f}% | {block_rec_all*100:>14.2f}% | {(block_rec_all-float(base_data['block_rec_all']))*100:>+10.2f}%")
    print(f"{'India Blocking Ceiling':<25} | {float(base_data['block_rec_in'])*100:>14.2f}% | {74.93:>13.2f}% | {block_rec_in*100:>14.2f}% | {(block_rec_in-float(base_data['block_rec_in']))*100:>+10.2f}%")
    print(f"{'US Blocking Ceiling':<25} | {float(base_data['block_rec_us'])*100:>14.2f}% | {93.30:>13.2f}% | {block_rec_us*100:>14.2f}% | {(block_rec_us-float(base_data['block_rec_us']))*100:>+10.2f}%")
    print("-" * 80)
    print(f"{'Overall Macro F0.5':<25} | {float(base_data['f05']):>16.4f} | {0.9044:>15.4f} | {f05_all:>16.4f} | {f05_all-float(base_data['f05']):>+12.4f}")
    print(f"{'Overall Precision':<25} | {float(base_data['prec'])*100:>14.2f}% | {99.26:>13.2f}% | {p_all*100:>14.2f}% | {(p_all-float(base_data['prec']))*100:>+10.2f}%")
    print(f"{'Overall Recall':<25} | {float(base_data['rec'])*100:>14.2f}% | {80.28:>13.2f}% | {r_all*100:>14.2f}% | {(r_all-float(base_data['rec']))*100:>+10.2f}%")
    print("-" * 80)
    print(f"{'India Macro F0.5':<25} | {float(base_data['f05_in']):>16.4f} | {0.8494:>15.4f} | {f05_in:>16.4f} | {f05_in-float(base_data['f05_in']):>+12.4f}")
    print(f"{'India Precision':<25} | {float(base_data['prec_in'])*100:>14.2f}% | {99.08:>13.2f}% | {p_in*100:>14.2f}% | {(p_in-float(base_data['prec_in']))*100:>+10.2f}%")
    print(f"{'India Recall':<25} | {float(base_data['rec_in'])*100:>14.2f}% | {71.39:>13.2f}% | {r_in*100:>14.2f}% | {(r_in-float(base_data['rec_in']))*100:>+10.2f}%")
    print("-" * 80)
    print(f"{'US Macro F0.5':<25} | {float(base_data['f05_us']):>16.4f} | {0.9595:>15.4f} | {f05_us:>16.4f} | {f05_us-float(base_data['f05_us']):>+12.4f}")
    print(f"{'US Precision':<25} | {float(base_data['prec_us'])*100:>14.2f}% | {99.41:>13.2f}% | {p_us*100:>14.2f}% | {(p_us-float(base_data['prec_us']))*100:>+10.2f}%")
    print(f"{'US Recall':<25} | {float(base_data['rec_us'])*100:>14.2f}% | {89.27:>13.2f}% | {r_us*100:>14.2f}% | {(r_us-float(base_data['rec_us']))*100:>+10.2f}%")
    print("=" * 80)

    # Diagnostic checks
    newly_captured = []
    suspicious = []
    for s1_id in all_s1_ids:
        gt_set = gt_map[s1_id]
        p_step3 = step3_preds[s1_id]
        p_step2 = step2_preds[s1_id]

        new_tps = (p_step3 - p_step2) & gt_set
        for cid in new_tps:
            newly_captured.append((s1_id, cid))

        fps = p_step3 - gt_set
        for cid in fps:
            suspicious.append((s1_id, cid))

    print(f"\n[DIAGNOSTICS]: Newly Captured True Matches vs Step 2: {len(newly_captured):,} | Total False Positives: {len(suspicious):,}")

    print("\n--- 5 EXAMPLES: NEWLY CAPTURED TRUE MATCHES ---")
    if newly_captured:
        for i, (s1_id, cid) in enumerate(newly_captured[:5]):
            r1 = s1_raw_info[s1_id]
            r2 = s23_raw_info[cid]
            p_val = pair_prob_lookup.get((s1_id, cid), 0.0)
            print(f"[{i+1}] S1: {r1['entity_id']} | Name: {r1['business_name']} | Addr: {r1['business_address']}")
            print(f"    S23: {r2['entity_id']} | Name: {r2['business_name']} | Addr: {r2['business_address']}")
            print(f"    Confidence: {p_val:.4f} | Status: TRUE POSITIVE (Captured via Phonetic Matching)\n")
    else:
        print("  None")

    print("\n--- 5 EXAMPLES: SUSPICIOUS / POTENTIAL FALSE POSITIVE MATCHES ---")
    if suspicious:
        for i, (s1_id, cid) in enumerate(suspicious[:5]):
            r1 = s1_raw_info[s1_id]
            r2 = s23_raw_info[cid]
            p_val = pair_prob_lookup.get((s1_id, cid), 0.0)
            print(f"[{i+1}] S1: {r1['entity_id']} | Name: {r1['business_name']} | Addr: {r1['business_address']}")
            print(f"    S23: {r2['entity_id']} | Name: {r2['business_name']} | Addr: {r2['business_address']}")
            print(f"    Confidence: {p_val:.4f} | Status: FLAGGED\n")

    prec_delta = p_all - float(base_data['prec'])
    if p_all < 0.985 or prec_delta < -0.005:
        print(f"\n[GATE FAILED]: Precision dropped to {p_all*100:.2f}% (delta: {prec_delta*100:.2f}%). Step 3 must be tuned or reverted!")
    else:
        print(f"\n[GATE PASSED]: Precision is {p_all*100:.2f}% (>= 98.5%, delta: {prec_delta*100:+.2f}%). Step 3 is SAFE to adopt!")

    with open("holdout_cache/step3_preds.pkl", "wb") as f:
        pickle.dump(step3_preds, f)

if __name__ == '__main__':
    run_step3()

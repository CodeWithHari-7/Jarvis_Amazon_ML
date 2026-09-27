import os
import sys
import time
from collections import defaultdict
import polars as pl
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.append('code/business_entity_resolution/src')
from blocking import normalize_record, BlockingIndex
from predict import EntityMatcher
from train_model import compute_macro_f05

print("=" * 80)
print("LEAK-FREE END-TO-END EVALUATION HARNESS ACROSS FULL CANDIDATE UNIVERSE")
print("=" * 80)
t0_total = time.time()

# 1. Load Ground Truth and sample 1,000 US and 1,000 India entities
print("\n[1] Ingesting Stratified Holdout Sample (1,000 US + 1,000 India)...")
gt = pl.read_csv("dataset/train/train_ground_truth.tsv", separator='\t')
s1 = pl.read_csv("dataset/train/train_source1.tsv", separator='\t')

s1_us_all = set(s1.filter(pl.col('country') == 'US')['entity_id'].to_list())
s1_in_all = set(s1.filter(pl.col('country') == 'India')['entity_id'].to_list())

gt_us = gt.filter(pl.col('source1_entity_id').is_in(s1_us_all))
gt_in = gt.filter(pl.col('source1_entity_id').is_in(s1_in_all))

# Stratified sample with proper singleton representation (5.58% singletons per true distribution)
N_EACH = 1000
N_SINGLE = int(N_EACH * 0.0558)
N_MATCH = N_EACH - N_SINGLE

sample_us_m = gt_us.filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != '')).sample(N_MATCH, seed=42)
sample_us_s = gt_us.filter(pl.col('matched_entity_ids').is_null() | (pl.col('matched_entity_ids') == '')).sample(N_SINGLE, seed=42)
sample_us = pl.concat([sample_us_m, sample_us_s])

sample_in_m = gt_in.filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != '')).sample(N_MATCH, seed=42)
sample_in_s = gt_in.filter(pl.col('matched_entity_ids').is_null() | (pl.col('matched_entity_ids') == '')).sample(N_SINGLE, seed=42)
sample_in = pl.concat([sample_in_m, sample_in_s])

eval_gt = pl.concat([sample_us, sample_in])
eval_s1_ids = eval_gt['source1_entity_id'].to_list()
eval_s1_df = s1.filter(pl.col('entity_id').is_in(set(eval_s1_ids)))

gt_mapping = {}
for r in eval_gt.to_dicts():
    m = r['matched_entity_ids']
    gt_mapping[r['source1_entity_id']] = set(m.split(',')) if m and str(m).strip() else set()

us_s1_ids = set(sample_us['source1_entity_id'].to_list())
in_s1_ids = set(sample_in['source1_entity_id'].to_list())

print(f"Sampled {len(eval_s1_ids):,} entities ({len(us_s1_ids):,} US, {len(in_s1_ids):,} India).")
print(f"Total true match pairs: {sum(len(v) for v in gt_mapping.values()):,}")

# Initialize trained GPU EntityMatcher
matcher = EntityMatcher("code/business_entity_resolution/src/model.pkl", threshold=0.85)
print(f"Loaded GPU EntityMatcher (Threshold: {matcher.threshold:.2f}, Max Matches: 10)")

# Function to evaluate one country against its full candidate pool
def evaluate_country(country_name: str, eval_ids: set, s2_path: str, s3_path: str):
    print("\n" + "-" * 75)
    print(f"EVALUATING {country_name.upper()} AGAINST FULL CANDIDATE POOL")
    print("-" * 75)
    t_c0 = time.time()
    
    # 1. Load full country candidates
    c_lower = country_name.lower()
    s2 = pl.read_csv(s2_path, separator='\t', columns=['entity_id', 'business_name', 'business_address', 'country']).filter(pl.col('country').str.to_lowercase() == c_lower)
    s3 = pl.read_csv(s3_path, separator='\t', columns=['entity_id', 'business_name', 'business_address', 'country']).filter(pl.col('country').str.to_lowercase() == c_lower)
    s23 = pl.concat([s2, s3]).unique(subset=['entity_id'])
    del s2, s3
    n_pool = len(s23)
    print(f"Loaded {n_pool:,} {country_name} candidate records in {time.time()-t_c0:.1f}s.")

    # 2. Build Inverted Index
    blocker = BlockingIndex()
    eids = s23['entity_id'].to_list()
    names = s23['business_name'].to_list()
    addrs = s23['business_address'].to_list()
    countries = s23['country'].to_list()
    s23_records = {}

    t_idx = time.time()
    for i in range(n_pool):
        rec = normalize_record(names[i], addrs[i], countries[i])
        blocker.add_record(eids[i], rec)
        s23_records[eids[i]] = rec
    del eids, names, addrs, countries, s23
    print(f"Indexed {n_pool:,} records in {time.time()-t_idx:.1f}s.")

    # 3. Retrieve candidates and score with model (NO POSITIVE INJECTION)
    country_s1_df = eval_s1_df.filter(pl.col('entity_id').is_in(eval_ids))
    s1_rows = country_s1_df.to_dicts()

    pred_mapping = {}
    hits_cand = 0
    tot_true = 0
    zero_cand_entities = 0

    t_eval = time.time()
    for r1_raw in s1_rows:
        s1_id = r1_raw['entity_id']
        r1 = normalize_record(r1_raw['business_name'], r1_raw['business_address'], r1_raw['country'])
        true_set = gt_mapping[s1_id]
        tot_true += len(true_set)

        # STRICT RETRIEVAL: only from blocker.retrieve_candidates
        cands = blocker.retrieve_candidates(r1, max_candidates=40)
        cands_set = set(cands)
        h = len(cands_set & true_set)
        hits_cand += h
        if len(true_set) > 0 and h == 0:
            zero_cand_entities += 1

        # Model inference on retrieved candidates
        cand_tuples = [(cid, s23_records[cid]) for cid in cands if cid in s23_records]
        matched_ids = matcher.predict_matches(r1, cand_tuples, max_matches=10)
        pred_mapping[s1_id] = set(matched_ids)

    cand_recall = (hits_cand / tot_true) * 100 if tot_true > 0 else 100.0
    zero_hit_rate = (zero_cand_entities / len(s1_rows)) * 100

    f05, prec, rec, s_acc = compute_macro_f05(eval_ids, gt_mapping, pred_mapping)
    print(f"Candidate Blocking Recall (Ceiling) : {cand_recall:.2f}%")
    print(f"Zero-Hit Entity Rate                : {zero_hit_rate:.2f}% ({zero_cand_entities}/{len(s1_rows)})")
    print(f"Downstream Macro F0.5               : {f05:.4f}")
    print(f"Downstream Precision                : {prec*100:.2f}%")
    print(f"Downstream Recall                   : {rec*100:.2f}%")
    print(f"Singleton Accuracy                  : {s_acc*100:.2f}%")
    print(f"Evaluation took {time.time()-t_eval:.2f}s.")

    return {
        'country': country_name,
        'cand_recall': cand_recall,
        'zero_hit_rate': zero_hit_rate,
        'f05': f05,
        'prec': prec,
        'rec': rec,
        's_acc': s_acc,
        'pred_map': pred_mapping
    }

# Run evaluations for US and India
s2_train_path = "dataset/train/train_source2.tsv"
s3_train_path = "dataset/train/train_source3.tsv"

res_us = evaluate_country("US", us_s1_ids, s2_train_path, s3_train_path)
res_in = evaluate_country("India", in_s1_ids, s2_train_path, s3_train_path)

# Overall Macro F0.5
combined_preds = {**res_us['pred_map'], **res_in['pred_map']}
ov_f05, ov_prec, ov_rec, ov_sacc = compute_macro_f05(set(eval_s1_ids), gt_mapping, combined_preds)

print("\n" + "=" * 80)
print("FINAL LEAK-FREE EVALUATION SUMMARY (REAL FULL-SCALE CORPUS):")
print("=" * 80)
print(f"{'Metric':<30} | {'US':<12} | {'India':<12} | {'OVERALL':<12}")
print("-" * 80)
print(f"{'Candidate Blocking Recall':<30} | {res_us['cand_recall']:>10.2f}% | {res_in['cand_recall']:>10.2f}% | {(res_us['cand_recall']+res_in['cand_recall'])/2:>10.2f}%")
print(f"{'Zero-Hit Entity Rate':<30} | {res_us['zero_hit_rate']:>10.2f}% | {res_in['zero_hit_rate']:>10.2f}% | {(res_us['zero_hit_rate']+res_in['zero_hit_rate'])/2:>10.2f}%")
print(f"{'Downstream Precision':<30} | {res_us['prec']*100:>10.2f}% | {res_in['prec']*100:>10.2f}% | {ov_prec*100:>10.2f}%")
print(f"{'Downstream Recall':<30} | {res_us['rec']*100:>10.2f}% | {res_in['rec']*100:>10.2f}% | {ov_rec*100:>10.2f}%")
print(f"{'Macro F0.5 Score':<30} | {res_us['f05']:>11.4f} | {res_in['f05']:>11.4f} | {ov_f05:>11.4f}")
print("=" * 80)
print(f"Total leak-free evaluation runtime: {time.time()-t0_total:.1f}s.")

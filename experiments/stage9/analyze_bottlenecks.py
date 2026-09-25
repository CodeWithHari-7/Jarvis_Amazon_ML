import pandas as pd
import numpy as np

print("=" * 70)
print("STAGE 9: BOTTLENECK ANALYSIS")
print("=" * 70)

# Load Ground Truth
gt_df = pd.read_csv("../../dataset/pipeline_cache/subset_gt.tsv", sep='\t', dtype=str).fillna("")
true_pairs = set()
for _, row in gt_df.iterrows():
    s1 = row['source1_entity_id']
    ms = row['matched_entity_ids']
    if ms:
        for m in ms.split(','):
            true_pairs.add((s1, m))

print(f"Total True Pairs (Ground Truth): {len(true_pairs)}")

# Load Candidates
cand_df = pd.read_parquet("../../dataset/pipeline_cache/candidate_pairs.parquet")
cand_pairs = set(zip(cand_df['source1_entity_id'], cand_df['source_entity_id']))

# Load Features and Model
import pickle
features_df = pd.read_parquet("../../dataset/pipeline_cache/candidate_features.parquet")
with open("../../dataset/processed/lgb_model.pkl", 'rb') as f:
    model = pickle.load(f)

feature_cols = ['name_fuzz_ratio', 'name_token_set', 'name_jw', 'addr_fuzz_ratio', 'addr_token_set', 'addr_num_overlap', 'cross_script']
features_df['prob'] = model.predict_proba(features_df[feature_cols])[:, 1]
matches = features_df[features_df['prob'] >= 0.89]

pred_pairs = set(zip(matches['source1_entity_id'], matches['source_entity_id']))

# 1. Blocking vs Matching Analysis
blocking_failures = true_pairs - cand_pairs
blocked_true = true_pairs.intersection(cand_pairs)
matching_failures = blocked_true - pred_pairs

tp = true_pairs.intersection(pred_pairs)
fp = pred_pairs - true_pairs

print("\n--- ERROR BUCKETING ---")
print(f"Total False Negatives : {len(true_pairs - pred_pairs)}")
print(f"  -> Blocking Failures: {len(blocking_failures)} ({(len(blocking_failures)/len(true_pairs))*100:.1f}% of total GT lost)")
print(f"  -> Matching Failures: {len(matching_failures)} ({(len(matching_failures)/len(true_pairs))*100:.1f}% of total GT lost)")
print(f"Total False Positives : {len(fp)}")

# 2. Source-Specific Analysis
s2_true = sum(1 for _, m in true_pairs if m.startswith("S2"))
s3_true = sum(1 for _, m in true_pairs if m.startswith("S3"))

s2_tp = sum(1 for _, m in tp if m.startswith("S2"))
s3_tp = sum(1 for _, m in tp if m.startswith("S3"))

s2_block_fail = sum(1 for _, m in blocking_failures if m.startswith("S2"))
s3_block_fail = sum(1 for _, m in blocking_failures if m.startswith("S3"))

s2_fp = sum(1 for _, m in fp if m.startswith("S2"))
s3_fp = sum(1 for _, m in fp if m.startswith("S3"))

print("\n--- SOURCE-SPECIFIC ANALYSIS ---")
print(f"S2 True Pairs: {s2_true} | TP: {s2_tp} | Block Fail: {s2_block_fail} | FP: {s2_fp}")
print(f"S3 True Pairs: {s3_true} | TP: {s3_tp} | Block Fail: {s3_block_fail} | FP: {s3_fp}")

if s2_true > 0:
    print(f"S2 Recall: {(s2_tp/s2_true)*100:.1f}%")
if s3_true > 0:
    print(f"S3 Recall: {(s3_tp/s3_true)*100:.1f}%")

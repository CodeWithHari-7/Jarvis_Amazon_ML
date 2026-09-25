import os
import pandas as pd
import numpy as np

print("=" * 70)
print("JARVIS_CHECKER STAGE 8A: PIPELINE VALIDATION")
print("=" * 70)

# Load files
results_path = "matching_results.tsv"
gt_path = r"dataset\pipeline_cache\subset_gt.tsv"
s1_path = r"dataset\pipeline_cache\subset_s1.tsv"

if not os.path.exists(results_path) or not os.path.exists(gt_path):
    print("Required files not found. Ensure the pipeline and subset creation have completed.")
    exit(1)

results_df = pd.read_csv(results_path, sep='\t', dtype=str).fillna("")
gt_df = pd.read_csv(gt_path, sep='\t', dtype=str).fillna("")
s1_df = pd.read_csv(s1_path, sep='\t', dtype=str)

# 1. Row counts check
print("\n[1] Basic Structure Checks")
print(f"  Input S1 rows          : {len(s1_df):,}")
print(f"  Result Output S1 rows  : {len(results_df):,}")
if len(s1_df) == len(results_df):
    print("  PASS: Output row count matches input.")
else:
    print("  FAIL: Missing S1 rows in output.")

# 2. Candidate and Prediction metrics
merged = pd.merge(gt_df, results_df, on='source1_entity_id', how='left', suffixes=('_gt', '_pred'))
merged['pred_list'] = merged['matched_entity_ids_pred'].apply(lambda x: set(str(x).split(',')) if x else set())
merged['gt_list'] = merged['matched_entity_ids_gt'].apply(lambda x: set(str(x).split(',')) if x else set())

total_tp = 0
total_fp = 0
total_fn = 0
f05_scores = []

for _, row in merged.iterrows():
    pred = row['pred_list']
    gt = row['gt_list']
    
    tp = len(pred.intersection(gt))
    fp = len(pred - gt)
    fn = len(gt - pred)
    
    total_tp += tp
    total_fp += fp
    total_fn += fn
    
    # Official Macro F0.5 per S1 entity
    if len(gt) == 0:
        if len(pred) == 0:
            f05_scores.append(1.0)
        else:
            f05_scores.append(0.0)
    else:
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        beta_sq = 0.25
        f05 = ((1 + beta_sq) * prec * rec / (beta_sq * prec + rec)) if (beta_sq * prec + rec) > 0 else 0.0
        f05_scores.append(f05)

print("\n[2] Matching Performance (Curated Subset)")
print(f"  True Positives (TP)  : {total_tp:,}")
print(f"  False Positives (FP) : {total_fp:,}")
print(f"  False Negatives (FN) : {total_fn:,}")

macro_f05 = np.mean(f05_scores)
prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

print(f"  Precision            : {prec:.4f}")
print(f"  Recall               : {rec:.4f}")
print(f"  Macro F0.5 (Official): {macro_f05:.4f}")

# 3. Multiple Matches vs Zero Matches
pred_counts = merged['pred_list'].apply(len)
print("\n[3] One-to-Many Decision Logic Check")
print(f"  S1 entities with 0 predicted matches  : {(pred_counts == 0).sum():,}")
print(f"  S1 entities with 1 predicted match    : {(pred_counts == 1).sum():,}")
print(f"  S1 entities with >1 predicted matches : {(pred_counts > 1).sum():,}")

# 4. Extracting Errors for Manual Inspection
print("\n[4] Manual Error Inspection (Sample)")
fps = merged[merged['pred_list'].apply(len) > merged['pred_list'].apply(lambda p: len(p.intersection(set())))] # Just finding any with FPs
fp_examples = []
fn_examples = []

for _, row in merged.iterrows():
    pred = row['pred_list']
    gt = row['gt_list']
    if len(pred - gt) > 0:
        fp_examples.append((row['source1_entity_id'], list(pred - gt)))
    if len(gt - pred) > 0:
        fn_examples.append((row['source1_entity_id'], list(gt - pred)))

print(f"\n  Found {len(fp_examples)} S1 entities with False Positives.")
for s1, fp in fp_examples[:3]:
    print(f"    S1 ID: {s1} predicted FP: {fp[0]}")

print(f"\n  Found {len(fn_examples)} S1 entities with False Negatives.")
for s1, fn in fn_examples[:3]:
    print(f"    S1 ID: {s1} missed FN: {fn[0]}")

print("\n=== PIPELINE VALIDATION COMPLETE ===")
if macro_f05 > 0.95 and len(s1_df) == len(results_df):
    print("\nFINAL DECISION: PASS")
    print("The pipeline is correct and reproduces the expected high F0.5 performance.")
else:
    print("\nFINAL DECISION: FIX REQUIRED")

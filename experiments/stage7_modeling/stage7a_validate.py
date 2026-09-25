import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os

base_dir = r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker\dataset"
pq_path = os.path.join(base_dir, "processed", "ml_candidate_pairs.parquet")
cand_df = pd.read_parquet(pq_path)

feat_cols = ['name_fuzz_ratio','name_token_set','name_jw',
             'addr_fuzz_ratio','addr_token_set','addr_num_overlap',
             'country_match','cross_script']

print("=== STAGE 7A FINAL REPORT ===")
total = len(cand_df)
pos = int(cand_df["label"].sum())
neg = total - pos
print(f"Total candidate pairs : {total:,}")
print(f"Positives (label=1)   : {pos:,}")
print(f"Negatives (label=0)   : {neg:,}")
s2p = len(cand_df[(cand_df["label"]==1) & (cand_df["source"]=="S2")])
s3p = len(cand_df[(cand_df["label"]==1) & (cand_df["source"]=="S3")])
print(f"S1->S2 positives      : {s2p:,}")
print(f"S1->S3 positives      : {s3p:,}")
print(f"Class ratio pos:neg   : 1:{neg // max(pos,1)}")
print(f"Columns               : {list(cand_df.columns)}")
print(f"Missing feature vals  : {cand_df[feat_cols].isnull().sum().sum()}")

# Validation checks
print("\n=== VALIDATION CHECKS ===")
gt_df = pd.read_csv(os.path.join(base_dir, "train", "train_ground_truth.tsv"), sep="\t", dtype=str)
gt_pos = set()
for _, row in gt_df[gt_df["matched_entity_ids"].notna()].iterrows():
    for m in str(row["matched_entity_ids"]).split(","):
        gt_pos.add((row["source1_entity_id"], m))

pos_df = cand_df[cand_df["label"]==1]
all_pos_in_gt = all(
    (r["source1_entity_id"], r["source_entity_id"]) in gt_pos
    for _, r in pos_df.iterrows()
)

checks = {
    "No duplicate pairs"               : cand_df.duplicated(subset=["source1_entity_id","source_entity_id"]).sum() == 0,
    "Labels are 0 or 1"                : cand_df["label"].isin([0,1]).all(),
    "Source is S2 or S3"               : cand_df["source"].isin(["S2","S3"]).all(),
    "All positives verified in GT"     : all_pos_in_gt,
    "Features are numeric"             : all(pd.api.types.is_numeric_dtype(cand_df[c]) for c in feat_cols),
    "No missing feature values"        : cand_df[feat_cols].isnull().sum().sum() == 0,
    "country_match=1 for all positives": cand_df[cand_df["label"]==1]["country_match"].min() == 1,
}
for chk, ok in checks.items():
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"  {tag}  {chk}")

print("\n=== POSITIVE FEATURE MEDIANS ===")
print(cand_df[cand_df["label"]==1][feat_cols].median().round(2).to_string())
print("\n=== NEGATIVE FEATURE MEDIANS ===")
print(cand_df[cand_df["label"]==0][feat_cols].median().round(2).to_string())

per_s1 = cand_df.groupby("source1_entity_id").size()
print(f"\nCands/S1  mean={per_s1.mean():.1f}  median={per_s1.median():.0f}  p95={per_s1.quantile(.95):.0f}  max={per_s1.max()}")
pos_per_s1 = cand_df[cand_df["label"]==1].groupby("source1_entity_id").size()
print(f"Pos/S1    mean={pos_per_s1.mean():.2f}  median={pos_per_s1.median():.0f}  max={pos_per_s1.max()}")
print(f"S1 with 0 positives: {pos - len(pos_per_s1):,}")

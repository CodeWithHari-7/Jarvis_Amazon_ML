import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import os

print("=" * 70)
print("STAGE 9 EXPERIMENT: S2 MISSINGNESS AWARENESS")
print("=" * 70)

def add_missingness_features(df):
    df['addr_missing'] = (df['addr_fuzz_ratio'] == 0).astype(int)
    df['name_missing'] = (df['name_fuzz_ratio'] == 0).astype(int)
    df['name_sim_x_addr_missing'] = df['name_token_set'] * df['addr_missing']
    df['addr_sim_x_name_missing'] = df['addr_token_set'] * df['name_missing']
    return df

# 1. LOAD TRAINING DATA
print("\n[1] Loading Training Data...")
train_df = pd.read_parquet("../../../dataset/processed/ml_train_candidates.parquet")
train_df = add_missingness_features(train_df)

base_features = ['name_fuzz_ratio', 'name_token_set', 'name_jw', 
                 'addr_fuzz_ratio', 'addr_token_set', 'addr_num_overlap', 'cross_script']
new_features = base_features + ['addr_missing', 'name_missing', 'name_sim_x_addr_missing', 'addr_sim_x_name_missing']

X_train_base = train_df[base_features]
X_train_new = train_df[new_features]
y_train = train_df['label']

# 2. TRAIN MODELS
print("\n[2] Training Models...")
t0 = time.time()
params = {'objective': 'binary', 'metric': 'auc', 'n_estimators': 500, 'max_depth': 6, 'random_state': 42, 'n_jobs': -1}
model_base = lgb.LGBMClassifier(**params)
model_base.fit(X_train_base, y_train)

model_new = lgb.LGBMClassifier(**params)
model_new.fit(X_train_new, y_train)
train_time = time.time() - t0
print(f"  Trained both models in {train_time:.2f}s")

# 3. LOAD VALIDATION SUBSET
print("\n[3] Loading Validation Subset...")
val_df = pd.read_parquet("../../../dataset/pipeline_cache/candidate_features.parquet")
val_df = add_missingness_features(val_df)

gt_df = pd.read_csv("../../../dataset/pipeline_cache/subset_gt.tsv", sep='\t', dtype=str).fillna("")
true_pairs = set()
for _, row in gt_df.iterrows():
    s1 = row['source1_entity_id']
    ms = row['matched_entity_ids']
    if ms:
        for m in ms.split(','):
            true_pairs.add((s1, m))
            
val_df['is_true'] = val_df.apply(lambda r: (r['source1_entity_id'], r['source_entity_id']) in true_pairs, axis=1).astype(int)

# 4. PREDICT & OPTIMIZE THRESHOLD
print("\n[4] Predicting and Optimizing...")
val_df['prob_base'] = model_base.predict_proba(val_df[base_features])[:, 1]
val_df['prob_new'] = model_new.predict_proba(val_df[new_features])[:, 1]

def evaluate(df, prob_col, threshold):
    pred = set(zip(df[df[prob_col] >= threshold]['source1_entity_id'], df[df[prob_col] >= threshold]['source_entity_id']))
    tp = len(pred.intersection(true_pairs))
    fp = len(pred - true_pairs)
    fn = len(true_pairs - pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f05 = ((1 + 0.25) * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0
    return prec, rec, f05, tp, fp, fn

# Base uses frozen 0.89
prec_b, rec_b, f05_b, tp_b, fp_b, fn_b = evaluate(val_df, 'prob_base', 0.89)

# New sweeps thresholds
best_t, best_f05 = 0, 0
for t in np.arange(0.1, 0.99, 0.01):
    _, _, f, _, _, _ = evaluate(val_df, 'prob_new', t)
    if f > best_f05:
        best_f05 = f
        best_t = t

prec_n, rec_n, f05_n, tp_n, fp_n, fn_n = evaluate(val_df, 'prob_new', best_t)

# 5. SOURCE-SPECIFIC METRICS
def eval_source(df, prob_col, threshold, source_prefix):
    sub_true = {p for p in true_pairs if p[1].startswith(source_prefix)}
    pred = set(zip(df[(df[prob_col] >= threshold) & (df['source_entity_id'].str.startswith(source_prefix))]['source1_entity_id'], 
                   df[(df[prob_col] >= threshold) & (df['source_entity_id'].str.startswith(source_prefix))]['source_entity_id']))
    tp = len(pred.intersection(sub_true))
    fp = len(pred - sub_true)
    fn = len(sub_true - pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f05 = ((1 + 0.25) * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0
    return prec, rec, f05, tp, fp, fn

b_s2_prec, b_s2_rec, b_s2_f05, b_s2_tp, b_s2_fp, b_s2_fn = eval_source(val_df, 'prob_base', 0.89, "S2")
b_s3_prec, b_s3_rec, b_s3_f05, b_s3_tp, b_s3_fp, b_s3_fn = eval_source(val_df, 'prob_base', 0.89, "S3")

n_s2_prec, n_s2_rec, n_s2_f05, n_s2_tp, n_s2_fp, n_s2_fn = eval_source(val_df, 'prob_new', best_t, "S2")
n_s3_prec, n_s3_rec, n_s3_f05, n_s3_tp, n_s3_fp, n_s3_fn = eval_source(val_df, 'prob_new', best_t, "S3")

print("\n" + "=" * 70)
print("EXPERIMENT RESULTS")
print("=" * 70)
print(f"Optimal Threshold | Baseline: 0.89 | Experiment: {best_t:.2f}")
print("-" * 70)
print("OVERALL")
print(f"  Baseline   -> Prec: {prec_b:.4f} | Rec: {rec_b:.4f} | F0.5: {f05_b:.4f} | FP: {fp_b:<4} | FN: {fn_b:<4}")
print(f"  Experiment -> Prec: {prec_n:.4f} | Rec: {rec_n:.4f} | F0.5: {f05_n:.4f} | FP: {fp_n:<4} | FN: {fn_n:<4}")
print("-" * 70)
print("S2 PERFORMANCE")
print(f"  Baseline   -> Prec: {b_s2_prec:.4f} | Rec: {b_s2_rec:.4f} | F0.5: {b_s2_f05:.4f} | FP: {b_s2_fp:<4} | FN: {b_s2_fn:<4}")
print(f"  Experiment -> Prec: {n_s2_prec:.4f} | Rec: {n_s2_rec:.4f} | F0.5: {n_s2_f05:.4f} | FP: {n_s2_fp:<4} | FN: {n_s2_fn:<4}")
print("-" * 70)
print("S3 PERFORMANCE")
print(f"  Baseline   -> Prec: {b_s3_prec:.4f} | Rec: {b_s3_rec:.4f} | F0.5: {b_s3_f05:.4f} | FP: {b_s3_fp:<4} | FN: {b_s3_fn:<4}")
print(f"  Experiment -> Prec: {n_s3_prec:.4f} | Rec: {n_s3_rec:.4f} | F0.5: {n_s3_f05:.4f} | FP: {n_s3_fp:<4} | FN: {n_s3_fn:<4}")

# 6. HARD NEGATIVE INSPECTION
print("\n[6] Generic-Name Hard Negatives Inspection")
mask = (val_df['addr_missing'] == 1) & (val_df['name_token_set'] > 80) & (val_df['is_true'] == 0)
hard_negs = val_df[mask]
base_hn_fp = (hard_negs['prob_base'] >= 0.89).sum()
new_hn_fp = (hard_negs['prob_new'] >= best_t).sum()

print(f"  Hard negatives (Addr missing + High Name Sim) total: {len(hard_negs)}")
print(f"  Baseline FP on these cases  : {base_hn_fp}")
print(f"  Experiment FP on these cases: {new_hn_fp}")

if f05_n > f05_b and (fp_b - fp_n) > 5:
    print("\nEXPERIMENT DECISION: KEEP")
else:
    print("\nEXPERIMENT DECISION: REJECT")

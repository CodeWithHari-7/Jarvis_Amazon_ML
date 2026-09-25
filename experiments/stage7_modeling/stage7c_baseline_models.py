import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import time
import tracemalloc
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix)

proc_dir  = r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker\dataset\processed"
pq_path   = os.path.join(proc_dir, "ml_train_candidates.parquet")

FEAT_COLS = ['name_fuzz_ratio','name_token_set','name_jw',
             'addr_fuzz_ratio','addr_token_set','addr_num_overlap',
             'cross_script']   # country_match excluded: zero variance (always 1)

print("=" * 65)
print("STAGE 7C  —  BASELINE MODEL TRAINING REPORT")
print("=" * 65)

# ── 1. LOAD & ENTITY-AWARE SPLIT ─────────────────────────────
print("\n[1] Loading training dataset ...")
df = pd.read_parquet(pq_path)
df = df.dropna(subset=FEAT_COLS)

# Entity-aware split: group by source1_entity_id
s1_ids = df['source1_entity_id'].unique()
np.random.seed(42)
np.random.shuffle(s1_ids)
split_idx = int(len(s1_ids) * 0.80)
train_ids = set(s1_ids[:split_idx])
val_ids   = set(s1_ids[split_idx:])

train_df = df[df['source1_entity_id'].isin(train_ids)].copy()
val_df   = df[df['source1_entity_id'].isin(val_ids)].copy()

print(f"  Train S1 entities   : {len(train_ids):,}")
print(f"  Val   S1 entities   : {len(val_ids):,}")
print(f"  Train candidates    : {len(train_df):,}  "
      f"(pos={train_df['label'].sum():,} neg={(train_df['label']==0).sum():,})")
print(f"  Val   candidates    : {len(val_df):,}  "
      f"(pos={val_df['label'].sum():,} neg={(val_df['label']==0).sum():,})")
print(f"  Train pos:neg ratio : 1:{(train_df['label']==0).sum()//max(train_df['label'].sum(),1)}")

X_train = train_df[FEAT_COLS].values
y_train = train_df['label'].values
X_val   = val_df[FEAT_COLS].values
y_val   = val_df['label'].values

# ── EVALUATION HELPER ─────────────────────────────────────────
def evaluate(name, y_true, y_pred, y_prob, source_col, runtime, mem_mb):
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    f05  = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0
    roc  = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0
    pr   = average_precision_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n  --- {name} ---")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  F0.5      : {f05:.4f}")
    print(f"  PR-AUC    : {pr:.4f}")
    print(f"  ROC-AUC   : {roc:.4f}")
    print(f"  Runtime   : {runtime:.2f}s")
    print(f"  RAM peak  : {mem_mb:.1f} MB")
    print(f"  Confusion Matrix (TN FP / FN TP):")
    print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    # S2 / S3 breakdown
    for src in ['S2', 'S3']:
        mask = val_df['source'] == src
        if mask.sum() == 0: continue
        p = precision_score(y_true[mask], y_pred[mask], zero_division=0)
        r = recall_score(y_true[mask], y_pred[mask], zero_division=0)
        print(f"  {src}  Prec={p:.3f}  Rec={r:.3f}")

    return {'Model': name, 'Prec': prec, 'Rec': rec,
            'F1': f1, 'F0.5': f05, 'PR-AUC': pr, 'ROC-AUC': roc,
            'Runtime': runtime, 'RAM_MB': mem_mb}

results = []

# ── 2. RULE-BASED BASELINE ────────────────────────────────────
print("\n[2] Rule-based baseline ...")
# Rule: name_token_set >= 80 AND addr_num_overlap >= 0.5
# Chosen from Phase 3 research medians, NOT tuned on validation labels
t0 = time.time()
rb_pred = ((val_df['name_token_set'] >= 80) &
           (val_df['addr_num_overlap'] >= 0.5)).astype(int).values
rb_prob = ((val_df['name_token_set'] / 100) * 0.5 +
            val_df['addr_num_overlap'] * 0.5).values
rt = time.time() - t0
results.append(evaluate("Rule-based (name_ts>=80 AND num_ovlp>=0.5)",
                         y_val, rb_pred, rb_prob, val_df['source'], rt, 0.0))

# ── 3. LOGISTIC REGRESSION ────────────────────────────────────
print("\n[3] Logistic Regression ...")
tracemalloc.start()
t0 = time.time()
lr = LogisticRegression(max_iter=2000, class_weight='balanced',
                        solver='lbfgs', random_state=42)
lr.fit(X_train, y_train)
rt = time.time() - t0
mem_mb = tracemalloc.get_traced_memory()[1] / 1e6
tracemalloc.stop()

lr_prob = lr.predict_proba(X_val)[:, 1]
lr_pred = lr.predict(X_val)
results.append(evaluate("Logistic Regression (balanced)",
                         y_val, lr_pred, lr_prob, val_df['source'], rt, mem_mb))

print("\n  LR Feature Coefficients:")
for feat, coef in sorted(zip(FEAT_COLS, lr.coef_[0]), key=lambda x: -abs(x[1])):
    print(f"    {feat:<25s} : {coef:+.4f}")

# ── 4. RANDOM FOREST ─────────────────────────────────────────
print("\n[4] Random Forest ...")
tracemalloc.start()
t0 = time.time()
rf = RandomForestClassifier(n_estimators=200, max_depth=12,
                             class_weight='balanced',
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rt = time.time() - t0
mem_mb = tracemalloc.get_traced_memory()[1] / 1e6
tracemalloc.stop()

rf_prob = rf.predict_proba(X_val)[:, 1]
rf_pred = rf.predict(X_val)
results.append(evaluate("Random Forest (200 trees, depth=12)",
                         y_val, rf_pred, rf_prob, val_df['source'], rt, mem_mb))

print("\n  RF Feature Importances:")
for feat, imp in sorted(zip(FEAT_COLS, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"    {feat:<25s} : {imp:.4f}")

# ── 5. PROBABILITY DISTRIBUTION ANALYSIS ─────────────────────
print("\n[5] Probability distribution analysis (Random Forest)")
for model_name, probs in [("LR", lr_prob), ("RF", rf_prob)]:
    print(f"\n  {model_name} — P(match) distribution:")
    for label, desc in [(1, "True Positives/FN"), (0, "True Negatives/FP")]:
        mask = y_val == label
        p = probs[mask]
        print(f"    label={label}  mean={p.mean():.3f}  median={np.median(p):.3f}  "
              f"p10={np.percentile(p,10):.3f}  p90={np.percentile(p,90):.3f}")

# False Positive pattern analysis
print("\n[6] RF False Positive pattern analysis (top 10)")
fp_mask = (rf_pred == 1) & (y_val == 0)
fp_df   = val_df[fp_mask].copy()
fp_df['prob'] = rf_prob[fp_mask]
fp_df = fp_df.sort_values('prob', ascending=False).head(10)
print(fp_df[['source','negative_type','name_token_set',
             'addr_token_set','addr_num_overlap','prob']].to_string(index=False))

print("\n[7] RF False Negative pattern analysis (top 10)")
fn_mask = (rf_pred == 0) & (y_val == 1)
fn_df   = val_df[fn_mask].copy()
fn_df['prob'] = rf_prob[fn_mask]
fn_df = fn_df.sort_values('prob').head(10)
print(fn_df[['source','name_token_set','addr_token_set',
             'addr_num_overlap','cross_script','prob']].to_string(index=False))

# ── 6. COMPARISON TABLE ───────────────────────────────────────
print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON TABLE")
print("=" * 65)
res_df = pd.DataFrame(results)
print(res_df[['Model','Prec','Rec','F1','F0.5','PR-AUC','ROC-AUC','Runtime']].to_string(index=False))

# ── 7. SAVE PROBABILITIES ─────────────────────────────────────
val_df = val_df.copy()
val_df['prob_lr'] = lr_prob
val_df['prob_rf'] = rf_prob
val_df['pred_rb'] = rb_pred
val_df['pred_lr'] = lr_pred
val_df['pred_rf'] = rf_pred
val_df.to_parquet(os.path.join(proc_dir, "val_predictions.parquet"), index=False)
print(f"\nSaved val_predictions.parquet  shape={val_df.shape}")

print("\n=== STAGE 7C COMPLETE ===")

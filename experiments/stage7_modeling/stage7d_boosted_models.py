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
import xgboost as xgb
import lightgbm as lgb
import shap

proc_dir  = r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker\dataset\processed"
pq_path   = os.path.join(proc_dir, "ml_train_candidates.parquet")

FEAT_COLS = ['name_fuzz_ratio','name_token_set','name_jw',
             'addr_fuzz_ratio','addr_token_set','addr_num_overlap',
             'cross_script']

print("=" * 70)
print("STAGE 7D  —  GRADIENT BOOSTED TREES COMPARISON")
print("=" * 70)

# ── 1. LOAD & ENTITY-AWARE SPLIT (identical to Stage 7C) ─────
print("\n[1] Loading and splitting data (entity-aware, 80/20) ...")
df = pd.read_parquet(pq_path).dropna(subset=FEAT_COLS)

s1_ids = df['source1_entity_id'].unique()
np.random.seed(42)
np.random.shuffle(s1_ids)
split_idx = int(len(s1_ids) * 0.80)
train_ids = set(s1_ids[:split_idx])
val_ids   = set(s1_ids[split_idx:])

train_df = df[df['source1_entity_id'].isin(train_ids)].copy()
val_df   = df[df['source1_entity_id'].isin(val_ids)].copy()

X_train = train_df[FEAT_COLS].values
y_train = train_df['label'].values
X_val   = val_df[FEAT_COLS].values
y_val   = val_df['label'].values

pos_count = y_train.sum()
neg_count = (y_train == 0).sum()
scale_pos = neg_count / pos_count
print(f"  Train: {len(train_df):,}  pos={pos_count:,}  neg={neg_count:,}")
print(f"  Val  : {len(val_df):,}  pos={y_val.sum():,}  neg={(y_val==0).sum():,}")
print(f"  scale_pos_weight for XGB/LGB: {scale_pos:.2f}")

all_results = []

# ── EVALUATION HELPER ─────────────────────────────────────────
def evaluate(name, y_true, y_pred, y_prob, runtime, mem_mb):
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    f05  = (1.25*prec*rec) / (0.25*prec+rec) if (prec+rec)>0 else 0
    roc  = roc_auc_score(y_true, y_prob)
    pr   = average_precision_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)
    print(f"\n  [{name}]")
    print(f"  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  F0.5={f05:.4f}")
    print(f"  PR-AUC={pr:.4f}  ROC-AUC={roc:.4f}")
    print(f"  Runtime={runtime:.2f}s  RAM={mem_mb:.1f}MB")
    print(f"  CM: TN={cm[0,0]:,} FP={cm[0,1]:,} | FN={cm[1,0]:,} TP={cm[1,1]:,}")
    for src in ['S2','S3']:
        m = val_df['source'] == src
        if m.sum():
            p = precision_score(y_true[m], y_pred[m], zero_division=0)
            r = recall_score(y_true[m], y_pred[m], zero_division=0)
            print(f"  {src}: Prec={p:.4f}  Rec={r:.4f}")
    all_results.append({'Model':name,'Prec':round(prec,4),'Rec':round(rec,4),
                        'F1':round(f1,4),'F0.5':round(f05,4),
                        'PR-AUC':round(pr,4),'ROC-AUC':round(roc,4),
                        'Runtime_s':round(runtime,2),'RAM_MB':round(mem_mb,1)})
    return y_prob

# ── 2. BASELINES (re-run for fair same-run comparison) ────────
print("\n[2] Logistic Regression baseline ...")
tracemalloc.start(); t0=time.time()
lr = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)
rt = time.time()-t0; mem=tracemalloc.get_traced_memory()[1]/1e6; tracemalloc.stop()
lr_prob = lr.predict_proba(X_val)[:,1]
evaluate("Logistic Regression", y_val, lr.predict(X_val), lr_prob, rt, mem)

print("\n[3] Random Forest baseline ...")
tracemalloc.start(); t0=time.time()
rf = RandomForestClassifier(n_estimators=200, max_depth=12,
                             class_weight='balanced', n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
rt = time.time()-t0; mem=tracemalloc.get_traced_memory()[1]/1e6; tracemalloc.stop()
rf_prob = rf.predict_proba(X_val)[:,1]
evaluate("Random Forest (200t, d=12)", y_val, rf.predict(X_val), rf_prob, rt, mem)

# ── 3. XGBOOST ───────────────────────────────────────────────
print("\n[4] XGBoost ...")
tracemalloc.start(); t0=time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    eval_metric='aucpr', early_stopping_rounds=30,
    random_state=42, n_jobs=-1, verbosity=0
)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], verbose=False)
rt = time.time()-t0; mem=tracemalloc.get_traced_memory()[1]/1e6; tracemalloc.stop()
xgb_prob = xgb_model.predict_proba(X_val)[:,1]
xgb_pred = (xgb_prob >= 0.5).astype(int)
evaluate("XGBoost (500t, lr=0.05, scale_pw)", y_val, xgb_pred, xgb_prob, rt, mem)

print("\n  XGB Feature Importances (gain):")
imp_xgb = dict(zip(FEAT_COLS, xgb_model.feature_importances_))
for k,v in sorted(imp_xgb.items(), key=lambda x:-x[1]):
    print(f"    {k:<25s}: {v:.4f}")

# ── 4. LIGHTGBM ──────────────────────────────────────────────
print("\n[5] LightGBM ...")
tracemalloc.start(); t0=time.time()
lgb_model = lgb.LGBMClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    random_state=42, n_jobs=-1, verbose=-1
)
lgb_model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(period=-1)])
rt = time.time()-t0; mem=tracemalloc.get_traced_memory()[1]/1e6; tracemalloc.stop()
lgb_prob = lgb_model.predict_proba(X_val)[:,1]
lgb_pred = (lgb_prob >= 0.5).astype(int)
evaluate("LightGBM (500t, lr=0.05, scale_pw)", y_val, lgb_pred, lgb_prob, rt, mem)

print("\n  LGB Feature Importances (gain):")
imp_lgb = dict(zip(FEAT_COLS, lgb_model.feature_importances_))
for k,v in sorted(imp_lgb.items(), key=lambda x:-x[1]):
    print(f"    {k:<25s}: {v:.4f}")

# ── 5. CLASS IMBALANCE VARIANTS ──────────────────────────────
print("\n[6] Class-imbalance handling comparison (LightGBM variants):")

# Variant A: no weighting (naive)
tracemalloc.start(); t0=time.time()
lgb_naive = lgb.LGBMClassifier(n_estimators=300, max_depth=6,
                                random_state=42, n_jobs=-1, verbose=-1)
lgb_naive.fit(X_train, y_train, callbacks=[lgb.log_evaluation(period=-1)])
rt=time.time()-t0; mem=tracemalloc.get_traced_memory()[1]/1e6; tracemalloc.stop()
p_naive = lgb_naive.predict_proba(X_val)[:,1]
evaluate("LightGBM (no class weight)", y_val, (p_naive>=0.5).astype(int), p_naive, rt, mem)

# Variant B: balanced sampling only
tracemalloc.start(); t0=time.time()
from sklearn.utils import resample
X_pos = X_train[y_train==1]; X_neg = X_train[y_train==0]
X_neg_ds = resample(X_neg, n_samples=len(X_pos)*3, random_state=42)
X_bal = np.vstack([X_pos, X_neg_ds])
y_bal = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg_ds))])
lgb_ds = lgb.LGBMClassifier(n_estimators=300, max_depth=6,
                              random_state=42, n_jobs=-1, verbose=-1)
lgb_ds.fit(X_bal, y_bal, callbacks=[lgb.log_evaluation(period=-1)])
rt=time.time()-t0; mem=tracemalloc.get_traced_memory()[1]/1e6; tracemalloc.stop()
p_ds = lgb_ds.predict_proba(X_val)[:,1]
evaluate("LightGBM (downsampled 1:3)", y_val, (p_ds>=0.5).astype(int), p_ds, rt, mem)

# ── 6. SHAP ANALYSIS (LightGBM best model) ───────────────────
print("\n[7] SHAP Analysis (LightGBM scale_pos_weight) ...")
X_val_df = pd.DataFrame(X_val, columns=FEAT_COLS)
explainer  = shap.TreeExplainer(lgb_model)
shap_vals  = explainer.shap_values(X_val_df)
# For binary: shap_values returns list [neg_class, pos_class] or single array
if isinstance(shap_vals, list):
    sv = shap_vals[1]
else:
    sv = shap_vals
mean_abs = np.abs(sv).mean(axis=0)
print("  SHAP Mean |value| per feature:")
for feat, sh in sorted(zip(FEAT_COLS, mean_abs), key=lambda x: -x[1]):
    print(f"    {feat:<25s}: {sh:.4f}")

# False positive pattern (LightGBM)
print("\n[8] LightGBM False Positive patterns (top 10):")
fp_mask = (lgb_pred == 1) & (y_val == 0)
fp_df = val_df[fp_mask].copy()
fp_df['prob'] = lgb_prob[fp_mask]
fp_df = fp_df.sort_values('prob', ascending=False).head(10)
print(fp_df[['source','negative_type','name_token_set',
             'addr_token_set','addr_num_overlap','prob']].to_string(index=False))

# ── 7. FINAL TABLE ───────────────────────────────────────────
print("\n" + "="*70)
print("FINAL MODEL COMPARISON TABLE")
print("="*70)
res_df = pd.DataFrame(all_results)
print(res_df.to_string(index=False))

# Save best model probabilities
val_df = val_df.copy()
val_df['prob_lr']  = lr_prob
val_df['prob_rf']  = rf_prob
val_df['prob_xgb'] = xgb_prob
val_df['prob_lgb'] = lgb_prob
val_df.to_parquet(os.path.join(proc_dir, "val_predictions_v2.parquet"), index=False)

# Save best LightGBM model
import pickle
with open(os.path.join(proc_dir, "lgb_model.pkl"), "wb") as f:
    pickle.dump(lgb_model, f)
print(f"\nSaved lgb_model.pkl and val_predictions_v2.parquet")
print("\n=== STAGE 7D COMPLETE ===")

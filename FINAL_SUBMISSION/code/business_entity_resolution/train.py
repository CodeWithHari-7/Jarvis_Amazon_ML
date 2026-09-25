"""
JARVIS_CHECKER — Complete Training Pipeline
Trains the LightGBM pair-classifier on training data.

Usage:
    python train.py [--sample N] [--output-model dataset/processed/lgb_model.pkl]

Arguments:
    --sample N: Train on N S1 entities (default: all, which may be 2M+ and take hours)
                For quick experiments use: --sample 10000
    --output-model: Path to save the trained model pickle
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import time
import argparse
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

# ── Parse arguments ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Train JARVIS_CHECKER entity resolution model')
parser.add_argument('--sample', type=int, default=None,
                    help='Number of S1 entities to sample for training (default: all)')
parser.add_argument('--output-model', type=str, default='dataset/processed/lgb_model.pkl',
                    help='Path to save trained model')
parser.add_argument('--train-dir', type=str, default='dataset/train',
                    help='Directory containing train TSV files')
parser.add_argument('--proc-dir', type=str, default='dataset/processed',
                    help='Directory to save processed features and models')
parser.add_argument('--threshold-sweep', action='store_true',
                    help='Run threshold sweep after training')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# ── Import pipeline modules ───────────────────────────────────────────────────
from src.cleaning.data_cleaning import load_and_clean_tsv
from src.normalization.normalization import normalize_record
from src.blocking.blocking import build_s23_index, generate_candidates_for_record
from src.features.feature_extraction import extract_features_for_pair, FEATURE_COLS

print("=" * 70)
print("JARVIS_CHECKER — TRAINING PIPELINE")
print("=" * 70)

TRAIN_S1 = os.path.join(args.train_dir, 'train_source1.tsv')
TRAIN_S2 = os.path.join(args.train_dir, 'train_source2.tsv')
TRAIN_S3 = os.path.join(args.train_dir, 'train_source3.tsv')
TRAIN_GT = os.path.join(args.train_dir, 'train_ground_truth.tsv')
PROC_DIR = args.proc_dir
os.makedirs(PROC_DIR, exist_ok=True)

for path in [TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT]:
    if not os.path.exists(path):
        print(f"ERROR: Required file not found: {path}")
        print("Please place the training TSV files in dataset/train/")
        sys.exit(1)

# ── 1. Load Ground Truth ──────────────────────────────────────────────────────
print("\n[1] Loading ground truth...")
t0 = time.time()
gt_df = pd.read_csv(TRAIN_GT, sep='\t', dtype=str, keep_default_na=False)
gt_df = gt_df[gt_df['matched_entity_ids'].notna() & (gt_df['matched_entity_ids'] != '')].copy()

# Build positive pair set and S1->matches map
gt_pos = set()
s1_to_matches = defaultdict(set)
for _, row in gt_df.iterrows():
    s1 = row['source1_entity_id']
    for m in str(row['matched_entity_ids']).split(','):
        m = m.strip()
        if m:
            gt_pos.add((s1, m))
            s1_to_matches[s1].add(m)

print(f"  GT rows: {len(gt_df):,} | Total positive pairs: {len(gt_pos):,}")
print(f"  S1 entities with at least 1 match: {len(s1_to_matches):,}")

# ── 2. Sample S1 entities ─────────────────────────────────────────────────────
print("\n[2] Loading and sampling S1 entities...")
t0 = time.time()
all_s1_records = load_and_clean_tsv(TRAIN_S1)
all_s1_records = [normalize_record(r) for r in all_s1_records]
print(f"  Total S1: {len(all_s1_records):,}")

np.random.seed(args.seed)
if args.sample and args.sample < len(all_s1_records):
    # Sample S1 entities, prefer those with known matches
    s1_with_matches = [r for r in all_s1_records if r['entity_id'] in s1_to_matches]
    s1_without_matches = [r for r in all_s1_records if r['entity_id'] not in s1_to_matches]
    
    # Take 70% from those with matches, 30% from those without
    n_with = min(int(args.sample * 0.7), len(s1_with_matches))
    n_without = min(args.sample - n_with, len(s1_without_matches))
    
    idx_with = np.random.choice(len(s1_with_matches), n_with, replace=False)
    idx_without = np.random.choice(len(s1_without_matches), n_without, replace=False)
    
    sampled_s1 = [s1_with_matches[i] for i in idx_with] + [s1_without_matches[i] for i in idx_without]
    print(f"  Sampled: {len(sampled_s1):,} S1 entities "
          f"({n_with} with matches, {n_without} without)")
else:
    sampled_s1 = all_s1_records
    print(f"  Using all {len(sampled_s1):,} S1 entities")

sampled_s1_ids = {r['entity_id'] for r in sampled_s1}
del all_s1_records

# ── 3. Load S2/S3 ─────────────────────────────────────────────────────────────
print("\n[3] Loading S2/S3 training data...")
t0 = time.time()
s2_records_raw = load_and_clean_tsv(TRAIN_S2)
s3_records_raw = load_and_clean_tsv(TRAIN_S3)
print(f"  Raw: S2={len(s2_records_raw):,} | S3={len(s3_records_raw):,}")

# Normalize
print("  Normalizing S2/S3...")
s2_records = [normalize_record(r) for r in s2_records_raw]
s3_records = [normalize_record(r) for r in s3_records_raw]
for r in s3_records:
    r['_source'] = 'S3'
for r in s2_records:
    r['_source'] = 'S2'
s23_records = s2_records + s3_records
s23_lookup = {r['entity_id']: r for r in s23_records}
print(f"  Normalized S2/S3: {len(s23_records):,} records in {time.time()-t0:.1f}s")
del s2_records_raw, s3_records_raw

# ── 4. Build blocking index ────────────────────────────────────────────────────
print("\n[4] Building blocking inverted indexes...")
t0 = time.time()
indexes = build_s23_index(s23_records)
print(f"  Indexes built in {time.time()-t0:.1f}s")

# ── 5. Generate candidates ─────────────────────────────────────────────────────
print("\n[5] Generating candidates (blocking)...")
t0 = time.time()
candidate_pairs = []
missed_positives = 0
recovered_positives = 0

for r in sampled_s1:
    s1_id = r['entity_id']
    cands = generate_candidates_for_record(r, indexes)
    
    # Ensure known positives are always in candidates (for training recall)
    known_matches = s1_to_matches.get(s1_id, set())
    in_s23 = known_matches & s23_lookup.keys()
    missed = in_s23 - cands
    missed_positives += len(missed)
    recovered_positives += len(in_s23 & cands)
    
    # Force positives into candidate set
    all_cands = cands | missed
    for cid in all_cands:
        candidate_pairs.append((s1_id, cid))

total_in_scope = recovered_positives + missed_positives
blocking_recall = recovered_positives / total_in_scope if total_in_scope > 0 else 0.0
print(f"  Candidates: {len(candidate_pairs):,}")
print(f"  Blocking recall: {blocking_recall:.4f} "
      f"({recovered_positives:,} / {total_in_scope:,} true pairs found)")
print(f"  Forced into candidates: {missed_positives:,} missed positives")
print(f"  Time: {time.time()-t0:.1f}s")

# ── 6. Feature extraction ──────────────────────────────────────────────────────
print("\n[6] Extracting features...")
t0 = time.time()
s1_lookup = {r['entity_id']: r for r in sampled_s1}

features_list = []
for s1_id, cid in candidate_pairs:
    r1 = s1_lookup.get(s1_id)
    r2 = s23_lookup.get(cid)
    if not r1 or not r2:
        continue
    
    feat = extract_features_for_pair(r1, r2)
    feat['source1_entity_id'] = s1_id
    feat['source_entity_id'] = cid
    feat['label'] = 1 if (s1_id, cid) in gt_pos else 0
    feat['source'] = r2.get('_source', 'S2' if str(cid).startswith('S2') else 'S3')
    features_list.append(feat)

feat_df = pd.DataFrame(features_list)
pos_count = feat_df['label'].sum()
neg_count = (feat_df['label'] == 0).sum()
print(f"  Feature rows: {len(feat_df):,} | Positives: {pos_count:,} | Negatives: {neg_count:,}")
print(f"  Class ratio: 1:{neg_count/pos_count:.1f}")
print(f"  Time: {time.time()-t0:.1f}s")

# Save training features
feat_df.to_parquet(os.path.join(PROC_DIR, 'ml_train_candidates.parquet'), index=False)
print(f"  Saved to {PROC_DIR}/ml_train_candidates.parquet")

# ── 7. Entity-aware split ──────────────────────────────────────────────────────
print("\n[7] Creating entity-aware train/validation split (80/20)...")
s1_ids_arr = feat_df['source1_entity_id'].unique()
np.random.seed(args.seed)
np.random.shuffle(s1_ids_arr)
split_idx = int(len(s1_ids_arr) * 0.80)
train_ids = set(s1_ids_arr[:split_idx])
val_ids = set(s1_ids_arr[split_idx:])

train_df = feat_df[feat_df['source1_entity_id'].isin(train_ids)].copy()
val_df = feat_df[feat_df['source1_entity_id'].isin(val_ids)].copy()

X_train = train_df[FEATURE_COLS].fillna(0).values
y_train = train_df['label'].values
X_val = val_df[FEATURE_COLS].fillna(0).values
y_val = val_df['label'].values

print(f"  Train: {len(train_df):,} rows ({y_train.sum():,} pos) | "
      f"Val: {len(val_df):,} rows ({y_val.sum():,} pos)")

# ── 8. Train LightGBM ─────────────────────────────────────────────────────────
print("\n[8] Training LightGBM model...")
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

t0 = time.time()
# Use no class-weight (per DECISIONS.md — class weighting degraded precision)
# But try with and without based on class ratio
scale_pos = neg_count / pos_count if pos_count > 0 else 1.0
print(f"  scale_pos_weight: {scale_pos:.2f}")

lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    random_state=args.seed,
    n_jobs=-1,
    verbose=-1,
)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(period=50),
    ]
)
train_time = time.time() - t0
print(f"  Training done in {train_time:.1f}s")
print(f"  Best iteration: {lgb_model.best_iteration_}")

# ── 9. Validation metrics ──────────────────────────────────────────────────────
print("\n[9] Validation metrics...")
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

val_probs = lgb_model.predict_proba(X_val)[:, 1]
val_df = val_df.copy()
val_df['prob'] = val_probs

def compute_macro_f05(df: pd.DataFrame, threshold: float) -> float:
    """Compute official macro-averaged F0.5 per S1 entity."""
    scores = []
    for s1_id, grp in df.groupby('source1_entity_id'):
        gt_pos_ = set(grp[grp['label'] == 1]['source_entity_id'])
        pred_pos_ = set(grp[grp['prob'] >= threshold]['source_entity_id'])
        
        if not gt_pos_:
            scores.append(1.0 if not pred_pos_ else 0.0)
            continue
        
        tp = len(gt_pos_ & pred_pos_)
        fp = len(pred_pos_ - gt_pos_)
        fn = len(gt_pos_ - pred_pos_)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f05 = 1.25 * prec * rec / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0.0
        scores.append(f05)
    return float(np.mean(scores)) if scores else 0.0

# Feature importances
print("\n  Feature importances:")
for feat, imp in sorted(zip(FEATURE_COLS, lgb_model.feature_importances_), key=lambda x: -x[1]):
    print(f"    {feat:<30s}: {imp:.4f}")

# Threshold sweep
print("\n[10] Threshold sweep for F0.5 optimization...")
best_t, best_f05 = 0.5, 0.0
results = []
thresholds = np.concatenate([
    np.arange(0.10, 0.50, 0.05),
    np.arange(0.50, 0.92, 0.02),
    np.arange(0.92, 0.99, 0.005),
])

for t in thresholds:
    y_pred = (val_probs >= t).astype(int)
    tp = int(((y_pred == 1) & (y_val == 1)).sum())
    fp = int(((y_pred == 1) & (y_val == 0)).sum())
    fn = int(((y_pred == 0) & (y_val == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    macro_f05 = compute_macro_f05(val_df, t)
    results.append({'t': round(t, 4), 'prec': prec, 'rec': rec, 'macro_f05': macro_f05,
                    'tp': tp, 'fp': fp, 'fn': fn})
    if macro_f05 > best_f05:
        best_f05, best_t = macro_f05, t

sweep_df = pd.DataFrame(results)
best_row = sweep_df.loc[sweep_df['macro_f05'].idxmax()]

print(f"\n  === THRESHOLD SWEEP RESULTS (top rows) ===")
top_rows = sweep_df.nlargest(10, 'macro_f05')
for _, row in top_rows.iterrows():
    print(f"  t={row['t']:.4f}  Prec={row['prec']:.4f}  Rec={row['rec']:.4f}  "
          f"MacroF05={row['macro_f05']:.4f}  TP={row['tp']}  FP={row['fp']}  FN={row['fn']}")

print(f"\n  === BEST THRESHOLD ===")
print(f"  Threshold: {best_row['t']:.4f}")
print(f"  Macro F0.5: {best_row['macro_f05']:.4f}")
print(f"  Precision: {best_row['prec']:.4f}")
print(f"  Recall: {best_row['rec']:.4f}")
print(f"  TP={best_row['tp']:.0f} FP={best_row['fp']:.0f} FN={best_row['fn']:.0f}")

sweep_df.to_csv(os.path.join(PROC_DIR, 'threshold_sweep.csv'), index=False)

# Singleton performance
print("\n  Singleton performance at best threshold:")
singleton_count = 0
singleton_correct = 0
for s1_id, grp in val_df.groupby('source1_entity_id'):
    gt_pos_ = set(grp[grp['label'] == 1]['source_entity_id'])
    pred_pos_ = set(grp[grp['prob'] >= best_t]['source_entity_id'])
    if not gt_pos_:
        singleton_count += 1
        if not pred_pos_:
            singleton_correct += 1

if singleton_count > 0:
    print(f"  S1 singletons in val: {singleton_count:,} | "
          f"Correctly predicted empty: {singleton_correct:,} "
          f"({100*singleton_correct/singleton_count:.1f}%)")

# ── 11. Save model and config ──────────────────────────────────────────────────
print(f"\n[11] Saving model...")
model_path = args.output_model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
with open(model_path, 'wb') as f:
    pickle.dump(lgb_model, f)
print(f"  Model saved: {model_path}")

# Save best threshold
threshold_path = os.path.join(PROC_DIR, 'best_threshold.txt')
with open(threshold_path, 'w') as f:
    f.write(str(round(float(best_row['t']), 4)))
print(f"  Best threshold saved: {threshold_path}")

print(f"\n=== TRAINING COMPLETE ===")
print(f"  Model: LightGBM (n_estimators={lgb_model.best_iteration_ or 500})")
print(f"  Features: {len(FEATURE_COLS)}")
print(f"  Best threshold: {best_row['t']:.4f}")
print(f"  Validation Macro F0.5: {best_row['macro_f05']:.4f}")
print(f"  Validation Precision: {best_row['prec']:.4f}")
print(f"  Validation Recall: {best_row['rec']:.4f}")
print(f"  Blocking recall: {blocking_recall:.4f}")
print(f"  Total training time: {time.time()-t0:.0f}s")

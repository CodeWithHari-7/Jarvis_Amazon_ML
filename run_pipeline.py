import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import yaml
import time
import polars as pl
import pandas as pd

# Pipeline Modules
from src.cleaning.data_cleaning import clean_dataset
from src.normalization.normalization import normalize_dataset
from src.blocking.blocking import generate_candidates
from src.features.feature_extraction import extract_features_for_pair
from src.models.inference import load_model, predict
from src.submission.format import format_submission

print("=" * 70)
print("JARVIS_CHECKER — STAGE 8 (END-TO-END PIPELINE)")
print("=" * 70)

# ── 1. LOAD CONFIGURATION ───────────────────────────────────────
print("\n[1] Loading Configuration...")
with open(os.path.join("configs", "final_config.yaml"), "r") as f:
    config = yaml.safe_load(f)

cache_dir = config['paths']['cache_dir']
os.makedirs(cache_dir, exist_ok=True)

# ── 2. DATA INGESTION & VALIDATION ────────────────────────────
print("\n[2] Reading Raw Data...")
t0 = time.time()
s1_df = pl.read_csv(config['paths']['s1_input'], separator='\t', ignore_errors=True, schema_overrides={'entity_id': pl.Utf8})
s2_df = pl.read_csv(config['paths']['s2_input'], separator='\t', ignore_errors=True, schema_overrides={'entity_id': pl.Utf8})
s3_df = pl.read_csv(config['paths']['s3_input'], separator='\t', ignore_errors=True, schema_overrides={'entity_id': pl.Utf8})

if config['pipeline'].get('sample_limit'):
    limit = config['pipeline']['sample_limit']
    print(f"  WARNING: Sampling S1, S2, and S3 to {limit} records for fast testing to avoid memory crashes.")
    s1_df = s1_df.head(limit)
    s2_df = s2_df.head(limit * 5)
    s3_df = s3_df.head(limit * 5)

assert 'entity_id' in s1_df.columns, "S1 missing entity_id"
assert 'entity_id' in s2_df.columns, "S2 missing entity_id"
assert 'entity_id' in s3_df.columns, "S3 missing entity_id"
print(f"  Loaded in {time.time()-t0:.2f}s (S1: {len(s1_df):,}, S2: {len(s2_df):,}, S3: {len(s3_df):,})")

# ── 3. CLEANING & NORMALIZATION ───────────────────────────────
print("\n[3] Cleaning and Normalization...")
t0 = time.time()

def process_source(df: pl.DataFrame, name: str) -> pl.DataFrame:
    df_clean = clean_dataset(df)
    df_norm = normalize_dataset(df_clean)
    # Cache to parquet
    out_path = os.path.join(cache_dir, f"{name}_normalized.parquet")
    df_norm.write_parquet(out_path)
    return df_norm

s1_norm = process_source(s1_df, "s1")
s2_norm = process_source(s2_df, "s2")
s3_norm = process_source(s3_df, "s3")
print(f"  Processed and cached normalized data in {time.time()-t0:.2f}s")

# ── 4. BLOCKING (CANDIDATE GENERATION) ────────────────────────
print("\n[4] Candidate Generation (Multi-Pass Blocking)...")
t0 = time.time()
s23_norm = pl.concat([s2_norm, s3_norm], how="align")
candidates = generate_candidates(s1_norm, s23_norm)

# Convert to DataFrame and Cache
cand_df = pd.DataFrame(list(candidates), columns=['source1_entity_id', 'source_entity_id'])
cand_df.to_parquet(os.path.join(cache_dir, "candidate_pairs.parquet"))
print(f"  Total Candidates: {len(cand_df):,}. Cached in {time.time()-t0:.2f}s")

# ── 5. FEATURE EXTRACTION ─────────────────────────────────────
print("\n[5] Feature Extraction (RapidFuzz)...")
t0 = time.time()

s1_lookup = {r['entity_id']: r for r in s1_norm.to_dicts()}
s23_lookup = {r['entity_id']: r for r in s23_norm.to_dicts()}

features = []
for idx, (s1_id, src_id) in enumerate(candidates):
    r1 = s1_lookup.get(s1_id)
    r2 = s23_lookup.get(src_id)
    if not r1 or not r2: continue
    
    feat = extract_features_for_pair(r1, r2)
    feat['source1_entity_id'] = s1_id
    feat['source_entity_id'] = src_id
    features.append(feat)

features_df = pd.DataFrame(features)
features_df.to_parquet(os.path.join(cache_dir, "candidate_features.parquet"))

# Validation Check
expected_cols = config['features']
missing_cols = [c for c in expected_cols if c not in features_df.columns]
assert not missing_cols, f"Missing feature columns: {missing_cols}"
print(f"  Extracted {len(features_df):,} feature rows in {time.time()-t0:.2f}s")

# ── 6. ML INFERENCE ───────────────────────────────────────────
print(f"\n[6] ML Inference ({config['model']['type']})...")
t0 = time.time()
model = load_model(config['paths']['model_path'])
probs = predict(model, features_df, expected_cols)
features_df['prob'] = probs
print(f"  Generated predictions in {time.time()-t0:.2f}s")

# ── 7. FORMAT SUBMISSION ──────────────────────────────────────
print(f"\n[7] Thresholding & Submission Formatting (t={config['model']['threshold']})...")
t0 = time.time()
s1_ids_list = s1_df['entity_id'].to_list()
result_df = format_submission(s1_ids_list, features_df['source1_entity_id'], features_df['source_entity_id'], features_df['prob'], config['model']['threshold'])

out_tsv = config['paths']['output_tsv']
result_df.to_csv(out_tsv, sep='\t', index=False)
print(f"  Formatted {len(result_df):,} S1 entries in {time.time()-t0:.2f}s")
print(f"  Submission saved to: {out_tsv}")

print("\n=== PIPELINE RUN COMPLETE ===")

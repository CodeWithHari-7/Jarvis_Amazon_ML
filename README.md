# JARVIS_CHECKER — Business Entity Resolution
## Amazon ML Challenge

---

## 1. Environment

- **Python**: 3.10+
- **OS**: Windows / Linux / macOS
- **RAM**: ≥ 8GB recommended for full inference

## 2. Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `lightgbm` — gradient boosted tree classifier
- `rapidfuzz` — fast string similarity (Levenshtein, Jaro-Winkler, token ratios)
- `pandas`, `numpy` — data processing
- `scikit-learn` — model evaluation utilities
- `pyarrow` — parquet caching

## 3. Dataset Structure

Place TSV files as follows:

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

All files are **tab-separated** with columns:
- `entity_id` — unique entity identifier (e.g., S1-00001)
- `business_name` — business name (may be in English, Hindi, French, etc.)
- `business_address` — full address string
- `country` — country name/code

Ground truth: `source1_entity_id<TAB>matched_entity_ids` (comma-separated S2/S3 IDs)

## 4. Reproduction Steps

### Step 1: Train the model

```bash
# Quick training on 10,000 S1 entities (fast, ~10-20 min)
python train.py --sample 10000

# Full training on all S1 entities (may take 1-4 hours)
python train.py
```

This saves:
- `dataset/processed/lgb_model.pkl` — trained LightGBM model
- `dataset/processed/best_threshold.txt` — optimal decision threshold
- `dataset/processed/threshold_sweep.csv` — full sweep results
- `dataset/processed/ml_train_candidates.parquet` — training feature matrix

### Step 2: Run test inference

```bash
python infer.py \
    --model dataset/processed/lgb_model.pkl \
    --output-dir output \
    --test-dir dataset/test \
    --chunksize 50000
```

This generates:
- `output/matching_results.tsv` — final submission
- `output/candidate_pairs.tsv` — all candidate pairs passed to the model

### Step 3: Validate submission

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Expected output: `VALIDATION: PASS`

## 5. Data Preprocessing

### Cleaning (`src/cleaning/data_cleaning.py`)
- Fill null fields with empty string
- Strip leading/trailing whitespace
- Preserve original values under `*_raw` keys

### Normalization (`src/normalization/normalization.py`)

**Business names:**
1. Convert accented Latin characters to ASCII (é→e, à→a, ç→c, etc.) — supports French
2. Lowercase
3. Replace `&` with `and`
4. Strip legal suffixes (Ltd, LLC, GmbH, SARL, etc.) — 40+ patterns
5. Remove non-alphanumeric characters (preserving Indic Unicode)
6. Normalize whitespace

**Addresses:**
1. Accented character conversion
2. Lowercase
3. Expand common abbreviations (St.→Street, Ave.→Avenue, etc.)
4. Remove non-alphanumeric characters
5. Normalize whitespace
6. Extract numeric sequences (house numbers, PIN codes)

**Country:**
- Standardize to canonical forms: `india`, `united states`, `france`, etc.
- Open-set: unknown countries kept as normalized strings

## 6. Blocking / Candidate Generation

Multi-pass strategy (`src/blocking/blocking.py`) with 5 complementary passes:

| Pass | Key | Notes |
|------|-----|-------|
| 1 | `country + name_prefix[:4]` | Primary name-based blocking |
| 2 | `country + name_prefix[:3]` | Catches short names (< 4 chars) |
| 3 | `country + addr_number` | Per address digit sequence |
| 4 | `country + name_token` | Individual name tokens (≥3 chars) |
| 5 | `country + addr_first/last_token` | City/locality matching |

All passes are UNIONed. Bucket size capped at 3,000 to prevent explosion.

**Benchmark (5k S1 sample):** ~96% blocking recall vs 2-pass baseline.

## 7. Feature Engineering

24 features extracted per (S1, S2/S3) candidate pair (`src/features/feature_extraction.py`):

**Name features (12):**
- `name_exact`, `name_ratio`, `name_partial`, `name_token_set`, `name_token_sort`
- `name_jw` (Jaro-Winkler), `name_token_jaccard`, `name_token_containment`
- `name_is_abbrev`, `name_len_diff`, `name_prefix4_match`, `name_both_present`

**Address features (8):**
- `addr_exact`, `addr_ratio`, `addr_token_set`, `addr_token_jaccard`
- `addr_num_overlap`, `addr_num_exact`, `addr_len_diff`, `addr_both_present`

**Country features (2):**
- `country_match`, `country_either_empty`

**Structural features (2):**
- `source_is_s2` (S2 vs S3 indicator), `cross_script` (Indic vs non-Indic)

## 8. Model

- **Type**: LightGBM `LGBMClassifier`
- **Parameters**: n_estimators=500, max_depth=6, lr=0.05, subsample=0.8, colsample_bytree=0.8
- **Early stopping**: 50 rounds on validation F0.5
- **Class weighting**: None (per experimental validation — weighting degraded precision)

## 9. Training Procedure

1. Load ground truth — build `(s1_id, s23_id)` positive pair set
2. Sample S1 entities (70% with known matches, 30% without)
3. Load all S2/S3 and normalize
4. Build blocking indexes
5. Generate candidates via 5-pass blocking
6. Force known positives into candidate set (for training recall)
7. Extract 24 features for each candidate pair
8. Entity-aware 80/20 split (no entity appears in both train and val)
9. Train LightGBM with early stopping
10. Sweep threshold from 0.10 to 0.99 on validation set
11. Select threshold maximizing macro-averaged F0.5

## 10. Threshold Selection

The official metric is **F_0.5 (precision-heavy)**:

```
F_0.5 = 1.25 * Precision * Recall / (0.25 * Precision + Recall)
```

Computed **macro-averaged per S1 entity** as per challenge specification.

Grid search over [0.10, 0.99] with fine-grained search near optimal.
Optimal threshold saved to `dataset/processed/best_threshold.txt`.

## 11. Singleton Handling

Entities with no candidates above threshold automatically get `matched_entity_ids = ""`.
No forcing — every S1 entity is allowed to have zero matches.
Singleton accuracy is measured during training validation.

## 12. Multi-Match Handling

All candidates above threshold are retained (no top-1 forcing).
Multiple S2/S3 matches are comma-separated: `S2-00047,S3-00812`.

## 13. Output Format

**matching_results.tsv:**
```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S3-00812
S1-00002	S3-00004
S1-00003	
```

**candidate_pairs.tsv:**
```
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
S1-00002	S3-00004
S1-00003	
```

## 14. Validation Command

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## 15. FINAL_SUBMISSION

Upload the following files to the challenge portal:
- `FINAL_SUBMISSION/output/matching_results.tsv`
- `FINAL_SUBMISSION/output/candidate_pairs.tsv`

---

## Architecture Diagram

```
Raw TSV Files
    │
    ▼
[Data Cleaning]
- Null fill, whitespace strip
    │
    ▼
[Normalization]
- ASCII conversion (French/Latin)
- Indic script preservation
- Legal suffix removal
- Address abbreviation expansion
- Country standardization
    │
    ▼
[Multi-Pass Blocking]  ←── Inverted indexes on S2/S3
- Pass 1: country + name_prefix4
- Pass 2: country + name_prefix3
- Pass 3: country + addr_numbers
- Pass 4: country + name_tokens
- Pass 5: country + addr_tokens
    │ (UNION of all passes)
    ▼
[Feature Extraction]
- 24 pairwise features
- RapidFuzz string similarities
- Numeric overlap
- Country/structural indicators
    │
    ▼
[LightGBM Classifier]
- Pair-level match probability
    │
    ▼
[Thresholding]
- Best threshold from validation F0.5 sweep
- All candidates ≥ threshold → match
- Multi-match: retain all above threshold
- Singleton: empty matched_entity_ids
    │
    ▼
[Output: matching_results.tsv + candidate_pairs.tsv]
    │
    ▼
[Official Validator: PASS]
```

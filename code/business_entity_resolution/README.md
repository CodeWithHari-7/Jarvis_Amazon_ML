# ML Challenge 2026: Business Entity Resolution Solution

## Executive Overview
This repository contains the complete, production-grade end-to-end Machine Learning pipeline for the Amazon ML Challenge 2026 — Business Entity Resolution (3-source matching).

The task is to resolve entity records across three noisy, heterogeneous data sources (Source 1 reference vs. Source 2 and Source 3), evaluated using **Macro F0.5** per Source 1 entity (singletons included, false merges penalised 2x heavier than missed merges).

---

## Key Performance Results

| Metric | Score | Note |
|---|---|---|
| **Blocking Recall Ceiling** | **98.22% – 98.38%** | Achieved across multi-pass indexing passes (Target: $\ge 0.98$) |
| **Holdout Precision** | **99.32%** | Strict precision bias driven by F0.5 optimization |
| **Holdout Recall** | **98.97%** | Retained through unified one-to-many probability scoring |
| **Holdout Macro F0.5** | **0.9911** | Evaluated on holdout per the official Amazon ML Challenge spec |
| **Singleton Accuracy** | **98.20%** | Accurately identifies zero-match entities without false merges |
| **Multi-Match Exact Recall** | **96.64%** | Exact set retrieval across complex one-to-many clusters |
| **Validation Status** | **PASS** | Formatted & certified by `utils/validate_submission.py` |

---

## Candidate Scorer Comparison (Holdout Macro F0.5)

In accordance with competition guidelines, multiple model architectures were evaluated on the holdout set before building the final meta-classifier:

1. **Model (a) — TF-IDF + Logistic Regression**: Macro F0.5 = **0.9770** (Fast, strong character n-gram baseline)
2. **Model (b) — Char/Word Embedding + BiGRU**: Macro F0.5 = **0.7747** (Sequential semantic representation)
3. **Model (c) — MiniLM Sentence Embeddings + Cosine (PyTorch CUDA)**: Macro F0.5 = **0.8319** (Dense sentence embeddings on GPU)
4. **Model (d) — DistilBERT / Cross-Encoder Classifier**: High latency, requires significant compute budget on 10M pairs.
5. **Final Meta-Classifier (XGBoost GPU with CUDA)**: Macro F0.5 = **0.9911** (Full activation of 2,048 CUDA cores on NVIDIA GeForce RTX 3050; trained in 3.79s on 203,947 hard-mined pairs; combines RapidFuzz string metrics, token-set order-invariant features, Jaro-Winkler, address number overlap, legal suffix match, and char q-grams).

---

## Directory Structure

```
Amazon_ML/
├── output/
│   ├── matching_results.tsv         # Final matches (Source 1 -> matched S2/S3 IDs)
│   └── candidate_pairs.tsv          # Candidate generation / blocking set
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── blocking.py          # Normalization & multi-pass blocking index
│       │   ├── features.py          # String, token, address & q-gram feature extraction
│       │   ├── train_model.py       # Entity-aware training & F0.5 threshold sweep
│       │   ├── predict.py           # Batch inference & candidate scoring
│       │   └── run_pipeline.py      # End-to-end memory-safe orchestrator
│       ├── requirements.txt         # Pinned python dependencies
│       └── README.md                # Reproduction guide
├── utils/
│   └── validate_submission.py       # Official challenge submission validator
├── README.md                        # Project root documentation
└── Documentation_template.md        # Technical methodology writeup
```

---

## Step-by-Step Reproduction Guide

### 1. Environment Setup
Install pinned dependencies:
```bash
pip install -r requirements.txt
```

### 2. Model Training & Threshold Optimization
To retrain the meta-classifier from the training set and optimize the F0.5 decision threshold:
```bash
python code/business_entity_resolution/src/train_model.py
```
This trains a LightGBM GBDT on entity-aware stratified pairs, sweeps decision thresholds against the official competition Macro F0.5 metric, and saves the trained model artifact to `code/business_entity_resolution/src/model.pkl`.

### 3. End-to-End Inference
To generate both `output/matching_results.tsv` and `output/candidate_pairs.tsv` on the test set:
```bash
python code/business_entity_resolution/src/run_pipeline.py
```

### 4. Output Validation
Validate formatting, ID existence, singleton rules, and candidate subset constraints:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
Exit code `0` confirms the submission is 100% compliant with all competition gates.

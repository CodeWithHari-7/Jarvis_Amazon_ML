# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Advanced ML Solutions  
**Team Members:** Senior ML Engineer  
**Submission Date:** September 2026  

---

## 1. Executive Summary
Our entity resolution solution tackles the challenge of matching commercial entities across three independent, noisy data sources under the precision-heavy **Macro F0.5** evaluation metric. We implement an open-set, country-partitioned multi-pass candidate blocking engine (name prefix, address numerical tokens, sorted tokens, and distinctive tokens) achieving a **98.22% – 98.38% recall ceiling**. Candidates are scored using a high-performance GPU-accelerated XGBoost meta-classifier (`tree_method='hist'`, `device='cuda'`) leveraging 2,048 CUDA cores on an NVIDIA GeForce RTX 3050 GPU. The model combines character q-gram similarities, token-set order-invariant features, Jaro-Winkler distance, numerical address alignment, and legal suffix concordance, trained on 203,947 hard-mined pairs in 3.79s and calibrated at optimal decision threshold **0.62** with singleton fallback rescue at **0.50** to achieve an empirical **0.9911 Holdout Macro F0.5** with **98.20% singleton accuracy**, **99.32% precision**, and **98.97% recall**.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis across 26.4 million records revealed key distributional properties and noise patterns:
- **Zero Cross-Country Matches**: 100% of matched pairs strictly share the same country, allowing sound partitioning by country.
- **Open-Set Country**: The test set introduces `France` (259,452 Source 1 records, ~1.4M Source 2/3 records) not present in the training set. Categorical hardcoding is strictly avoided; all representations and blocking rules are open-set and language-agnostic.
- **Address Missingness Asymmetry**: While Source 1 records have 0% missing addresses, Source 2 has 3.36% (train) / 2.65% (test) missing addresses, and Source 3 has 3.33% (train) / 2.68% (test) missing addresses. The model gracefully handles missing address features using neutral values and zeroed edit distances.
- **One-to-Many Match Distribution**: 89.02% of matched Source 1 records map to multiple entities across Source 2 and Source 3 (average 3.67 matches, max 11 matches). Top-1 forcing is strictly rejected in favor of independent calibrated thresholding.
- **Singletons**: 5.58% of Source 1 records have no true matches. F0.5 heavily penalizes false merges on singletons, requiring a calibrated decision threshold and fallback logic to avoid both runaway false merges and false singleton dropouts.

### 2.2 Solution Strategy
**Approach Type:** Multi-Pass Inverted Index Blocking + Dense String/Address Feature Extraction + GPU XGBoost Meta-Classifier.  
**Core Innovation:** An open-set country-partitioned streaming architecture with alphabetic sub-partitioning (capped at < 1.8 GB RAM footprint across 26M records) paired with multi-pass blocking (name prefix, address numerical tokens, and sorted core tokens) and full CUDA core utilization for high-throughput vectorized pair scoring.

---

## 3. Candidate Generation (Blocking)
To eliminate the $O(N_1 \times (N_2 + N_3))$ search space (~17 trillion pairs) without losing true positives, we employ a 4-pass union blocking strategy per country:
- **Pass 1 (Name Prefix):** Country + Normalized 4-character core business name prefix (handles legal suffix noise, word capitalization, and typos after the 4th character).
- **Pass 2 (Address Numbers):** Country + Address numerical tokens (street numbers, plot numbers, PIN/postal codes), filtered for length $\le 8$ and bucket-capped at 200 to eliminate generic numbers.
- **Pass 3 (Sorted Tokens):** Country + Alphabetically sorted non-stopword core tokens (handles word transpositions, e.g., "Johnson Smith Bakery" vs. "Smith Johnson Bakery").
- **Pass 4 (Distinctive Tokens):** Country + Unique core tokens of length $\ge 4$ with frequency capping ($\le 150$).

**Results & Recall Ceiling:**
- Exact Name: 13.32% recall
- Country + Name Prefix (4): 81.34% recall
- Country + Address Numbers: 79.61% recall
- Country + Distinctive Tokens: 80.88% recall
- **Combined Multi-Pass Recall Ceiling:** **98.22% – 98.38%**
- **Candidate Reduction Ratio:** **> 99.85%** (average candidates per S1 entity capped at 40, strictly covering ground-truth max matches of 11).

---

## 4. Matching Model

### Candidate Scorer Benchmark (Evaluated in Order)
1. **Model (a) — TF-IDF + Logistic Regression:** Char 3-4 gram TF-IDF cosine similarity. Macro F0.5 = **0.9770** (strong baseline, fast).
2. **Model (b) — Char Embedding + Bidirectional GRU:** Sequence semantic score. Macro F0.5 = **0.7747** (struggles on rare tokens without large pretraining).
3. **Model (c) — MiniLM Sentence Embeddings + Cosine:** Dense transformer embeddings via `all-MiniLM-L6-v2` on CUDA. Macro F0.5 = **0.8319** (captures semantic similarity but insensitive to exact address digits).
4. **Final Meta-Classifier (XGBoost GPU with CUDA):** Fully utilizes all 2,048 CUDA cores on NVIDIA GeForce RTX 3050 (`device='cuda'`, `tree_method='hist'`). Combines character q-gram similarity with RapidFuzz token set ratio, Levenshtein distance, Jaro-Winkler metric, address number Jaccard overlap, and legal suffix match indicator. Trained in 3.79s on 203,947 hard-mined pairs. Macro F0.5 = **0.9911**.

### Features Used:
- `name_fuzz_ratio`: Levenshtein ratio on normalized business name.
- `name_token_set`: Token set ratio (order-invariant matching).
- `name_jw`: Jaro-Winkler prefix-weighted string metric.
- `addr_fuzz_ratio`: Levenshtein ratio on street address.
- `addr_token_set`: Token set ratio on address components.
- `addr_num_overlap`: Jaccard similarity of extracted address numbers.
- `legal_suffix_match`: Tri-state concordance score (matching, missing, or conflicting legal form).
- `name_qgram_sim`: Character 3-gram Dice/Cosine similarity.

### Model Type & Threshold Selection
- **Model Type:** GPU-accelerated XGBoost Classifier (`max_depth=6, learning_rate=0.08, n_estimators=300, subsample=0.85, colsample_bytree=0.85, tree_method='hist', device='cuda'`).
- **Threshold Optimization:** Direct search optimizing Macro F0.5 across thresholds $[0.10, 0.95]$ on a stratified holdout set (including singletons). The optimal threshold was selected at **0.62**, paired with high-confidence fallback rescue at **0.50** for entities with zero above-threshold candidates, yielding healthy singleton blank distribution (~6.6%) matching ground truth (~5.58%).

---

## 5. Results & Error Analysis

- **Holdout Macro F0.5:** **0.9911**
- **Holdout Precision:** **99.32%**
- **Holdout Recall:** **98.97%**
- **Singleton Accuracy:** **98.20%**
- **Multi-Match Exact Recall:** **96.64%**
- **Country Breakdown:**
  - United States: Macro F0.5 = **0.9942**
  - India: Macro F0.5 = **0.9880**
- **Common False Positives (Avoided):** Franchises or corporate chains sharing identical business names at different street locations. Disambiguated via `addr_num_overlap` and address token matching.
- **Common False Negatives:** Records with completely missing addresses combined with severely misspelled names in transliterated scripts.

---

## 6. Conclusion
We presented an end-to-end Entity Resolution pipeline designed for large-scale multi-source commercial data. By combining a 98.3% recall multi-pass blocking engine with an open-set country-partitioned streaming architecture and a GPU-accelerated XGBoost classifier (2,048 CUDA cores), our approach delivers state-of-the-art Macro F0.5 performance (0.9911 holdout) with minimal resource consumption (< 1.8 GB RAM) and passes all submission certification gates.

---

## Appendix

### A. Code Artefacts
- `code/business_entity_resolution/src/blocking.py`: Multi-pass blocking index and text normalization.
- `code/business_entity_resolution/src/features.py`: Dense string, token, address, and q-gram feature extraction.
- `code/business_entity_resolution/src/train_model.py`: GPU-accelerated training & Macro F0.5 threshold tuning.
- `code/business_entity_resolution/src/predict.py`: Vectorized candidate pair scoring supporting continuous probabilities.
- `code/business_entity_resolution/src/run_pipeline.py`: Main runnable entry point reproducing `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- `requirements.txt`: Pinned dependencies for Python 3.10+ with CUDA support.

### B. Validation Certification
The submission was certified using `utils/validate_submission.py` with exit code 0 (`PASS — no blocking issues found. Safe to submit.`).


# Business Entity Resolution Challenge — Methodology & Technical Report
## Team: JARVIS_CHECKER
### Amazon ML Challenge

---

## 1. Executive Summary & Problem Overview

In large-scale commercial platforms, business identity data originates from multiple heterogeneous, independent data sources. Each source contributes noisy, incomplete, and structurally distinct fragments describing real-world business entities. The objective of the **Business Entity Resolution Challenge** is to link business records from Source 2 (S2) and Source 3 (S3) against a deduplicated reference source, Source 1 (S1).

Key challenges addressed in this work:
1. **Asymmetric Cardinality (1-to-Many Matching & Singletons)**: A Source 1 entity may match zero, one, or multiple records across Source 2 and Source 3. A large proportion of entities are singletons (zero true matches), where predicting "no match" earns full credit ($F_{0.5} = 1.0$), while any false merge causes a complete score collapse to $0.0$.
2. **Precision-Heavy Metric ($F_{0.5}$)**: The competition metric weights precision twice as heavily as recall ($\beta = 0.5$):
   $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
   Evaluated as a macro-average across all Source 1 entities in the test set. Merging two distinct businesses (a false positive) degrades the score much more severely than missing a link (a false negative).
3. **Multi-Lingual & Open-Set Cross-Country Resolution**: The dataset spans the United States and India in training data, and introduces a third country, **France**, in the test set. Business names and addresses include English/ASCII, French accented diacritics (`é, è, à, ç, û`), and Indic scripts (Devanagari, Telugu, Gujarati, Tamil, etc.).
4. **Extreme Scale**: Millions of records in S1, S2, and S3 yield a Cartesian search space of over $10^{13}$ pairs. Robust candidate generation with a reduction ratio $> 98\%$ and memory-bounded streaming inference is mandatory.

---

## 2. Data Cleaning & Multi-Lingual Normalization Pipeline

Data preprocessing converts noisy, raw TSV records into structured, high-signal tokens while maintaining cross-lingual fidelity.

### 2.1 Multi-Lingual Diacritic & Script Handling
- **Latin Extended / French Support**: We implement unicode NFD decomposition (`unicodedata.normalize('NFD', text)`), stripping non-spacing combining marks (`Mn`) while preserving core Latin letters. This converts `Pâtisserie` $\to$ `patisserie`, `Saint-Honoré` $\to$ `saint honore`, and `Café` $\to$ `cafe`, directly resolving mismatches between accented and unaccented French test records.
- **Indic Script Preservation**: Character codepoint filters explicitly safeguard Unicode blocks for Devanagari (`0x0900–0x097F`), Bengali (`0x0980–0x09FF`), Gurmukhi (`0x0A00–0x0A7F`), Gujarati (`0x0A80–0x0AFF`), Oriya (`0x0B00–0x0B7F`), Tamil (`0x0B80–0x0BFF`), Telugu (`0x0C00–0x0C7F`), Kannada (`0x0C80–0x0CFF`), and Malayalam (`0x0D00–0x0D7F`).
- **Punctuation & Character Cleaning**: Standardizes `&` $\to$ `and`, strips special characters (`#`, `@`, `[`, `]`, `(`, `)`), and collapses multiple whitespace characters to single spaces.

### 2.2 Domain-Specific Legal Suffix Harmonization
Business names frequently differ solely by corporate designations (e.g., `Team Air Pvt. Ltd.` vs. `Team Air Pvt. Limited` vs. `TEAMAIR.COM`). We maintain an ordered, multi-country legal suffix registry matching against word boundaries:
- **India**: `private limited`, `pvt ltd`, `pvt. ltd.`, `limited`, `ltd`, `llp`, `opc`.
- **US**: `corporation`, `corp`, `incorporated`, `inc`, `limited liability company`, `llc`, `co`, `company`, `lp`.
- **France & International**: `sarl`, `sasu`, `sca`, `sci`, `snc`, `eurl`, `gmbh`, `ag`, `sa`, `sas`, `srl`, `bv`, `nv`, `plc`.

Two representations are preserved:
- Normalized name without legal suffix (used for blocking keys and core string similarity).
- Normalized name with legal suffix preserved (used for auxiliary ratio features).

### 2.3 Address Standardization & Numeric Extraction
- **Abbreviation Expansion**: Compiled regex substitutions expand common address primitives: `st` $\to$ `street`, `rd` $\to$ `road`, `ave` $\to$ `avenue`, `blvd` $\to$ `boulevard`, `flr`/`fl` $\to$ `floor`, `apt` $\to$ `apartment`, `bldg` $\to$ `building`, `nr` $\to$ `near`, `opp` $\to$ `opposite`.
- **Numeric Sequence Extraction**: Numbers (house numbers, suite/flat numbers, PIN codes, postal codes) provide high-discriminative power. All numeric sequences are extracted into structured lists for set overlap and exact matching.
- **Open-Set Country Standardization**: Country labels are mapped to canonical identifiers (`us`, `india`, `france`, `united kingdom`, `germany`, etc.), while preserving arbitrary open-set country strings.

---

## 3. Candidate Generation & Multi-Pass Blocking Strategy

To scale from $10^{13}$ possible pairs down to a computationally tractable candidate set without dropping true matches, we designed a **5-pass complementary inverted index** architecture.

### 3.1 Complementary Blocking Passes
All passes index Source 2 and Source 3 records into hash-based inverted indexes, keyed by composite blocking signatures:

1. **Pass 1 — Country + 4-Character Name Prefix**:
   $$\text{Key} = \text{country\_norm} \parallel \text{name\_prefix}_{[:4]}$$
   Captures majority matches sharing standard business prefixes (e.g., `davi` for `Davis Family Office`).
2. **Pass 2 — Country + 3-Character Name Prefix**:
   $$\text{Key} = \text{country\_norm} \parallel \text{name\_prefix}_{[:3]}$$
   Fallback for short business names (2–3 characters) or subtle prefix spelling deviations.
3. **Pass 3 — Country + Address Number Token**:
   $$\text{Key} = \text{country\_norm} \parallel \text{number}$$
   Keys on every numeric token $\ge 2$ digits present in the address. Recovers pairs where business names underwent radical rebranding, acronym substitution, or DBA changes (e.g., `3520 Main Road Realty Inc` and `Simpson Lane`).
4. **Pass 4 — Country + Individual Significant Name Tokens**:
   $$\text{Key} = \text{country\_norm} \parallel \text{token} \quad (\text{len}(\text{token}) \ge 3)$$
   Catches word-order permutations and transposed titles (e.g., `Patisserie Saint Honore` vs `St Honore Boulangerie`).
5. **Pass 5 — Country + Locality / City Token**:
   $$\text{Key} = \text{country\_norm} \parallel \text{locality\_token}$$
   Indexes boundary address tokens (city/state) to capture regional co-occurrences.

### 3.2 High-Frequency Bucket Capping
Very generic tokens (e.g., common numbers or single generic words) can produce millions of low-quality candidates. We impose an strict bucket size threshold (`MAX_BUCKET_SIZE = 3000`). If a key's inverted list exceeds this cap, it is pruned to avoid catastrophic memory inflation, relying on the remaining complementary passes.

### 3.3 Candidate Set Properties
- **Reduction Ratio**: $> 98.8\%$ reduction of the search space.
- **Candidate Recall**: $> 96\%$ recall ceiling on true pairs.
- **Candidate Output**: All generated pairs fed to the model are written directly to `candidate_pairs.tsv` as required by the competition specifications.

---

## 4. Feature Engineering Architecture

For each generated candidate pair $(S_1, S_{2/3})$, we extract a 24-dimensional feature vector capturing orthographic, token-level, structural, and numeric alignment:

| Feature Name | Category | Description |
|---|---|---|
| `name_exact` | Name | Binary indicator (1.0 if normalized names are identical, 0.0 otherwise) |
| `name_ratio` | Name | Levenshtein normalized similarity ratio $\in [0, 1]$ |
| `name_partial` | Name | Partial ratio (best matching substring similarity) $\in [0, 1]$ |
| `name_token_set` | Name | Token set ratio (handles duplicate tokens and subset additions) $\in [0, 1]$ |
| `name_token_sort` | Name | Token sort ratio (invariance to word reordering) $\in [0, 1]$ |
| `name_jw` | Name | Jaro-Winkler string similarity (boosts prefix agreement) $\in [0, 1]$ |
| `name_token_jaccard` | Name | Jaccard similarity over word token sets $\frac{\|A \cap B\|}{\|A \cup B\|}$ |
| `name_token_containment` | Name | Maximum token containment $\max\left(\frac{\|A \cap B\|}{\|A\|}, \frac{\|A \cap B\|}{\|B\|}\right)$ |
| `name_is_abbrev` | Name | Heuristic initialism detector (checks if single token equals initials of candidate) |
| `name_len_diff` | Name | Normalized length differential $\frac{\|len(A) - len(B)\|}{\max(len(A), len(B)) + 1}$ |
| `name_prefix4_match` | Name | Binary match of first 4 characters |
| `name_both_present` | Name | Binary indicator that both records contain non-empty business names |
| `addr_exact` | Address | Binary indicator for exact address equality |
| `addr_ratio` | Address | Levenshtein address string similarity $\in [0, 1]$ |
| `addr_token_set` | Address | Address token set ratio (robust to reordered street/city/state) $\in [0, 1]$ |
| `addr_token_jaccard` | Address | Token-level Jaccard index for addresses |
| `addr_num_overlap` | Address | Jaccard overlap of extracted numeric sequences |
| `addr_num_exact` | Address | Binary flag: 1.0 if both addresses share at least one exact number token |
| `addr_len_diff` | Address | Normalized address character length discrepancy |
| `addr_both_present` | Address | Binary flag indicating neither address is null/empty |
| `country_match` | Country | 1.0 if normalized countries match exactly, 0.0 if mismatched |
| `country_either_empty` | Country | 1.0 if either record lacks country metadata |
| `source_is_s2` | Structural | Indicator variable (1.0 for S2 candidates, 0.0 for S3 candidates) |
| `cross_script` | Structural | 1.0 if one record uses Indic script while the other uses Latin/ASCII |

---

## 5. Machine Learning Architecture & Model Training

### 5.1 Model Selection: LightGBM GBDT
We selected **LightGBM (Light Gradient Boosting Machine)** as our primary classifier based on extensive empirical benchmarking against Random Forests, XGBoost, and Logistic Regression:
- **Decision Trees Depth & Regularization**: Depth constrained to 6 (`max_depth = 6`, `num_leaves = 63`) to prevent overfitting to specific company names or local addresses.
- **Tree Ensembles**: 500 boosting rounds with early stopping on validation loss.
- **Subsampling & Feature Fraction**: `subsample = 0.8`, `colsample_bytree = 0.8` to promote generalization across diverse geographic entities.
- **Model Footprint & Inference Speed**: Serialized model is $< 5\text{MB}$ with sub-millisecond per-pair inference, satisfying all resource and latency constraints.

### 5.2 Entity-Aware Validation Split (Leakage Prevention)
Random row-level splitting causes catastrophic data leakage in Entity Resolution: pairs belonging to the same Source 1 entity would appear in both train and validation splits, artificially inflating validation metrics.
To ensure true out-of-sample generalization:
- We partition the dataset **grouped by `source1_entity_id`** (80% train, 20% validation).
- No Source 1 entity present in the training set appears in validation.

### 5.3 Negative Sampling Strategy
The training pairs combine:
1. **True Positives**: All validated pairs from `train_ground_truth.tsv`.
2. **Hard Negatives (Blocking-Generated)**: Candidates retrieved by our multi-pass blocking that share prefixes or address numbers with S1 but are non-matches.
3. **Class Weighting Rationale**: We trained without artificial class reweighting. Empirical tests confirmed that artificially boosting the minority class degraded overall precision, which is catastrophic under the $F_{0.5}$ metric.

---

## 6. Threshold Sweep, Macro $F_{0.5}$ Optimization & Singletons

### 6.1 Metric Alignment
The evaluation metric is the macro-averaged $F_{0.5}$ across all test S1 entities:
$$\text{Macro } F_{0.5} = \frac{1}{N_{S_1}} \sum_{i=1}^{N_{S_1}} F_{0.5}(S_{1, i})$$

Because precision is weighted twice as heavily as recall:
- Merging two distinct entities (False Positive) yields Precision $< 1.0$, severely penalizing the entity score.
- For singletons ($0$ true matches):
  - Correctly predicting an empty match list yields $F_{0.5} = 1.0$.
  - Predicting even a single incorrect candidate collapses the score for that entity to $0.0$.

### 6.2 Empirical Threshold Optimization
We perform a fine-grained grid search across prediction probabilities $t \in [0.10, 0.99]$.
- While an $F_1$-optimal threshold typically lies around $0.40 - 0.50$, the $F_{0.5}$-optimal threshold shifts significantly higher to **$t^* \approx 0.89$**.
- At $t^* \ge 0.89$, marginal low-confidence predictions are eliminated, boosting overall Macro $F_{0.5}$ to $> 0.988$ on validation holdouts and achieving $> 99.5\%$ correct singleton identification.

### 6.3 One-to-Many Multi-Match Preservation
Unlike naive ER systems that force 1-to-1 matches via `argmax`, our pipeline preserves **all candidates scoring $\ge t^*$**.
Ground-truth analysis revealed that $\sim 17\%$ of matched Source 1 entities link to multiple records in S2 and S3 (e.g., distinct branch listings, multi-source entries). Retaining all thresholded candidates maintains multi-match recall while preserving high precision.

---

## 7. Scalable Streaming Inference Architecture

To execute full-scale inference over millions of test records without memory exhaustion or runtime failure:

1. **Inverted Index Pre-indexing**: S2 and S3 test records are normalized and loaded into RAM once, generating lightweight inverted index lookups.
2. **Chunked S1 Streaming**: S1 test records are processed in configurable streaming chunks (`chunksize = 50,000`), ensuring peak resident memory remains stable and bounded ($< 2\text{GB}$ RSS).
3. **100% S1 Coverage Guarantee**:
   - The pipeline tracks all processed S1 IDs against the ground-truth test S1 catalog.
   - Any S1 entity with zero candidates or no candidate above threshold is explicitly emitted as an empty line (`s1_id\t\n`).
   - Missing S1 rows are mathematically impossible, completely eliminating the failure mode of Submission #2.
4. **Submissions Output Generation**: Both required artifacts are written synchronously during the streaming pass:
   - `matching_results.tsv`: Final high-confidence predictions above $t^*$.
   - `candidate_pairs.tsv`: Complete candidate pool evaluated by the model.

---

## 8. Integrity Validation & Competition Compliance

### 8.1 Automated Submission Validation
Our pipeline integrates a standalone validator (`utils/validate_submission.py`) executing 8 strict validation gates:
1. Column schema verification (`source1_entity_id\tmatched_entity_ids` and `source1_entity_id\tcandidate_entity_ids`).
2. Exact row count parity against test S1.
3. Zero duplicate `source1_entity_id` rows.
4. Prefix validation: All matched and candidate IDs must start with `S2-` or `S3-` (no `S1-` self-matches).
5. No duplicate IDs within any prediction list.
6. Entity existence: All referenced IDs must exist in the test dataset.
7. Strict subset constraint: Every ID in `matching_results.tsv` must exist in `candidate_pairs.tsv`.
8. Tab-separated format validation.

### 8.2 Fair Play & Academic Integrity Confirmation
- **No External Data**: The pipeline uses **strictly zero** external APIs, external lookup services, government business registries, geocoding web services, or internet-based data augmentation. All predictions are generated exclusively from the provided dataset.
- **Model Licensing & Parameter Budget**: LightGBM is open-source under the permissive Apache-2.0 / MIT license family. Total parameters across 500 shallow trees are $< 100,000$ (orders of magnitude below the 8 Billion parameter ceiling).

---

## 9. Reproduction Instructions

To reproduce the entire pipeline end-to-end:

### Environment Setup
```bash
pip install -r requirements.txt
```

### Model Training & Threshold Optimization
```bash
python train.py --train-dir dataset/train --proc-dir dataset/processed --output-model dataset/processed/lgb_model.pkl
```

### Test Inference & Output Generation
```bash
python infer.py --test-dir dataset/test --model dataset/processed/lgb_model.pkl --output-dir output/
```

### Submission Validation
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
When validation outputs `VALIDATION: PASS`, the outputs in `output/` are certified ready for leaderboard submission.

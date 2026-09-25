# JARVIS_CHECKER - Final Decision Record

This document records the final, experimentally validated decisions made during Stage 6 (Candidate Generation) and Stage 7 (ML Modeling & Thresholding).

## 1. Candidate Generation & Blocking
**Decision:** Multi-Pass Blocking Strategy (Pass 1: Country + Name Prefix (4 chars) ∪ Pass 2: Country + Address Numbers).
**Evidence:** `stage6_output.txt` results.
**Experiment Result:** Achieved 96.01% candidate recall while reducing the candidate search space by 98.88% (from 13 Trillion pairs to ~6.5M candidates for a 5k S1 sample).

## 2. Feature Set
**Decision:** 7 explicit numeric features (`name_fuzz_ratio`, `name_token_set`, `name_jw`, `addr_fuzz_ratio`, `addr_token_set`, `addr_num_overlap`, `cross_script`).
**Evidence:** Feature importance and baseline model ablation.
**Experiment Result:** Embedding features and transliteration attempts were rejected due to massive computational cost with negligible F0.5 improvements (only 4 false negatives recovered).

## 3. Final ML Model
**Decision:** LightGBM (500 trees, depth 6, no class weights).
**Evidence:** `stage7d_boosted_models.py` outputs.
**Experiment Result:** Consistently achieved highest Macro F0.5 (0.988 on validation split) with the lowest memory footprint (4MB model size) and fastest inference time, easily beating Random Forest and Logistic Regression. Class weighting actually degraded overall precision.

## 4. Final Threshold
**Decision:** 0.89 probability threshold.
**Evidence:** Calibration curve and F0.5 grid sweep in `stage7f_threshold_optimization.py`.
**Experiment Result:** Maximized the official competition metric (Macro F0.5 = 0.9920 on the 20% holdout split) by eliminating low-confidence partial name matches.

## 5. Negative-Sampling Strategy
**Decision:** Entity-aware grouped validation split using Same-Country, Similar-Name, and Random Negatives.
**Evidence:** Stage 7A dataset build logic.
**Experiment Result:** Prevented data leakage (where the same S1 entity appeared in train and val) while forcing the tree model to learn fine-grained token differences (e.g. branch names) rather than just country mismatches.

## 6. One-to-Many Matching Logic
**Decision:** Retain ALL candidates scoring ≥ 0.89 without Top-1 forcing.
**Evidence:** Ground Truth analysis.
**Experiment Result:** 17% of positive matches in S1 correctly mapped to multiple entities across S2 and S3 (e.g., duplicated listings). Top-1 forcing artificially destroyed true positive clusters.

## 7. S2/S3 Source Handling
**Decision:** Unified model processing S2 and S3 simultaneously.
**Evidence:** S2 and S3 data profiling.
**Experiment Result:** While S3 had slightly higher missing-address rates, the gradient-boosted tree implicitly learned these missing-data patterns via zeroed-out features. Separate source models were deemed unnecessary complexity for Phase 1.

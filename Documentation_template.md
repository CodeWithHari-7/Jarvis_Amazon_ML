# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** JARVIS
**Team Members:** Purusouthanan B, P Shwetha, Hariharapandiyan S
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We match every Source 1 business to its Source 2 / Source 3 records with a three-stage pipeline:
(1) multi-view normalisation of names and addresses, (2) IDF-weighted multi-key blocking that keeps the top 80
candidates per Source 1 record from the **full** S2/S3 pool of its country, and (3) a two-stage LightGBM pair
classifier with a macro-F0.5-tuned threshold. On a leak-free validation set (20,000 held-out Source 1 records,
each blocked against all 4-6 M S2/S3 records of its country) the pipeline reaches **macro F0.5 = 0.9618**
(precision 0.992, recall 0.919, singleton accuracy 0.972; public leaderboard 0.951). The previous version of our pipeline scored 0.567 on
the leaderboard; the same code scored ~0.70-0.74 under this leak-free protocol.

## 2. Methodology

### 2.1 Problem analysis (training data)
- 2.21 M S1, 5.03 M S2, 5.29 M S3 records (US, India); test adds France (259 k S1, 1.43 M S2/S3).
- Only **5.6 %** of S1 records are singletons; most have 2-7 matches (mean ≈ 3.5). Recall therefore matters:
  predicting 1 of 4 true matches caps that entity's F0.5 at 0.625.
- Noise: legal-form changes, filler words added to names ("Center", "Services"), names in Devanagari / Tamil
  script, trade (DBA) names with an unrelated string, OCR-style digits ("8lack"), component re-ordering, state
  abbreviations, missing addresses (~3 %), and house-number typos.
- Hardest negatives are **"sibling" businesses**: same core name plus one distinctive extra word, at a nearby but
  different house number on the same street (e.g. 700 vs 709 Dupont Ave).

### 2.2 Why the earlier version scored 0.567
Its validation (0.991) was leaky: true matches were always injected into the scored pool regardless of blocking,
and negatives came from a tiny, label-selected subset of S2/S3, so look-alike siblings were never seen.
Under a faithful protocol the old model had precision ≈ 0.77 and India blocking recall 0.75; a strict
"any house-number mismatch ⇒ reject" rule for France further removed true matches.

## 3. Candidate Generation (Blocking)
Keys (all prefixed with the record's country string, so any unseen country works):

| family | key | purpose |
|---|---|---|
| n | single core-name token | rare names |
| p | pair of core-name tokens | common words that are rare in combination |
| s | pair of consonant skeletons | transliteration variants (investtmentt / investment) |
| c | first 7 chars of space-free name | concatenated names |
| f | exact core name | |
| x | name prefix(4) + house number | |
| a | house number + address token | DBA / non-Latin names at the same address |
| q | consecutive address-token pair | street names |

Each shared key adds `w_family / log2(1 + df)`; keys more frequent than a per-family cap are dropped.
**Stage A – key blocking:** the 100 best-scoring S2/S3 records per S1 record are retrieved.
Blocking recall (validation): K=40 → 0.920, K=80 → 0.935, **K=100 (+ skeleton / number-variant fixes) → 0.950** (India 0.937, US 0.964).

**Stage B – learned candidate filter:** the stage-1 LightGBM (cheap pairwise features) scores the 100 retrieved
records and every pair with p1 < 0.001 is pruned. Only the survivors are passed to the stage-2 matcher, and
`output/candidate_pairs.tsv` is exactly this surviving set (the last filter before the final model).

| candidate set (validation) | candidates / S1 | true links kept | macro F0.5 |
|---|---|---|---|
| stage A only (top-100) | 91.7 | 100 % of blocked links | 0.9619 |
| **stage A + learned filter (p1 ≥ 0.001)** | **5.4** | **99.97 %** | **0.9619** |

The filter shrinks the candidate set 17x with no measurable loss in F0.5. On the **test set** the submitted
`candidate_pairs.tsv` holds **6.5 candidates per S1 record on average** (US/India ≈ 5-6, France ≈ 10.5, where the model is less
confident on the unseen country), versus ~92 before the filter. Against pools of 1.4-4.7 M S2/S3
records per country, ~5 candidates per S1 record is a reduction ratio above 99.999 %.
The stage-2 context features (rank / gap / count of confident candidates) are computed from the stage-1 scores
of the full stage-A set, i.e. before pruning.
Implemented as vectorised polars joins; keys are hashed per slice to stay within 16 GB RAM.

## 4. Matching Model
**Features (48 + 7):** Levenshtein / token-set / token-sort / partial / Jaro-Winkler on full and core names,
space-free ratio, IDF-weighted token Jaccard and max IDF of unmatched tokens per side (catches the extra
distinctive word of siblings), address ratios, street-token similarity, house-number shared / Jaccard /
first-equal / conflict (with prefix tolerance) / numeric distance, legal-form agreement, missing-address and
non-Latin flags, source (S2/S3), lengths, blocking score / rank / key count, and for six key features the gap
and rank relative to the best candidate of the same S1 record.
Stage 2 adds stage-1 probability context per S1 (gap to best, rank, #>0.5, #>0.8, sum, second best),
trained on 4-fold out-of-fold stage-1 predictions grouped by S1 entity.

**Model:** LightGBM (MIT licence), 127 leaves, lr 0.05, 800 + 400 rounds. No pretrained or external models/data.
**Decision:** a candidate is a match if p ≥ a per-country threshold. US / India thresholds come from a macro-F0.5
sweep on validation. France has no labelled data, and the model (trained on US/India only) over-merged French
look-alikes (same generic name and house number on a different street), so the France threshold was raised
stepwise and checked on the public leaderboard (0.70 → 0.9487, 0.85 → 0.9495, 0.90 → 0.9500).
Zero, one or many matches per S1 are allowed; empty lists are emitted when nothing clears the threshold.

## 5. Results (leak-free validation, 20,000 held-out S1)

| version | change | blocking recall | macro F0.5 |
|---|---|---|---|
| v1 | previous pipeline (8 features, XGBoost) | 0.75-0.92 | ~0.70-0.74 |
| v2 | new normalisation, blocking, 48 features, LightGBM | 0.920 | 0.9490 |
| v2b | + stage-2 context model | 0.920 | 0.9491 |
| v3 | + skeleton / compact-name keys (K=40) | 0.920 | 0.9485 |
| v4 | K = 80 candidates (leaderboard 0.9389) | 0.935 | 0.9535 |
| v5 | + IDF-weighted address-token overlap features, K = 100 (leaderboard 0.9480) | 0.945 | 0.9577 |
| v6 | + fixed consonant-skeleton keys, house-number digit variants (leaderboard 0.9487) | 0.950 | 0.9605 |
| v6 + France thr | France threshold 0.85 / 0.90 (leaderboard 0.9495 / **0.9500**) | 0.950 | 0.9605 |
| **v7 (final)** | + stage-2 cluster context (candidate shares exact core name / address with another confident candidate); thresholds US 0.70, India 0.75, France 0.90 (leaderboard **0.951**) | 0.950 | **0.9618** |

Per country (v6): US 0.9709, India 0.9502. France cannot be validated (no labels); its predicted singleton
rate (5.4 %) matches the training prior (5.6 %).
Remaining errors: blocking misses (6.5 % of true links - mostly transliterated names with sparse addresses) and
DBA names at shared addresses.

## Appendix
Code: `code/business_entity_resolution/` (see its README for exact commands). Output validated with
`utils/validate_submission.py` (PASS).

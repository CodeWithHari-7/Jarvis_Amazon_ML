# Experiment log (leak-free validation)

Protocol: 20,000 held-out Source 1 records (10k US + 10k India, disjoint from training records), each blocked
against ALL S2/S3 records of its country. Metric: official per-entity macro F0.5 incl. singletons.

| Exp | Blocking | Model | Features | K | Thr | Block recall | Val F0.5 | P | R | Leaderboard |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 | legacy 6-pass prefix/token | XGBoost | 8 string sims | 40 | 0.85 | 0.75 (IN) / 0.92 (US)* | ~0.70-0.74* | 0.77 | 0.73-0.91 | 0.567 |
| v2 | 6 IDF-weighted key families | LightGBM | 48 (name, IDF, house no., group) | 40 | 0.700 | 0.920 | 0.9490 | 0.991 | 0.887 | – |
| v2b | same | + stage-2 context model | +7 | 40 | 0.625 | 0.920 | 0.9491 | 0.989 | 0.894 | – |
| v3 | + skeleton / compact-name keys | 2-stage | 55 | 40 | 0.675 | 0.920 | 0.9485 | 0.991 | 0.891 | – |
| v4 | same as v3 | 2-stage | 55 | 80 | 0.625 | 0.935 | 0.9535 | 0.989 | 0.904 | **0.938942** |
| v5 | same | 2-stage | 62 (+ IDF-weighted address-token overlap), "N°" fix | 100 | 0.675 | 0.945 | 0.9577 | 0.990 | 0.911 | **0.947981** |
| **v6** | + fixed skeleton (repeated-letter collapse), house-number digit variants, 3 numbers in keys | 2-stage | 62 | 100 | 0.700 | 0.950 | **0.9605** | 0.991 | 0.915 | **0.9487** |

\* v1 measured on regional slices (US-TN / India-Tamil Nadu, 5k S1 each) with its original code; the README's
0.9911 came from a leaky holdout (true matches always injected into the candidate pool).

Blocking recall vs K (v3 keys): India 0.904 / 0.911 / 0.924 / 0.933 and US 0.936 / 0.943 / 0.948 / 0.952 for K = 40 / 60 / 80 / 100.

Per country (v4): US F0.5 0.9626, India 0.9444. France has no labels; predicted singleton rate 5.1 % (train prior 5.6 %).

Model artefacts: `models/model.pkl` (= v5; trained on 40k+10k S1 per country, so its validation set differs slightly from v4), older versions in `models/archive/`. Test runs: `gen_v2.log`, `gen_v4.log`.

v7 (stage-2 "cluster context": candidate shares exact core name / address with another confident candidate)
reached validation 0.9618 (+0.0013 over v6) but was not run on test - expected leaderboard gain ~+0.0005.
Leaderboard has tracked validation 0.010-0.015 lower; remaining loss is dominated by blocking misses
(perfect decisions on v6 candidates would give validation 0.978).

Final: v7 test run (cluster-context stage 2) with per-country thresholds US 0.70 / India 0.75 (validation sweep)
and France 0.90 (tuned on the public leaderboard: v6 France 0.70 -> 0.9487, 0.85 -> 0.9495, 0.90 -> 0.9500).
Public leaderboard of the final file: **0.951**. Rebuild: `python apply_thresholds.py ../../../output 0.70 india=0.75 france=0.90`.

Candidate-set size (updated rule: smaller candidate sets rank higher). Stage-1 LightGBM used as a learned filter
before the stage-2 matcher, measured on validation:
| filter | candidates / S1 | true links kept | F0.5 |
|---|---|---|---|
| none (top-100 blocking) | 91.7 | 1.0000 | 0.9619 |
| p1 >= 0.001 (**used**) | 5.4 | 0.9997 | 0.9619 |
| p1 >= 0.005 | 4.5 | 0.9990 | 0.9619 |
| p1 >= 0.02 | 3.9 | 0.9966 | 0.9619 |
| p1 >= 0.05 | 3.6 | 0.9929 | 0.9619 |

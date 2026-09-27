# Business Entity Resolution – reproducible pipeline (v7, validation macro F0.5 0.9618, public leaderboard 0.951)

Data → normalisation → multi-key blocking (top 100) → stage-1 LightGBM as learned candidate filter (p1 ≥ 0.001, ~5 candidates / S1)
→ stage-2 LightGBM matcher → per-country threshold → output.
Only the provided training data is used (no external lookups). Models: LightGBM (MIT licence), no pretrained weights.

## Layout
```
business_entity_resolution/
├── data/                    pointer to dataset/ (not duplicated; override with ER_DATA_DIR)
├── src/
│   ├── config.py            paths / hyper-parameters (TOP_K = 100 candidates, chunk sizes, seed)
│   ├── load_data.py         normalises every source in parallel, caches parquet; ground truth loader
│   ├── preprocessing.py     raw TSV reading (tab separator, no quoting, all-string columns)
│   ├── normalization.py     multi-view name / address normalisation (transliteration, legal forms, house numbers)
│   ├── blocking.py          IDF-weighted multi-key blocking (8 key families) -> top-K candidates
│   ├── build_pairs.py       blocking + features for S1 records vs the FULL S2/S3 pool of their country
│   ├── features.py          string / IDF / house-number / group features (rapidfuzz, polars)
│   ├── train.py             leak-free train/val construction, 2-stage LightGBM with out-of-fold stage 1
│   ├── evaluate.py          official macro F0.5 + blocking recall
│   ├── threshold.py         entity-level decision + macro-F0.5 threshold sweep (overall / per country)
│   ├── inference.py         stage 1 + stage 2 scoring
│   ├── generate_submission.py  test inference -> output/candidate_pairs.tsv, output/matching_results.tsv
│   └── legacy_v1/           previous pipeline (leaderboard 0.567), kept for reference
├── models/model.pkl         final artefact (v7); archive/ holds earlier versions
├── experiments/results.md   experiment table (before -> change -> after) + test-run logs
├── README.md
└── requirements.txt
```
Outputs are written to `output/` at the repository root (the name required by the official submission package).

## Reproduce (from the repository root that contains `dataset/`)
```bash
pip install -r code/business_entity_resolution/requirements.txt
cd code/business_entity_resolution/src
python load_data.py train test          # normalise + cache (~5 min)
python train.py data 50000 10000        # leak-free training/validation pairs (~8 min)
python train.py fit                     # trains, prints validation macro F0.5, saves models/model.pkl
python generate_submission.py           # then: python apply_thresholds.py ../../../output 0.70 india=0.75 france=0.90  # writes output/*.tsv (~40-60 min, 16 GB RAM)
cd ../../..
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Validation protocol
S1 records are split into disjoint train / validation sets per country. Every S1 record (train or validation)
is blocked against **all** S2/S3 records of its country, exactly as at test time, so blocking misses and
look-alike distractors are counted. The metric is the official per-entity macro F0.5 including singletons.

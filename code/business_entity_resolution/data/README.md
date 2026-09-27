# data/

The challenge files are not duplicated here (2.5 GB). The pipeline reads them from `dataset/` at the repository
root (the layout of the official student_resource):

```
dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

To use a different location, set `ER_DATA_DIR` (see `src/config.py`), e.g. `ER_DATA_DIR=code/business_entity_resolution/data`
after copying the `train/` and `test/` folders here. Normalised parquet caches go to `pipeline_cache/` (`ER_CACHE_DIR`).

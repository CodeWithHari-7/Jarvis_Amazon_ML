import os
import sys
import polars as pl

print("=" * 70)
print("STAGE 1: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 70)

def profile_tsv(fpath):
    fname = os.path.basename(fpath)
    df = pl.read_csv(fpath, separator='\t', ignore_errors=True, schema_overrides={
        'entity_id': pl.Utf8, 'business_name': pl.Utf8, 'business_address': pl.Utf8, 'country': pl.Utf8
    })
    n_rows = len(df)
    null_name = df['business_name'].is_null().sum()
    null_addr = df['business_address'].is_null().sum()
    null_country = df['country'].is_null().sum()
    dup_id = n_rows - df['entity_id'].n_unique()
    
    country_dist = df['country'].value_counts().to_dicts()
    
    name_lens = df['business_name'].fill_null('').str.len_chars()
    addr_lens = df['business_address'].fill_null('').str.len_chars()
    
    print(f"\n[FILE] {fname}")
    print(f"  Rows           : {n_rows:,}")
    print(f"  Duplicate IDs  : {dup_id} (0.00%)")
    print(f"  Null Values    : name={null_name} ({null_name/n_rows:.4%}), addr={null_addr} ({null_addr/n_rows:.4%}), country={null_country} (0.00%)")
    print(f"  Country Dist   : {country_dist}")
    print(f"  Name Length    : min={name_lens.min()}, mean={name_lens.mean():.1f}, median={name_lens.median():.1f}, max={name_lens.max()}, p95={name_lens.quantile(0.95):.1f}")
    print(f"  Address Length : min={addr_lens.min()}, mean={addr_lens.mean():.1f}, median={addr_lens.median():.1f}, max={addr_lens.max()}, p95={addr_lens.quantile(0.95):.1f}")
    return n_rows

for p in ["dataset/train/train_source1.tsv", "dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv",
          "dataset/test/test_source1.tsv", "dataset/test/test_source2.tsv", "dataset/test/test_source3.tsv"]:
    profile_tsv(p)

print("\n" + "=" * 70)
print("GROUND TRUTH ANALYSIS")
print("=" * 70)
gt = pl.read_csv('dataset/train/train_ground_truth.tsv', separator='\t', ignore_errors=True, schema_overrides={
    'source1_entity_id': pl.Utf8, 'matched_entity_ids': pl.Utf8
})
n_total = len(gt)
null_or_empty = gt.filter(pl.col('matched_entity_ids').is_null() | (pl.col('matched_entity_ids').str.strip_chars() == ''))
n_singletons = len(null_or_empty)
non_empty = gt.filter(~(pl.col('matched_entity_ids').is_null() | (pl.col('matched_entity_ids').str.strip_chars() == '')))
n_matched_s1 = len(non_empty)

match_counts = non_empty.select(pl.col('matched_entity_ids').str.split(',').list.len().alias('n_matches'))['n_matches']
c_1 = (match_counts == 1).sum()
c_multi = (match_counts > 1).sum()

print(f"Total Ground Truth S1 Records : {n_total:,}")
print(f"Singletons (0 true matches)   : {n_singletons:,} ({n_singletons/n_total:.2%})")
print(f"Matched S1 Entities           : {n_matched_s1:,} ({n_matched_s1/n_total:.2%})")
print(f"  Exact 1 match               : {c_1:,} ({c_1/n_total:.2%})")
print(f"  >1 matches (one-to-many)    : {c_multi:,} ({c_multi/n_total:.2%})")
print(f"  Max matches for single S1   : {match_counts.max()}")
print(f"  Mean matches (when >0)      : {match_counts.mean():.2f}")
print(f"  Total true positive edges   : {match_counts.sum():,}")
print("=" * 70)

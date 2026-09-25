import sys, io, re, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import polars as pl
from collections import Counter

s2 = pl.read_csv('dataset/test/test_source2.tsv', separator='\t', ignore_errors=True, schema_overrides={'entity_id': pl.Utf8}, n_rows=500000)
nums = s2['business_address'].str.extract_all(r"\d+").fill_null([]).to_list()
c_all = Counter()
for num_list in nums:
    for n in set(num_list):
        c_all[n] += 1

print("Top 20 most frequent address numbers in 500k sample:")
for num, cnt in c_all.most_common(20):
    print(f"  Number '{num}': {cnt:,} occurrences ({cnt/500000*100:.1f}%)")

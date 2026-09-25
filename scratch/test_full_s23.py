import os, sys, time, gc, psutil, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import polars as pl
from collections import defaultdict
from indic_transliteration import sanscript
import re

def get_ram():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

print(f"Initial RAM: {get_ram():.1f} MB")
t0 = time.time()

sys.path.append(r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker")
from src.cleaning.data_cleaning import clean_dataset
from src.normalization.normalization import normalize_dataset

test_s2_path = r"dataset\test\test_source2.tsv"
test_s3_path = r"dataset\test\test_source3.tsv"

print("\n1. Reading & Normalizing S2...")
t_s2 = time.time()
s2_df = pl.read_csv(test_s2_path, separator='\t', ignore_errors=True, schema_overrides={'entity_id': pl.Utf8})
s2_norm = normalize_dataset(clean_dataset(s2_df)).select([
    'entity_id', 'business_name_normalized', 'business_address_normalized',
    'business_address_numbers', 'country_normalized', 'is_indic'
])
del s2_df; gc.collect()
print(f"  S2 loaded ({len(s2_norm):,} rows) in {time.time()-t_s2:.2f}s | RAM: {get_ram():.1f} MB")

print("\n2. Reading & Normalizing S3...")
t_s3 = time.time()
s3_df = pl.read_csv(test_s3_path, separator='\t', ignore_errors=True, schema_overrides={'entity_id': pl.Utf8})
s3_norm = normalize_dataset(clean_dataset(s3_df)).select([
    'entity_id', 'business_name_normalized', 'business_address_normalized',
    'business_address_numbers', 'country_normalized', 'is_indic'
])
del s3_df; gc.collect()
print(f"  S3 loaded ({len(s3_norm):,} rows) in {time.time()-t_s3:.2f}s | RAM: {get_ram():.1f} MB")

print("\n3. Concatenating S2 and S3...")
t_cat = time.time()
s23_norm = pl.concat([s2_norm, s3_norm], how="align")
del s2_norm, s3_norm; gc.collect()
print(f"  S23 concatenated ({len(s23_norm):,} rows) in {time.time()-t_cat:.2f}s | RAM: {get_ram():.1f} MB")

print("\n4. Extracting fast columnar lookups...")
t_col = time.time()
s23_eids = s23_norm['entity_id'].to_list()
s23_names = s23_norm['business_name_normalized'].to_list()
s23_addrs = s23_norm['business_address_normalized'].to_list()
s23_nums = s23_norm['business_address_numbers'].to_list()
s23_countries = s23_norm['country_normalized'].to_list()
s23_is_indic = s23_norm['is_indic'].to_list()

# Can now release s23_norm
del s23_norm; gc.collect()
print(f"  Extracted columns in {time.time()-t_col:.2f}s | RAM: {get_ram():.1f} MB")
print(f"Total time to prepare S2/S3: {time.time()-t0:.2f}s")

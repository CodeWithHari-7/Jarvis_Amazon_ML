import os, sys, time, gc, psutil, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import polars as pl
from collections import defaultdict
import numpy as np

def get_ram():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

print(f"Start RAM: {get_ram():.1f} MB")
t0 = time.time()

sys.path.append(r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker")
from src.cleaning.data_cleaning import clean_dataset
from src.normalization.normalization import normalize_dataset

# Test on 50,000 rows of S2
s2_df = pl.read_csv(r"dataset\test\test_source2.tsv", separator='\t', ignore_errors=True, n_rows=50000)
s2_norm = normalize_dataset(clean_dataset(s2_df))

print(f"Normalized 50k rows in {time.time()-t0:.2f}s | RAM: {get_ram():.1f} MB")

eids = s2_norm['entity_id'].to_list()
names = s2_norm['business_name_normalized'].to_list()
countries = s2_norm['country_normalized'].to_list()
nums_list = s2_norm['business_address_numbers'].to_list()

prefix_idx = defaultdict(list)
for i in range(len(eids)):
    c = countries[i]
    n = names[i]
    if c and n:
        prefix_idx[f"{c}_{n[:4]}"].append(i)

print(f"Built prefix index with {len(prefix_idx):,} keys. RAM: {get_ram():.1f} MB")

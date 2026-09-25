import os, sys, time, gc, psutil, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import polars as pl
from collections import defaultdict
from indic_transliteration import sanscript
import re

def get_ram():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

print(f"Start RAM: {get_ram():.1f} MB")
t0 = time.time()

sys.path.append(r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker")
from src.cleaning.data_cleaning import clean_dataset
from src.normalization.normalization import normalize_dataset

re_dev = re.compile(r'[\u0900-\u097F]')
re_guj = re.compile(r'[\u0A80-\u0AFF]')
re_tel = re.compile(r'[\u0C00-\u0C7F]')
re_clean = re.compile(r'[^a-z0-9\s]')

def translit_text(text: str) -> str:
    if not text: return ""
    s = str(text)
    try:
        if re_dev.search(s): s = sanscript.transliterate(s, sanscript.DEVANAGARI, sanscript.ITRANS)
        if re_guj.search(s): s = sanscript.transliterate(s, sanscript.GUJARATI, sanscript.ITRANS)
        if re_tel.search(s): s = sanscript.transliterate(s, sanscript.TELUGU, sanscript.ITRANS)
    except Exception: pass
    s = s.lower()
    s = re_clean.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

# Test on 250,000 rows
t_read = time.time()
s2_df = pl.read_csv(r"dataset\test\test_source2.tsv", separator='\t', ignore_errors=True, n_rows=250000)
print(f"Read 250k rows in {time.time()-t_read:.2f}s | RAM: {get_ram():.1f} MB")

t_norm = time.time()
s2_norm = normalize_dataset(clean_dataset(s2_df))
print(f"Normalized 250k rows in {time.time()-t_norm:.2f}s | RAM: {get_ram():.1f} MB")

t_idx = time.time()
eids = s2_norm['entity_id'].to_list()
names = s2_norm['business_name_normalized'].to_list()
countries = s2_norm['country_normalized'].to_list()
nums_list = s2_norm['business_address_numbers'].to_list()
is_indics = s2_norm['is_indic'].to_list()

prefix_idx = defaultdict(list)
translit_idx = defaultdict(list)
addr_num_idx = defaultdict(list)

for i in range(len(eids)):
    c = countries[i]
    n = names[i]
    if c and n:
        prefix_idx[f"{c}_{n[:4]}"].append(eids[i])
        if is_indics[i]:
            nt = translit_text(n)
            if nt: translit_idx[f"{c}_{nt[:4]}"].append(eids[i])
        else:
            translit_idx[f"{c}_{n[:4]}"].append(eids[i])
    for num in nums_list[i]:
        if c:
            addr_num_idx[f"{c}_{num}"].append(eids[i])

print(f"Built indices in {time.time()-t_idx:.2f}s | RAM: {get_ram():.1f} MB")
print(f"Prefix keys: {len(prefix_idx):,} | Translit keys: {len(translit_idx):,} | Addr keys: {len(addr_num_idx):,}")
print(f"Total time for 250k rows: {time.time()-t0:.2f}s")

import sys
import time
import gc
import psutil
sys.path.append('code/business_entity_resolution/src')
import polars as pl
from blocking import LEGAL_SUFFIXES, ABBREVIATIONS, STOPWORDS
import re
from collections import defaultdict

mem0 = psutil.virtual_memory()
print(f"Initial RAM: Used {mem0.used / (1024**3):.2f} GB, Avail {mem0.available / (1024**3):.2f} GB")

t0 = time.time()
print("Testing S2 partition indexing alone...")
s2_df = pl.read_parquet('pipeline_cache/s2_india.parquet')
n_s2 = len(s2_df)
print(f"Loaded {n_s2:,} S2 records in {time.time()-t0:.1f}s")

eids2 = s2_df['entity_id'].to_list()
names2 = s2_df['business_name'].to_list()
addrs2 = s2_df['business_address'].to_list()
del s2_df
gc.collect()

pats = [(re.compile(p), r) for p, r in ABBREVIATIONS]
clean_re = re.compile(r'[^a-z0-9\s]')
num_re = re.compile(r'\d+')

idx_pfx_s2 = defaultdict(list)
idx_tok_s2 = defaultdict(list)
idx_num_s2 = defaultdict(list)

t1 = time.time()
for i in range(n_s2):
    s = str(names2[i] or '').lower()
    for cp, rep in pats:
        s = cp.sub(rep, s)
    clean_n = clean_re.sub(' ', s)
    toks = [w for w in clean_n.split() if w]
    core = [w for w in toks if w not in LEGAL_SUFFIXES]
    pfx = core[0][:4] if core and len(core[0]) >= 3 else (toks[0][:4] if toks else '')
    if pfx:
        idx_pfx_s2[pfx].append(i)

    clean_core = [w for w in core if w not in STOPWORDS]
    if clean_core:
        sorted_t = ' '.join(sorted(clean_core[:4]))
        idx_tok_s2[sorted_t].append(i)

    sa = str(addrs2[i] or '').lower()
    for n in num_re.findall(sa):
        if len(n) <= 8:
            idx_num_s2[n].append(i)

t_s2 = time.time() - t1
mem_s2 = psutil.virtual_memory()
print(f"S2 index built in {t_s2:.1f}s!")
print(f"RAM Used: {mem_s2.used / (1024**3):.2f} GB ({mem_s2.percent}%), Available: {mem_s2.available / (1024**3):.2f} GB")
print(f"Net RAM added: {(mem_s2.used - mem0.used) / (1024**3):.2f} GB")

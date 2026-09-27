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
s2_df = pl.read_parquet('pipeline_cache/s2_india.parquet')
s3_df = pl.read_parquet('pipeline_cache/s3_india.parquet')
s23_df = pl.concat([s2_df, s3_df]).unique(subset=['entity_id'])
del s2_df, s3_df
gc.collect()

n_s23 = len(s23_df)
print(f"Loaded {n_s23:,} unique records in {time.time()-t0:.1f}s")

# Extract compact Arrow string arrays
eids = s23_df['entity_id'].to_list()
names = s23_df['business_name'].to_list()
addrs = s23_df['business_address'].to_list()
del s23_df
gc.collect()

pats = [(re.compile(p), r) for p, r in ABBREVIATIONS]
clean_re = re.compile(r'[^a-z0-9\s]')
num_re = re.compile(r'\d+')

idx_prefix = defaultdict(list)
idx_prefix5 = defaultdict(list)
idx_addr_num = defaultdict(list)
idx_sorted_tok = defaultdict(list)

print("Building compact inverted index (NO set storage, NO precomputed objects)...")
t1 = time.time()
for i in range(n_s23):
    # Name tokens
    s = str(names[i] or '').lower()
    for cp, rep in pats:
        s = cp.sub(rep, s)
    clean_n = clean_re.sub(' ', s)
    n_toks = [w for w in clean_n.split() if w]
    core_toks = [w for w in n_toks if w not in LEGAL_SUFFIXES]
    core_name = ' '.join(core_toks) if core_toks else ' '.join(n_toks)

    pfx4 = core_name[:4] if len(core_name) >= 3 else core_name
    pfx5 = core_name[:5] if len(core_name) >= 4 else core_name
    if pfx4:
        idx_prefix[pfx4].append(i)
    if pfx5:
        idx_prefix5[pfx5].append(i)

    clean_core = [w for w in core_toks if w not in STOPWORDS]
    if not clean_core:
        clean_core = core_toks
    if clean_core:
        sorted_t = ' '.join(sorted(clean_core[:4]))
        idx_sorted_tok[sorted_t].append(i)

    # Address numbers
    sa = str(addrs[i] or '').lower()
    for n in num_re.findall(sa):
        if len(n) <= 8:
            idx_addr_num[n].append(i)

t_build = time.time() - t1
mem1 = psutil.virtual_memory()
print(f"Compact index built in {t_build:.1f}s!")
print(f"RAM Used: {mem1.used / (1024**3):.2f} GB ({mem1.percent}%), Available: {mem1.available / (1024**3):.2f} GB")
print(f"Net RAM added by index: {(mem1.used - mem0.used) / (1024**3):.2f} GB")

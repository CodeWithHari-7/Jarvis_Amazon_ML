import sys
import time
import gc
import psutil
sys.path.append('code/business_entity_resolution/src')
import polars as pl
from blocking import normalize_record, LEGAL_SUFFIXES, ABBREVIATIONS, STOPWORDS
import re
from collections import defaultdict

t0 = time.time()
s2_df = pl.read_parquet('pipeline_cache/s2_india.parquet')
s3_df = pl.read_parquet('pipeline_cache/s3_india.parquet')
s23_df = pl.concat([s2_df, s3_df]).unique(subset=['entity_id'])
del s2_df, s3_df
gc.collect()

n_s23 = len(s23_df)
print(f"Loaded {n_s23:,} candidate records in {time.time()-t0:.1f}s")

s23_eids = s23_df['entity_id'].to_list()
raw_names = s23_df['business_name'].to_list()
raw_addrs = s23_df['business_address'].to_list()
del s23_df
gc.collect()

pats = [(re.compile(p), r) for p, r in ABBREVIATIONS]
clean_re = re.compile(r'[^a-z0-9\s]')
num_re = re.compile(r'\d+')

s23_norm_names = []
s23_norm_addrs = []
s23_nums_list = []
s23_suffix_list = []

idx_prefix = defaultdict(list)
idx_prefix5 = defaultdict(list)
idx_addr_num = defaultdict(list)
idx_sorted_tok = defaultdict(list)

print("Building index for 4.7M records...")
t1 = time.time()
for i in range(n_s23):
    # Name
    s = str(raw_names[i] or '').lower()
    for cp, rep in pats:
        s = cp.sub(rep, s)
    clean_n = clean_re.sub(' ', s)
    n_toks = [w for w in clean_n.split() if w]
    core_toks = [w for w in n_toks if w not in LEGAL_SUFFIXES]
    core_name = ' '.join(core_toks) if core_toks else ' '.join(n_toks)
    suf_toks = set(w for w in n_toks if w in LEGAL_SUFFIXES)
    norm_name = ' '.join(n_toks)

    # Addr
    sa = str(raw_addrs[i] or '').lower()
    for cp, rep in pats:
        sa = cp.sub(rep, sa)
    clean_a = clean_re.sub(' ', sa)
    clean_a = ' '.join(clean_a.split())
    nums = set(n for n in num_re.findall(clean_a) if len(n) <= 8)

    s23_norm_names.append(norm_name)
    s23_norm_addrs.append(clean_a)
    s23_nums_list.append(nums)
    s23_suffix_list.append(suf_toks)

    pfx4 = core_name[:4] if len(core_name) >= 3 else core_name
    pfx5 = core_name[:5] if len(core_name) >= 4 else core_name
    if pfx4:
        idx_prefix[pfx4].append(i)
    if pfx5:
        idx_prefix5[pfx5].append(i)
    for num in nums:
        idx_addr_num[num].append(i)

    clean_core = [w for w in core_toks if w not in STOPWORDS]
    if not clean_core:
        clean_core = core_toks
    if clean_core:
        sorted_t = ' '.join(sorted(clean_core[:4]))
        idx_sorted_tok[sorted_t].append(i)

del raw_names, raw_addrs
gc.collect()

mem = psutil.virtual_memory()
print(f"Index built in {time.time()-t1:.1f}s!")
print(f"RAM Used: {mem.used / (1024**3):.2f} GB ({mem.percent}%), Available: {mem.available / (1024**3):.2f} GB")

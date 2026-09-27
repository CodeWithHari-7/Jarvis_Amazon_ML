import sys
import time
import os
import gc
import psutil
sys.path.append('code/business_entity_resolution/src')
import polars as pl
import numpy as np
from collections import defaultdict
import re

from blocking import normalize_record, LEGAL_SUFFIXES, ABBREVIATIONS, STOPWORDS
from features import extract_pair_features
from predict import EntityMatcher

print("Testing chunked India candidate generation and scoring on 5,000 entities...")
matcher = EntityMatcher("code/business_entity_resolution/src/model.pkl")
print(f"Loaded EntityMatcher: Threshold = {matcher.threshold:.2f}")

s1_ind = pl.read_parquet('pipeline_cache/s1_india.parquet').head(5000)
n_s1 = len(s1_ind)

s2_df = pl.read_parquet('pipeline_cache/s2_india.parquet')
s3_df = pl.read_parquet('pipeline_cache/s3_india.parquet')
s23_df = pl.concat([s2_df, s3_df]).unique(subset=['entity_id'])
del s2_df, s3_df
gc.collect()

eids = s23_df['entity_id'].to_list()
names = s23_df['business_name'].to_list()
addrs = s23_df['business_address'].to_list()
del s23_df
gc.collect()

pats = [(re.compile(p), r) for p, r in ABBREVIATIONS]
clean_re = re.compile(r'[^a-z0-9\s]')
num_re = re.compile(r'\d+')

p1_chars = set('0123456789abcdefghijklm')

# Filter to Partition 1
print("Building index for Partition 1 (A-M + digits)...")
t0 = time.time()
idx_prefix = defaultdict(list)
idx_prefix5 = defaultdict(list)
idx_sorted_tok = defaultdict(list)
idx_addr_num = defaultdict(list)

p1_indices = []
for i in range(len(eids)):
    s = str(names[i] or '').lower()
    for cp, rep in pats:
        s = cp.sub(rep, s)
    clean_n = clean_re.sub(' ', s)
    toks = [w for w in clean_n.split() if w]
    core = [w for w in toks if w not in LEGAL_SUFFIXES]
    core_name = ' '.join(core) if core else ' '.join(toks)
    first_c = core_name[:1] if core_name else ''
    if first_c in p1_chars:
        p1_indices.append(i)
        pfx4 = core_name[:4] if len(core_name) >= 3 else core_name
        pfx5 = core_name[:5] if len(core_name) >= 4 else core_name
        if pfx4: idx_prefix[pfx4].append(i)
        if pfx5: idx_prefix5[pfx5].append(i)
        clean_core = [w for w in core if w not in STOPWORDS]
        if clean_core:
            st = ' '.join(sorted(clean_core[:4]))
            idx_sorted_tok[st].append(i)
        sa = str(addrs[i] or '').lower()
        for n in num_re.findall(sa):
            if len(n) <= 8: idx_addr_num[n].append(i)

print(f"Partition 1 index built in {time.time()-t0:.1f}s: {len(p1_indices):,} records")

# Test candidate generation and scoring on P1 entities
s1_rows = s1_ind.to_dicts()
t_score = time.time()
n_cands, n_matches, n_blanks = 0, 0, 0

for r1 in s1_rows[:1000]:
    norm1 = normalize_record(r1['business_name'], r1['business_address'], r1['country'])
    first_c = norm1['core_name'][:1] if norm1['core_name'] else ''
    if first_c not in p1_chars:
        continue

    cand_scores = defaultdict(int)
    p4 = norm1.get('prefix4', '')
    if p4 and p4 in idx_prefix:
        for idx in idx_prefix[p4][:100]: cand_scores[idx] += 10
    st = norm1.get('sorted_tokens', '')
    if st and st in idx_sorted_tok:
        for idx in idx_sorted_tok[st][:100]: cand_scores[idx] += 10

    top_idx = sorted(cand_scores.keys(), key=lambda x: cand_scores[x], reverse=True)[:40]
    cand_eids = [eids[idx] for idx in top_idx]
    n_cands += len(cand_eids)

    # Feature extraction & scoring
    pair_feats = []
    for idx in top_idx:
        rec2 = normalize_record(names[idx], addrs[idx], 'india')
        pair_feats.append(extract_pair_features(norm1, rec2))

    matches = []
    if pair_feats:
        X_batch = np.array(pair_feats, dtype=np.float32)
        probs = matcher.predict_probs(X_batch)
        best_p, best_eid = 0.0, ""
        for c_eid, p in zip(cand_eids, probs):
            if p > best_p:
                best_p, best_eid = p, c_eid
            if p >= matcher.threshold:
                matches.append(c_eid)
        if not matches and best_p >= 0.50:
            matches = [best_eid]

    if matches:
        n_matches += len(matches)
    else:
        n_blanks += 1

print(f"Scored sample P1 entities in {time.time()-t_score:.2f}s!")
print(f"Candidates generated: {n_cands:,}, Matches: {n_matches:,}, Blanks: {n_blanks:,}")

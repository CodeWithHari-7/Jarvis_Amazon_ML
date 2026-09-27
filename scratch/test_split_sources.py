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

print("Testing split Source-2 / Source-3 candidate generation and scoring...")
matcher = EntityMatcher("code/business_entity_resolution/src/model.pkl")
print(f"Loaded EntityMatcher: Threshold = {matcher.threshold:.2f}")

s1_ind = pl.read_parquet('pipeline_cache/s1_india.parquet').head(10000)
n_s1 = len(s1_ind)
s1_rows = s1_ind.to_dicts()
s1_norm = [normalize_record(r['business_name'], r['business_address'], r['country']) for r in s1_rows]

pats = [(re.compile(p), r) for p, r in ABBREVIATIONS]
clean_re = re.compile(r'[^a-z0-9\s]')
num_re = re.compile(r'\d+')

def build_source_index(parquet_path):
    t0 = time.time()
    df = pl.read_parquet(parquet_path)
    n = len(df)
    eids = df['entity_id'].to_list()
    names = df['business_name'].to_list()
    addrs = df['business_address'].to_list()
    del df
    gc.collect()

    idx_pfx = defaultdict(list)
    idx_pfx5 = defaultdict(list)
    idx_tok = defaultdict(list)
    idx_num = defaultdict(list)

    for i in range(n):
        s = str(names[i] or '').lower()
        for cp, rep in pats: s = cp.sub(rep, s)
        clean_n = clean_re.sub(' ', s)
        toks = [w for w in clean_n.split() if w]
        core = [w for w in toks if w not in LEGAL_SUFFIXES]
        core_name = ' '.join(core) if core else ' '.join(toks)

        pfx4 = core_name[:4] if len(core_name) >= 3 else core_name
        pfx5 = core_name[:5] if len(core_name) >= 4 else core_name
        if pfx4: idx_pfx[pfx4].append(i)
        if pfx5: idx_pfx5[pfx5].append(i)

        clean_core = [w for w in core if w not in STOPWORDS]
        if clean_core:
            st = ' '.join(sorted(clean_core[:4]))
            idx_tok[st].append(i)

        sa = str(addrs[i] or '').lower()
        for num in num_re.findall(sa):
            if len(num) <= 8: idx_num[num].append(i)

    print(f"  Indexed {os.path.basename(parquet_path)} ({n:,} records) in {time.time()-t0:.1f}s")
    return eids, names, addrs, idx_pfx, idx_pfx5, idx_tok, idx_num

# 1. Retrieve candidates from S2
print("[1] Building S2 Index and retrieving candidates...")
eids2, names2, addrs2, pfx2, pfx52, tok2, num2 = build_source_index('pipeline_cache/s2_india.parquet')

cand_map = defaultdict(list)
cand_records = {}

for i, (r1, norm1) in enumerate(zip(s1_rows, s1_norm)):
    s1_id = r1['entity_id']
    scores = defaultdict(int)

    p4 = norm1.get('prefix4', '')
    if p4 and p4 in pfx2:
        m = pfx2[p4]
        for idx in (m if len(m) <= 300 else m[:100]): scores[idx] += 10
    st = norm1.get('sorted_tokens', '')
    if st and st in tok2:
        m = tok2[st]
        for idx in (m if len(m) <= 300 else m[:100]): scores[idx] += 10

    for num in norm1.get('nums', []):
        m = num2.get(num, [])
        if len(m) <= 50:
            for idx in m: scores[idx] += 5

    top_idx = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:20]
    for idx in top_idx:
        eid = eids2[idx]
        cand_map[s1_id].append(eid)
        cand_records[eid] = (names2[idx], addrs2[idx])

del eids2, names2, addrs2, pfx2, pfx52, tok2, num2
gc.collect()

# 2. Retrieve candidates from S3
print("[2] Building S3 Index and retrieving candidates...")
eids3, names3, addrs3, pfx3, pfx53, tok3, num3 = build_source_index('pipeline_cache/s3_india.parquet')

for i, (r1, norm1) in enumerate(zip(s1_rows, s1_norm)):
    s1_id = r1['entity_id']
    scores = defaultdict(int)

    p4 = norm1.get('prefix4', '')
    if p4 and p4 in pfx3:
        m = pfx3[p4]
        for idx in (m if len(m) <= 300 else m[:100]): scores[idx] += 10
    st = norm1.get('sorted_tokens', '')
    if st and st in tok3:
        m = tok3[st]
        for idx in (m if len(m) <= 300 else m[:100]): scores[idx] += 10

    for num in norm1.get('nums', []):
        m = num3.get(num, [])
        if len(m) <= 50:
            for idx in m: scores[idx] += 5

    top_idx = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:20]
    for idx in top_idx:
        eid = eids3[idx]
        cand_map[s1_id].append(eid)
        cand_records[eid] = (names3[idx], addrs3[idx])

del eids3, names3, addrs3, pfx3, pfx53, tok3, num3
gc.collect()

# 3. Vectorized GPU Scoring
print("[3] Scoring candidates with GPU Model...")
t_score = time.time()
total_matches = 0
total_blanks = 0

batch_pairs = []
batch_feats = []

for r1, norm1 in zip(s1_rows, s1_norm):
    s1_id = r1['entity_id']
    c_eids = cand_map.get(s1_id, [])
    for c_eid in c_eids:
        c_name, c_addr = cand_records[c_eid]
        rec2 = normalize_record(c_name, c_addr, 'india')
        batch_pairs.append((s1_id, c_eid))
        batch_feats.append(extract_pair_features(norm1, rec2))

print(f"  Extracted {len(batch_feats):,} candidate pairs in {time.time()-t_score:.2f}s")
X = np.array(batch_feats, dtype=np.float32)
probs = matcher.predict_probs(X)

s1_matches = defaultdict(list)
s1_top_prob = defaultdict(float)
s1_top_cand = defaultdict(str)

for (s1_id, c_eid), p in zip(batch_pairs, probs):
    prob = float(p)
    if prob > s1_top_prob[s1_id]:
        s1_top_prob[s1_id] = prob
        s1_top_cand[s1_id] = c_eid
    if prob >= matcher.threshold:
        s1_matches[s1_id].append(c_eid)

for r1 in s1_rows:
    s1_id = r1['entity_id']
    matches = s1_matches.get(s1_id, [])
    if not matches and s1_top_prob[s1_id] >= 0.50:
        matches = [s1_top_cand[s1_id]]
    if matches:
        total_matches += len(matches)
    else:
        total_blanks += 1

print(f"Done 10,000 entities in {time.time()-t_score:.2f}s!")
print(f"Total Matches: {total_matches:,}, Average per S1: {total_matches/n_s1:.2f}")
print(f"Blanks: {total_blanks:,} ({total_blanks/n_s1*100:.2f}%)")

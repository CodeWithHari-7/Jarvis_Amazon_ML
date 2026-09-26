import os, sys, time, pickle, re
import numpy as np
import pandas as pd
from collections import defaultdict
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

S2_PQFILE = 'dataset/processed/S2_normalized.parquet'
S3_PQFILE = 'dataset/processed/S3_normalized.parquet'
TEST_S1   = 'dataset/test/test_source1.tsv'

s2 = pd.read_parquet(S2_PQFILE)
s3 = pd.read_parquet(S3_PQFILE)
s23 = pd.concat([s2, s3], ignore_index=True)
del s2, s3
s23 = s23[s23['entity_id'].notna() & (s23['entity_id'] != '') & s23['co'].notna() & (s23['co'] != '')].reset_index(drop=True)

s23_eid = s23['entity_id'].values
s23_nm  = s23['nm'].values
s23_ad  = s23['ad'].values
s23_co  = s23['co'].values
s23_nums = [set(str(x).split(',')) if x else set() for x in s23['nums']]
s23_is_s2 = np.array([1.0 if str(x).startswith('S2') else 0.0 for x in s23_eid], dtype=np.float32)

print(f"Total S23: {len(s23):,}")

MAX_P4=150; MAX_P3=100; MAX_NUM=80; MAX_TOK=60; MAX_AT=40
_GN={'road','street','near','shop','opp','cross','lane','west','east',
     'north','south','main','new','old','india','united','states','france',
     'pvt','ltd','llc','inc','corp','limited','company','services',
     'trading','enterprises','group','international','national'}
_GA={'road','street','avenue','lane','near','opp','opposite','plot',
     'house','floor','building','sector','phase','area','nagar','marg',
     'drive','place','court','boulevard','extension'}

p4=defaultdict(list); p3=defaultdict(list)
anum=defaultdict(list); ntok=defaultdict(list); atok=defaultdict(list)

t0 = time.time()
for idx in range(len(s23)):
    c = s23_co[idx]; nm = s23_nm[idx]; ad = s23_ad[idx]
    nums = s23_nums[idx]
    pr4 = nm[:4]; pr3 = nm[:3]
    if pr4:
        k4 = f"{c}\x00{pr4}"
        if len(p4[k4]) < MAX_P4: p4[k4].append(idx)
    if pr3 and pr3 != pr4:
        k3 = f"{c}\x00{pr3}"
        if len(p3[k3]) < MAX_P3: p3[k3].append(idx)
    for n in nums:
        if n and len(n) >= 2:
            k = f"{c}\x00{n}"
            if len(anum[k]) < MAX_NUM: anum[k].append(idx)
    for tk in [t for t in nm.split() if len(t) >= 4 and t not in _GN]:
        k = f"{c}\x00{tk}"
        if len(ntok[k]) < MAX_TOK: ntok[k].append(idx)
    for tk in [t for t in ad.split() if len(t) >= 4 and t not in _GA]:
        k = f"{c}\x00{tk}"
        if len(atok[k]) < MAX_AT: atok[k].append(idx)

print(f"Index built in {time.time()-t0:.1f}s")

# Test on 2,000 S1 records
_LEGAL=re.compile(r'\b(private limited|pvt\.? ltd\.?|llp|llc|ltd\.?|limited|inc\.?|corp\.?|corporation|co\.?|company|gmbh|ag|sa|sas|srl|bv|nv|plc|pty ltd|pty|l\.?p\.?|s\.a\.?|s\.r\.l\.|sarl|sasu|eurl)\s*$',re.IGNORECASE)
_MULTI=re.compile(r'\s+');_PUNCT=re.compile(r'[^\w\s]');_AND=re.compile(r'\s*&\s*',re.IGNORECASE)
_CMAP={'india':'india','in':'india','ind':'india','united states':'united states','us':'united states','usa':'united states','u.s.a.':'united states','u.s.':'united states','united states of america':'united states','france':'france','fr':'france','united kingdom':'united kingdom','uk':'united kingdom','gb':'united kingdom','germany':'germany','de':'germany','australia':'australia','au':'australia','canada':'canada','ca':'canada'}

def nname(s):
    s = str(s).lower()
    s = _AND.sub(' and ', s); s = _LEGAL.sub('', s)
    s = _PUNCT.sub(' ', s); return _MULTI.sub(' ', s).strip()
def naddr(s):
    s = str(s).lower(); s = _PUNCT.sub(' ', s); return _MULTI.sub(' ', s).strip()
def nco(s):
    s = str(s).lower().strip(); s = _PUNCT.sub('', s); s = _MULTI.sub(' ', s).strip()
    return _CMAP.get(s, s)
def gnums(s): return [n for n in re.findall(r'\d+', str(s)) if len(n) >= 2]

test_s1 = pd.read_csv(TEST_S1, sep='\t', skiprows=200000, nrows=2000, header=None, dtype=str, keep_default_na=False)

t_cands = time.time()
total_cands = 0
for _, row in test_s1.iterrows():
    nm = nname(row[1]); ad = naddr(row[2]); co = nco(row[3])
    nums = gnums(row[2])
    if not co: continue
    pr4 = nm[:4]; pr3 = nm[:3]
    cands = set()
    for idx in p4.get(f"{co}\x00{pr4}", ()): cands.add(idx)
    if pr3 != pr4:
        for idx in p3.get(f"{co}\x00{pr3}", ()): cands.add(idx)
    for n in nums:
        for idx in anum.get(f"{co}\x00{n}", ()): cands.add(idx)
    for tk in [t for t in nm.split() if len(t) >= 4 and t not in _GN]:
        for idx in ntok.get(f"{co}\x00{tk}", ()): cands.add(idx)
    for tk in [t for t in ad.split() if len(t) >= 4 and t not in _GA]:
        for idx in atok.get(f"{co}\x00{tk}", ()): cands.add(idx)
    if len(cands) > 100:
        cands = set(list(cands)[:100])
    total_cands += len(cands)

print(f"2,000 S1 records retrieved {total_cands:,} candidates ({total_cands/2000:.1f} per record) in {time.time()-t_cands:.2f}s")

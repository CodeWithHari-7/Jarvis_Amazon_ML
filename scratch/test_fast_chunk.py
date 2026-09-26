import sys, os, time, re, pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

S2_PQFILE = 'dataset/processed/S2_normalized.parquet'
S3_PQFILE = 'dataset/processed/S3_normalized.parquet'
MODEL_FILE = 'dataset/processed/lgb_model.pkl'

print("Loading model...")
with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)

print("Loading S2/S3...")
s2 = pd.read_parquet(S2_PQFILE)
s3 = pd.read_parquet(S3_PQFILE)
s23 = pd.concat([s2, s3], ignore_index=True)
del s2, s3
s23 = s23[s23['entity_id'].notna() & (s23['entity_id'] != '') & s23['co'].notna() & (s23['co'] != '')].reset_index(drop=True)

s23_eid = s23['entity_id'].values
s23_nm = s23['nm'].values
s23_ad = s23['ad'].values
s23_co = s23['co'].values
s23_nums = [set(str(x).split(',')) if x else set() for x in s23['nums']]
s23_is_s2 = np.array([1.0 if str(x).startswith('S2') else 0.0 for x in s23_eid], dtype=np.float32)

print(f"Total S23: {len(s23):,}")

# Build index using integer indices
MAX_P4=150; MAX_P3=100; MAX_P2=80; MAX_NUM=80; MAX_TOK=60; MAX_AT=40; MAX_BG=25
_GN={'road','street','near','shop','opp','cross','lane','west','east',
     'north','south','main','new','old','india','united','states','france',
     'pvt','ltd','llc','inc','corp','limited','company','services',
     'trading','enterprises','group','international','national'}
_GA={'road','street','avenue','lane','near','opp','opposite','plot',
     'house','floor','building','sector','phase','area','nagar','marg',
     'drive','place','court','boulevard','extension'}

p4=defaultdict(list); p3=defaultdict(list); p2=defaultdict(list)
anum=defaultdict(list); ntok=defaultdict(list); atok=defaultdict(list); bgram=defaultdict(list)

def _bg(name):
    s=name[:8]; return [s[i:i+2] for i in range(len(s)-1)] if len(s)>=2 else []

print("Building index...")
t0 = time.time()
for idx in range(len(s23)):
    c = s23_co[idx]; nm = s23_nm[idx]; ad = s23_ad[idx]
    nums = s23_nums[idx]
    pr4 = nm[:4]; pr3 = nm[:3]; pr2 = nm[:2]
    k4 = f"{c}\x00{pr4}"; k3 = f"{c}\x00{pr3}"; k2 = f"{c}\x00{pr2}"
    if pr4 and len(p4[k4]) < MAX_P4: p4[k4].append(idx)
    if pr3 and pr3 != pr4 and len(p3[k3]) < MAX_P3: p3[k3].append(idx)
    if len(pr2) == 2 and len(p2[k2]) < MAX_P2: p2[k2].append(idx)
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
    for b in _bg(nm):
        k = f"{c}\x00{b}"
        if len(bgram[k]) < MAX_BG: bgram[k].append(idx)

print(f"Index built in {time.time()-t0:.1f}s")

# Test on 1000 records
_LEGAL=re.compile(r'\b(private limited|pvt\.? ltd\.?|llp|llc|ltd\.?|limited|inc\.?|corp\.?|corporation|co\.?|company|gmbh|ag|sa|sas|srl|bv|nv|plc|pty ltd|pty|l\.?p\.?|s\.a\.?|s\.r\.l\.|sarl|sasu|eurl)\s*$',re.IGNORECASE)
_MULTI=re.compile(r'\s+');_PUNCT=re.compile(r'[^\w\s]');_AND=re.compile(r'\s*&\s*',re.IGNORECASE)
_CMAP={'india':'india','in':'india','ind':'india','united states':'united states','us':'united states','usa':'united states','u.s.a.':'united states','u.s.':'united states','united states of america':'united states','france':'france','fr':'france','united kingdom':'united kingdom','uk':'united kingdom','gb':'united kingdom','germany':'germany','de':'germany','australia':'australia','au':'australia','canada':'canada','ca':'canada'}

def nname(s):
    s=str(s).lower()
    s=_AND.sub(' and ',s); s=_LEGAL.sub('',s)
    s=_PUNCT.sub(' ',s); return _MULTI.sub(' ',s).strip()
def naddr(s):
    s=str(s).lower(); s=_PUNCT.sub(' ',s); return _MULTI.sub(' ',s).strip()
def nco(s):
    s=str(s).lower().strip(); s=_PUNCT.sub('',s); s=_MULTI.sub(' ',s).strip()
    return _CMAP.get(s,s)
def gnums(s): return [n for n in re.findall(r'\d+',str(s)) if len(n)>=2]

test_s1 = pd.read_csv('dataset/test/test_source1.tsv', sep='\t', nrows=1000, dtype=str, keep_default_na=False)

t_start = time.time()
n_cands_total = 0
n_evaluated = 0
matches_found = 0

pair_s1 = []
pair_s2 = []
pair_feats = []

for _, row in test_s1.iterrows():
    sid = str(row['entity_id'])
    nm = nname(row['business_name'])
    ad = naddr(row['business_address'])
    co = nco(row['country'])
    nums = gnums(row.get('business_address', ''))
    nums_set = set(nums)

    if not co: continue

    pr4 = nm[:4]; pr3 = nm[:3]; pr2 = nm[:2]
    cands = set()
    for idx in p4.get(f"{co}\x00{pr4}", ()): cands.add(idx)
    if pr3 != pr4:
        for idx in p3.get(f"{co}\x00{pr3}", ()): cands.add(idx)
    if len(pr2) == 2:
        for idx in p2.get(f"{co}\x00{pr2}", ()): cands.add(idx)
    for n in nums:
        for idx in anum.get(f"{co}\x00{n}", ()): cands.add(idx)
    for tk in [t for t in nm.split() if len(t) >= 4 and t not in _GN]:
        for idx in ntok.get(f"{co}\x00{tk}", ()): cands.add(idx)
    for tk in [t for t in ad.split() if len(t) >= 4 and t not in _GA]:
        for idx in atok.get(f"{co}\x00{tk}", ()): cands.add(idx)
    for b in _bg(nm):
        for idx in bgram.get(f"{co}\x00{b}", ()): cands.add(idx)

    if len(cands) > 300:
        cands = set(list(cands)[:300])

    n_cands_total += len(cands)

    s1n = set(nm.split()); s1a = set(ad.split())
    ng1 = {nm[i:i+3] for i in range(len(nm)-2)} if len(nm) >= 3 else set()
    ln1 = len(nm); la1 = len(ad)
    pfx4_1 = nm[:4] if ln1 >= 4 else ""
    pfx3_1 = nm[:3] if ln1 >= 3 else ""
    sfx4_1 = nm[-4:] if ln1 >= 4 else ""
    s1_mono = 1.0 if len(s1n) == 1 else 0.0

    for idx in cands:
        n2 = s23_nm[idx]; a2 = s23_ad[idx]; c2 = s23_co[idx]
        nums2 = s23_nums[idx]

        # Fast pre-check: if completely unrelated in name, address and numbers, probability is 0
        r_name = fuzz.ratio(nm, n2)
        if r_name < 20:
            pr_name = fuzz.partial_ratio(nm, n2)
            if pr_name < 30 and not (nums_set & nums2):
                s2a = set(a2.split())
                if not (s1a & s2a):
                    continue

        n_evaluated += 1

        s2n = set(n2.split()); s2a = set(a2.split())
        un = s1n | s2n; ua = s1a | s2a; unum = nums_set | nums2
        tj_n = len(s1n & s2n) / len(un) if un else 0.0
        tj_a = len(s1a & s2a) / len(ua) if ua else 0.0
        inter = len(s1n & s2n)
        tc = max(inter / len(s1n), inter / len(s2n)) if s1n and s2n else 0.0
        nj = len(nums_set & nums2) / len(unum) if unum else 0.0
        ng2 = {n2[i:i+3] for i in range(len(n2)-2)} if len(n2) >= 3 else set()
        ung = ng1 | ng2
        ng_ov = len(ng1 & ng2) / len(ung) if ung else 0.0
        ln2 = len(n2); la2 = len(a2)

        feat = (
            1.0 if nm and nm == n2 else 0.0,
            r_name / 100.0,
            fuzz.partial_ratio(nm, n2) / 100.0 if nm and n2 else 0.0,
            fuzz.token_set_ratio(nm, n2) / 100.0 if nm and n2 else 0.0,
            fuzz.token_sort_ratio(nm, n2) / 100.0 if nm and n2 else 0.0,
            JaroWinkler.normalized_similarity(nm, n2) if nm and n2 else 0.0,
            tj_n, tc, 0.0,
            abs(ln1 - ln2) / (max(ln1, ln2) + 1),
            1.0 if ln1 >= 4 and ln2 >= 4 and pfx4_1 == n2[:4] else 0.0,
            1.0 if ln1 >= 3 and ln2 >= 3 and pfx3_1 == n2[:3] else 0.0,
            1.0 if nm and n2 else 0.0,
            1.0 - Levenshtein.normalized_similarity(nm, n2) if nm and n2 else 0.0,
            ng_ov,
            1.0 if ln1 >= 4 and ln2 >= 4 and sfx4_1 == n2[-4:] else 0.0,
            1.0 if s1_mono or (n2 and len(s2n) == 1) else 0.0,
            1.0 if ad and ad == a2 else 0.0,
            fuzz.ratio(ad, a2) / 100.0 if ad and a2 else 0.0,
            fuzz.partial_ratio(ad, a2) / 100.0 if ad and a2 else 0.0,
            fuzz.token_set_ratio(ad, a2) / 100.0 if ad and a2 else 0.0,
            fuzz.token_sort_ratio(ad, a2) / 100.0 if ad and a2 else 0.0,
            JaroWinkler.normalized_similarity(ad, a2) if ad and a2 else 0.0,
            tj_a, nj,
            1.0 if nums_set and nums_set & nums2 else 0.0,
            abs(la1 - la2) / (max(la1, la2) + 1) if (ad or a2) else 0.0,
            1.0 if ad and a2 else 0.0,
            1.0 if nums_set and nums2 else 0.0,
            1.0 if nums_set and nums_set == nums2 else 0.0,
            1.0 if co and co == c2 else 0.0,
            1.0 if not co or not c2 else 0.0,
            s23_is_s2[idx], 0.0,
            1.0 if ln1 >= 4 and ln2 >= 4 and sfx4_1 == n2[-4:] else 0.0
        )
        pair_s1.append(sid)
        pair_s2.append(s23_eid[idx])
        pair_feats.append(feat)

t_extract = time.time() - t_start
print(f"1,000 S1 records: total candidates = {n_cands_total:,}, evaluated = {n_evaluated:,} (dropped {n_cands_total - n_evaluated:,}) in {t_extract:.2f}s")

if pair_feats:
    X = np.array(pair_feats, dtype=np.float32)
    t_pred = time.time()
    probs = model.predict_proba(X)[:, 1]
    print(f"LGBM predict on {len(X):,} pairs took: {time.time()-t_pred:.2f}s")
    matches = (probs >= 0.92).sum()
    print(f"Matches found (prob >= 0.92): {matches:,}")

total_time = time.time() - t_start
print(f"Total time for 1,000 S1 records: {total_time:.2f}s (Rate: {1000/total_time:.1f} rec/s)")
extrapolated_20k = (20000 / 1000) * total_time
print(f"Time for 20,000 records: {extrapolated_20k:.1f}s ({extrapolated_20k/60:.2f} min)")

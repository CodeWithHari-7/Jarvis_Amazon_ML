import os, time, re
import pandas as pd
from collections import defaultdict

print("=" * 60)
print("DIAGNOSTIC: MEASURING CANDIDATE RECALL ON 1000 TRAIN S1")
print("=" * 60)

s1_tr = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', dtype=str, nrows=1000, keep_default_na=False)
gt_tr = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False)
gt_map = dict(zip(gt_tr['source1_entity_id'], gt_tr['matched_entity_ids']))

sample_gt = {}
all_gt_pairs = set()
for sid in s1_tr['entity_id']:
    m_str = gt_map.get(sid, '')
    ms = set(m.strip() for m in m_str.split(',') if m.strip())
    sample_gt[sid] = ms
    for m in ms:
        all_gt_pairs.add((sid, m))

print(f"Total S1: {len(sample_gt):,} | S1 with matches: {sum(1 for v in sample_gt.values() if v)} | True Pairs: {len(all_gt_pairs):,}")

# Load S2 and S3 normalized parquets
print("Loading S2 and S3 parquet...")
s2 = pd.read_parquet('dataset/processed/S2_normalized.parquet')
s3 = pd.read_parquet('dataset/processed/S3_normalized.parquet')
s23 = pd.concat([s2, s3], ignore_index=True)
del s2, s3

s23 = s23[s23['entity_id'].notna() & (s23['entity_id'] != '') & s23['co'].notna() & (s23['co'] != '')].reset_index(drop=True)
print(f"Total S23 records: {len(s23):,}")

# Build Full Index (including p2, bgram)
print("Building Full Blocking Index...")
MAX_P4=150; MAX_P3=100; MAX_P2=60; MAX_NUM=80; MAX_TOK=60; MAX_AT=40; MAX_BG=40
_GN={'road','street','near','shop','opp','cross','lane','west','east','north','south','main','new','old','india','united','states','france','pvt','ltd','llc','inc','corp','limited','company','services','trading','enterprises','group','international','national'}
_GA={'road','street','avenue','lane','near','opp','opposite','plot','house','floor','building','sector','phase','area','nagar','marg','drive','place','court','boulevard','extension'}

def _bg(name):
    s=name[:8]
    return [s[i:i+2] for i in range(len(s)-1)] if len(s)>=2 else []

p4=defaultdict(list); p3=defaultdict(list); p2=defaultdict(list)
anum=defaultdict(list); ntok=defaultdict(list); atok=defaultdict(list); bgram=defaultdict(list)

s23_eid = s23['entity_id'].values
s23_nm = s23['nm'].values
s23_ad = s23['ad'].values
s23_co = s23['co'].values
s23_nums = [set(str(x).split(',')) if x else set() for x in s23['nums']]
s23_pr4 = s23['pr4'].values
s23_pr3 = s23['pr3'].values
s23_pr2 = s23['pr2'].values

for idx in range(len(s23)):
    c = s23_co[idx]; nm = s23_nm[idx]; ad = s23_ad[idx]
    pr4 = s23_pr4[idx]; pr3 = s23_pr3[idx]; pr2 = s23_pr2[idx]
    k4 = f"{c}\x00{pr4}"; k3 = f"{c}\x00{pr3}"; k2 = f"{c}\x00{pr2}"
    if pr4 and len(p4[k4]) < MAX_P4: p4[k4].append(idx)
    if pr3 and pr3 != pr4 and len(p3[k3]) < MAX_P3: p3[k3].append(idx)
    if len(pr2) == 2 and len(p2[k2]) < MAX_P2: p2[k2].append(idx)
    for n in s23_nums[idx]:
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

# Test candidate generation on sample
print("\nTesting candidate generation...")
_LEGAL=re.compile(r'\b(private limited|pvt\.? ltd\.?|llp|llc|ltd\.?|limited|inc\.?|corp\.?|corporation|co\.?|company|gmbh|ag|sa|sas|srl|bv|nv|plc|pty ltd|pty|l\.?p\.?|s\.a\.?|s\.r\.l\.|sarl|sasu|eurl)\s*$',re.IGNORECASE)
_MULTI=re.compile(r'\s+');_PUNCT=re.compile(r'[^\w\s]');_AND=re.compile(r'\s*&\s*',re.IGNORECASE)
_CMAP={'india':'india','in':'india','ind':'india','united states':'united states','us':'united states','usa':'united states','u.s.a.':'united states','u.s.':'united states','united states of america':'united states','france':'france','fr':'france','united kingdom':'united kingdom','uk':'united kingdom','gb':'united kingdom','germany':'germany','de':'germany','australia':'australia','au':'australia','canada':'canada','ca':'canada'}

def nname(s):
    s = str(s).lower(); s = _AND.sub(' and ', s); s = _LEGAL.sub('', s)
    s = _PUNCT.sub(' ', s); return _MULTI.sub(' ', s).strip()
def naddr(s):
    s = str(s).lower(); s = _PUNCT.sub(' ', s); return _MULTI.sub(' ', s).strip()
def nco(s):
    s = str(s).lower().strip(); s = _PUNCT.sub('', s); s = _MULTI.sub(' ', s).strip()
    return _CMAP.get(s, s)
def gnums(s): return [n for n in re.findall(r'\d+', str(s)) if len(n) >= 2]

# Config 1: Fast (infer_fast.py current)
hits_fast = 0
for _, row in s1_tr.iterrows():
    sid = str(row['entity_id'])
    gt_targets = sample_gt.get(sid, set())
    if not gt_targets: continue
    nm = nname(row['business_name']); ad = naddr(row['business_address']); co = nco(row['country'])
    nums = gnums(row['business_address'])
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
    cand_eids = set(s23_eid[i] for i in list(cands)[:100])
    hits_fast += len(cand_eids & gt_targets)

# Config 2: Full (infer.py style with p2, bgram, max_cands=300)
hits_full = 0
for _, row in s1_tr.iterrows():
    sid = str(row['entity_id'])
    gt_targets = sample_gt.get(sid, set())
    if not gt_targets: continue
    nm = nname(row['business_name']); ad = naddr(row['business_address']); co = nco(row['country'])
    nums = gnums(row['business_address'])
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
    cand_eids = set(s23_eid[i] for i in list(cands)[:300])
    hits_full += len(cand_eids & gt_targets)

print(f"\nRESULTS:")
print(f"Total Ground Truth Matches: {len(all_gt_pairs):,}")
print(f"Fast Blocking Recall (infer_fast): {hits_fast:,} / {len(all_gt_pairs):,} ({hits_fast/len(all_gt_pairs)*100:.2f}%)")
print(f"Full Blocking Recall (infer.py):   {hits_full:,} / {len(all_gt_pairs):,} ({hits_full/len(all_gt_pairs)*100:.2f}%)")

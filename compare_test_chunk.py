import os, sys, re, pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

print("=" * 65)
print("TESTING 1,000 ENTITIES FROM ROWS 200K-201K")
print("=" * 65)

# Load S1 rows 200,000 to 201,000
test_s1 = pd.read_csv('dataset/test/test_source1.tsv', sep='\t', dtype=str,
                      skiprows=range(1, 200001), nrows=1000, keep_default_na=False)

print(f"Loaded {len(test_s1)} entities: {test_s1.iloc[0]['entity_id']} to {test_s1.iloc[-1]['entity_id']}")

# Load model
with open('dataset/processed/lgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load S2 and S3 parquets (test)
s2 = pd.read_parquet('dataset/processed/S2_normalized.parquet')
s3 = pd.read_parquet('dataset/processed/S3_normalized.parquet')
s23 = pd.concat([s2, s3], ignore_index=True)
del s2, s3
s23 = s23[s23['entity_id'].notna() & (s23['entity_id'] != '') & s23['co'].notna() & (s23['co'] != '')].reset_index(drop=True)

s23_eid = s23['entity_id'].values
s23_nm = s23['nm'].values
s23_ad = s23['ad'].values
s23_co = s23['co'].values
s23_nums = [set(str(x).split(',')) if x else set() for x in s23['nums']]
s23_pr4 = s23['pr4'].values
s23_pr3 = s23['pr3'].values
s23_pr2 = s23['pr2'].values
s23_is_s2 = np.array([1.0 if str(x).startswith('S2') else 0.0 for x in s23_eid], dtype=np.float32)

print(f"Loaded {len(s23):,} S23 records.")

# Build Index
MAX_P4=150; MAX_P3=100; MAX_P2=60; MAX_NUM=80; MAX_TOK=60; MAX_AT=40; MAX_BG=40
_GN={'road','street','near','shop','opp','cross','lane','west','east','north','south','main','new','old','india','united','states','france','pvt','ltd','llc','inc','corp','limited','company','services','trading','enterprises','group','international','national'}
_GA={'road','street','avenue','lane','near','opp','opposite','plot','house','floor','building','sector','phase','area','nagar','marg','drive','place','court','boulevard','extension'}

def _bg(name):
    s = name[:8]; return [s[i:i+2] for i in range(len(s)-1)] if len(s) >= 2 else []

p4 = defaultdict(list); p3 = defaultdict(list); p2 = defaultdict(list)
anum = defaultdict(list); ntok = defaultdict(list); atok = defaultdict(list); bgram = defaultdict(list)

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

# Let's run full candidates vs fast candidates and count how many S1 get matches >= 0.92
matched_infer_fast = 0
matched_full = 0

for _, row in test_s1.iterrows():
    sid = str(row['entity_id'])
    nm = nname(row['business_name']); ad = naddr(row['business_address']); co = nco(row['country'])
    nums = gnums(row['business_address'])
    nums_set = set(nums)
    pr4 = nm[:4]; pr3 = nm[:3]; pr2 = nm[:2]

    # Fast cands (infer_fast)
    cands_fast = set()
    for idx in p4.get(f"{co}\x00{pr4}", ()): cands_fast.add(idx)
    if pr3 != pr4:
        for idx in p3.get(f"{co}\x00{pr3}", ()): cands_fast.add(idx)
    for n in nums:
        for idx in anum.get(f"{co}\x00{n}", ()): cands_fast.add(idx)
    for tk in [t for t in nm.split() if len(t) >= 4 and t not in _GN]:
        for idx in ntok.get(f"{co}\x00{tk}", ()): cands_fast.add(idx)
    for tk in [t for t in ad.split() if len(t) >= 4 and t not in _GA]:
        for idx in atok.get(f"{co}\x00{tk}", ()): cands_fast.add(idx)
    cands_fast = list(cands_fast)[:100]

    # Full cands (infer.py)
    cands_full = set(cands_fast)
    if len(pr2) == 2:
        for idx in p2.get(f"{co}\x00{pr2}", ()): cands_full.add(idx)
    for b in _bg(nm):
        for idx in bgram.get(f"{co}\x00{b}", ()): cands_full.add(idx)
    cands_full = list(cands_full)[:300]

    # Score fast
    has_m_fast = False
    s1n = set(nm.split()); s1a = set(ad.split())
    ng1 = {nm[i:i+3] for i in range(len(nm)-2)} if len(nm) >= 3 else set()
    ln1 = len(nm); la1 = len(ad)
    pfx4_1 = nm[:4] if ln1 >= 4 else ""; pfx3_1 = nm[:3] if ln1 >= 3 else ""; sfx4_1 = nm[-4:] if ln1 >= 4 else ""
    s1_mono = 1.0 if len(s1n) == 1 else 0.0

    feats_fast = []
    for idx in cands_fast:
        n2 = s23_nm[idx]; a2 = s23_ad[idx]; c2 = s23_co[idx]; nums2 = s23_nums[idx]
        has_num = bool(nums_set & nums2); r_name = fuzz.ratio(nm, n2)
        if not has_num and r_name < 30:
            if not any(w in n2 for w in s1n if len(w) >= 4): continue
        s2n = set(n2.split()); s2a = set(a2.split())
        un = s1n | s2n; ua = s1a | s2a; unum = nums_set | nums2
        tj_n = len(s1n & s2n) / len(un) if un else 0.0; tj_a = len(s1a & s2a) / len(ua) if ua else 0.0
        inter = len(s1n & s2n); tc = max(inter / len(s1n), inter / len(s2n)) if s1n and s2n else 0.0
        nj = len(nums_set & nums2) / len(unum) if unum else 0.0
        ng2 = {n2[k:k+3] for k in range(len(n2)-2)} if len(n2) >= 3 else set()
        ung = ng1 | ng2; ng_ov = len(ng1 & ng2) / len(ung) if ung else 0.0
        ln2 = len(n2); la2 = len(a2)
        feats_fast.append((
            1.0 if nm and nm == n2 else 0.0, r_name / 100.0,
            fuzz.partial_ratio(nm, n2) / 100.0 if nm and n2 else 0.0,
            fuzz.token_set_ratio(nm, n2) / 100.0 if nm and n2 else 0.0,
            fuzz.token_sort_ratio(nm, n2) / 100.0 if nm and n2 else 0.0,
            JaroWinkler.normalized_similarity(nm, n2) if nm and n2 else 0.0,
            tj_n, tc, 0.0, abs(ln1 - ln2) / (max(ln1, ln2) + 1),
            1.0 if ln1 >= 4 and ln2 >= 4 and pfx4_1 == n2[:4] else 0.0,
            1.0 if ln1 >= 3 and ln2 >= 3 and pfx3_1 == n2[:3] else 0.0,
            1.0 if nm and n2 else 0.0, 1.0 - Levenshtein.normalized_similarity(nm, n2) if nm and n2 else 0.0,
            ng_ov, 1.0 if ln1 >= 4 and ln2 >= 4 and sfx4_1 == n2[-4:] else 0.0,
            1.0 if s1_mono or (n2 and len(s2n) == 1) else 0.0,
            1.0 if ad and ad == a2 else 0.0, fuzz.ratio(ad, a2) / 100.0 if ad and a2 else 0.0,
            fuzz.partial_ratio(ad, a2) / 100.0 if ad and a2 else 0.0,
            fuzz.token_set_ratio(ad, a2) / 100.0 if ad and a2 else 0.0,
            fuzz.token_sort_ratio(ad, a2) / 100.0 if ad and a2 else 0.0,
            JaroWinkler.normalized_similarity(ad, a2) if ad and a2 else 0.0,
            tj_a, nj, 1.0 if has_num else 0.0, abs(la1 - la2) / (max(la1, la2) + 1) if (ad or a2) else 0.0,
            1.0 if ad and a2 else 0.0, 1.0 if nums_set and nums2 else 0.0,
            1.0 if nums_set and nums_set == nums2 else 0.0, 1.0 if co and co == c2 else 0.0,
            1.0 if not co or not c2 else 0.0, s23_is_s2[idx], 0.0,
            1.0 if ln1 >= 4 and ln2 >= 4 and sfx4_1 == n2[-4:] else 0.0
        ))
    if feats_fast:
        pr = model.predict_proba(np.array(feats_fast, dtype=np.float32))[:, 1]
        if (pr >= 0.92).any(): matched_infer_fast += 1

    # Score full (NO pre-filter, max_cands=300)
    feats_full = []
    for idx in cands_full:
        n2 = s23_nm[idx]; a2 = s23_ad[idx]; c2 = s23_co[idx]; nums2 = s23_nums[idx]
        has_num = bool(nums_set & nums2); r_name = fuzz.ratio(nm, n2)
        s2n = set(n2.split()); s2a = set(a2.split())
        un = s1n | s2n; ua = s1a | s2a; unum = nums_set | nums2
        tj_n = len(s1n & s2n) / len(un) if un else 0.0; tj_a = len(s1a & s2a) / len(ua) if ua else 0.0
        inter = len(s1n & s2n); tc = max(inter / len(s1n), inter / len(s2n)) if s1n and s2n else 0.0
        nj = len(nums_set & nums2) / len(unum) if unum else 0.0
        ng2 = {n2[k:k+3] for k in range(len(n2)-2)} if len(n2) >= 3 else set()
        ung = ng1 | ng2; ng_ov = len(ng1 & ng2) / len(ung) if ung else 0.0
        ln2 = len(n2); la2 = len(a2)
        feats_full.append((
            1.0 if nm and nm == n2 else 0.0, r_name / 100.0,
            fuzz.partial_ratio(nm, n2) / 100.0 if nm and n2 else 0.0,
            fuzz.token_set_ratio(nm, n2) / 100.0 if nm and n2 else 0.0,
            fuzz.token_sort_ratio(nm, n2) / 100.0 if nm and n2 else 0.0,
            JaroWinkler.normalized_similarity(nm, n2) if nm and n2 else 0.0,
            tj_n, tc, 0.0, abs(ln1 - ln2) / (max(ln1, ln2) + 1),
            1.0 if ln1 >= 4 and ln2 >= 4 and pfx4_1 == n2[:4] else 0.0,
            1.0 if ln1 >= 3 and ln2 >= 3 and pfx3_1 == n2[:3] else 0.0,
            1.0 if nm and n2 else 0.0, 1.0 - Levenshtein.normalized_similarity(nm, n2) if nm and n2 else 0.0,
            ng_ov, 1.0 if ln1 >= 4 and ln2 >= 4 and sfx4_1 == n2[-4:] else 0.0,
            1.0 if s1_mono or (n2 and len(s2n) == 1) else 0.0,
            1.0 if ad and ad == a2 else 0.0, fuzz.ratio(ad, a2) / 100.0 if ad and a2 else 0.0,
            fuzz.partial_ratio(ad, a2) / 100.0 if ad and a2 else 0.0,
            fuzz.token_set_ratio(ad, a2) / 100.0 if ad and a2 else 0.0,
            fuzz.token_sort_ratio(ad, a2) / 100.0 if ad and a2 else 0.0,
            JaroWinkler.normalized_similarity(ad, a2) if ad and a2 else 0.0,
            tj_a, nj, 1.0 if has_num else 0.0, abs(la1 - la2) / (max(la1, la2) + 1) if (ad or a2) else 0.0,
            1.0 if ad and a2 else 0.0, 1.0 if nums_set and nums2 else 0.0,
            1.0 if nums_set and nums_set == nums2 else 0.0, 1.0 if co and co == c2 else 0.0,
            1.0 if not co or not c2 else 0.0, s23_is_s2[idx], 0.0,
            1.0 if ln1 >= 4 and ln2 >= 4 and sfx4_1 == n2[-4:] else 0.0
        ))
    if feats_full:
        pr = model.predict_proba(np.array(feats_full, dtype=np.float32))[:, 1]
        if (pr >= 0.92).any(): matched_full += 1

print(f"\nRESULTS on 1,000 S1 from rows 200k-201k:")
print(f"  infer_fast (current): {matched_infer_fast} / 1000 matched ({matched_infer_fast/10:.1f}%)")
print(f"  Full Blocking (infer.py style): {matched_full} / 1000 matched ({matched_full/10:.1f}%)")

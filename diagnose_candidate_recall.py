import os, sys, re, pickle
import pandas as pd
from collections import defaultdict
from rapidfuzz import fuzz

# Load 200 train S1 and their true matches
s1_df = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', dtype=str, nrows=200, keep_default_na=False)
gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False)
gt_map = dict(zip(gt_df['source1_entity_id'], gt_df['matched_entity_ids']))

target_matches = {}
all_target_eids = set()
for sid in s1_df['entity_id']:
    m_str = gt_map.get(sid, '')
    ms = [m.strip() for m in m_str.split(',') if m.strip()]
    target_matches[sid] = ms
    all_target_eids.update(ms)

# Load records
s23_records = {}
for chunk in pd.read_csv('dataset/train/train_source2.tsv', sep='\t', dtype=str, chunksize=100000, keep_default_na=False):
    sub = chunk[chunk['entity_id'].isin(all_target_eids)]
    for _, r in sub.iterrows(): s23_records[r['entity_id']] = r.to_dict()
for chunk in pd.read_csv('dataset/train/train_source3.tsv', sep='\t', dtype=str, chunksize=100000, keep_default_na=False):
    sub = chunk[chunk['entity_id'].isin(all_target_eids)]
    for _, r in sub.iterrows(): s23_records[r['entity_id']] = r.to_dict()

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
def _bg(name):
    s = name[:8]; return [s[i:i+2] for i in range(len(s)-1)] if len(s) >= 2 else []

_GN={'road','street','near','shop','opp','cross','lane','west','east','north','south','main','new','old','india','united','states','france','pvt','ltd','llc','inc','corp','limited','company','services','trading','enterprises','group','international','national'}
_GA={'road','street','avenue','lane','near','opp','opposite','plot','house','floor','building','sector','phase','area','nagar','marg','drive','place','court','boulevard','extension'}

# Test each pass on all true pairs
total_pairs = 0
hit_p4 = 0; hit_p3 = 0; hit_p2 = 0; hit_num = 0; hit_ntok = 0; hit_atok = 0; hit_bg = 0
hit_fast_union = 0; hit_full_union = 0
killed_by_prefilter = 0

for _, r1 in s1_df.iterrows():
    sid = r1['entity_id']
    nm1 = nname(r1['business_name']); ad1 = naddr(r1['business_address']); co1 = nco(r1['country'])
    nums1 = set(gnums(r1['business_address']))
    s1n = set(nm1.split()); s1a = set(ad1.split())
    ntok1 = [t for t in s1n if len(t) >= 4 and t not in _GN]
    atok1 = [t for t in s1a if len(t) >= 4 and t not in _GA]
    bg1 = set(_bg(nm1))

    for tid in target_matches.get(sid, []):
        r2 = s23_records.get(tid)
        if not r2: continue
        total_pairs += 1
        nm2 = nname(r2['business_name']); ad2 = naddr(r2['business_address']); co2 = nco(r2['country'])
        nums2 = set(gnums(r2['business_address']))
        s2n = set(nm2.split()); s2a = set(ad2.split())
        ntok2 = [t for t in s2n if len(t) >= 4 and t not in _GN]
        atok2 = [t for t in s2a if len(t) >= 4 and t not in _GA]
        bg2 = set(_bg(nm2))

        m_p4 = bool(nm1[:4] and nm1[:4] == nm2[:4])
        m_p3 = bool(nm1[:3] and nm1[:3] == nm2[:3])
        m_p2 = bool(nm1[:2] and nm1[:2] == nm2[:2])
        m_num = bool(nums1 & nums2)
        m_ntok = bool(set(ntok1) & set(ntok2))
        m_atok = bool(set(atok1) & set(atok2))
        m_bg = bool(bg1 & bg2)

        if m_p4: hit_p4 += 1
        if m_p3: hit_p3 += 1
        if m_p2: hit_p2 += 1
        if m_num: hit_num += 1
        if m_ntok: hit_ntok += 1
        if m_atok: hit_atok += 1
        if m_bg: hit_bg += 1

        is_fast = m_p4 or m_p3 or m_num or m_ntok or m_atok
        is_full = is_fast or m_p2 or m_bg

        if is_fast: hit_fast_union += 1
        if is_full: hit_full_union += 1

        # Check pre-filter
        has_num = bool(nums1 & nums2)
        r_name = fuzz.ratio(nm1, nm2)
        if not has_num and r_name < 30:
            if not any(w in nm2 for w in s1n if len(w) >= 4):
                killed_by_prefilter += 1

print(f"Total True Pairs: {total_pairs}")
print(f"  P4 match (first 4 chars):     {hit_p4:>4} ({hit_p4/total_pairs*100:.1f}%)")
print(f"  P3 match (first 3 chars):     {hit_p3:>4} ({hit_p3/total_pairs*100:.1f}%)")
print(f"  P2 match (first 2 chars):     {hit_p2:>4} ({hit_p2/total_pairs*100:.1f}%)")
print(f"  Address Numbers match:        {hit_num:>4} ({hit_num/total_pairs*100:.1f}%)")
print(f"  Name Token match (>=4 chars): {hit_ntok:>4} ({hit_ntok/total_pairs*100:.1f}%)")
print(f"  Addr Token match (>=4 chars): {hit_atok:>4} ({hit_atok/total_pairs*100:.1f}%)")
print(f"  Name Bigram match:            {hit_bg:>4} ({hit_bg/total_pairs*100:.1f}%)")
print("-" * 50)
print(f"Fast Union Recall (infer_fast): {hit_fast_union:>4} ({hit_fast_union/total_pairs*100:.1f}%)")
print(f"Full Union Recall (infer.py):   {hit_full_union:>4} ({hit_full_union/total_pairs*100:.1f}%)")
print(f"Pairs killed by prefilter:      {killed_by_prefilter:>4} ({killed_by_prefilter/total_pairs*100:.1f}%)")

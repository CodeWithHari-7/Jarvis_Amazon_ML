import os, sys, re, pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

print("=" * 65)
print("GROUND TRUTH FAILURE ANALYSIS ON 200 TRAIN S1")
print("=" * 65)

# Load 200 train S1
s1_df = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', dtype=str, nrows=200, keep_default_na=False)
gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False)
gt_map = dict(zip(gt_df['source1_entity_id'], gt_df['matched_entity_ids']))

# Target matches for these 200 S1
target_matches = {}
all_target_eids = set()
for sid in s1_df['entity_id']:
    m_str = gt_map.get(sid, '')
    ms = [m.strip() for m in m_str.split(',') if m.strip()]
    target_matches[sid] = ms
    all_target_eids.update(ms)

print(f"200 S1 entities | Total True Matches: {len(all_target_eids):,}")

# Load model
with open('dataset/processed/lgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load S2 and S3 from train
print("Loading train S2/S3 for these targets...")
s2_records = {}
for chunk in pd.read_csv('dataset/train/train_source2.tsv', sep='\t', dtype=str, chunksize=100000, keep_default_na=False):
    sub = chunk[chunk['entity_id'].isin(all_target_eids)]
    for _, r in sub.iterrows():
        s2_records[r['entity_id']] = r.to_dict()
    if len(s2_records) == len([e for e in all_target_eids if e.startswith('S2')]):
        break

s3_records = {}
for chunk in pd.read_csv('dataset/train/train_source3.tsv', sep='\t', dtype=str, chunksize=100000, keep_default_na=False):
    sub = chunk[chunk['entity_id'].isin(all_target_eids)]
    for _, r in sub.iterrows():
        s3_records[r['entity_id']] = r.to_dict()
    if len(s3_records) == len([e for e in all_target_eids if e.startswith('S3')]):
        break

print(f"Loaded {len(s2_records):,} S2 targets and {len(s3_records):,} S3 targets.")

# Normalization
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

# Now for each true pair, compute features and model probability!
print("\nEvaluating model probabilities on TRUE MATCHES:")
probs = []
low_prob_pairs = []

for _, r1 in s1_df.iterrows():
    sid = r1['entity_id']
    nm1 = nname(r1['business_name']); ad1 = naddr(r1['business_address']); co1 = nco(r1['country'])
    nums1 = set(gnums(r1['business_address']))
    s1n = set(nm1.split()); s1a = set(ad1.split())
    ng1 = {nm1[i:i+3] for i in range(len(nm1)-2)} if len(nm1) >= 3 else set()
    ln1 = len(nm1); la1 = len(ad1)
    pfx4_1 = nm1[:4] if ln1 >= 4 else ""
    pfx3_1 = nm1[:3] if ln1 >= 3 else ""
    sfx4_1 = nm1[-4:] if ln1 >= 4 else ""
    s1_mono = 1.0 if len(s1n) == 1 else 0.0

    for tid in target_matches.get(sid, []):
        r2 = s2_records.get(tid) or s3_records.get(tid)
        if not r2: continue
        nm2 = nname(r2['business_name']); ad2 = naddr(r2['business_address']); co2 = nco(r2['country'])
        nums2 = set(gnums(r2['business_address']))
        s2n = set(nm2.split()); s2a = set(ad2.split())
        un = s1n | s2n; ua = s1a | s2a; unum = nums1 | nums2
        tj_n = len(s1n & s2n) / len(un) if un else 0.0
        tj_a = len(s1a & s2a) / len(ua) if ua else 0.0
        inter = len(s1n & s2n)
        tc = max(inter / len(s1n), inter / len(s2n)) if s1n and s2n else 0.0
        nj = len(nums1 & nums2) / len(unum) if unum else 0.0
        ng2 = {nm2[k:k+3] for k in range(len(nm2)-2)} if len(nm2) >= 3 else set()
        ung = ng1 | ng2
        ng_ov = len(ng1 & ng2) / len(ung) if ung else 0.0
        ln2 = len(nm2); la2 = len(ad2)

        feat = [
            1.0 if nm1 and nm1 == nm2 else 0.0,
            fuzz.ratio(nm1, nm2) / 100.0 if nm1 and nm2 else 0.0,
            fuzz.partial_ratio(nm1, nm2) / 100.0 if nm1 and nm2 else 0.0,
            fuzz.token_set_ratio(nm1, nm2) / 100.0 if nm1 and nm2 else 0.0,
            fuzz.token_sort_ratio(nm1, nm2) / 100.0 if nm1 and nm2 else 0.0,
            JaroWinkler.normalized_similarity(nm1, nm2) if nm1 and nm2 else 0.0,
            tj_n, tc, 0.0,
            abs(ln1 - ln2) / (max(ln1, ln2) + 1),
            1.0 if ln1 >= 4 and ln2 >= 4 and pfx4_1 == nm2[:4] else 0.0,
            1.0 if ln1 >= 3 and ln2 >= 3 and pfx3_1 == nm2[:3] else 0.0,
            1.0 if nm1 and nm2 else 0.0,
            1.0 - Levenshtein.normalized_similarity(nm1, nm2) if nm1 and nm2 else 0.0,
            ng_ov,
            1.0 if ln1 >= 4 and ln2 >= 4 and sfx4_1 == nm2[-4:] else 0.0,
            1.0 if s1_mono or (nm2 and len(s2n) == 1) else 0.0,
            1.0 if ad1 and ad1 == ad2 else 0.0,
            fuzz.ratio(ad1, ad2) / 100.0 if ad1 and ad2 else 0.0,
            fuzz.partial_ratio(ad1, ad2) / 100.0 if ad1 and ad2 else 0.0,
            fuzz.token_set_ratio(ad1, ad2) / 100.0 if ad1 and ad2 else 0.0,
            fuzz.token_sort_ratio(ad1, ad2) / 100.0 if ad1 and ad2 else 0.0,
            JaroWinkler.normalized_similarity(ad1, ad2) if ad1 and ad2 else 0.0,
            tj_a, nj,
            1.0 if nums1 & nums2 else 0.0,
            abs(la1 - la2) / (max(la1, la2) + 1) if (ad1 or ad2) else 0.0,
            1.0 if ad1 and ad2 else 0.0,
            1.0 if nums1 and nums2 else 0.0,
            1.0 if nums1 and nums1 == nums2 else 0.0,
            1.0 if co1 and co1 == co2 else 0.0,
            1.0 if not co1 or not co2 else 0.0,
            1.0 if tid.startswith('S2') else 0.0,
            0.0,
            1.0 if ln1 >= 4 and ln2 >= 4 and sfx4_1 == nm2[-4:] else 0.0
        ]
        p = model.predict_proba(np.array([feat], dtype=np.float32))[0, 1]
        probs.append(p)
        if p < 0.92:
            low_prob_pairs.append((sid, tid, p, nm1, nm2, ad1, ad2, co1, co2))

probs = np.array(probs)
print(f"Total True Pairs Evaluated: {len(probs)}")
print(f"Mean Probability: {probs.mean():.4f} | Median: {np.median(probs):.4f}")
print(f"Pairs with prob >= 0.92: {(probs >= 0.92).sum()} ({(probs >= 0.92).sum()/len(probs)*100:.1f}%)")
print(f"Pairs with prob >= 0.80: {(probs >= 0.80).sum()} ({(probs >= 0.80).sum()/len(probs)*100:.1f}%)")
print(f"Pairs with prob >= 0.50: {(probs >= 0.50).sum()} ({(probs >= 0.50).sum()/len(probs)*100:.1f}%)")
print(f"Pairs with prob < 0.50:  {(probs < 0.50).sum()} ({(probs < 0.50).sum()/len(probs)*100:.1f}%)")

if low_prob_pairs:
    print("\nSample True Pairs with prob < 0.92:")
    for sid, tid, p, nm1, nm2, ad1, ad2, c1, c2 in low_prob_pairs[:5]:
        print(f"  [{sid} -> {tid}] prob={p:.4f}")
        print(f"    S1: {nm1} | {ad1} | {c1}")
        print(f"    S2/3: {nm2} | {ad2} | {c2}")

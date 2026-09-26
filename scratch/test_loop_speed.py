import time, pickle
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

# Benchmark the optimized candidate loop
with open('dataset/processed/lgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Mock 1000 records with 50 candidates each (50,000 candidate evaluations)
nm = 'sharma general store'
ad = '123 mg road bangalore'
co = 'india'
nums_set = {'123'}
s1n = set(nm.split()); s1a = set(ad.split())
ng1 = {nm[i:i+3] for i in range(len(nm)-2)}
ln1 = len(nm); la1 = len(ad)
pfx4_1 = nm[:4]; pfx3_1 = nm[:3]; sfx4_1 = nm[-4:]
s1_mono = 1.0 if len(s1n) == 1 else 0.0

cands_n2 = ['sharma stores pvt ltd', 'shiv om jewellers', 'shree ganesh sweets', 'sharma medical store', 'random shop name'] * 10000
cands_a2 = ['123 mg road near temple bangalore', '45 park street kolkata', '88 nehru road delhi', '123 mg road bangalore', 'opp bus stand mumbai'] * 10000
cands_c2 = ['india'] * 50000
cands_nums = [{'123'}, {'45'}, {'88'}, {'123'}, set()] * 10000

t0 = time.time()
pair_feats = []
n_eval = 0

for i in range(50000):
    n2 = cands_n2[i]; a2 = cands_a2[i]; c2 = cands_c2[i]; nums2 = cands_nums[i]

    # Fast filter
    has_num = bool(nums_set & nums2)
    r_name = fuzz.ratio(nm, n2)
    if not has_num and r_name < 30:
        if not any(w in n2 for w in s1n if len(w) >= 4):
            continue

    n_eval += 1
    s2n = set(n2.split()); s2a = set(a2.split())
    un = s1n | s2n; ua = s1a | s2a; unum = nums_set | nums2
    tj_n = len(s1n & s2n) / len(un) if un else 0.0
    tj_a = len(s1a & s2a) / len(ua) if ua else 0.0
    inter = len(s1n & s2n)
    tc = max(inter / len(s1n), inter / len(s2n)) if s1n and s2n else 0.0
    nj = len(nums_set & nums2) / len(unum) if unum else 0.0
    ng2 = {n2[k:k+3] for k in range(len(n2)-2)} if len(n2) >= 3 else set()
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
        1.0 if has_num else 0.0,
        abs(la1 - la2) / (max(la1, la2) + 1) if (ad or a2) else 0.0,
        1.0 if ad and a2 else 0.0,
        1.0 if nums_set and nums2 else 0.0,
        1.0 if nums_set and nums_set == nums2 else 0.0,
        1.0 if co and co == c2 else 0.0,
        1.0 if not co or not c2 else 0.0,
        1.0, 0.0,
        1.0 if ln1 >= 4 and ln2 >= 4 and sfx4_1 == n2[-4:] else 0.0
    )
    pair_feats.append(feat)

t_loop = time.time() - t0
X = np.array(pair_feats, dtype=np.float32)
t_pred = time.time()
probs = model.predict_proba(X)[:, 1]
t_pred = time.time() - t_pred

print(f"Evaluated {n_eval:,}/{50000:,} pairs in {t_loop:.2f}s, predict took {t_pred:.2f}s, total: {t_loop+t_pred:.2f}s")

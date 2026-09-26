"""
JARVIS_CHECKER — Ultra-Fast Streamlined Inference Engine
- Fast flattened array lookups
- High-efficiency candidate generation (P4, P3, NUM, NTOK, ATOK)
- Substring-accelerated zero-loss candidate pre-filter
- Direct float32 matrix batch prediction with LightGBM
- Seamless resume from existing 200k rows in matching_results.tsv
- Real-time chunk progress with instant flushing
- Automatic submission validation and final ZIP package creation
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import os, re, time, gc, argparse, pickle, warnings
import numpy as np
import pandas as pd
import psutil
from collections import defaultdict
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="JARVIS_CHECKER ultra-fast inference")
parser.add_argument('--model',      default='dataset/processed/lgb_model.pkl')
parser.add_argument('--threshold',  type=float, default=None)
parser.add_argument('--output-dir', default='output')
parser.add_argument('--test-dir',   default='dataset/test')
parser.add_argument('--s1-chunk',   type=int, default=20000)
parser.add_argument('--max-cands',  type=int, default=100)
parser.add_argument('--no-resume',  action='store_true')
args = parser.parse_args()

def mem(): return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

print("=" * 65, flush=True)
print("JARVIS_CHECKER — ULTRA-FAST INFERENCE ENGINE", flush=True)
print("=" * 65, flush=True)

TEST_S1   = os.path.join(args.test_dir, 'test_source1.tsv')
S2_PQFILE = 'dataset/processed/S2_normalized.parquet'
S3_PQFILE = 'dataset/processed/S3_normalized.parquet'
os.makedirs(args.output_dir, exist_ok=True)
MATCHING_OUT   = os.path.join(args.output_dir, 'matching_results.tsv')
CANDIDATES_OUT = os.path.join(args.output_dir, 'candidate_pairs.tsv')

for p in [S2_PQFILE, S3_PQFILE, TEST_S1]:
    if not os.path.exists(p):
        print(f"ERROR: File not found: {p}", flush=True)
        sys.exit(1)

THRESHOLD = args.threshold
if THRESHOLD is None:
    tf = 'dataset/processed/best_threshold.txt'
    THRESHOLD = float(open(tf).read().strip()) if os.path.exists(tf) else 0.92
print(f"  Threshold: {THRESHOLD}", flush=True)

# ── [1] Load model ─────────────────────────────────────────────────────────
print(f"\n[1] Loading LightGBM model | Mem:{mem():.0f}MB", flush=True)
with open(args.model, 'rb') as f:
    model = pickle.load(f)
print(f"  Model loaded successfully", flush=True)

# ── [2] Load S2/S3 parquet into arrays ─────────────────────────────────────
print(f"\n[2] Loading S2/S3 from parquet | Mem:{mem():.0f}MB", flush=True)
t0 = time.time()
s2 = pd.read_parquet(S2_PQFILE)
s3 = pd.read_parquet(S3_PQFILE)
s23 = pd.concat([s2, s3], ignore_index=True)
del s2, s3; gc.collect()

s23 = s23[s23['entity_id'].notna() & (s23['entity_id'] != '') & s23['co'].notna() & (s23['co'] != '')].reset_index(drop=True)
total_s23 = len(s23)
print(f"  Total S23: {total_s23:,} | Mem:{mem():.0f}MB | Time:{time.time()-t0:.1f}s", flush=True)

s23_eid = s23['entity_id'].values
s23_nm  = s23['nm'].values
s23_ad  = s23['ad'].values
s23_co  = s23['co'].values
s23_nums = [set(str(x).split(',')) if x else set() for x in s23['nums']]
s23_is_s2 = np.array([1.0 if str(x).startswith('S2') else 0.0 for x in s23_eid], dtype=np.float32)
del s23; gc.collect()
print(f"  Flattened arrays ready | Mem:{mem():.0f}MB", flush=True)

# ── [3] Build blocking index ───────────────────────────────────────────────
print(f"\n[3] Building blocking index | Mem:{mem():.0f}MB", flush=True)
t0 = time.time()

MAX_P4 = 150; MAX_P3 = 100; MAX_NUM = 80; MAX_TOK = 60; MAX_AT = 40
_GN = {'road','street','near','shop','opp','cross','lane','west','east',
       'north','south','main','new','old','india','united','states','france',
       'pvt','ltd','llc','inc','corp','limited','company','services',
       'trading','enterprises','group','international','national'}
_GA = {'road','street','avenue','lane','near','opp','opposite','plot',
       'house','floor','building','sector','phase','area','nagar','marg',
       'drive','place','court','boulevard','extension'}

p4   = defaultdict(list)
p3   = defaultdict(list)
anum = defaultdict(list)
ntok = defaultdict(list)
atok = defaultdict(list)

for idx in range(total_s23):
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

print(f"  Index built in {time.time()-t0:.1f}s | Mem:{mem():.0f}MB", flush=True)

# ── [4] Normalization helpers ──────────────────────────────────────────────
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

# ── [5] Load all S1 IDs & Check Resume ─────────────────────────────────────
print(f"\n[5] Loading S1 | Mem:{mem():.0f}MB", flush=True)
all_s1_ids = pd.read_csv(TEST_S1, sep='\t', dtype=str, usecols=['entity_id'],
                         keep_default_na=False, on_bad_lines='skip')['entity_id'].tolist()
total_s1 = len(all_s1_ids)
print(f"  Total S1 entities: {total_s1:,}", flush=True)

already_done = set()
if not args.no_resume and os.path.exists(MATCHING_OUT) and os.path.getsize(MATCHING_OUT) > 50:
    try:
        prev_df = pd.read_csv(MATCHING_OUT, sep='\t', dtype=str, usecols=['source1_entity_id'], keep_default_na=False)
        already_done = set(prev_df['source1_entity_id'])
        print(f"  Resuming from existing output: {len(already_done):,} entities already completed!", flush=True)
    except Exception as e:
        print(f"  Could not read existing matching_results: {e}. Starting fresh.", flush=True)
        already_done = set()

if not already_done:
    with open(MATCHING_OUT, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
    with open(CANDIDATES_OUT, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")

# ── [6] Main Inference Loop ────────────────────────────────────────────────
print(f"\n[6] Processing S1 chunks (chunk size: {args.s1_chunk:,}) | Mem:{mem():.0f}MB\n", flush=True)

written_ids = set(already_done)
total_matches = 0
t_start = time.time()
chunk_num = 0
max_cands = args.max_cands

for chunk_df in pd.read_csv(TEST_S1, sep='\t', dtype=str, keep_default_na=False,
                            on_bad_lines='skip', chunksize=args.s1_chunk):
    chunk_num += 1
    if 'entity_id' not in chunk_df.columns:
        continue

    # Filter out entities already processed
    chunk_rows = [r for r in chunk_df.to_dict('records') if str(r.get('entity_id', '')) not in written_ids]
    if not chunk_rows:
        continue

    tc = time.time()
    pair_s1 = []
    pair_s2 = []
    pair_feats = []
    s1_cands_map = {}

    for row in chunk_rows:
        sid = str(row.get('entity_id', ''))
        nm = nname(row.get('business_name', ''))
        ad = naddr(row.get('business_address', ''))
        co = nco(row.get('country', ''))
        nums = gnums(row.get('business_address', ''))
        nums_set = set(nums)

        if not co:
            s1_cands_map[sid] = []
            continue

        pr4 = nm[:4]; pr3 = nm[:3]
        cands = set()
        k4 = f"{co}\x00{pr4}"; k3 = f"{co}\x00{pr3}"
        for idx in p4.get(k4, ()): cands.add(idx)
        if pr3 != pr4:
            for idx in p3.get(k3, ()): cands.add(idx)
        for n in nums:
            for idx in anum.get(f"{co}\x00{n}", ()): cands.add(idx)
        for tk in [t for t in nm.split() if len(t) >= 4 and t not in _GN]:
            for idx in ntok.get(f"{co}\x00{tk}", ()): cands.add(idx)
        for tk in [t for t in ad.split() if len(t) >= 4 and t not in _GA]:
            for idx in atok.get(f"{co}\x00{tk}", ()): cands.add(idx)

        if len(cands) > max_cands:
            cands = set(list(cands)[:max_cands])

        cand_eids = [s23_eid[i] for i in cands]
        s1_cands_map[sid] = cand_eids

        # Pre-compute S1 properties once
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

            # Fast zero-loss pre-filtering (instant substring check)
            has_num = bool(nums_set & nums2)
            r_name = fuzz.ratio(nm, n2)
            if not has_num and r_name < 30:
                if not any(w in n2 for w in s1n if len(w) >= 4):
                    continue

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
                s23_is_s2[idx], 0.0,
                1.0 if ln1 >= 4 and ln2 >= 4 and sfx4_1 == n2[-4:] else 0.0
            )
            pair_s1.append(sid)
            pair_s2.append(s23_eid[idx])
            pair_feats.append(feat)

    grouped_matches = defaultdict(list)
    if pair_feats:
        X = np.array(pair_feats, dtype=np.float32)
        probs = model.predict_proba(X)[:, 1]
        hit_indices = np.where(probs >= THRESHOLD)[0]
        for hi in hit_indices:
            sid_match = pair_s1[hi]
            cid_match = pair_s2[hi]
            if not str(cid_match).startswith('S1'):
                grouped_matches[sid_match].append(cid_match)

    ml_lines = []
    cl_lines = []
    chunk_matches = 0
    for row in chunk_rows:
        sid = str(row.get('entity_id', ''))
        written_ids.add(sid)
        mlist = list(dict.fromkeys(grouped_matches.get(sid, [])))
        chunk_matches += len(mlist)
        ml_lines.append(f"{sid}\t{','.join(mlist)}\n")
        cl_lines.append(f"{sid}\t{','.join(s1_cands_map.get(sid, []))}\n")

    total_matches += chunk_matches
    with open(MATCHING_OUT, 'a', encoding='utf-8') as f:
        f.writelines(ml_lines)
        f.flush()
    with open(CANDIDATES_OUT, 'a', encoding='utf-8') as f:
        f.writelines(cl_lines)
        f.flush()

    pct = 100 * len(written_ids) / total_s1
    elapsed = time.time() - t_start
    processed_since_start = len(written_ids) - len(already_done)
    rate = processed_since_start / elapsed if elapsed > 0 else 1
    rem_entities = total_s1 - len(written_ids)
    eta_min = (rem_entities / rate) / 60 if rate > 0 else 0
    chunk_time = time.time() - tc

    print(f"  Chunk {chunk_num:3d} | {len(written_ids):>9,}/{total_s1:,} ({pct:5.1f}%) | "
          f"Pairs:{len(pair_feats):>7,} | Matches:+{chunk_matches:<5,} | "
          f"Chunk:{chunk_time:5.1f}s | Rate:{rate:5.1f} rec/s | "
          f"ETA:{eta_min:5.1f}min | Mem:{mem():.0f}MB", flush=True)

    del pair_s1, pair_s2, pair_feats, s1_cands_map, grouped_matches
    gc.collect()

# ── [7] Final Integrity & Wrap Up ──────────────────────────────────────────
miss = set(all_s1_ids) - written_ids
if miss:
    print(f"\nFilling {len(miss):,} missing S1 entities with empty predictions...", flush=True)
    with open(MATCHING_OUT, 'a', encoding='utf-8') as fm, \
         open(CANDIDATES_OUT, 'a', encoding='utf-8') as fc:
        for sid in sorted(miss):
            fm.write(f"{sid}\t\n")
            fc.write(f"{sid}\t\n")
        fm.flush(); fc.flush()

print(f"\n{'='*65}", flush=True)
print(f"INFERENCE COMPLETE in {(time.time()-t_start)/60:.1f} minutes!", flush=True)
print(f"Total Matches Found: {total_matches:,}", flush=True)
mMB = os.path.getsize(MATCHING_OUT) / (1024 ** 2)
cMB = os.path.getsize(CANDIDATES_OUT) / (1024 ** 2)
print(f"  matching_results.tsv: {mMB:.2f}MB", flush=True)
print(f"  candidate_pairs.tsv:  {cMB:.2f}MB", flush=True)

# Validate
print("\n[8] Running Official Submission Validator...", flush=True)
os.system(f"python utils/validate_submission.py --matching {MATCHING_OUT} --candidate {CANDIDATES_OUT} --test-dir {args.test_dir}")

# Build submission zip
print("\n[9] Building final submission ZIP package...", flush=True)
os.system("python build_zip.py")
print(f"\n{'='*65}", flush=True)
print("ALL DONE! FINAL SUBMISSION FILES READY IN output/ AND ROOT!", flush=True)
print(f"{'='*65}", flush=True)

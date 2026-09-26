"""
JARVIS_CHECKER — Fast Parquet-Based Inference
Requires: preprocess_s23.py to have been run first.
Loads S2/S3 from compact parquet files (~800MB each) instead of raw TSV.
Index built from parquet. S1 processed in chunks. Peak RAM ~6-8GB.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import os, re, time, gc, argparse, warnings
import numpy as np
import pandas as pd
import psutil
from collections import defaultdict

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--model',      default='dataset/processed/lgb_model.pkl')
parser.add_argument('--threshold',  type=float, default=None)
parser.add_argument('--output-dir', default='output')
parser.add_argument('--test-dir',   default='dataset/test')
parser.add_argument('--s1-chunk',   type=int, default=20000)
args = parser.parse_args()

from src.models.inference import load_model
from src.features.feature_extraction import FEATURE_COLS
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

def mem(): return psutil.Process(os.getpid()).memory_info().rss/1024**2

print("=" * 60)
print("JARVIS_CHECKER — PARQUET-BASED INFERENCE")
print("=" * 60)

TEST_S1   = os.path.join(args.test_dir, 'test_source1.tsv')
S2_PQFILE = 'dataset/processed/S2_normalized.parquet'
S3_PQFILE = 'dataset/processed/S3_normalized.parquet'
os.makedirs(args.output_dir, exist_ok=True)
MATCHING_OUT   = os.path.join(args.output_dir, 'matching_results.tsv')
CANDIDATES_OUT = os.path.join(args.output_dir, 'candidate_pairs.tsv')

# Check parquet files
for p in [S2_PQFILE, S3_PQFILE]:
    if not os.path.exists(p):
        print(f"ERROR: {p} not found. Run: python preprocess_s23.py first!")
        sys.exit(1)

THRESHOLD = args.threshold
if THRESHOLD is None:
    tf='dataset/processed/best_threshold.txt'
    THRESHOLD=float(open(tf).read().strip()) if os.path.exists(tf) else 0.92
print(f"  Threshold: {THRESHOLD}")

# ── [1] Load model ─────────────────────────────────────────────────────────
print(f"\n[1] Loading model | Mem:{mem():.0f}MB")
model = load_model(args.model)
print(f"  Features: {len(FEATURE_COLS)}")

# ── [2] Load S2/S3 parquet into memory ────────────────────────────────────
print(f"\n[2] Loading S2/S3 from parquet | Mem:{mem():.0f}MB")
t0=time.time()
s2=pd.read_parquet(S2_PQFILE); print(f"  S2: {len(s2):,} rows | Mem:{mem():.0f}MB")
s3=pd.read_parquet(S3_PQFILE); print(f"  S3: {len(s3):,} rows | Mem:{mem():.0f}MB")
s23=pd.concat([s2,s3],ignore_index=True); del s2,s3; gc.collect()
s23=s23[s23['entity_id'].notna() & (s23['entity_id']!='') & s23['co'].notna() & (s23['co']!='')]
print(f"  Total: {len(s23):,} | Mem:{mem():.0f}MB | Time:{time.time()-t0:.0f}s")

# Lookup dict: entity_id -> row index (use iloc for fast access)
s23=s23.reset_index(drop=True)
eid2idx={eid:i for i,eid in enumerate(s23['entity_id'])}

# ── [3] Build blocking index from parquet ──────────────────────────────────
print(f"\n[3] Building blocking index | Mem:{mem():.0f}MB")
t0=time.time()

MAX_P4=150;MAX_P3=100;MAX_P2=80;MAX_NUM=80;MAX_TOK=60;MAX_AT=40;MAX_BG=25
_GN={'road','street','near','shop','opp','cross','lane','west','east',
     'north','south','main','new','old','india','united','states','france',
     'pvt','ltd','llc','inc','corp','limited','company','services',
     'trading','enterprises','group','international','national'}
_GA={'road','street','avenue','lane','near','opp','opposite','plot',
     'house','floor','building','sector','phase','area','nagar','marg',
     'drive','place','court','boulevard','extension'}

p4=defaultdict(list);p3=defaultdict(list);p2=defaultdict(list)
anum=defaultdict(list);ntok=defaultdict(list);atok=defaultdict(list);bgram=defaultdict(list)

def _bg(name):
    s=name[:8]; return [s[i:i+2] for i in range(len(s)-1)] if len(s)>=2 else []

for row in s23.itertuples():
    eid=row.entity_id; c=row.co; nm=row.nm; ad=row.ad
    nums=str(row.nums).split(',') if row.nums else []
    pr4=row.pr4; pr3=row.pr3; pr2=row.pr2
    k4=f"{c}\x00{pr4}"; k3=f"{c}\x00{pr3}"; k2=f"{c}\x00{pr2}"
    if pr4 and len(p4[k4])<MAX_P4: p4[k4].append(eid)
    if pr3 and pr3!=pr4 and len(p3[k3])<MAX_P3: p3[k3].append(eid)
    if len(pr2)==2 and len(p2[k2])<MAX_P2: p2[k2].append(eid)
    for n in nums:
        if n and len(n)>=2:
            k=f"{c}\x00{n}"
            if len(anum[k])<MAX_NUM: anum[k].append(eid)
    for tk in [t for t in nm.split() if len(t)>=4 and t not in _GN]:
        k=f"{c}\x00{tk}"
        if len(ntok[k])<MAX_TOK: ntok[k].append(eid)
    for tk in [t for t in ad.split() if len(t)>=4 and t not in _GA]:
        k=f"{c}\x00{tk}"
        if len(atok[k])<MAX_AT: atok[k].append(eid)
    for b in _bg(nm):
        k=f"{c}\x00{b}"
        if len(bgram[k])<MAX_BG: bgram[k].append(eid)

print(f"  Index built in {time.time()-t0:.0f}s | Mem:{mem():.0f}MB")

# ── Helpers ────────────────────────────────────────────────────────────────
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

def get_cands(nm,ad,co,nums,ntok_list,atok_list):
    cands=set()
    if not co: return cands
    pr4=nm[:4]; pr3=nm[:3]; pr2=nm[:2]
    for eid in p4.get(f"{co}\x00{pr4}",[]): cands.add(eid)
    if pr3!=pr4:
        for eid in p3.get(f"{co}\x00{pr3}",[]): cands.add(eid)
    if len(pr2)==2:
        for eid in p2.get(f"{co}\x00{pr2}",[]): cands.add(eid)
    for n in nums:
        for eid in anum.get(f"{co}\x00{n}",[]): cands.add(eid)
    for tk in ntok_list:
        for eid in ntok.get(f"{co}\x00{tk}",[]): cands.add(eid)
    for tk in atok_list:
        for eid in atok.get(f"{co}\x00{tk}",[]): cands.add(eid)
    for b in _bg(nm):
        for eid in bgram.get(f"{co}\x00{b}",[]): cands.add(eid)
    if len(cands)>300: cands=set(list(cands)[:300])
    return cands

def feats(n1,a1,c1,nums1_set,n2,a2,c2,nums2_set,eid2):
    s1n=set(n1.split());s2n=set(n2.split())
    s1a=set(a1.split());s2a=set(a2.split())
    un=s1n|s2n;ua=s1a|s2a;unum=nums1_set|nums2_set
    tj_n=len(s1n&s2n)/len(un) if un else 0.
    tj_a=len(s1a&s2a)/len(ua) if ua else 0.
    inter=len(s1n&s2n)
    tc=max(inter/len(s1n),inter/len(s2n)) if s1n and s2n else 0.
    nj=len(nums1_set&nums2_set)/len(unum) if unum else 0.
    ng1={n1[i:i+3] for i in range(len(n1)-2)} if len(n1)>=3 else set()
    ng2={n2[i:i+3] for i in range(len(n2)-2)} if len(n2)>=3 else set()
    ung=ng1|ng2; ng_ov=len(ng1&ng2)/len(ung) if ung else 0.
    ln1=len(n1);ln2=len(n2);la1=len(a1);la2=len(a2)
    return {
        'name_exact':1. if n1 and n1==n2 else 0.,
        'name_ratio':fuzz.ratio(n1,n2)/100. if n1 and n2 else 0.,
        'name_partial':fuzz.partial_ratio(n1,n2)/100. if n1 and n2 else 0.,
        'name_token_set':fuzz.token_set_ratio(n1,n2)/100. if n1 and n2 else 0.,
        'name_token_sort':fuzz.token_sort_ratio(n1,n2)/100. if n1 and n2 else 0.,
        'name_jw':JaroWinkler.normalized_similarity(n1,n2) if n1 and n2 else 0.,
        'name_token_jaccard':tj_n,'name_token_containment':tc,'name_is_abbrev':0.,
        'name_len_diff':abs(ln1-ln2)/(max(ln1,ln2)+1),
        'name_prefix4_match':1. if ln1>=4 and ln2>=4 and n1[:4]==n2[:4] else 0.,
        'name_prefix3_match':1. if ln1>=3 and ln2>=3 and n1[:3]==n2[:3] else 0.,
        'name_both_present':1. if n1 and n2 else 0.,
        'name_levenshtein_norm':1.-Levenshtein.normalized_similarity(n1,n2) if n1 and n2 else 0.,
        'name_char3gram_overlap':ng_ov,
        'name_suffix_match':1. if ln1>=4 and ln2>=4 and n1[-4:]==n2[-4:] else 0.,
        'name_mono_token':1. if (n1 and len(n1.split())==1) or (n2 and len(n2.split())==1) else 0.,
        'addr_exact':1. if a1 and a1==a2 else 0.,
        'addr_ratio':fuzz.ratio(a1,a2)/100. if a1 and a2 else 0.,
        'addr_partial':fuzz.partial_ratio(a1,a2)/100. if a1 and a2 else 0.,
        'addr_token_set':fuzz.token_set_ratio(a1,a2)/100. if a1 and a2 else 0.,
        'addr_token_sort':fuzz.token_sort_ratio(a1,a2)/100. if a1 and a2 else 0.,
        'addr_jw':JaroWinkler.normalized_similarity(a1,a2) if a1 and a2 else 0.,
        'addr_token_jaccard':tj_a,'addr_num_overlap':nj,
        'addr_num_exact':1. if nums1_set and nums1_set&nums2_set else 0.,
        'addr_len_diff':abs(la1-la2)/(max(la1,la2)+1) if (a1 or a2) else 0.,
        'addr_both_present':1. if a1 and a2 else 0.,
        'both_have_numbers':1. if nums1_set and nums2_set else 0.,
        'numbers_exactly_match':1. if nums1_set and nums1_set==nums2_set else 0.,
        'country_match':1. if c1 and c1==c2 else 0.,
        'country_either_empty':1. if not c1 or not c2 else 0.,
        'source_is_s2':1. if str(eid2).startswith('S2') else 0.,
        'cross_script':0.,'name_suffix_match_struct':1. if ln1>=4 and ln2>=4 and n1[-4:]==n2[-4:] else 0.,
    }

# ── [4] Load all S1 IDs ────────────────────────────────────────────────────
print(f"\n[4] Loading S1 | Mem:{mem():.0f}MB")
all_s1_ids=pd.read_csv(TEST_S1,sep='\t',dtype=str,usecols=['entity_id'],
    keep_default_na=False,on_bad_lines='skip')['entity_id'].tolist()
total_s1=len(all_s1_ids)
print(f"  Total S1: {total_s1:,}")

# ── [5] Process S1 chunks ──────────────────────────────────────────────────
print(f"\n[5] Processing S1 | chunk={args.s1_chunk:,} | Mem:{mem():.0f}MB")
with open(MATCHING_OUT,'w',encoding='utf-8') as f: f.write("source1_entity_id\tmatched_entity_ids\n")
with open(CANDIDATES_OUT,'w',encoding='utf-8') as f: f.write("source1_entity_id\tcandidate_entity_ids\n")

written=set(); tot_m=0; stats={'0':0,'1':0,'m':0}
t_start=time.time()

for cn,chunk in enumerate(pd.read_csv(TEST_S1,sep='\t',dtype=str,
        keep_default_na=False,on_bad_lines='skip',chunksize=args.s1_chunk)):
    if 'entity_id' not in chunk.columns: continue
    tc=time.time()
    for c in ['business_name','business_address','country']:
        if c not in chunk.columns: chunk[c]=''

    pairs=[]; s1_cands={}
    for _,row in chunk.fillna('').iterrows():
        sid=str(row['entity_id'])
        nm=nname(row['business_name'])
        ad=naddr(row['business_address'])
        co=nco(row['country'])
        nums=gnums(row.get('business_address',''))
        nums_set=set(nums)
        ntok_list=[t for t in nm.split() if len(t)>=4 and t not in _GN]
        atok_list=[t for t in ad.split() if len(t)>=4 and t not in _GA]
        cands=get_cands(nm,ad,co,nums,ntok_list,atok_list)
        s1_cands[sid]=cands
        for cid in cands:
            idx=eid2idx.get(cid)
            if idx is None: continue
            r2=s23.iloc[idx]
            n2=str(r2['nm']); a2=str(r2['ad']); c2=str(r2['co'])
            nums2=set(str(r2['nums']).split(',')) if r2['nums'] else set()
            f=feats(nm,ad,co,nums_set,n2,a2,c2,nums2,cid)
            f['s1id']=sid; f['s2id']=cid
            pairs.append(f)

    grouped={}
    if pairs:
        fdf=pd.DataFrame(pairs)
        for col in FEATURE_COLS:
            if col not in fdf.columns: fdf[col]=0.
        probs=model.predict_proba(fdf[FEATURE_COLS].fillna(0).values)[:,1]
        fdf['prob']=probs
        hit=fdf[fdf['prob']>=THRESHOLD]
        grouped=hit.groupby('s1id')['s2id'].apply(list).to_dict()

    ml,cl=[],[]
    for _,row in chunk.fillna('').iterrows():
        sid=str(row['entity_id']); written.add(sid)
        mlist=list(dict.fromkeys(m for m in grouped.get(sid,[]) if not str(m).startswith('S1')))
        mc=len(mlist); tot_m+=mc
        if mc==0: stats['0']+=1
        elif mc==1: stats['1']+=1
        else: stats['m']+=1
        ml.append(f"{sid}\t{','.join(mlist)}\n")
        cl.append(f"{sid}\t{','.join(sorted(s1_cands.get(sid,set())))}\n")

    with open(MATCHING_OUT,'a',encoding='utf-8') as f: f.writelines(ml)
    with open(CANDIDATES_OUT,'a',encoding='utf-8') as f: f.writelines(cl)

    pct=100*len(written)/total_s1
    elapsed=time.time()-t_start
    rate=len(written)/elapsed if elapsed>0 else 1
    eta=(total_s1-len(written))/rate/60
    print(f"  Chunk {cn+1:4d} | {len(written):>9,}/{total_s1:,} ({pct:.1f}%) "
          f"| {len(pairs):,} pairs | {time.time()-tc:.1f}s | ETA:{eta:.0f}min | Mem:{mem():.0f}MB")
    del pairs; gc.collect()

miss=set(all_s1_ids)-written
if miss:
    with open(MATCHING_OUT,'a',encoding='utf-8') as fm, \
         open(CANDIDATES_OUT,'a',encoding='utf-8') as fc:
        for sid in sorted(miss): fm.write(f"{sid}\t\n"); fc.write(f"{sid}\t\n")

rt=time.time()-t_start
print(f"\n{'='*60}")
print(f"DONE in {rt:.0f}s ({rt/60:.1f}min) | Matches={tot_m:,}")
mMB=os.path.getsize(MATCHING_OUT)/1024**2
cMB=os.path.getsize(CANDIDATES_OUT)/1024**2
print(f"matching_results.tsv: {mMB:.2f}MB | candidate_pairs.tsv: {cMB:.2f}MB")

# Integrity
print("\n[6] Integrity check...")
mdf=pd.read_csv(MATCHING_OUT,sep='\t',dtype=str,keep_default_na=False)
ok=(len(mdf)==total_s1 and mdf['source1_entity_id'].nunique()==len(mdf)
    and set(mdf['source1_entity_id'])==set(all_s1_ids))
inv=sum(1 for ms in mdf['matched_entity_ids'] if ms
        for m in ms.split(',') if m and not(m.startswith('S2') or m.startswith('S3')))
ok=ok and inv==0
if ok:
    print("  ALL CHECKS PASSED")
    print(f"\n{'='*60}")
    print("  SUBMISSION FILES READY!")
    print(f"  Upload to leaderboard: {os.path.abspath(MATCHING_OUT)}")
    print(f"  Then run: python build_zip.py")
    print(f"{'='*60}")
else:
    print(f"  FAIL rows={len(mdf)} ok={ok} inv={inv}"); sys.exit(1)

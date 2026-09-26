"""
JARVIS_CHECKER — Step 1: Pre-process S2/S3 to parquet
Normalizes all S2/S3 records and saves 6 key columns to a compressed parquet.
Run once — takes ~10 min, saves ~800MB on disk.
After this, infer.py loads from parquet (much faster, controlled RAM).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
import os, re, time, gc
import pandas as pd
import psutil

def mem(): return psutil.Process(os.getpid()).memory_info().rss/1024**2

_LEGAL = re.compile(
    r'\b(private limited|pvt\.? ltd\.?|llp|llc|ltd\.?|limited|inc\.?|corp\.?|'
    r'corporation|co\.?|company|gmbh|ag|sa|sas|srl|bv|nv|plc|pty ltd|pty|'
    r'l\.?p\.?|s\.a\.?|s\.r\.l\.|sarl|sasu|eurl)\s*$', re.IGNORECASE)
_MULTI=re.compile(r'\s+'); _PUNCT=re.compile(r'[^\w\s]'); _AND=re.compile(r'\s*&\s*',re.IGNORECASE)
_CMAP={'india':'india','in':'india','ind':'india',
       'united states':'united states','us':'united states','usa':'united states',
       'u.s.a.':'united states','u.s.':'united states','united states of america':'united states',
       'france':'france','fr':'france',
       'united kingdom':'united kingdom','uk':'united kingdom','gb':'united kingdom',
       'germany':'germany','de':'germany','australia':'australia','au':'australia',
       'canada':'canada','ca':'canada'}

def norm_name(s):
    s=s.fillna('').astype(str).str.lower()
    s=s.str.replace(_AND,' and ',regex=True)
    s=s.str.replace(_LEGAL,'',regex=True)
    s=s.str.replace(_PUNCT,' ',regex=True)
    return s.str.replace(_MULTI,' ',regex=True).str.strip()

def norm_addr(s):
    s=s.fillna('').astype(str).str.lower()
    s=s.str.replace(_PUNCT,' ',regex=True)
    return s.str.replace(_MULTI,' ',regex=True).str.strip()

def norm_co(s):
    s=s.fillna('').astype(str).str.lower().str.strip()
    s=s.str.replace(_PUNCT,'',regex=True).str.replace(_MULTI,' ',regex=True).str.strip()
    return s.map(lambda x: _CMAP.get(x,x))

def extract_nums(s):
    nums=[n for n in re.findall(r'\d+',str(s)) if len(n)>=2]
    return ','.join(nums) if nums else ''

TEST_DIR = 'dataset/test'
OUT_DIR  = 'dataset/processed'
os.makedirs(OUT_DIR, exist_ok=True)

for src_file, tag in [('test_source2.tsv','S2'),('test_source3.tsv','S3')]:
    path    = os.path.join(TEST_DIR, src_file)
    outpath = os.path.join(OUT_DIR, f'{tag}_normalized.parquet')
    if os.path.exists(outpath):
        sz = os.path.getsize(outpath)/1024**2
        print(f"  {tag}: already exists ({sz:.0f}MB) — skipping")
        continue

    print(f"\nProcessing {tag} | Mem:{mem():.0f}MB")
    t0=time.time(); chunks=[]
    for i,df in enumerate(pd.read_csv(path,sep='\t',dtype=str,chunksize=500000,
                           keep_default_na=False,on_bad_lines='skip')):
        for c in ['entity_id','business_name','business_address','country']:
            if c not in df.columns: df[c]=''
        out=pd.DataFrame()
        out['entity_id'] = df['entity_id'].fillna('').astype(str)
        out['nm']  = norm_name(df['business_name'])
        out['ad']  = norm_addr(df['business_address'])
        out['co']  = norm_co(df['country'])
        out['nums']= df['business_address'].map(extract_nums)
        out['pr4'] = out['nm'].str[:4]
        out['pr3'] = out['nm'].str[:3]
        out['pr2'] = out['nm'].str[:2]
        chunks.append(out)
        print(f"  {tag} chunk {i+1}: {len(df):,} rows | Mem:{mem():.0f}MB")
        gc.collect()

    full=pd.concat(chunks,ignore_index=True)
    full.to_parquet(outpath,index=False,compression='snappy')
    sz=os.path.getsize(outpath)/1024**2
    print(f"  {tag}: {len(full):,} rows saved → {outpath} ({sz:.0f}MB) in {time.time()-t0:.0f}s")
    del full,chunks; gc.collect()

print("\nDone! Now run: python infer.py")

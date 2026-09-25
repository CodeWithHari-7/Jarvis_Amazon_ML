import sys, sqlite3, time, os, psutil, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import polars as pl
from indic_transliteration import sanscript
import re
import pandas as pd

sys.path.append(r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker")
from src.cleaning.data_cleaning import clean_dataset
from src.normalization.normalization import normalize_dataset
from src.features.feature_extraction import extract_features_for_pair
from src.models.inference import load_model, predict

db_path = r"dataset\pipeline_cache\s23_test.db"
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Test 5,000 S1 records
t0 = time.time()
test_s1_path = r"dataset\test\test_source1.tsv"
s1_test_sample = pl.read_csv(test_s1_path, separator='\t', ignore_errors=True, n_rows=5000)
s1_norm = normalize_dataset(clean_dataset(s1_test_sample)).to_dicts()

re_dev = re.compile(r'[\u0900-\u097F]')
re_guj = re.compile(r'[\u0A80-\u0AFF]')
re_tel = re.compile(r'[\u0C00-\u0C7F]')
re_clean = re.compile(r'[^a-z0-9\s]')

def translit_text(text: str) -> str:
    if not text: return ""
    s = str(text)
    try:
        if re_dev.search(s): s = sanscript.transliterate(s, sanscript.DEVANAGARI, sanscript.ITRANS)
        if re_guj.search(s): s = sanscript.transliterate(s, sanscript.GUJARATI, sanscript.ITRANS)
        if re_tel.search(s): s = sanscript.transliterate(s, sanscript.TELUGU, sanscript.ITRANS)
    except Exception: pass
    s = s.lower()
    s = re_clean.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

model = load_model(r"dataset\processed\lgb_model.pkl")
feat_cols = ['name_fuzz_ratio', 'name_token_set', 'name_jw', 
             'addr_fuzz_ratio', 'addr_token_set', 'addr_num_overlap', 'cross_script']

total_cands = 0
matches_found = 0

for r1 in s1_norm:
    s1_id = r1['entity_id']
    c = r1['country_normalized']
    n = r1['business_name_normalized']
    indic = r1['is_indic']
    
    cand_eids = set()
    if c and n:
        cur.execute("SELECT eid FROM blocks WHERE k = ? LIMIT 200", (f"P_{c}_{n[:4]}",))
        for row in cur.fetchall(): cand_eids.add(row[0])
        
        nt = translit_text(n) if indic else n
        if nt:
            cur.execute("SELECT eid FROM blocks WHERE k = ? LIMIT 200", (f"T_{c}_{nt[:4]}",))
            for row in cur.fetchall(): cand_eids.add(row[0])
            
    for num in r1['business_address_numbers']:
        if c and len(num) >= 2:
            cur.execute("SELECT eid FROM blocks WHERE k = ? LIMIT 200", (f"A_{c}_{num}",))
            for row in cur.fetchall(): cand_eids.add(row[0])
            
    total_cands += len(cand_eids)
    if not cand_eids: continue
    
    # Batch fetch candidate attributes
    placeholders = ",".join("?" * len(cand_eids))
    cur.execute(f"SELECT eid, name, addr, nums, is_indic FROM s23 WHERE eid IN ({placeholders})", list(cand_eids))
    cand_rows = cur.fetchall()
    
    feats = []
    cand_ids = []
    for m, m_name, m_addr, m_nums_str, m_indic in cand_rows:
        r2 = {
            'business_name_normalized': m_name,
            'business_address_normalized': m_addr,
            'business_address_numbers': m_nums_str.split(',') if m_nums_str else [],
            'is_indic': bool(m_indic)
        }
        feat = extract_features_for_pair(r1, r2)
        feats.append(feat)
        cand_ids.append(m)
        
    feat_df = pd.DataFrame(feats)
    probs = predict(model, feat_df, feat_cols)
    for cid, prob in zip(cand_ids, probs):
        if prob >= 0.89:
            matches_found += 1

t_total = time.time() - t0
print(f"Processed 5,000 S1 records in {t_total:.2f}s | Candidates: {total_cands:,} | Matches: {matches_found} | RAM: {psutil.Process(os.getpid()).memory_info().rss/1024**2:.1f} MB")
conn.close()

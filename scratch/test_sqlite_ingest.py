import sys, sqlite3, time, os, psutil, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import polars as pl
from indic_transliteration import sanscript
import re

db_path = r"dataset\pipeline_cache\s23_test.db"
if os.path.exists(db_path): os.remove(db_path)

conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("PRAGMA synchronous = OFF")
cur.execute("PRAGMA journal_mode = OFF")
cur.execute("PRAGMA cache_size = -64000") # 64MB cache
cur.execute("CREATE TABLE s23 (eid TEXT PRIMARY KEY, name TEXT, addr TEXT, nums TEXT, is_indic INT)")
cur.execute("CREATE TABLE blocks (k TEXT, eid TEXT)")
conn.commit()

sys.path.append(r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker")
from src.cleaning.data_cleaning import clean_dataset
from src.normalization.normalization import normalize_dataset

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

t0 = time.time()
test_s2_path = r"dataset\test\test_source2.tsv"
s2_df = pl.read_csv(test_s2_path, separator='\t', ignore_errors=True, schema_overrides={'entity_id': pl.Utf8}, n_rows=500000)
s2_norm = normalize_dataset(clean_dataset(s2_df)).select([
    'entity_id', 'business_name_normalized', 'business_address_normalized',
    'business_address_numbers', 'country_normalized', 'is_indic'
])

eids = s2_norm['entity_id'].to_list()
names = s2_norm['business_name_normalized'].to_list()
addrs = s2_norm['business_address_normalized'].to_list()
nums_list = s2_norm['business_address_numbers'].to_list()
countries = s2_norm['country_normalized'].to_list()
is_indics = s2_norm['is_indic'].to_list()

records = []
block_records = []

for i in range(len(eids)):
    eid = eids[i]
    c = countries[i]
    n = names[i]
    indic = 1 if is_indics[i] else 0
    nums_str = ",".join(nums_list[i])
    records.append((eid, n, addrs[i], nums_str, indic))
    
    if c and n:
        block_records.append((f"P_{c}_{n[:4]}", eid))
        if indic:
            nt = translit_text(n)
            if nt: block_records.append((f"T_{c}_{nt[:4]}", eid))
        else:
            block_records.append((f"T_{c}_{n[:4]}", eid))
            
    for num in nums_list[i]:
        if c and len(num) >= 2:
            block_records.append((f"A_{c}_{num}", eid))

cur.executemany("INSERT OR IGNORE INTO s23 VALUES (?, ?, ?, ?, ?)", records)
cur.executemany("INSERT INTO blocks VALUES (?, ?)", block_records)
conn.commit()

t_insert = time.time() - t0
print(f"Ingested 500k rows + {len(block_records):,} block pairs into SQLite in {t_insert:.2f}s | DB size: {os.path.getsize(db_path)/1024**2:.1f} MB | RAM: {psutil.Process(os.getpid()).memory_info().rss/1024**2:.1f} MB")

t0 = time.time()
cur.execute("CREATE INDEX idx_blocks_k ON blocks(k)")
conn.commit()
print(f"Indexed in {time.time()-t0:.2f}s")

conn.close()
os.remove(db_path)

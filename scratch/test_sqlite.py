import sys, sqlite3, time, os, psutil, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import polars as pl

db_path = r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker\scratch\test.db"
if os.path.exists(db_path): os.remove(db_path)

conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("PRAGMA synchronous = OFF")
cur.execute("PRAGMA journal_mode = OFF")
cur.execute("CREATE TABLE s23 (eid TEXT, name TEXT, addr TEXT, nums TEXT, is_indic INT)")
cur.execute("CREATE TABLE idx_prefix (k TEXT, eid TEXT)")
cur.execute("CREATE TABLE idx_addr (k TEXT, eid TEXT)")

t0 = time.time()
# Test inserting 100k rows
records = [(f"S2-{i}", f"Company {i}", f"Address {i}", "123", 0) for i in range(100000)]
pfx_records = [(f"in_comp", f"S2-{i}") for i in range(100000)]

cur.executemany("INSERT INTO s23 VALUES (?, ?, ?, ?, ?)", records)
cur.executemany("INSERT INTO idx_prefix VALUES (?, ?)", pfx_records)
conn.commit()

print(f"Inserted 100k rows in {time.time()-t0:.2f}s")

t0 = time.time()
cur.execute("CREATE INDEX idx_k ON idx_prefix(k)")
conn.commit()
print(f"Indexed in {time.time()-t0:.2f}s")

t0 = time.time()
cur.execute("SELECT eid FROM idx_prefix WHERE k = ? LIMIT 500", ("in_comp",))
res = cur.fetchall()
print(f"Queried 500 rows in {time.time()-t0:.4f}s. Result count: {len(res)}")

conn.close()
os.remove(db_path)

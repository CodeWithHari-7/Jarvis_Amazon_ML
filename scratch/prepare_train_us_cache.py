import os
import sys
import time
import multiprocessing as mp
import polars as pl

sys.path.append('code/business_entity_resolution/src')
from blocking import normalize_record

def process_chunk(chunk_rows):
    res = []
    for r in chunk_rows:
        eid = r['entity_id']
        rec = normalize_record(r['business_name'], r['business_address'], r['country'])
        # Store essential fields as strings/lists for fast parquet serialization
        res.append({
            'entity_id': eid,
            'norm_name': rec['norm_name'],
            'core_name': rec['core_name'],
            'prefix4': rec['prefix4'],
            'prefix5': rec['prefix5'],
            'sorted_tokens': rec['sorted_tokens'],
            'distinct_tokens': list(rec['distinct_tokens']),
            'collapsed_tokens': list(rec['collapsed_tokens']),
            'norm_addr': rec['norm_addr'],
            'nums': list(rec['nums']),
            'key_nums': list(rec['key_nums'])
        })
    return res

def main():
    print("=" * 80)
    print("PREPARING PARQUET CACHE FOR TRAIN US S2+S3 (FAST MULTIPROCESSING)")
    print("=" * 80)
    t0 = time.time()
    
    os.makedirs("train_cache", exist_ok=True)
    out_path = "train_cache/s23_us_normalized.parquet"
    if os.path.exists(out_path):
        print(f"Cache already exists at {out_path}.")
        return

    print("Loading raw Train S2 and S3 for US...")
    s2 = pl.read_csv("dataset/train/train_source2.tsv", separator='\t').filter(pl.col('country') == 'US')
    s3 = pl.read_csv("dataset/train/train_source3.tsv", separator='\t').filter(pl.col('country') == 'US')
    s23 = pl.concat([s2, s3]).unique(subset=['entity_id'])
    del s2, s3
    n_total = len(s23)
    print(f"Loaded {n_total:,} US records in {time.time()-t0:.1f}s.")

    rows = s23.to_dicts()
    del s23

    n_cpus = max(1, mp.cpu_count() - 1)
    chunk_size = (n_total + n_cpus - 1) // n_cpus
    chunks = [rows[i:i + chunk_size] for i in range(0, n_total, chunk_size)]
    print(f"Processing in parallel across {n_cpus} workers (chunk size: {chunk_size:,})...")

    t_mp = time.time()
    with mp.Pool(n_cpus) as pool:
        chunk_results = pool.map(process_chunk, chunks)

    print(f"Multiprocessing complete in {time.time()-t_mp:.1f}s. Flattening...")
    all_recs = []
    for cr in chunk_results:
        all_recs.extend(cr)
    del chunk_results

    print(f"Converting {len(all_recs):,} records to Polars DataFrame...")
    df = pl.DataFrame(all_recs)
    del all_recs
    df.write_parquet(out_path, compression="zstd")
    print(f"Saved normalized cache to {out_path} ({os.path.getsize(out_path)/(1024*1024):.1f} MB) in {time.time()-t0:.1f}s.")

if __name__ == '__main__':
    mp.freeze_support()
    main()

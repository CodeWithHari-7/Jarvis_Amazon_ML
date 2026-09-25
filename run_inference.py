import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os, time, gc, psutil
import pandas as pd
import polars as pl
from collections import defaultdict

from src.cleaning.data_cleaning import clean_dataset
from src.normalization.normalization import normalize_dataset
from src.features.feature_extraction import extract_features_for_pair
from src.models.inference import load_model, predict

def get_memory():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

print("=" * 70)
print("JARVIS_CHECKER — STAGE 8B (FULL TEST INFERENCE)")
print("=" * 70)

test_s1_path = r"dataset\test\test_source1.tsv"
test_s2_path = r"dataset\test\test_source2.tsv"
test_s3_path = r"dataset\test\test_source3.tsv"
out_path = "matching_results.tsv"
model_path = r"dataset\processed\lgb_model.pkl"
threshold = 0.89

print(f"\n[1] Loading and Normalizing S2 & S3 ... Memory: {get_memory():.1f} MB")
t0 = time.time()
s2_df = pl.read_csv(test_s2_path, separator='\t', ignore_errors=True, schema_overrides={'entity_id': pl.Utf8})
s2_norm = normalize_dataset(clean_dataset(s2_df)).select([
    'entity_id', 'business_name_normalized', 'business_address_normalized',
    'business_address_numbers', 'country_normalized', 'is_indic'
])
del s2_df; gc.collect()

s3_df = pl.read_csv(test_s3_path, separator='\t', ignore_errors=True, schema_overrides={'entity_id': pl.Utf8})
s3_norm = normalize_dataset(clean_dataset(s3_df)).select([
    'entity_id', 'business_name_normalized', 'business_address_normalized',
    'business_address_numbers', 'country_normalized', 'is_indic'
])
del s3_df; gc.collect()

s23_norm = pl.concat([s2_norm, s3_norm], how="align")
del s2_norm, s3_norm; gc.collect()

s23_records = s23_norm.to_dicts()
del s23_norm; gc.collect()

# We need random access to s23_records. Build a dict.
s23_lookup = {r['entity_id']: r for r in s23_records}
print(f"  Loaded {len(s23_lookup):,} S2/S3 records in {time.time()-t0:.2f}s. Memory: {get_memory():.1f} MB")

print("\n[2] Building Inverted Indices ...")
t0 = time.time()
prefix_idx = defaultdict(list)
addr_num_idx = defaultdict(list)

for r in s23_records:
    eid = r['entity_id']
    c = r.get('country_normalized', '')
    
    n = r.get('business_name_normalized', '')
    if c and n:
        prefix_idx[f"{c}_{n[:4]}"].append(eid)
        
    nums = r.get('business_address_numbers', [])
    for num in nums:
        if c:
            addr_num_idx[f"{c}_{num}"].append(eid)

print(f"  Built indices in {time.time()-t0:.2f}s. Memory: {get_memory():.1f} MB")

print("\n[3] Loading ML Model ...")
model = load_model(model_path)
features_list = ['name_fuzz_ratio', 'name_token_set', 'name_jw', 
                 'addr_fuzz_ratio', 'addr_token_set', 'addr_num_overlap', 'cross_script']

print("\n[4] Processing S1 in Chunks ...")
chunksize = 250000
total_s1 = 0
total_cands = 0
total_matches = 0

stats = {
    '0_match': 0,
    '1_match': 0,
    'multi_match': 0,
    'max_match': 0,
    's2_matches': 0,
    's3_matches': 0
}

# Write header
with open(out_path, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")

# Initialize timing for chunks
start_time = time.time()

for chunk_num, chunk_df in enumerate(pd.read_csv(test_s1_path, sep='\t', dtype=str, chunksize=chunksize, on_bad_lines='skip')):
    t_chunk = time.time()
    s1_df = pl.from_pandas(chunk_df).fill_null("")
    s1_norm = normalize_dataset(clean_dataset(s1_df)).to_dicts()
    
    chunk_s1_count = len(s1_norm)
    total_s1 += chunk_s1_count
    
    candidates = [] # list of tuples (s1_id, r1, s23_id)
    # Blocking
    for r in s1_norm:
        s1_id = r['entity_id']
        c = r.get('country_normalized', '')
        cands_set = set()
        
        n = r.get('business_name_normalized', '')
        if c and n:
            key = f"{c}_{n[:4]}"
            for m in prefix_idx.get(key, []): cands_set.add(m)
            
        nums = r.get('business_address_numbers', [])
        for num in nums:
            if c:
                key = f"{c}_{num}"
                for m in addr_num_idx.get(key, []): cands_set.add(m)
        
        for m in cands_set:
            candidates.append((s1_id, r, m))
            
    total_cands += len(candidates)
    
    if not candidates:
        # No candidates for any S1 in this chunk (unlikely but possible)
        lines = [f"{r['entity_id']}\t\n" for r in s1_norm]
        with open(out_path, "a", encoding="utf-8") as f:
            f.writelines(lines)
        stats['0_match'] += chunk_s1_count
        continue

    # Feature extraction
    features = []
    for s1_id, r1, src_id in candidates:
        r2 = s23_lookup.get(src_id)
        if not r2: continue
        feat = extract_features_for_pair(r1, r2)
        feat['source1_entity_id'] = s1_id
        feat['source_entity_id'] = src_id
        features.append(feat)
        
    feat_df = pd.DataFrame(features)
    
    # Inference
    probs = predict(model, feat_df, features_list)
    feat_df['prob'] = probs
    
    # Thresholding
    matches = feat_df[feat_df['prob'] >= threshold]
    
    # Grouping
    grouped = matches.groupby('source1_entity_id')['source_entity_id'].apply(list).to_dict()
    
    # Output formatting & Stats
    lines = []
    for r in s1_norm:
        s1_id = r['entity_id']
        m_list = grouped.get(s1_id, [])
        match_count = len(m_list)
        total_matches += match_count
        
        if match_count == 0:
            stats['0_match'] += 1
            lines.append(f"{s1_id}\t\n")
        else:
            if match_count == 1: stats['1_match'] += 1
            else: stats['multi_match'] += 1
            if match_count > stats['max_match']: stats['max_match'] = match_count
            
            for m in m_list:
                if str(m).startswith('S2'): stats['s2_matches'] += 1
                elif str(m).startswith('S3'): stats['s3_matches'] += 1
                
            lines.append(f"{s1_id}\t{','.join(m_list)}\n")
            
    with open(out_path, "a", encoding="utf-8") as f:
        f.writelines(lines)
        
    print(f"  Processed {total_s1:>9,} S1 entities... (Chunk Cands: {len(candidates):>8,} | Time: {time.time()-t_chunk:>5.1f}s | Mem: {get_memory():.1f}MB)")
    
    # Cleanup
    del s1_df, s1_norm, candidates, features, feat_df, matches, grouped
    gc.collect()

print("\n" + "=" * 70)
print("POST-RUN STATISTICS")
print("=" * 70)
print(f"Total S1 entities processed    : {total_s1:,}")
print(f"Total candidate pairs generated: {total_cands:,}")
print(f"Total predicted matches        : {total_matches:,}")
print(f"Average matches per S1         : {total_matches/total_s1:.4f}")
print(f"Maximum matches per S1         : {stats['max_match']:,}")
print("-" * 30)
print(f"S1 with 0 matches              : {stats['0_match']:,} ({(stats['0_match']/total_s1)*100:.1f}%)")
print(f"S1 with 1 match                : {stats['1_match']:,} ({(stats['1_match']/total_s1)*100:.1f}%)")
print(f"S1 with >1 matches             : {stats['multi_match']:,} ({(stats['multi_match']/total_s1)*100:.1f}%)")
print("-" * 30)
print(f"Matches to S2                  : {stats['s2_matches']:,}")
print(f"Matches to S3                  : {stats['s3_matches']:,}")
print(f"Output File Size               : {os.path.getsize(out_path)/1024**2:.2f} MB")
print(f"Total Runtime                  : {time.time()-start_time:.1f} seconds")
print("=== FULL TEST INFERENCE COMPLETE ===")

import polars as pl
import pandas as pd
import numpy as np
import time
import os
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

base_dir = r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker\dataset"

print("--- 1. LOADING DATA SAMPLE & GROUND TRUTH ---")
start = time.time()
gt_df = pd.read_csv(os.path.join(base_dir, "train", "train_ground_truth.tsv"), sep='\t', dtype=str)
sample_gt = gt_df[gt_df['matched_entity_ids'].notna()].sample(5000, random_state=42)
s1_ids = set(sample_gt['source1_entity_id'])

target_s23 = set()
for _, row in sample_gt.iterrows():
    for m in str(row['matched_entity_ids']).split(','):
        target_s23.add(m)

# Load S1 sample
s1_chunks = []
for chunk in pd.read_csv(os.path.join(base_dir, "train", "train_source1.tsv"), sep='\t', dtype=str, chunksize=100000):
    s1_chunks.append(chunk[chunk['entity_id'].isin(s1_ids)])
s1_df = pd.concat(s1_chunks)

# Load S2/S3 sample (targets + some noise)
s2_chunks = []
s2_noise_collected = 0
for chunk in pd.read_csv(os.path.join(base_dir, "train", "train_source2.tsv"), sep='\t', dtype=str, chunksize=100000):
    s2_chunks.append(chunk[chunk['entity_id'].isin(target_s23)])
    if s2_noise_collected < 50000:
        noise = chunk[~chunk['entity_id'].isin(target_s23)].head(50000 - s2_noise_collected)
        s2_chunks.append(noise)
        s2_noise_collected += len(noise)
s2_df = pd.concat(s2_chunks)

s3_chunks = []
s3_noise_collected = 0
for chunk in pd.read_csv(os.path.join(base_dir, "train", "train_source3.tsv"), sep='\t', dtype=str, chunksize=100000):
    s3_chunks.append(chunk[chunk['entity_id'].isin(target_s23)])
    if s3_noise_collected < 50000:
        noise = chunk[~chunk['entity_id'].isin(target_s23)].head(50000 - s3_noise_collected)
        s3_chunks.append(noise)
        s3_noise_collected += len(noise)
s3_df = pd.concat(s3_chunks)

s23_df = pd.concat([s2_df, s3_df]).drop_duplicates(subset=['entity_id'])

# Normalize logic
def norm(s): return re.sub(r"[^a-z0-9\s\u0900-\u097F\u0A80-\u0AFF\u0C00-\u0C7F]", " ", str(s).lower()).strip()
def extract_nums(s): return re.findall(r"\d+", str(s))

s1_df['norm_name'] = s1_df['business_name'].apply(norm)
s1_df['country_norm'] = s1_df['country'].apply(norm)
s1_df['addr_nums'] = s1_df['business_address'].apply(extract_nums)

s23_df['norm_name'] = s23_df['business_name'].apply(norm)
s23_df['country_norm'] = s23_df['country'].apply(norm)
s23_df['addr_nums'] = s23_df['business_address'].apply(extract_nums)

print(f"Loaded {len(s1_df)} S1 entities and {len(s23_df)} S2/S3 candidates in {time.time()-start:.2f}s.")

# Build ground truth map
true_pairs = set()
for _, row in sample_gt.iterrows():
    if row['source1_entity_id'] in s1_df['entity_id'].values:
        for m in str(row['matched_entity_ids']).split(','):
            if m in s23_df['entity_id'].values:
                true_pairs.add((row['source1_entity_id'], m))
total_true = len(true_pairs)
print(f"Total True Pairs in sample universe: {total_true}")

s1_list = s1_df.to_dict('records')
s23_list = s23_df.to_dict('records')

results = []

def run_inverted_index(name, key_func):
    start_t = time.time()
    idx = defaultdict(list)
    for r in s23_list:
        keys = key_func(r)
        if not isinstance(keys, list): keys = [keys]
        for k in keys:
            if k: idx[k].append(r['entity_id'])
            
    cands_per_s1 = []
    found = 0
    total_cands = 0
    
    for r1 in s1_list:
        keys = key_func(r1)
        if not isinstance(keys, list): keys = [keys]
        cands = set()
        for k in keys:
            if k in idx:
                cands.update(idx[k])
        cands_per_s1.append(len(cands))
        total_cands += len(cands)
        for c in cands:
            if (r1['entity_id'], c) in true_pairs:
                found += 1
                
    run_t = time.time() - start_t
    recall = found / total_true if total_true else 0
    rr = 1 - (total_cands / (len(s1_list) * len(s23_list)))
    p95 = np.percentile(cands_per_s1, 95) if cands_per_s1 else 0
    
    results.append({
        'Method': name,
        'Recall': f"{recall*100:.2f}%",
        'Cands': total_cands,
        'Avg Cands': np.mean(cands_per_s1),
        'P95 Cands': p95,
        'Reduction': f"{rr*100:.4f}%",
        'Time (s)': f"{run_t:.2f}"
    })
    print(f"Finished {name}: Recall={recall*100:.2f}% | Cands={total_cands} | Time={run_t:.2f}s")

print("\n--- 2. RUNNING BLOCKING EXPERIMENTS ---")

# Strategy A: Exact Normalized Name
run_inverted_index("A. Exact Name", lambda r: r['norm_name'])

# Strategy B: Country + Exact Name
run_inverted_index("B. Country + Name", lambda r: f"{r['country_norm']}_{r['norm_name']}")

# Strategy C: Country + Name Prefix (first 4 chars)
run_inverted_index("C. Country + Name Prefix (4)", lambda r: f"{r['country_norm']}_{r['norm_name'][:4]}" if len(r['norm_name'])>=4 else "")

# Strategy D: Country + Name Tokens (>3 chars)
run_inverted_index("D. Country + Name Tokens", lambda r: [f"{r['country_norm']}_{t}" for t in r['norm_name'].split() if len(t)>3])

# Strategy F: Country + Address Numbers
run_inverted_index("F. Country + Addr Num", lambda r: [f"{r['country_norm']}_{num}" for num in r['addr_nums']])

# Multi-pass: Strategy C UNION Strategy F
run_inverted_index("Multi-Pass (Prefix UNION Addr Num)", lambda r: 
    ([f"P_{r['country_norm']}_{r['norm_name'][:4]}"] if len(r['norm_name'])>=4 else []) + 
    [f"A_{r['country_norm']}_{num}" for num in r['addr_nums']]
)

# Strategy E: TF-IDF Character 3-gram
print("Running Strategy E: TF-IDF Char 3-gram (Cosine > 0.6)...")
start_t = time.time()
vec = TfidfVectorizer(analyzer='char', ngram_range=(3,3), min_df=2)
# We add country string to name to force country match loosely
s23_text = [f"{r['country_norm']} {r['norm_name']}" for r in s23_list]
s1_text = [f"{r['country_norm']} {r['norm_name']}" for r in s1_list]

vec.fit(s23_text + s1_text)
s23_mat = vec.transform(s23_text)
s1_mat = vec.transform(s1_text)

# Sparse matrix multiplication for cosine similarity
sim_mat = s1_mat.dot(s23_mat.T)
# Filter by threshold 0.6
sim_mat.data[sim_mat.data < 0.6] = 0
sim_mat.eliminate_zeros()
rows, cols = sim_mat.nonzero()

total_cands_tfidf = len(rows)
found_tfidf = 0
s23_id_array = np.array([r['entity_id'] for r in s23_list])
cands_per_s1_tfidf = defaultdict(int)

for r, c in zip(rows, cols):
    s1_id = s1_list[r]['entity_id']
    s23_id = s23_id_array[c]
    cands_per_s1_tfidf[s1_id] += 1
    if (s1_id, s23_id) in true_pairs:
        found_tfidf += 1

recall_tfidf = found_tfidf / total_true if total_true else 0
cands_dist_tfidf = list(cands_per_s1_tfidf.values()) + [0]*(len(s1_list) - len(cands_per_s1_tfidf))
rr_tfidf = 1 - (total_cands_tfidf / (len(s1_list) * len(s23_list)))

results.append({
    'Method': 'E. TF-IDF Char 3-gram (>0.6)',
    'Recall': f"{recall_tfidf*100:.2f}%",
    'Cands': total_cands_tfidf,
    'Avg Cands': np.mean(cands_dist_tfidf),
    'P95 Cands': np.percentile(cands_dist_tfidf, 95),
    'Reduction': f"{rr_tfidf*100:.4f}%",
    'Time (s)': f"{time.time() - start_t:.2f}"
})
print(f"Finished E. TF-IDF: Recall={recall_tfidf*100:.2f}% | Cands={total_cands_tfidf} | Time={time.time() - start_t:.2f}s")

res_df = pd.DataFrame(results)
print("\n=== FINAL BLOCKING BENCHMARK ===")
print(res_df.to_string())

res_df.to_csv("blocking_benchmark.csv", index=False)

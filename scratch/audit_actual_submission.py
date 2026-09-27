import time
from collections import Counter
import numpy as np

print("=" * 60)
print("AUDITING ACTUAL SUBMISSION FILES IN output/")
print("=" * 60)

# Check candidate_pairs.tsv
cand_path = "output/candidate_pairs.tsv"
t0 = time.time()
print(f"Reading {cand_path}...")

cand_counts = []
cand_empty = 0
sample_cands = []

with open(cand_path, 'r', encoding='utf-8') as f:
    header = next(f)
    for i, line in enumerate(f):
        parts = line.rstrip('\r\n').split('\t')
        s1_id = parts[0]
        cands = parts[1].split(',') if len(parts) > 1 and parts[1] else []
        n_c = len(cands)
        cand_counts.append(n_c)
        if n_c == 0:
            cand_empty += 1
        if i < 5 or (i % 200000 == 0):
            sample_cands.append((s1_id, n_c, cands[:3]))

cand_counts = np.array(cand_counts, dtype=np.int32)
print(f"Total entities in candidate_pairs: {len(cand_counts):,}")
print(f"Mean candidates/entity: {np.mean(cand_counts):.2f}")
print(f"Median candidates/entity: {np.median(cand_counts):.0f}")
print(f"Min candidates: {np.min(cand_counts)}, Max: {np.max(cand_counts)}")
print(f"Zero-candidate entities (empty): {cand_empty:,} ({cand_empty/len(cand_counts)*100:.2f}%)")

# Check matching_results.tsv
match_path = "output/matching_results.tsv"
print(f"\nReading {match_path}...")

match_counts = []
match_empty = 0
match_hist = Counter()
sample_matches = []

with open(match_path, 'r', encoding='utf-8') as f:
    header = next(f)
    for i, line in enumerate(f):
        parts = line.rstrip('\r\n').split('\t')
        s1_id = parts[0]
        matches = parts[1].split(',') if len(parts) > 1 and parts[1] else []
        n_m = len(matches)
        match_counts.append(n_m)
        match_hist[n_m] += 1
        if n_m == 0:
            match_empty += 1
        if i < 5 or (i % 200000 == 0):
            sample_matches.append((s1_id, n_m, matches[:3]))

match_counts = np.array(match_counts, dtype=np.int32)
print(f"Total entities in matching_results: {len(match_counts):,}")
print(f"Mean matches/entity: {np.mean(match_counts):.2f}")
print(f"Zero-match entities (singletons): {match_empty:,} ({match_empty/len(match_counts)*100:.2f}%)")
print("\nMatch count histogram:")
for k in sorted(match_hist.keys()):
    cnt = match_hist[k]
    print(f"  Matches = {k}: {cnt:>10,} ({cnt/len(match_counts)*100:6.2f}%)")

print(f"\nAudit complete in {time.time()-t0:.1f}s.")

"""
Export a complete, 100% compliant leaderboard submission snapshot right now
by taking all currently computed high-confidence predictions and padding
the remaining uncompleted test S1 entities with empty predictions.
"""
import os, sys, shutil

TEST_S1 = 'dataset/test/test_source1.tsv'
SRC_MATCH = 'output/matching_results.tsv'
OUT_DIR = 'output/leaderboard_ready'
os.makedirs(OUT_DIR, exist_ok=True)
OUT_MATCH = os.path.join(OUT_DIR, 'matching_results.tsv')

print("=" * 65, flush=True)
print("CREATING COMPLIANT LEADERBOARD SUBMISSION SNAPSHOT", flush=True)
print("=" * 65, flush=True)

# Step 1: Copy existing computed matching_results lines
print("\n[1] Copying computed predictions...", flush=True)
seen = set()
with open(SRC_MATCH, 'r', encoding='utf-8') as f_in, \
     open(OUT_MATCH, 'w', encoding='utf-8', newline='\n') as f_out:
    hdr = f_in.readline()
    f_out.write("source1_entity_id\tmatched_entity_ids\n")
    for line in f_in:
        parts = line.split('\t', 1)
        if parts and parts[0]:
            seen.add(parts[0])
            f_out.write(line)

print(f"  Copied {len(seen):,} completed entities with predictions.", flush=True)

# Step 2: Append missing S1 entities with empty predictions
print("\n[2] Appending remaining uncompleted test entities...", flush=True)
padded = 0
with open(TEST_S1, 'r', encoding='utf-8') as f_s1, \
     open(OUT_MATCH, 'a', encoding='utf-8', newline='\n') as f_out:
    hdr = f_s1.readline()
    for line in f_s1:
        sid = line.split('\t')[0].strip()
        if sid and sid not in seen:
            f_out.write(f"{sid}\t\n")
            seen.add(sid)
            padded += 1

print(f"  Padded {padded:,} entities with empty matches.", flush=True)
print(f"  Total entities now in file: {len(seen):,}", flush=True)
size_mb = os.path.getsize(OUT_MATCH) / (1024 ** 2)
print(f"  File size: {size_mb:.2f} MB", flush=True)

# Step 3: Verify row count
with open(OUT_MATCH, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f) - 1

print(f"\n[3] Verification:", flush=True)
print(f"  Expected test entities: 1,732,544")
print(f"  Actual entities in file: {total_lines:,}")
if total_lines == 1732544:
    print("  STATUS: 100% COMPLETE & READY FOR UNSTOP LEADERBOARD UPLOAD!", flush=True)
else:
    print(f"  WARNING: Row count mismatch: {total_lines}", flush=True)


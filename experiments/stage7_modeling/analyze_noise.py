import pandas as pd
import random
import os

base_dir = r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker\dataset"

print("Loading ground truth...")
gt_df = pd.read_csv(os.path.join(base_dir, "train", "train_ground_truth.tsv"), sep='\t', dtype=str)

# We just need to load a subset to find examples. Let's sample 1000 pairs.
sample_gt = gt_df.dropna().sample(2000, random_state=42)
s1_ids = set(sample_gt['source1_entity_id'])

# Gather all target IDs
target_ids_s2 = set()
target_ids_s3 = set()

for _, row in sample_gt.iterrows():
    matches = str(row['matched_entity_ids']).split(',')
    for m in matches:
        if m.startswith('S2-'):
            target_ids_s2.add(m)
        elif m.startswith('S3-'):
            target_ids_s3.add(m)

print(f"Looking for {len(s1_ids)} S1, {len(target_ids_s2)} S2, {len(target_ids_s3)} S3 records...")

# Load chunks of S1, S2, S3 and filter
def filter_dataset(path, id_set, id_col='entity_id'):
    print(f"Filtering {path}...")
    chunks = []
    found_count = 0
    try:
        for chunk in pd.read_csv(path, sep='\t', dtype=str, on_bad_lines='skip', chunksize=200000):
            filtered = chunk[chunk[id_col].isin(id_set)]
            chunks.append(filtered)
            found_count += len(filtered)
            if found_count >= len(id_set):
                break
    except Exception as e:
        print(f"Error filtering {path}: {e}")
    return pd.concat(chunks) if chunks else pd.DataFrame()

s1_data = filter_dataset(os.path.join(base_dir, "train", "train_source1.tsv"), s1_ids)
s2_data = filter_dataset(os.path.join(base_dir, "train", "train_source2.tsv"), target_ids_s2)
s3_data = filter_dataset(os.path.join(base_dir, "train", "train_source3.tsv"), target_ids_s3)

# Build a lookup dictionary
lookup = {}
for df in [s1_data, s2_data, s3_data]:
    if df.empty: continue
    for _, row in df.iterrows():
        lookup[row['entity_id']] = row.to_dict()

# Print pairs to examine noise
count = 0
with open("noise_examples.txt", "w", encoding="utf-8") as f:
    for _, row in sample_gt.iterrows():
        s1_id = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).split(',')
        if s1_id not in lookup: continue
        s1_rec = lookup[s1_id]
        
        for m in matches:
            if m in lookup:
                m_rec = lookup[m]
                f.write(f"--- MATCH PAIR ---\n")
                f.write(f"S1 ({s1_id}): Name: {s1_rec.get('business_name')} | Addr: {s1_rec.get('business_address')} | Country: {s1_rec.get('country')}\n")
                f.write(f"Target ({m}): Name: {m_rec.get('business_name')} | Addr: {m_rec.get('business_address')} | Country: {m_rec.get('country')}\n")
                f.write("\n")
                count += 1
                if count >= 200:
                    break
        if count >= 200:
            break

print("Noise examples extracted to noise_examples.txt")

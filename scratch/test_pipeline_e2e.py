"""
End-to-end integration test for JARVIS_CHECKER pipeline.
Generates realistic synthetic data, tests train.py, infer.py, and validate_submission.py.
"""
import os
import shutil
import subprocess
import sys
import pandas as pd
import numpy as np

TEST_DATA_DIR = 'test_env'
os.makedirs(f'{TEST_DATA_DIR}/train', exist_ok=True)
os.makedirs(f'{TEST_DATA_DIR}/test', exist_ok=True)
os.makedirs(f'{TEST_DATA_DIR}/output', exist_ok=True)
os.makedirs(f'{TEST_DATA_DIR}/processed', exist_ok=True)

print("=" * 70)
print("1. GENERATING REALISTIC SYNTHETIC DATA")
print("=" * 70)

# Training S1 records
s1_train = [
    {"entity_id": "S1-0001", "business_name": "Davis Family Office", "business_address": "88 Olive Circle, Lebanon, TN", "country": "US"},
    {"entity_id": "S1-0002", "business_name": "MS Consultancy Corp", "business_address": "Shymala Appts 1St Floor Flat No. 4 Opp Ratna Hospital Sb Road In Haveli, Pune, Maharashtra", "country": "India"},
    {"entity_id": "S1-0003", "business_name": "3520 Main Road Realty Inc", "business_address": "TX, 158 Simpson Lane, Somerset", "country": "US"},
    {"entity_id": "S1-0004", "business_name": "Team Air Pvt. Ltd.", "business_address": "Flat No:101, Anuska Towers, Opp. Mercedes Benz Show Room, Lakdi- Ka, -Pool, Hyderabad, Telangana", "country": "India"},
    {"entity_id": "S1-0005", "business_name": "Lone Star Singleton Inc", "business_address": "404 Nowhere Rd, Austin, TX", "country": "US"},
    {"entity_id": "S1-0006", "business_name": "Mumbai Spices Center", "business_address": "Shop 12, Crawford Market, Mumbai", "country": "India"},
    {"entity_id": "S1-0007", "business_name": "Alpha Beta Technologies", "business_address": "100 Silicon Blvd, San Jose, CA", "country": "US"},
    {"entity_id": "S1-0008", "business_name": "Delta Pharma Ltd", "business_address": "Plot 55 IDA Mallapur, Hyderabad", "country": "India"},
    {"entity_id": "S1-0009", "business_name": "Universal Logistics Group", "business_address": "Terminal 4 JFK Airport, Queens, NY", "country": "US"},
    {"entity_id": "S1-0010", "business_name": "Evergreen Nursery", "business_address": "NH 44 Near Toll Plaza, Nagpur", "country": "India"},
]

# Training S2 records
s2_train = [
    {"entity_id": "S2-0001", "business_name": "Davis Family Offie", "business_address": "88 OLIVE CIR, LEBANON, TN", "country": "US"},
    {"entity_id": "S2-0002", "business_name": "MS [Consultancy]", "business_address": "SHYMALA APPTS 1-1ST FLOOR FLAT NO. 4 OPP RATNA HOSPITAL SB ROAD IN HAVELI, PUNE, Maharashtra", "country": "India"},
    {"entity_id": "S2-0003", "business_name": "3520 MAIN ROAD REALTY INC", "business_address": "SIMPSON LANE, SOMERSET, TX", "country": "US"},
    {"entity_id": "S2-0004", "business_name": "TEAMAIR.COM", "business_address": "FLAT NO:101, ANUSKA TOWERS, OPP. MERCEDES BENZ SHOW ROOM, LAKDI- KA, -POOL, Telangana", "country": "India"},
    {"entity_id": "S2-0005", "business_name": "Unrelated Business LLC", "business_address": "999 Random St, Denver, CO", "country": "US"},
    {"entity_id": "S2-0006", "business_name": "Mumbai Spices Centre", "business_address": "Shop No 12 Crawford Mkt, Mumbai, MH", "country": "India"},
    {"entity_id": "S2-0007", "business_name": "Alpha Beta Tech Inc", "business_address": "100 Silicon Boulevard, San Jose, California", "country": "US"},
    {"entity_id": "S2-0008", "business_name": "Delta Pharma Limited", "business_address": "Plot 55 IDA Mallapur, Hyderabad, TS", "country": "India"},
    {"entity_id": "S2-0009", "business_name": "Universal Logistics", "business_address": "JFK Airport Term 4, Queens, New York", "country": "US"},
    {"entity_id": "S2-0010", "business_name": "Evergreen Nursery Farm", "business_address": "NH-44 Toll Plaza, Nagpur, Maharashtra", "country": "India"},
]

# Training S3 records
s3_train = [
    {"entity_id": "S3-0001", "business_name": "Davis Family (Office)", "business_address": "88 Olive Cir, Lebanon, Tennessee", "country": "US"},
    {"entity_id": "S3-0002", "business_name": "M5 Consultancy Corp", "business_address": "Pune, MH, Pune, Shymala Appts 1St Floor Flat No. 4 Opp Ratna Hospital Sb Road In Haveli", "country": "India"},
    {"entity_id": "S3-0003", "business_name": "3520 MAIN ROAD Realty INC", "business_address": "158 Simpson Lane, Somerset, Texas", "country": "US"},
    {"entity_id": "S3-0004", "business_name": "Team Air Pvt. Limited", "business_address": "Flat No:1-101, Anuska Towers, Opp. Mercedes Benz Show Room, Lakdi- Ka, -Pool, Hyderabad, Andhra Pradesh", "country": "India"},
    {"entity_id": "S3-0005", "business_name": "Completely Different Store", "business_address": "123 Main St, Miami, FL", "country": "US"},
    {"entity_id": "S3-0006", "business_name": "Bombay Spice Co", "business_address": "12 Crawford Mkt, Mumbai", "country": "India"},
    {"entity_id": "S3-0007", "business_name": "Alpha Beta Tech", "business_address": "Silicon Blvd #100, San Jose", "country": "US"},
    {"entity_id": "S3-0008", "business_name": "Delta Pharmaceuticals", "business_address": "IDA Mallapur Plot No 55, Hyderabad", "country": "India"},
    {"entity_id": "S3-0009", "business_name": "Universal Logistics Corp", "business_address": "Queens NY, JFK Terminal 4", "country": "US"},
    {"entity_id": "S3-0010", "business_name": "Evergreen Plant Nursery", "business_address": "Near Toll Plaza NH 44, Nagpur", "country": "India"},
]

# Training Ground Truth
gt_train = [
    {"source1_entity_id": "S1-0001", "matched_entity_ids": "S2-0001,S3-0001"},
    {"source1_entity_id": "S1-0002", "matched_entity_ids": "S2-0002,S3-0002"},
    {"source1_entity_id": "S1-0003", "matched_entity_ids": "S2-0003,S3-0003"},
    {"source1_entity_id": "S1-0004", "matched_entity_ids": "S2-0004,S3-0004"},
    {"source1_entity_id": "S1-0005", "matched_entity_ids": ""}, # singleton
    {"source1_entity_id": "S1-0006", "matched_entity_ids": "S2-0006"}, # single match
    {"source1_entity_id": "S1-0007", "matched_entity_ids": "S2-0007,S3-0007"},
    {"source1_entity_id": "S1-0008", "matched_entity_ids": "S2-0008,S3-0008"},
    {"source1_entity_id": "S1-0009", "matched_entity_ids": "S2-0009,S3-0009"},
    {"source1_entity_id": "S1-0010", "matched_entity_ids": "S2-0010,S3-0010"},
]

# Save training files
pd.DataFrame(s1_train).to_csv(f'{TEST_DATA_DIR}/train/train_source1.tsv', sep='\t', index=False)
pd.DataFrame(s2_train).to_csv(f'{TEST_DATA_DIR}/train/train_source2.tsv', sep='\t', index=False)
pd.DataFrame(s3_train).to_csv(f'{TEST_DATA_DIR}/train/train_source3.tsv', sep='\t', index=False)
pd.DataFrame(gt_train).to_csv(f'{TEST_DATA_DIR}/train/train_ground_truth.tsv', sep='\t', index=False)

# Test records including France!
s1_test = [
    {"entity_id": "S1-T001", "business_name": "Boulangerie Patisserie Saint-Honore", "business_address": "15 Rue Saint-Honore, 75001 Paris", "country": "France"},
    {"entity_id": "S1-T002", "business_name": "Davis Family Office", "business_address": "88 Olive Circle, Lebanon, TN", "country": "US"},
    {"entity_id": "S1-T003", "business_name": "Team Air Pvt. Ltd.", "business_address": "Flat No:101, Anuska Towers, Lakdi- Ka, Hyderabad", "country": "India"},
    {"entity_id": "S1-T004", "business_name": "Test Singleton Entity", "business_address": "999 Ghost St, Unknown City", "country": "US"},
    {"entity_id": "S1-T005", "business_name": "Cafe de Paris SARL", "business_address": "22 Boulevard Haussmann, Paris", "country": "France"},
]

s2_test = [
    {"entity_id": "S2-T001", "business_name": "Boulangerie Pâtisserie Saint-Honoré", "business_address": "15 RUE SAINT HONORE, 75001 PARIS", "country": "France"},
    {"entity_id": "S2-T002", "business_name": "Davis Family Offie", "business_address": "88 OLIVE CIR, LEBANON, TN", "country": "US"},
    {"entity_id": "S2-T003", "business_name": "TEAMAIR.COM", "business_address": "FLAT NO:101, ANUSKA TOWERS, HYDERABAD", "country": "India"},
    {"entity_id": "S2-T005", "business_name": "Café de Paris", "business_address": "22 BD HAUSSMANN, 75009 PARIS", "country": "France"},
    {"entity_id": "S2-T099", "business_name": "Noise Company", "business_address": "123 Random Way", "country": "US"},
]

s3_test = [
    {"entity_id": "S3-T001", "business_name": "St Honore Boulangerie", "business_address": "15 Rue St Honore, Paris", "country": "France"},
    {"entity_id": "S3-T002", "business_name": "Davis Family (Office)", "business_address": "88 Olive Cir, Lebanon, Tennessee", "country": "US"},
    {"entity_id": "S3-T003", "business_name": "Team Air Pvt. Limited", "business_address": "Flat No:1-101, Anuska Towers, Hyderabad", "country": "India"},
    {"entity_id": "S3-T005", "business_name": "Cafe de Paris S.A.R.L.", "business_address": "22 Boulevard Haussmann, Paris", "country": "France"},
    {"entity_id": "S3-T099", "business_name": "Another Distractor", "business_address": "456 Distractor St", "country": "India"},
]

pd.DataFrame(s1_test).to_csv(f'{TEST_DATA_DIR}/test/test_source1.tsv', sep='\t', index=False)
pd.DataFrame(s2_test).to_csv(f'{TEST_DATA_DIR}/test/test_source2.tsv', sep='\t', index=False)
pd.DataFrame(s3_test).to_csv(f'{TEST_DATA_DIR}/test/test_source3.tsv', sep='\t', index=False)

print("Synthetic dataset created successfully.")

print("\n" + "=" * 70)
print("2. RUNNING TRAIN.PY ON SYNTHETIC DATA")
print("=" * 70)
cmd_train = [
    sys.executable, "train.py",
    "--train-dir", f"{TEST_DATA_DIR}/train",
    "--proc-dir", f"{TEST_DATA_DIR}/processed",
    "--output-model", f"{TEST_DATA_DIR}/processed/lgb_model.pkl",
]
ret_train = subprocess.run(cmd_train, capture_output=True, text=True, encoding='utf-8')
print("STDOUT:")
print(ret_train.stdout)
if ret_train.stderr:
    print("STDERR:")
    print(ret_train.stderr)
assert ret_train.returncode == 0, f"train.py failed with returncode {ret_train.returncode}"

print("\n" + "=" * 70)
print("3. RUNNING INFER.PY ON SYNTHETIC TEST SET")
print("=" * 70)
cmd_infer = [
    sys.executable, "infer.py",
    "--test-dir", f"{TEST_DATA_DIR}/test",
    "--model", f"{TEST_DATA_DIR}/processed/lgb_model.pkl",
    "--output-dir", f"{TEST_DATA_DIR}/output",
    "--chunksize", "10",
]
ret_infer = subprocess.run(cmd_infer, capture_output=True, text=True, encoding='utf-8')
print("STDOUT:")
print(ret_infer.stdout)
if ret_infer.stderr:
    print("STDERR:")
    print(ret_infer.stderr)
assert ret_infer.returncode == 0, f"infer.py failed with returncode {ret_infer.returncode}"

print("\n" + "=" * 70)
print("4. RUNNING OFFICIAL VALIDATION CHECK")
print("=" * 70)
cmd_val = [
    sys.executable, "utils/validate_submission.py",
    "--matching", f"{TEST_DATA_DIR}/output/matching_results.tsv",
    "--candidate", f"{TEST_DATA_DIR}/output/candidate_pairs.tsv",
    "--test-dir", f"{TEST_DATA_DIR}/test",
]
ret_val = subprocess.run(cmd_val, capture_output=True, text=True, encoding='utf-8')
print("STDOUT:")
print(ret_val.stdout)
if ret_val.stderr:
    print("STDERR:")
    print(ret_val.stderr)
assert ret_val.returncode == 0, f"Validator failed with returncode {ret_val.returncode}"

print("\n" + "=" * 70)
print("ALL TESTS PASSED WITH EXIT CODE 0!")
print("=" * 70)


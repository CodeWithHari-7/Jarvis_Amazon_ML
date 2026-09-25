# pyrefly: ignore [missing-import]
import polars as pl
import time
import os
import pytest
from src.cleaning.data_cleaning import clean_dataset
from src.normalization.normalization import normalize_dataset

# Configuration
base_dir = r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker\dataset"

with open("preprocessing_report.txt", "w", encoding="utf-8") as f:
    f.write("=== PREPROCESSING BENCHMARK & QUALITY REPORT ===\n\n")

    # 1. Load Data
    start = time.time()
    # Read a robust representative chunk (100k rows) from S1 and S2
    s1_raw = pl.read_csv(os.path.join(base_dir, "train", "train_source1.tsv"), separator='\t', null_values=[""], n_rows=100000)
    s2_raw = pl.read_csv(os.path.join(base_dir, "train", "train_source2.tsv"), separator='\t', null_values=[""], n_rows=100000)
    f.write(f"Data Load (100k rows each): {time.time() - start:.3f} seconds\n\n")

    # 2. Data Cleaning
    start = time.time()
    s1_clean = clean_dataset(s1_raw)
    s2_clean = clean_dataset(s2_raw)
    f.write(f"Data Cleaning (200k rows total): {time.time() - start:.3f} seconds\n")
    f.write(f"S2 Null Addresses before cleaning: {s2_raw['business_address'].null_count()}\n")
    f.write(f"S2 Null Addresses after cleaning (filled empty string): {s2_clean['business_address_clean'].null_count()}\n\n")

    # 3. Normalization
    start = time.time()
    s1_norm = normalize_dataset(s1_clean)
    s2_norm = normalize_dataset(s2_clean)
    f.write(f"Data Normalization & Regex Extraction (200k rows total): {time.time() - start:.3f} seconds\n\n")

    # 4. Example Output Analysis
    f.write("=== NORMALIZATION EXAMPLES (REAL DATA) ===\n")
    for row in s2_norm.head(10).to_dicts():
        f.write(f"Raw Name   : {row.get('business_name_raw', '')}\n")
        f.write(f"Norm Name  : {row.get('business_name_normalized', '')}\n")
        f.write(f"Raw Addr   : {row.get('business_address_raw', '')}\n")
        f.write(f"Norm Addr  : {row.get('business_address_normalized', '')}\n")
        f.write(f"Extracted #: {row.get('business_address_numbers', [])}\n")
        f.write(f"Is Indic   : {row.get('is_indic', False)}\n")
        f.write("-" * 40 + "\n")

    # 5. Performance Metrics
    f.write("\n=== MEMORY & PERFORMANCE ===\n")
    s1_mem_mb = s1_norm.estimated_size() / (1024 * 1024)
    f.write(f"Memory Usage for S1 100k processed rows (incl. raw+clean+norm cols): {s1_mem_mb:.2f} MB\n")
    # Extrapolate for full S2 (5M rows)
    f.write(f"Extrapolated RAM required for full S2 normalization (5M rows): {s1_mem_mb * 50:.2f} MB\n")
    
# Run Pytest Programmatically
pytest_res = pytest.main(["tests/test_preprocessing.py", "-v"])
with open("preprocessing_report.txt", "a", encoding="utf-8") as f:
    f.write(f"\nUnit Tests Passed: {pytest_res == 0}\n")

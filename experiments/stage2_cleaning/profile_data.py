import pandas as pd
import json
import os
import sys

def profile_tsv(file_path):
    print(f"Profiling {file_path}...")
    try:
        df = pd.read_csv(file_path, sep='\t', dtype=str, on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    num_rows = len(df)
    num_cols = len(df.columns)
    cols = list(df.columns)
    
    stats = {
        "filename": os.path.basename(file_path),
        "format": "tsv",
        "rows": int(num_rows),
        "columns": int(num_cols),
        "column_names": cols,
        "missing_pct": {k: float(v) for k, v in (df.isnull().sum() / num_rows).items()},
        "unique_values": {k: int(v) for k, v in df.nunique().items()},
        "duplicates": int(df.duplicated().sum()),
        "file_size_mb": float(os.path.getsize(file_path) / (1024 * 1024))
    }
    
    print(json.dumps(stats, indent=2))
    print(f"Sample data for {os.path.basename(file_path)}:")
    print(df.head(5).to_string())
    print("-" * 50)

def main():
    base_dir = r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker\dataset"
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.tsv')]
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.tsv')]
    
    for f in train_files + test_files:
        profile_tsv(f)

if __name__ == "__main__":
    main()

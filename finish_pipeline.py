"""
JARVIS_CHECKER — Auto-Finish Pipeline
Runs: Inference → Validation → ZIP Packaging
Run this AFTER training is complete.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import os, subprocess, shutil, zipfile, time

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE, 'dataset', 'processed', 'lgb_model.pkl')
OUTPUT_DIR = os.path.join(BASE, 'output')
MATCHING = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
CANDIDATES = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
TEST_DIR = os.path.join(BASE, 'dataset', 'test')
FINAL_DIR = os.path.join(BASE, 'FINAL_SUBMISSION')

def run(cmd, desc):
    print(f"\n{'='*70}")
    print(f"  {desc}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, shell=True, cwd=BASE)
    if result.returncode != 0:
        print(f"\nERROR: {desc} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"  Done: {desc}")

# Step 1: Check model exists
if not os.path.exists(MODEL):
    print(f"ERROR: Model not found at {MODEL}. Run train.py first!")
    sys.exit(1)
print(f"Model found: {MODEL}")

# Step 2: Run Inference
run(
    'python -u infer.py --model dataset/processed/lgb_model.pkl '
    '--output-dir output --chunksize 3000 --test-dir dataset/test',
    "STEP 1/3: Running full test inference"
)

# Step 3: Validate submission
run(
    'python -u utils/validate_submission.py '
    '--matching output/matching_results.tsv '
    '--candidate output/candidate_pairs.tsv '
    '--test-dir dataset/test',
    "STEP 2/3: Validating submission files"
)

# Step 4: Build FINAL_SUBMISSION structure
print(f"\n{'='*70}")
print("  STEP 3/3: Building final submission ZIP")
print(f"{'='*70}")

os.makedirs(FINAL_DIR, exist_ok=True)

out_dest = os.path.join(FINAL_DIR, 'output')
os.makedirs(out_dest, exist_ok=True)
shutil.copy2(MATCHING,   os.path.join(out_dest, 'matching_results.tsv'))
shutil.copy2(CANDIDATES, os.path.join(out_dest, 'candidate_pairs.tsv'))
print(f"  Copied output TSVs to {out_dest}")

code_dest = os.path.join(FINAL_DIR, 'code', 'business_entity_resolution')
os.makedirs(code_dest, exist_ok=True)

for item in ['src', 'configs', 'requirements.txt', 'README.md', 'train.py', 'infer.py']:
    src_path = os.path.join(BASE, item)
    if not os.path.exists(src_path):
        continue
    dst_path = os.path.join(code_dest, item)
    if os.path.isdir(src_path):
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
        shutil.copytree(src_path, dst_path,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pkl'))
    else:
        shutil.copy2(src_path, dst_path)

doc_src = os.path.join(BASE, 'Documentation_template.md')
if os.path.exists(doc_src):
    shutil.copy2(doc_src, os.path.join(FINAL_DIR, 'Documentation_template.md'))

zip_path = os.path.join(BASE, 'JARVIS_CHECKER_submission.zip')
count = 0
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(FINAL_DIR):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for fname in files:
            if fname.endswith('.pyc'):
                continue
            full = os.path.join(root, fname)
            arcname = os.path.relpath(full, FINAL_DIR)
            z.write(full, arcname=arcname)
            count += 1

zip_size = os.path.getsize(zip_path) / 1024**2
print(f"\n  ZIP created: {zip_path}")
print(f"  Files: {count} | Size: {zip_size:.2f} MB")

print(f"\n{'='*70}")
print("  ZIP CONTENTS:")
print(f"{'='*70}")
with zipfile.ZipFile(zip_path, 'r') as z:
    for name in sorted(z.namelist()):
        info = z.getinfo(name)
        print(f"  {name:60s}  {info.file_size/1024:.1f} KB")

print(f"\n{'='*70}")
print("  PROJECT COMPLETE - READY FOR SUBMISSION")
print(f"{'='*70}")
print(f"\n  Submit this ZIP to Unstop/Amazon ML Challenge portal:")
print(f"     {zip_path}")
print(f"\n  OR submit these 2 TSV files if the portal accepts them directly:")
print(f"     output/matching_results.tsv")
print(f"     output/candidate_pairs.tsv")
print(f"{'='*70}\n")

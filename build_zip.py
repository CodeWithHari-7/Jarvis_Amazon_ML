"""
Build the final JARVIS_CHECKER_submission.zip ready for Amazon ML Challenge portal.
Run this AFTER inference completes.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
import os, shutil, zipfile

BASE     = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE, 'output')
MATCHING = os.path.join(OUT_DIR, 'matching_results.tsv')
CANDS    = os.path.join(OUT_DIR, 'candidate_pairs.tsv')
FINAL    = os.path.join(BASE, 'FINAL_SUBMISSION')
ZIP_PATH = os.path.join(BASE, 'JARVIS_CHECKER_submission.zip')

print("=" * 60)
print("Building JARVIS_CHECKER_submission.zip")
print("=" * 60)

# Check output files exist
for p in [MATCHING, CANDS]:
    if not os.path.exists(p):
        print(f"ERROR: Missing {p}")
        print("Run inference first:  python infer.py")
        sys.exit(1)
    sz = os.path.getsize(p)/1024**2
    print(f"  Found: {os.path.basename(p)} ({sz:.2f} MB)")

# Build directory structure
if os.path.exists(FINAL): shutil.rmtree(FINAL)
os.makedirs(FINAL)

# output/
out_dest = os.path.join(FINAL, 'output')
os.makedirs(out_dest)
shutil.copy2(MATCHING, out_dest)
shutil.copy2(CANDS,    out_dest)

# code/business_entity_resolution/
code_dest = os.path.join(FINAL, 'code', 'business_entity_resolution')
os.makedirs(code_dest)
for item in ['src', 'configs', 'train.py', 'infer.py',
             'requirements.txt', 'README.md']:
    src = os.path.join(BASE, item)
    if not os.path.exists(src):
        print(f"  Skipping missing: {item}")
        continue
    dst = os.path.join(code_dest, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pkl','*.parquet'))
    else:
        shutil.copy2(src, dst)

# Documentation
doc = os.path.join(BASE, 'Documentation_template.md')
if os.path.exists(doc):
    shutil.copy2(doc, FINAL)

# Build ZIP
count = 0
with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(FINAL):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for fname in files:
            if fname.endswith('.pyc'): continue
            full    = os.path.join(root, fname)
            arcname = os.path.relpath(full, FINAL)
            z.write(full, arcname=arcname)
            count += 1

zip_mb = os.path.getsize(ZIP_PATH)/1024**2
print(f"\n{'='*60}")
print(f"  ZIP created: {ZIP_PATH}")
print(f"  Files: {count} | Size: {zip_mb:.2f} MB")
print(f"{'='*60}")
print("\n  ZIP structure:")
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    for n in sorted(z.namelist()):
        kb = z.getinfo(n).file_size/1024
        print(f"    {n:<55}  {kb:.0f} KB")

print(f"\n{'='*60}")
print("  SUBMISSION READY!")
print(f"{'='*60}")
print(f"\n  >>> LEADERBOARD: Upload this file to Unstop portal:")
print(f"      {MATCHING}")
print(f"\n  >>> FINAL PACKAGE: Upload this ZIP:")
print(f"      {ZIP_PATH}")
print(f"{'='*60}\n")

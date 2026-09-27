"""Builds the final submission zip from a chosen output folder and model.

  python make_package.py <matching_dir> <model_pkl> <zip_name>
  e.g. python make_package.py output_v7_fr090 code/business_entity_resolution/models/archive/model_v7_cluster.pkl JARVIS_submission.zip

candidate_pairs.tsv is taken from output_v7/ (v7 run) or output_v6/ depending on the model, since thresholds do not
change the candidate set.
"""
import os
import sys
import zipfile

match_dir, model, name = sys.argv[1:4]
cand_dir = sys.argv[4] if len(sys.argv) > 4 else ("output_v7" if "v7" in model else "output_v6")
if os.path.exists(name):
    os.remove(name)
z = zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED)
z.write(os.path.join(match_dir, "matching_results.tsv"), "output/matching_results.tsv")
z.write(os.path.join(cand_dir, "candidate_pairs.tsv"), "output/candidate_pairs.tsv")
z.write("Documentation_template.md")
for r, _, fs in os.walk("code/business_entity_resolution"):
    for f in fs:
        p = os.path.join(r, f)
        if "__pycache__" in p or "archive" in p or (r.endswith("models") and f == "model.pkl"):
            continue
        z.write(p)
z.write(model, "code/business_entity_resolution/models/model.pkl")
z.close()
n = zipfile.ZipFile(name).namelist()
print(f"{name}: {len(n)} files, {len(n) - len(set(n))} duplicates, {os.path.getsize(name) / 1e6:.0f} MB")
print(f"  matching <- {match_dir}, candidates <- {cand_dir}, model <- {model}")

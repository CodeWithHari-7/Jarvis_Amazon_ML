"""
JARVIS_CHECKER — Submission Package Builder
Creates the final competition zip archive adhering strictly to the official challenge structure:

<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""
import os
import sys
import zipfile
import argparse

def build_zip(team_name: str = "JARVIS_CHECKER", source_dir: str = "FINAL_SUBMISSION", output_zip: str = None):
    if output_zip is None:
        output_zip = f"{team_name}_submission.zip"
    
    print(f"Building submission zip: {output_zip} from {source_dir}...")
    
    # Check that required files exist
    required_rel_paths = [
        "output/matching_results.tsv",
        "output/candidate_pairs.tsv",
        "code/business_entity_resolution/src",
        "code/business_entity_resolution/README.md",
        "code/business_entity_resolution/requirements.txt",
        "Documentation_template.md",
    ]
    
    for rel_path in required_rel_paths:
        full_path = os.path.join(source_dir, rel_path)
        if not os.path.exists(full_path):
            print(f"WARNING: Required path does not exist yet: {full_path}")
    
    count = 0
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.pyc') or '__pycache__' in root:
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, source_dir)
                z.write(full_path, arcname=rel_path)
                count += 1
                
    zip_size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"Successfully created {output_zip} ({count} files, {zip_size_mb:.2f} MB)")
    print("\nZip Archive Structure:")
    with zipfile.ZipFile(output_zip, 'r') as z:
        for name in sorted(z.namelist()):
            print(f"  {name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Package submission directory into official ZIP")
    parser.add_argument('--team', default="JARVIS_CHECKER", help="Team name for zip filename")
    parser.add_argument('--source-dir', default="FINAL_SUBMISSION", help="Directory with final submission files")
    parser.add_argument('--output-zip', default=None, help="Output zip filename")
    args = parser.parse_args()
    build_zip(args.team, args.source_dir, args.output_zip)

import polars as pl

df = pl.read_csv('dataset/test/test_source1.tsv', separator='\t', columns=['country'])
counts = df.group_by('country').len()
print("=== Ground Truth Test Set by Country ===")
for r in counts.iter_rows():
    print(f"  {r[0]}: {r[1]:,} rows ({r[1]/100000:.2f} Lakh)")
print(f"Total test entities required: {len(df):,} ({len(df)/100000:.2f} Lakh)\n")

with open('output/matching_results.tsv', 'r', encoding='utf-8') as f:
    match_lines = sum(1 for _ in f)
print("=== Current Output Files ===")
print(f"  matching_results.tsv: {match_lines:,} lines ({match_lines/100000:.2f} Lakh)")

with open('output/candidate_pairs.tsv', 'r', encoding='utf-8') as f:
    cand_lines = sum(1 for _ in f)
print(f"  candidate_pairs.tsv:  {cand_lines:,} lines ({cand_lines/100000:.2f} Lakh)")

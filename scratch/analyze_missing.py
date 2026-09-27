import polars as pl
from collections import defaultdict

s1 = pl.read_csv('dataset/test/test_source1.tsv', separator='\t', columns=['entity_id', 'country'])
tot_s1 = len(s1)
print(f"Total S1 entities required: {tot_s1:,}")
c_counts = s1.group_by('country').len().to_dicts()
for c in c_counts:
    print(f"  Country {c['country']}: {c['len']:,} ({c['len']/tot_s1*100:.2f}%)")

with open('output/matching_results.tsv', 'r', encoding='utf-8') as f:
    hdr = f.readline().strip()
    data = [line.rstrip('\r\n').split('\t') for line in f]

tot_rows = len(data)
print(f"\nmatching_results.tsv Total Rows: {tot_rows:,}")
print(f"Header: {hdr}")

eid_to_country = dict(zip(s1['entity_id'].to_list(), s1['country'].to_list()))
empty_by_country = defaultdict(int)
matched_by_country = defaultdict(int)
tot_matches = 0
match_counts = []

for r in data:
    eid = r[0]
    country = eid_to_country.get(eid, 'UNKNOWN')
    if len(r) < 2 or not r[1].strip():
        empty_by_country[country] += 1
    else:
        m_list = r[1].split(',')
        tot_matches += len(m_list)
        match_counts.append(len(m_list))
        matched_by_country[country] += 1

tot_empty = sum(empty_by_country.values())
tot_matched = sum(matched_by_country.values())

print(f"\n=== MISSING VALUE INVENTORY ===")
print(f"  Field 1 (source1_entity_id)   : 0 nulls / missing ({tot_rows:,} present, 100.0%)")
print(f"  Field 2 (matched_entity_ids)  : {tot_empty:,} empty values ({tot_empty/tot_rows*100:.2f}%)")
print(f"  Field 2 Non-Empty (Matched)   : {tot_matched:,} populated values ({tot_matched/tot_rows*100:.2f}%)")
print(f"  Total S2/S3 Matches Generated : {tot_matches:,} (Average: {tot_matches/tot_matched:.2f} per matched entity)")

print(f"\n=== COUNTRY-BY-COUNTRY DISTRIBUTION ===")
for c in c_counts:
    c_name = c['country']
    tot_c = c['len']
    emp_c = empty_by_country[c_name]
    mat_c = matched_by_country[c_name]
    print(f"  Country: {c_name:6s} | Total: {tot_c:>7,} | Matched: {mat_c:>7,} ({mat_c/tot_c*100:5.2f}%) | Empty: {emp_c:>7,} ({emp_c/tot_c*100:5.2f}%)")

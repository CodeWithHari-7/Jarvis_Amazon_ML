"""
Step 2 & 3 Remediation:
Applies street-number exact mismatch veto, core name token veto, and max-match capping (<=6)
to output/matching_results.tsv to eliminate over-matching while preserving valid matches.
"""

import os
import sys
import re
import time
from collections import defaultdict
import numpy as np
import polars as pl
from rapidfuzz import fuzz

sys.path.append('code/business_entity_resolution/src')
from blocking import LEGAL_SUFFIXES, STOPWORDS

print("=" * 80)
print("APPLYING HARD VETO FEATURES AND OVER-MATCHING REMEDIATION")
print("=" * 80)
t0 = time.time()

num_re = re.compile(r'\b\d+[a-zA-Z]?\b')
clean_re = re.compile(r'[^a-z0-9\s]')

def extract_nums(addr: str) -> set:
    if not addr: return set()
    nums = set()
    for n in num_re.findall(str(addr).lower()):
        if len(n) <= 6:
            nums.add(n)
    return nums

def get_core_tokens(name: str) -> set:
    if not name: return set()
    clean = clean_re.sub(' ', str(name).lower())
    toks = [w for w in clean.split() if w not in LEGAL_SUFFIXES and w not in STOPWORDS and len(w) >= 3]
    return set(toks)

# 1. Load S1 test info
print("\n[1] Loading Test Source 1 records...")
s1_test = pl.read_csv("dataset/test/test_source1.tsv", separator='\t')
s1_recs = {}
for r in s1_test.to_dicts():
    s1_recs[r['entity_id']] = {
        'name': r['business_name'] or '',
        'addr': r['business_address'] or '',
        'country': r['country'] or '',
        'nums': extract_nums(r['business_address']),
        'core_tokens': get_core_tokens(r['business_name'])
    }
print(f"  Loaded {len(s1_recs):,} Source 1 records.")

# 2. Read current matching_results.tsv and collect matched IDs
print("\n[2] Reading current output/matching_results.tsv...")
raw_matches = {}
all_matched_s23 = set()
with open("output/matching_results.tsv", 'r', encoding='utf-8') as f:
    hdr = f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        eid = p[0]
        if len(p) >= 2 and p[1].strip():
            m_list = [x.strip() for x in p[1].split(',') if x.strip()]
            raw_matches[eid] = m_list
            all_matched_s23.update(m_list)
        else:
            raw_matches[eid] = []

print(f"  Total S1 in file: {len(raw_matches):,}")
print(f"  Total matched S2/S3 entity pool to inspect: {len(all_matched_s23):,} unique IDs")

# 3. Load only the necessary S2/S3 records
print("\n[3] Ingesting matched S2/S3 records for veto checking...")
t_load = time.time()
s2_test = pl.read_csv("dataset/test/test_source2.tsv", separator='\t').filter(pl.col('entity_id').is_in(all_matched_s23))
s3_test = pl.read_csv("dataset/test/test_source3.tsv", separator='\t').filter(pl.col('entity_id').is_in(all_matched_s23))

s23_recs = {}
for r in s2_test.to_dicts():
    s23_recs[r['entity_id']] = {
        'name': r['business_name'] or '',
        'addr': r['business_address'] or '',
        'nums': extract_nums(r['business_address']),
        'core_tokens': get_core_tokens(r['business_name'])
    }
for r in s3_test.to_dicts():
    s23_recs[r['entity_id']] = {
        'name': r['business_name'] or '',
        'addr': r['business_address'] or '',
        'nums': extract_nums(r['business_address']),
        'core_tokens': get_core_tokens(r['business_name'])
    }
print(f"  Loaded {len(s23_recs):,} S2/S3 records in {time.time()-t_load:.1f}s.")

# 4. Apply Hard Veto Rules and Cap <= 6
print("\n[4] Applying Hard Veto Rules & Capping <= 6 matches...")
cleaned_matches = {}
pruned_street_nums = 0
pruned_core_names = 0
pruned_caps = 0

country_before = defaultdict(list)
country_after = defaultdict(list)

for eid, m_list in raw_matches.items():
    r1 = s1_recs[eid]
    c_name = r1['country']
    country_before[c_name].append(len(m_list))
    
    if not m_list:
        cleaned_matches[eid] = []
        country_after[c_name].append(0)
        continue

    surviving = []
    for mid in m_list:
        r2 = s23_recs.get(mid)
        if not r2:
            surviving.append(mid)
            continue

        # Hard Veto 1: Street Number Discrepancy
        nums1 = r1['nums']
        nums2 = r2['nums']
        if nums1 and nums2 and not (nums1 & nums2):
            # Street numbers exist in both addresses but have zero overlap
            # Check if address string similarity is low
            addr_sim = fuzz.token_set_ratio(r1['addr'].lower(), r2['addr'].lower())
            if addr_sim < 75:
                pruned_street_nums += 1
                continue  # VETO!

        # Hard Veto 2: Core Name Token Mismatch
        c1 = r1['core_tokens']
        c2 = r2['core_tokens']
        if c1 and c2 and not (c1 & c2):
            name_sim = fuzz.ratio(r1['name'].lower(), r2['name'].lower())
            if name_sim < 65:
                pruned_core_names += 1
                continue  # VETO!

        surviving.append(mid)

    # Hard Rule 3: Cap at <= 6 matches
    if len(surviving) > 6:
        pruned_caps += (len(surviving) - 6)
        surviving = surviving[:6]

    cleaned_matches[eid] = surviving
    country_after[c_name].append(len(surviving))

print(f"  Total false merges pruned by Street Number Veto : {pruned_street_nums:,}")
print(f"  Total false merges pruned by Core Name Veto     : {pruned_core_names:,}")
print(f"  Total false merges pruned by Match Cap (<=6)    : {pruned_caps:,}")
print(f"  Total false merges eliminated                   : {pruned_street_nums + pruned_core_names + pruned_caps:,}")

# 5. Country Breakdown Comparison
print("\n" + "=" * 80)
print("COUNTRY-WISE MATCH STATISTICS: BEFORE VS AFTER REMEDIATION")
print("=" * 80)
print(f"{'Country':<10} | {'Metric':<25} | {'Before Remediation':<20} | {'After Remediation':<20}")
print("-" * 80)

for c in ['US', 'India', 'France']:
    b_arr = np.array(country_before[c])
    a_arr = np.array(country_after[c])
    print(f"{c:<10} | {'Average Matches/Entity':<25} | {np.mean(b_arr):>18.2f} | {np.mean(a_arr):>18.2f}")
    print(f"{'':<10} | {'Singleton Rate':<25} | {np.mean(b_arr==0)*100:>17.2f}% | {np.mean(a_arr==0)*100:>17.2f}%")
    print(f"{'':<10} | {'> 6 Matches Rate':<25} | {np.mean(b_arr>6)*100:>17.2f}% | {np.mean(a_arr>6)*100:>17.2f}%")
    print(f"{'':<10} | {'>= 11 Matches Rate':<25} | {np.mean(b_arr>=11)*100:>17.2f}% | {np.mean(a_arr>=11)*100:>17.2f}%")
    print(f"{'':<10} | {'Max Matches in Entity':<25} | {np.max(b_arr):>18} | {np.max(a_arr):>18}")
    print("-" * 80)

# Overall
all_b = np.concatenate([np.array(v) for v in country_before.values()])
all_a = np.concatenate([np.array(v) for v in country_after.values()])
print(f"{'OVERALL':<10} | {'Average Matches/Entity':<25} | {np.mean(all_b):>18.2f} | {np.mean(all_a):>18.2f}")
print(f"{'':<10} | {'Singleton Rate':<25} | {np.mean(all_b==0)*100:>17.2f}% | {np.mean(all_a==0)*100:>17.2f}%")
print(f"{'':<10} | {'> 6 Matches Rate':<25} | {np.mean(all_b>6)*100:>17.2f}% | {np.mean(a_arr>6)*100:>17.2f}%")
print(f"{'':<10} | {'Max Matches in Entity':<25} | {np.max(all_b):>18} | {np.max(all_a):>18}")

# 6. Write Remediated matching_results.tsv
print("\n[6] Writing remediated output/matching_results.tsv...")
out_tsv = "output/matching_results.tsv"
with open(out_tsv, 'w', encoding='utf-8') as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for eid in s1_test['entity_id'].to_list():
        m_str = ",".join(cleaned_matches.get(eid, []))
        f.write(f"{eid}\t{m_str}\n")

print(f"  Successfully wrote {len(s1_recs):,} rows to {out_tsv} in {time.time()-t0:.1f}s.")
print("=" * 80)

"""
Test France over-matching remediation:
Evaluates street number veto, street name overlap, and match capping on France entities.
"""

import polars as pl
from collections import defaultdict
import numpy as np
import re
from rapidfuzz import fuzz

num_re = re.compile(r'\b\d+[a-zA-Z]?\b')

def extract_numbers(addr: str) -> set:
    if not addr: return set()
    nums = set()
    for n in num_re.findall(str(addr).lower()):
        if len(n) <= 6:
            nums.add(n)
    return nums

# Load sample of high-match France entities
s1_test = pl.read_csv("dataset/test/test_source1.tsv", separator='\t').filter(pl.col('country') == 'France')
s1_map = {r['entity_id']: r for r in s1_test.to_dicts()}

# Read current matching results for France
fr_matches = {}
with open("output/matching_results.tsv", 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        eid = p[0]
        if eid in s1_map:
            m_list = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) >= 2 and p[1].strip() else []
            fr_matches[eid] = m_list

print(f"Total France S1 entities: {len(fr_matches):,}")
match_counts = [len(m) for m in fr_matches.values()]
print(f"Current France Avg Matches: {np.mean(match_counts):.2f}")
print(f"Current France >6 matches: {np.sum(np.array(match_counts) > 6):,} ({np.mean(np.array(match_counts) > 6)*100:.2f}%)")
print(f"Current France >=11 matches: {np.sum(np.array(match_counts) >= 11):,} ({np.mean(np.array(match_counts) >= 11)*100:.2f}%)")

# Sample 500 entities with >6 matches to evaluate veto rule impact
high_eids = [eid for eid, m in fr_matches.items() if len(m) > 6][:500]
needed_s23 = set()
for eid in high_eids:
    needed_s23.update(fr_matches[eid])

s2_test = pl.read_csv("dataset/test/test_source2.tsv", separator='\t').filter(pl.col('entity_id').is_in(needed_s23))
s3_test = pl.read_csv("dataset/test/test_source3.tsv", separator='\t').filter(pl.col('entity_id').is_in(needed_s23))
s23_map = {r['entity_id']: r for r in s2_test.to_dicts()}
s23_map.update({r['entity_id']: r for r in s3_test.to_dicts()})

print(f"\nEvaluating Veto Rules on 500 high-match entities (originally {sum(len(fr_matches[eid]) for eid in high_eids):,} matches, avg {sum(len(fr_matches[eid]) for eid in high_eids)/500:.1f} per entity):")

for cap in [6, 8, None]:
    for apply_veto in [False, True]:
        filtered_counts = []
        for eid in high_eids:
            r1 = s1_map[eid]
            nums1 = extract_numbers(r1['business_address'])
            kept = []
            for mid in fr_matches[eid]:
                r2 = s23_map.get(mid, {})
                nums2 = extract_numbers(r2.get('business_address'))
                
                if apply_veto:
                    # Street number exact mismatch veto
                    if nums1 and nums2 and not (nums1 & nums2):
                        # Numbers exist in both but are completely different (e.g. 6 vs 19)
                        # Check address string similarity
                        addr_sim = fuzz.token_set_ratio(str(r1['business_address'] or '').lower(), str(r2.get('business_address') or '').lower())
                        if addr_sim < 75:
                            continue  # VETO!
                kept.append(mid)
            if cap:
                kept = kept[:cap]
            filtered_counts.append(len(kept))
        
        avg_k = np.mean(filtered_counts)
        tot_k = sum(filtered_counts)
        print(f"  Veto={str(apply_veto):<5} | Cap={str(cap):<4} -> Total Matches: {tot_k:>5,} | Avg Matches: {avg_k:4.2f} (Pruned {sum(len(fr_matches[eid]) for eid in high_eids) - tot_k:,} false merges)")

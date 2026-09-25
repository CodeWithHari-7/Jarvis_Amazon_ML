import pandas as pd
import os

print("--- ERROR INSPECTION ---")
s1_df = pd.read_csv("dataset/pipeline_cache/subset_s1.tsv", sep='\t', dtype=str).set_index("entity_id")
s2_df = pd.read_csv("dataset/pipeline_cache/subset_s2.tsv", sep='\t', dtype=str).set_index("entity_id")
s3_df = pd.read_csv("dataset/pipeline_cache/subset_s3.tsv", sep='\t', dtype=str).set_index("entity_id")
cand_features = pd.read_parquet("dataset/pipeline_cache/candidate_features.parquet")
results_df = pd.read_csv("matching_results.tsv", sep='\t', dtype=str).fillna("")

def print_pair(s1_id, src_id):
    try:
        r1 = s1_df.loc[s1_id]
        if src_id.startswith("S2"): r2 = s2_df.loc[src_id]
        elif src_id.startswith("S3"): r2 = s3_df.loc[src_id]
        else: r2 = None
    except KeyError:
        print(f"ID not found: {s1_id} / {src_id}")
        return

    feat = cand_features[(cand_features['source1_entity_id']==s1_id) & (cand_features['source_entity_id']==src_id)]
    
    print(f"\nS1: {s1_id} | Name: {r1.get('business_name')} | Addr: {r1.get('business_address')} | Country: {r1.get('country')}")
    if r2 is not None:
        print(f"S2/3: {src_id} | Name: {r2.get('business_name')} | Addr: {r2.get('business_address')} | Country: {r2.get('country')}")
    else:
        print(f"S2/3: {src_id} Not found in subset")
        
    if not feat.empty:
        res_row = results_df[results_df['source1_entity_id']==s1_id]
        if not res_row.empty and src_id in str(res_row['matched_entity_ids'].values[0]).split(','):
            print(f"Features: TS={feat['name_token_set'].values[0]} | AddrTS={feat['addr_token_set'].values[0]} | NumOverlap={feat['addr_num_overlap'].values[0]:.2f} | Prob=PREDICTED_MATCH")
        else:
            print(f"Features: TS={feat['name_token_set'].values[0]} | AddrTS={feat['addr_token_set'].values[0]} | NumOverlap={feat['addr_num_overlap'].values[0]:.2f} | Prob=PREDICTED_NON_MATCH")
    else:
        print(f"Features: NOT GENERATED AS CANDIDATE (Blocking Failure)")

print("\n--- FALSE POSITIVES ---")
print_pair("S1-924757726", "S2-410425647")
print_pair("S1-690282030", "S2-282513358")
print_pair("S1-853204970", "S3-444254145")

print("\n--- FALSE NEGATIVES ---")
print_pair("S1-219754944", "S3-468071974")
print_pair("S1-970876847", "S2-944913305")
print_pair("S1-703657788", "S3-56331466")

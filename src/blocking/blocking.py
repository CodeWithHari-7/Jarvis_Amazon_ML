import polars as pl
from collections import defaultdict
import time

def generate_candidates(s1_df: pl.DataFrame, s23_df: pl.DataFrame) -> set:
    """
    Generates candidate pairs (s1_id, src_id) using the frozen Multi-Pass Blocking strategy:
    Pass 1: country_norm + norm_name[:4]
    Pass 2: country_norm + addr_num
    Returns a deduplicated set of (s1_id, src_id).
    """
    s1_records = s1_df.to_dicts()
    s23_records = s23_df.to_dicts()
    
    print("  Building inverted index for S2/S3...")
    t0 = time.time()
    
    # Pass 1: Prefix Index
    prefix_idx = defaultdict(list)
    # Pass 2: Address Num Index
    addr_num_idx = defaultdict(list)
    
    for r in s23_records:
        eid = r['entity_id']
        c = r.get('country_normalized', '')
        
        # Pass 1 Key
        n = r.get('business_name_normalized', '')
        if c and n:
            prefix_idx[f"{c}_{n[:4]}"].append(eid)
            
        # Pass 2 Key
        nums = r.get('business_address_numbers', [])
        for num in nums:
            if c:
                addr_num_idx[f"{c}_{num}"].append(eid)
                
    print(f"  Index built in {time.time()-t0:.2f}s")
    
    print("  Querying index for S1...")
    t0 = time.time()
    candidates = set()
    
    for r in s1_records:
        s1_id = r['entity_id']
        c = r.get('country_normalized', '')
        
        # Pass 1 Lookup
        n = r.get('business_name_normalized', '')
        if c and n:
            key = f"{c}_{n[:4]}"
            for match_id in prefix_idx.get(key, []):
                candidates.add((s1_id, match_id))
                
        # Pass 2 Lookup
        nums = r.get('business_address_numbers', [])
        for num in nums:
            if c:
                key = f"{c}_{num}"
                for match_id in addr_num_idx.get(key, []):
                    candidates.add((s1_id, match_id))
                    
    print(f"  Candidates generated in {time.time()-t0:.2f}s")
    print(f"  Total unique candidate pairs: {len(candidates):,}")
    
    return candidates

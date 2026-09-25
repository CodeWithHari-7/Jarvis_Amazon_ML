from rapidfuzz import fuzz, distance

def calculate_numeric_overlap(nums1, nums2):
    s1, s2 = set(nums1), set(nums2)
    if not s1 and not s2: 
        return 0.0
    return len(s1.intersection(s2)) / max(len(s1.union(s2)), 1)

def extract_features_for_pair(r1: dict, r2: dict) -> dict:
    """
    Generates feature vector for ML inference.
    """
    n1, n2 = r1.get('business_name_normalized', ''), r2.get('business_name_normalized', '')
    a1, a2 = r1.get('business_address_normalized', ''), r2.get('business_address_normalized', '')
    
    cross_script = 1 if r1.get('is_indic', False) != r2.get('is_indic', False) else 0
    
    return {
        'name_fuzz_ratio': fuzz.ratio(n1, n2) if n1 and n2 else 0,
        'name_token_set': fuzz.token_set_ratio(n1, n2) if n1 and n2 else 0,
        'name_jw': distance.JaroWinkler.normalized_similarity(n1, n2) if n1 and n2 else 0,
        'addr_fuzz_ratio': fuzz.ratio(a1, a2) if a1 and a2 else 0,
        'addr_token_set': fuzz.token_set_ratio(a1, a2) if a1 and a2 else 0,
        'addr_num_overlap': calculate_numeric_overlap(r1.get('business_address_numbers', []), r2.get('business_address_numbers', [])),
        'cross_script': cross_script
    }

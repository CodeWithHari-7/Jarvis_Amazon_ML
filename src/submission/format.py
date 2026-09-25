import pandas as pd

def format_submission(s1_ids: list, candidate_s1_ids: pd.Series, candidate_src_ids: pd.Series, probs: pd.Series, threshold: float) -> pd.DataFrame:
    """
    Applies the threshold to probabilities and formats the output into (source1_entity_id, matched_entity_ids).
    Any S1 with no matches above threshold will get an empty string for matched_entity_ids.
    """
    df = pd.DataFrame({
        'source1_entity_id': candidate_s1_ids,
        'source_entity_id': candidate_src_ids,
        'prob': probs
    })
    
    # 1. Apply Threshold (No top-1 forcing)
    matches = df[df['prob'] >= threshold]
    
    # 2. Group By S1 and join matches with commas
    grouped = matches.groupby('source1_entity_id')['source_entity_id'].apply(lambda x: ','.join(x)).reset_index()
    grouped.columns = ['source1_entity_id', 'matched_entity_ids']
    
    # 3. Ensure all original S1 IDs are present (even those with 0 matches)
    all_s1 = pd.DataFrame({'source1_entity_id': s1_ids})
    result = pd.merge(all_s1, grouped, on='source1_entity_id', how='left')
    result['matched_entity_ids'] = result['matched_entity_ids'].fillna("")
    
    return result

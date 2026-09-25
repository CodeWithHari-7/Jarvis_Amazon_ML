"""
Data cleaning module for JARVIS_CHECKER Business Entity Resolution.
Converts raw TSV records into clean dicts ready for normalization.
"""
import pandas as pd
from typing import List


def clean_record(row: dict) -> dict:
    """
    Clean a single raw record dict.
    - Fills missing fields with empty string
    - Strips whitespace
    - Preserves original values under *_raw keys
    """
    result = {}
    for key in ['entity_id', 'business_name', 'business_address', 'country']:
        val = row.get(key)
        if val is None or (isinstance(val, float) and str(val) == 'nan'):
            val = ''
        val = str(val).strip()
        result[key] = val
        result[f'{key}_raw'] = val  # preserve raw
    return result


def load_and_clean_tsv(path: str) -> List[dict]:
    """
    Loads a TSV file and returns a list of cleaned record dicts.
    Always uses tab separator.
    """
    df = pd.read_csv(
        path, sep='\t', dtype=str,
        keep_default_na=False, on_bad_lines='skip'
    )
    # Ensure expected columns exist
    for col in ['entity_id', 'business_name', 'business_address', 'country']:
        if col not in df.columns:
            df[col] = ''
    return [clean_record(row) for row in df.to_dict('records')]


def load_tsv_chunked(path: str, chunksize: int = 100_000):
    """
    Generator that yields cleaned record dicts from a large TSV file in chunks.
    """
    for chunk in pd.read_csv(
        path, sep='\t', dtype=str,
        keep_default_na=False, on_bad_lines='skip',
        chunksize=chunksize
    ):
        for col in ['entity_id', 'business_name', 'business_address', 'country']:
            if col not in chunk.columns:
                chunk[col] = ''
        for row in chunk.to_dict('records'):
            yield clean_record(row)

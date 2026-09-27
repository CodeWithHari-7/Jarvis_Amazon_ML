"""Raw-file reading (before normalisation).

- TSVs are read with an explicit tab separator and quote_char=None (names contain stray quotes).
- All columns are kept as strings; entity ids keep their S1-/S2-/S3- prefixes; empty fields become null.
- Country is kept as an open-set string label (France appears only in test) - never filtered or one-hot encoded.
"""
import os

import polars as pl

COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: str) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema_length=0)


def read_source(data_dir: str, split: str, source: int) -> pl.DataFrame:
    """Raw records of dataset/{split}/{split}_source{source}.tsv with the four expected columns."""
    return read_tsv(os.path.join(data_dir, split, f"{split}_source{source}.tsv")).select(COLUMNS)

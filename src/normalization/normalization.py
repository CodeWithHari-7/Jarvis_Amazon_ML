# pyrefly: ignore [missing-import]
import polars as pl
import re

def normalize_text(text_col: pl.Expr) -> pl.Expr:
    """
    Lowercases, removes special punctuation, and normalizes whitespace.
    Does NOT remove legal suffixes, as TokenSetRatio handles them efficiently.
    """
    expr = text_col.str.to_lowercase()
    # Preserve English alphanumeric and Indic Unicode ranges
    expr = expr.str.replace_all(r"[^a-z0-9\s\u0900-\u097F\u0A80-\u0AFF\u0C00-\u0C7F]", " ")
    expr = expr.str.replace_all(r"\s+", " ").str.strip_chars()
    return expr

def extract_numbers(text_col: pl.Expr) -> pl.Expr:
    """
    Extracts purely numeric sequences. Extremely high value for address matching.
    """
    return text_col.str.extract_all(r"\d+").fill_null([])

def is_indic_script(text_col: pl.Expr) -> pl.Expr:
    """
    Checks if the string contains Indic scripts (Devanagari, Telugu, Gujarati, etc.).
    """
    return text_col.str.contains(r"[\u0900-\u097F\u0A80-\u0AFF\u0C00-\u0C7F]")

def normalize_dataset(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        normalize_text(pl.col("business_name_clean")).alias("business_name_normalized"),
        normalize_text(pl.col("business_address_clean")).alias("business_address_normalized"),
        normalize_text(pl.col("country_clean")).alias("country_normalized"),
        extract_numbers(pl.col("business_address_clean")).alias("business_address_numbers"),
        is_indic_script(pl.col("business_name_clean")).alias("is_indic")
    ])
    return df

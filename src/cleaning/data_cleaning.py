import polars as pl

def clean_dataset(df: pl.DataFrame) -> pl.DataFrame:
    """
    Cleans raw dataset by preserving original columns and creating _clean columns.
    Rules:
    - Replace nulls with empty strings to prevent downstream type errors.
    - Strip leading/trailing whitespaces.
    - No rows are aggressively dropped (preserves maximum recall).
    """
    df = df.with_columns([
        pl.col("business_name").alias("business_name_raw"),
        pl.col("business_address").alias("business_address_raw"),
        pl.col("country").alias("country_raw")
    ])
    
    df = df.with_columns([
        pl.col("business_name").fill_null("").str.strip_chars().alias("business_name_clean"),
        pl.col("business_address").fill_null("").str.strip_chars().alias("business_address_clean"),
        pl.col("country").fill_null("").str.strip_chars().alias("country_clean")
    ])
    return df

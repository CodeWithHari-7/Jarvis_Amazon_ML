# pyrefly: ignore [missing-import]
import polars as pl
from src.cleaning.data_cleaning import clean_dataset
from src.normalization.normalization import normalize_dataset
from src.features.feature_extraction import extract_features_for_pair

def test_cleaning():
    df = pl.DataFrame({
        "business_name": [" A ", None], 
        "business_address": ["", " B "], 
        "country": [None, "US"]
    })
    clean = clean_dataset(df)
    assert clean["business_name_clean"][0] == "A"
    assert clean["business_name_clean"][1] == ""

def test_normalization():
    df = pl.DataFrame({
        "business_name_clean": ["Davis Family Office", "सुप्रीम"], 
        "business_address_clean": ["0226 87 ST", "1104 Freedley"], 
        "country_clean": ["US", "India"]
    })
    norm = normalize_dataset(df)
    assert norm["business_name_normalized"][0] == "davis family office"
    assert norm["business_address_numbers"][0].to_list() == ["0226", "87"]
    assert norm["is_indic"][1] == True

def test_features():
    r1 = {
        "business_name_normalized": "a b", 
        "business_address_normalized": "1 st", 
        "business_address_numbers": ["1"], 
        "is_indic": False, 
        "country_normalized": "us"
    }
    r2 = {
        "business_name_normalized": "b a", 
        "business_address_normalized": "2 st", 
        "business_address_numbers": ["2"], 
        "is_indic": False, 
        "country_normalized": "us"
    }
    feats = extract_features_for_pair(r1, r2)
    assert feats["name_token_set"] == 100
    assert feats["addr_num_overlap"] == 0.0

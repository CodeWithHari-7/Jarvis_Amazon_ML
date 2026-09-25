"""
Unit tests for JARVIS_CHECKER entity resolution modules.
"""
from src.cleaning.data_cleaning import clean_record
from src.normalization.normalization import (
    normalize_record,
    normalize_business_name,
    normalize_address,
    normalize_country,
    _to_ascii_compatible
)
from src.blocking.blocking import build_s23_index, generate_candidates_for_record
from src.features.feature_extraction import extract_features_for_pair, FEATURE_COLS


def test_cleaning():
    row = {"entity_id": "S1-001", "business_name": "  A Corp  ", "business_address": None, "country": "US"}
    clean = clean_record(row)
    assert clean["business_name"] == "A Corp"
    assert clean["business_address"] == ""
    assert clean["country"] == "US"


def test_french_normalization():
    # Test French diacritics stripping
    name = "Boulangerie Pâtisserie Saint-Honoré SARL"
    norm = normalize_business_name(name)
    assert "patisserie" in norm
    assert "saint honore" in norm
    assert "sarl" not in norm  # Legal suffix stripped


def test_indic_normalization():
    # Test Indic preservation
    row = {
        "entity_id": "S1-002",
        "business_name": "सुप्रीम ट्रेडर्स Pvt Ltd",
        "business_address": "Flat 101, Lakdi-Ka-Pool, Hyderabad",
        "country": "India"
    }
    norm = normalize_record(row)
    assert "सुप्रीम" in norm["business_name_normalized"]
    assert norm["is_indic"] is True
    assert "101" in norm["business_address_numbers"]
    assert norm["country_normalized"] == "india"


def test_blocking_candidate_generation():
    s23_records = [
        normalize_record({
            "entity_id": "S2-001",
            "business_name": "Davis Family Offie",
            "business_address": "88 OLIVE CIR, LEBANON, TN",
            "country": "US"
        }),
        normalize_record({
            "entity_id": "S3-001",
            "business_name": "Random Other Corp",
            "business_address": "123 Main St",
            "country": "US"
        })
    ]
    indexes = build_s23_index(s23_records)
    
    s1_rec = normalize_record({
        "entity_id": "S1-001",
        "business_name": "Davis Family Office",
        "business_address": "88 Olive Circle, Lebanon, TN",
        "country": "US"
    })
    cands = generate_candidates_for_record(s1_rec, indexes)
    assert "S2-001" in cands


def test_features():
    r1 = normalize_record({
        "entity_id": "S1-001",
        "business_name": "Davis Family Office",
        "business_address": "88 Olive Circle, Lebanon, TN",
        "country": "US"
    })
    r2 = normalize_record({
        "entity_id": "S2-001",
        "business_name": "Davis Family Offie",
        "business_address": "88 OLIVE CIR, LEBANON, TN",
        "country": "US"
    })
    feats = extract_features_for_pair(r1, r2)
    assert feats["name_jw"] > 0.90
    assert feats["country_match"] == 1.0
    assert feats["addr_num_exact"] == 1.0
    for col in FEATURE_COLS:
        assert col in feats, f"Missing feature: {col}"

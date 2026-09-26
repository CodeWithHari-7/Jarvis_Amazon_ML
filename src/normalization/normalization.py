"""
Normalization module for JARVIS_CHECKER Business Entity Resolution.
Handles multi-language business names and addresses including:
- English / ASCII
- Indic scripts (Devanagari, Telugu, Gujarati, etc.)
- French / Latin extended (accent characters: é, è, à, ç, ü, etc.)
- Open-set countries
"""
import re
import unicodedata

# ── LEGAL SUFFIX TABLE ────────────────────────────────────────────────────────
# Sorted longest-first so we strip the longest matching suffix
_LEGAL_SUFFIXES = sorted([
    "private limited", "pvt ltd", "pvt. ltd.", "pvt. ltd", "pvt ltd.",
    "limited liability company", "limited liability partnership",
    "llp", "llc", "ltd.", "ltd", "limited", "inc.", "inc", "corp.", "corp",
    "corporation", "co.", "co", "company", "gmbh", "ag", "sa", "sas", "srl",
    "bv", "nv", "plc", "pty ltd", "pty", "lp", "lp.", "l.p.",
    "s.a.", "s.r.l.", "s.a.s.", "s.p.a.",
    "opc", "opc pvt ltd",
    # French legal
    "sarl", "sasu", "sca", "sci", "snc", "eurl",
], key=len, reverse=True)

_LEGAL_SUFFIX_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(s) for s in _LEGAL_SUFFIXES) + r')\s*$',
    re.IGNORECASE
)

# ── NORMALISATION HELPERS ─────────────────────────────────────────────────────
_MULTI_WS = re.compile(r'\s+')
_PUNCT_BUT_KEEP = re.compile(r"[^\w\s]")   # remove punctuation, keep word chars + space
_AND_PATTERN = re.compile(r'\s*&\s*', re.IGNORECASE)
_NUMBERED_PREFIX = re.compile(r'^\d+[\.\-]\s*')

# ── COUNTRY NORMALISATION MAP ─────────────────────────────────────────────────
_COUNTRY_MAP = {
    # India
    "india": "india", "in": "india", "ind": "india",
    # United States
    "united states": "united states", "us": "united states",
    "usa": "united states", "u.s.a.": "united states", "u.s.": "united states",
    "united states of america": "united states",
    # France
    "france": "france", "fr": "france",
    # United Kingdom
    "united kingdom": "united kingdom", "uk": "united kingdom",
    "gb": "united kingdom", "great britain": "united kingdom",
    # Germany
    "germany": "germany", "de": "germany", "deutschland": "germany",
    # Australia
    "australia": "australia", "au": "australia",
    # Canada
    "canada": "canada", "ca": "canada",
}

# ── ADDRESS ABBREVIATION MAP ──────────────────────────────────────────────────
_ADDR_ABBREV = {
    r'\bst\.?\b': 'street',
    r'\bave\.?\b': 'avenue',
    r'\brd\.?\b': 'road',
    r'\bblvd\.?\b': 'boulevard',
    r'\bdr\.?\b': 'drive',
    r'\bln\.?\b': 'lane',
    r'\bpl\.?\b': 'place',
    r'\bcrt\.?\b': 'court',
    r'\bct\.?\b': 'court',
    r'\bnr\.?\b': 'near',
    r'\bno\.?\b': 'number',
    r'\bph\.?\b': 'phase',
    r'\bsec\.?\b': 'sector',
    r'\bextn\.?\b': 'extension',
    r'\bext\.?\b': 'extension',
    r'\bflr\.?\b': 'floor',
    r'\bfl\.?\b': 'floor',
    r'\bopp\.?\b': 'opposite',
    r'\bapt\.?\b': 'apartment',
    r'\bbldg\.?\b': 'building',
}
_ADDR_ABBREV_COMPILED = [(re.compile(pat, re.IGNORECASE), rep) for pat, rep in _ADDR_ABBREV.items()]


def _to_ascii_compatible(text: str) -> str:
    """
    Converts accented Latin characters to ASCII equivalents (NFD decomposition).
    Preserves Indic script characters exactly.
    French: é->e, è->e, à->a, ç->c, etc.
    """
    result = []
    for ch in text:
        cp = ord(ch)
        # Preserve Indic scripts (Devanagari, Gujarati, Telugu, Kannada, Tamil, Malayalam, Bengali)
        if (0x0900 <= cp <= 0x097F or  # Devanagari
            0x0980 <= cp <= 0x09FF or  # Bengali
            0x0A00 <= cp <= 0x0A7F or  # Gurmukhi
            0x0A80 <= cp <= 0x0AFF or  # Gujarati
            0x0B00 <= cp <= 0x0B7F or  # Oriya
            0x0B80 <= cp <= 0x0BFF or  # Tamil
            0x0C00 <= cp <= 0x0C7F or  # Telugu
            0x0C80 <= cp <= 0x0CFF or  # Kannada
            0x0D00 <= cp <= 0x0D7F):   # Malayalam
            result.append(ch)
        elif cp > 127:
            # Try NFD decomposition to strip diacritics (handles French/Latin extended)
            decomposed = unicodedata.normalize('NFD', ch)
            ascii_chars = [c for c in decomposed if unicodedata.category(c) != 'Mn' and ord(c) < 128]
            if ascii_chars:
                result.extend(ascii_chars)
            else:
                result.append(ch)  # Keep if no ASCII equivalent found
        else:
            result.append(ch)
    return ''.join(result)


def normalize_business_name(name: str, strip_legal: bool = True) -> str:
    """
    Full normalization pipeline for business names.
    Returns normalized string.
    """
    if not name or not isinstance(name, str):
        return ''
    
    # Step 1: Convert accented chars to ASCII-compatible (preserve Indic)
    text = _to_ascii_compatible(name)
    
    # Step 2: Lowercase
    text = text.lower()
    
    # Step 3: & -> and
    text = _AND_PATTERN.sub(' and ', text)
    
    # Step 4: Strip legal suffixes BEFORE removing punctuation
    if strip_legal:
        text = _LEGAL_SUFFIX_PATTERN.sub('', text).strip()
    
    # Step 5: Remove punctuation (keep alphanumeric, spaces, Indic chars)
    # Build char-level filter
    cleaned = []
    for ch in text:
        cp = ord(ch)
        if (ch.isalnum() or ch == ' ' or 
            0x0900 <= cp <= 0x0D7F):  # all Indic ranges
            cleaned.append(ch)
        else:
            cleaned.append(' ')
    text = ''.join(cleaned)
    
    # Step 6: Normalize whitespace
    text = _MULTI_WS.sub(' ', text).strip()
    
    return text


def normalize_address(addr: str) -> str:
    """
    Normalization pipeline for business addresses.
    """
    if not addr or not isinstance(addr, str):
        return ''
    
    # Step 1: Convert accented Latin chars  
    text = _to_ascii_compatible(addr)
    
    # Step 2: Lowercase
    text = text.lower()
    
    # Step 3: Apply address abbreviation expansions
    for pattern, replacement in _ADDR_ABBREV_COMPILED:
        text = pattern.sub(replacement, text)
    
    # Step 4: Remove punctuation except digits/letters/spaces/Indic
    cleaned = []
    for ch in text:
        cp = ord(ch)
        if (ch.isalnum() or ch == ' ' or 0x0900 <= cp <= 0x0D7F):
            cleaned.append(ch)
        else:
            cleaned.append(' ')
    text = ''.join(cleaned)
    
    # Step 5: Normalize whitespace
    text = _MULTI_WS.sub(' ', text).strip()
    
    return text


def normalize_country(country: str) -> str:
    """
    Standardize country names to a canonical lowercase form.
    Unknown countries are kept as-is (normalized) for open-set handling.
    """
    if not country or not isinstance(country, str):
        return ''
    
    text = _to_ascii_compatible(country).lower().strip()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text).strip()
    text = _MULTI_WS.sub(' ', text).strip()
    
    return _COUNTRY_MAP.get(text, text)


def extract_numbers(text: str) -> list:
    """
    Extracts all numeric sequences from a text string.
    Critical for address matching (house numbers, PIN codes, etc.)
    Only returns numbers with >= 2 digits to avoid noise from single-digit tokens.
    """
    if not text or not isinstance(text, str):
        return []
    nums = re.findall(r'\d+', text)
    return [n for n in nums if len(n) >= 1]


def extract_tokens(text: str) -> set:
    """
    Tokenize normalized text into a set of tokens.
    Filters out very short tokens (< 2 chars) to reduce noise.
    """
    if not text or not isinstance(text, str):
        return set()
    return {tok for tok in text.split() if len(tok) >= 2}


def is_indic_script(text: str) -> bool:
    """
    Returns True if the text contains Indic script characters.
    """
    if not text or not isinstance(text, str):
        return False
    return bool(re.search(r'[\u0900-\u0D7F]', text))


def normalize_record(record: dict) -> dict:
    """
    Applies full normalization to a raw record dict.
    Expects keys: entity_id, business_name, business_address, country
    Returns enriched dict with *_normalized, *_tokens, etc.
    """
    def _safe_str(val):
        if val is None or (isinstance(val, float) and str(val) == 'nan'):
            return ''
        s = str(val).strip()
        return '' if s.lower() == 'nan' else s

    name_raw = _safe_str(record.get('business_name'))
    addr_raw = _safe_str(record.get('business_address'))
    country_raw = _safe_str(record.get('country'))
    entity_id = _safe_str(record.get('entity_id'))
    
    name_norm = normalize_business_name(name_raw)
    name_norm_full = normalize_business_name(name_raw, strip_legal=False)  # for features
    addr_norm = normalize_address(addr_raw)
    country_norm = normalize_country(country_raw)
    
    addr_numbers = extract_numbers(addr_raw)
    name_tokens = extract_tokens(name_norm)
    addr_tokens = extract_tokens(addr_norm)
    
    # Name prefix (first 4 chars) — used for blocking
    name_prefix4 = name_norm[:4] if len(name_norm) >= 4 else name_norm
    name_prefix3 = name_norm[:3] if len(name_norm) >= 3 else name_norm
    
    return {
        **record,
        'business_name_normalized': name_norm,
        'business_name_normalized_full': name_norm_full,
        'business_address_normalized': addr_norm,
        'country_normalized': country_norm,
        'business_address_numbers': addr_numbers,
        'name_tokens': name_tokens,
        'addr_tokens': addr_tokens,
        'name_prefix4': name_prefix4,
        'name_prefix3': name_prefix3,
        'is_indic': is_indic_script(name_raw),
    }


def normalize_dataframe_records(records: list) -> list:
    """
    Normalizes a list of record dicts in-place.
    """
    return [normalize_record(r) for r in records]

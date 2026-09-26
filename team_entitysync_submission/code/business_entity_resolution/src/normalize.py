"""
Text normalization module for Business Entity Resolution.
Handles multi-lingual (US, India, France) normalization of business names and addresses.
"""

import re
import unicodedata
from typing import Tuple, Optional

# Regex patterns
RE_URL_PREFIX = re.compile(r"^https?://(?:www\.)?|^www\.", re.IGNORECASE)
RE_DOMAIN_SUFFIX = re.compile(r"\.(?:com|org|net|in|fr|co\.in|co|io|biz|info)(?:/.*)?$", re.IGNORECASE)
RE_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
RE_WHITESPACE = re.compile(r"\s+")

# Legal suffix list across US, India, France
LEGAL_SUFFIXES = [
    # Multi-word suffixes
    "private limited", "pvt ltd", "pvt limited", "private ltd",
    "limited liability company", "llc",
    "public limited company", "plc",
    "limited liability partnership", "llp",
    "societe a responsabilite limitee", "societe anonyme",
    "societe par actions simplifiee", "societe civile immobiliere",
    "sarl", "sasu", "sas", "eurl", "sci", "snc", "gie",
    "corporation", "corp", "incorporated", "inc",
    "limited", "ltd", "holdings", "company", "co"
]

# Suffix pattern matching at end of string
RE_SUFFIX_END = re.compile(
    r"\b(?:" + "|".join(re.escape(s) for s in sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r")\s*$",
    re.IGNORECASE
)

# Address abbreviations mapping
ADDR_ABBR = {
    # US / International
    "st": "street", "str": "street",
    "rd": "road",
    "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bvd": "boulevard", "bd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "hwy": "highway",
    "pkwy": "parkway",
    "ste": "suite",
    "apt": "apartment",
    "bldg": "building",
    "fl": "floor",
    "sq": "square",
    # India
    "opp": "opposite",
    "nr": "near",
    "indl": "industrial",
    "ind": "industrial",
    "est": "estate",
    "sec": "sector",
    "dist": "district",
    "po": "post",
    # France
    "imp": "impasse",
    "chem": "chemin",
    "all": "allee",
    "pl": "place",
    "rte": "route",
    "r": "rue"
}

# Postal code patterns: 5 digits (US/France), 6 digits (India)
RE_POSTAL = re.compile(r"\b(\d{5}|\d{6})\b")


def strip_accents(text: str) -> str:
    """Decompose Unicode characters and remove accent diacritics."""
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def normalize_business_name(name: Optional[str]) -> Tuple[str, str]:
    """
    Normalize business name.
    
    Returns:
    - raw_clean: cleaned name with legal suffix preserved
    - core_name: name with trailing legal suffix stripped
    """
    if not name or not isinstance(name, str) or str(name).lower() == "nan":
        return "", ""
    
    # 1. Lowercase and strip accents
    s = strip_accents(name.lower().strip())
    
    # 2. Strip URL and domain patterns
    s = RE_URL_PREFIX.sub("", s)
    s = RE_DOMAIN_SUFFIX.sub("", s)
    
    # 3. Standardize symbols
    s = s.replace("&", " and ")
    s = s.replace("@", " at ")
    s = s.replace("+", " ")
    
    # 4. Remove punctuation
    s = RE_PUNCT.sub(" ", s)
    raw_clean = RE_WHITESPACE.sub(" ", s).strip()
    
    # 5. Extract core name by stripping trailing legal suffix (iteratively up to 2 times for e.g. "co ltd")
    core = raw_clean
    for _ in range(2):
        new_core = RE_SUFFIX_END.sub("", core).strip()
        if new_core and len(new_core) >= 2:
            core = new_core
        else:
            break
            
    core_name = RE_WHITESPACE.sub(" ", core).strip()
    if not core_name:
        core_name = raw_clean
        
    return raw_clean, core_name


def normalize_address(address: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Normalize address string.
    
    Returns:
    - clean_address: normalized address tokens
    - postal_code: extracted postal code if found
    """
    if not address or not isinstance(address, str) or str(address).lower() == "nan":
        return "", None
    
    s = strip_accents(address.lower().strip())
    
    # Find postal code
    postal_matches = RE_POSTAL.findall(s)
    postal_code = postal_matches[-1] if postal_matches else None
    
    # Punctuation & symbols
    s = s.replace("&", " and ")
    s = s.replace("-", " ")
    s = RE_PUNCT.sub(" ", s)
    tokens = s.split()
    
    # Expand abbreviations
    expanded = [ADDR_ABBR.get(t, t) for t in tokens]
    clean_address = " ".join(expanded).strip()
    
    return clean_address, postal_code

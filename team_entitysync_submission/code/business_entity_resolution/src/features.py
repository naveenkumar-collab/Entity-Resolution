"""
Pairwise Feature Engineering Module.
Extracts high-signal name, address, and interaction similarity features using RapidFuzz.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import rapidfuzz
from rapidfuzz import fuzz

FEATURE_NAMES = [
    # Name features
    "name_exact_clean",
    "name_exact_core",
    "name_compact_match",
    "name_fuzz_ratio",
    "name_fuzz_partial_ratio",
    "name_fuzz_token_sort",
    "name_fuzz_token_set",
    "name_core_fuzz_ratio",
    "name_len_diff",
    "name_len_ratio",
    "name_token_jaccard",
    "name_char3gram_jaccard",
    
    # Address features
    "is_s23_addr_missing",
    "is_both_addr_present",
    "addr_fuzz_token_sort",
    "addr_fuzz_token_set",
    "addr_token_jaccard",
    "postal_exact_match",
    "postal_mismatch",
    "number_match",
    
    # Interaction / Source
    "name_addr_geom_mean",
    "source_is_s2",
    "source_is_s3"
]


def char_ngrams(s: str, n: int = 3) -> set:
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def extract_pair_features(
    s1_nc: str, s1_core: str, s1_ca: str, s1_pc: Optional[str],
    m_nc: str, m_core: str, m_ca: str, m_pc: Optional[str],
    target_id: str
) -> List[float]:
    """
    Extract pairwise feature vector for a candidate pair (S1, S2/3).
    """
    # 1. Name features
    exact_clean = 1.0 if s1_nc == m_nc and s1_nc else 0.0
    exact_core = 1.0 if s1_core == m_core and s1_core else 0.0
    
    # Compact match (space stripped)
    s1_cmp = s1_core.replace(" ", "")
    m_cmp = m_core.replace(" ", "")
    cmp_match = 1.0 if s1_cmp == m_cmp and s1_cmp else 0.0
    
    fuzz_ratio = fuzz.ratio(s1_nc, m_nc) / 100.0
    fuzz_partial = fuzz.partial_ratio(s1_nc, m_nc) / 100.0
    fuzz_tsort = fuzz.token_sort_ratio(s1_nc, m_nc) / 100.0
    fuzz_tset = fuzz.token_set_ratio(s1_nc, m_nc) / 100.0
    core_fuzz_ratio = fuzz.ratio(s1_core, m_core) / 100.0
    
    len_diff = float(abs(len(s1_nc) - len(m_nc)))
    max_len = max(len(s1_nc), len(m_nc))
    len_ratio = min(len(s1_nc), len(m_nc)) / max_len if max_len > 0 else 0.0
    
    toks1 = set(s1_core.split())
    toks2 = set(m_core.split())
    tok_jaccard = jaccard(toks1, toks2)
    
    gram1 = char_ngrams(s1_core, 3)
    gram2 = char_ngrams(m_core, 3)
    gram_jaccard = jaccard(gram1, gram2)
    
    # 2. Address features
    s23_missing = 1.0 if not m_ca else 0.0
    both_present = 1.0 if (s1_ca and m_ca) else 0.0
    
    if both_present:
        addr_tsort = fuzz.token_sort_ratio(s1_ca, m_ca) / 100.0
        addr_tset = fuzz.token_set_ratio(s1_ca, m_ca) / 100.0
        addr_tok1 = set(s1_ca.split())
        addr_tok2 = set(m_ca.split())
        addr_tok_jaccard = jaccard(addr_tok1, addr_tok2)
        
        # Postal match
        if s1_pc and m_pc:
            postal_match = 1.0 if s1_pc == m_pc else 0.0
            postal_mismatch = 1.0 if s1_pc != m_pc else 0.0
        else:
            postal_match = 0.0
            postal_mismatch = 0.0
            
        # Street number match
        nums1 = {w for w in s1_ca.split() if w.isdigit()}
        nums2 = {w for w in m_ca.split() if w.isdigit()}
        num_match = 1.0 if (nums1 and nums2 and (nums1 & nums2)) else 0.0
    else:
        addr_tsort = 0.0
        addr_tset = 0.0
        addr_tok_jaccard = 0.0
        postal_match = 0.0
        postal_mismatch = 0.0
        num_match = 0.0
        
    # 3. Interactions
    if both_present:
        geom_mean = np.sqrt(fuzz_tset * addr_tset)
    else:
        geom_mean = fuzz_tset * 0.5
        
    is_s2 = 1.0 if target_id.startswith("S2-") else 0.0
    is_s3 = 1.0 if target_id.startswith("S3-") else 0.0
    
    return [
        exact_clean,
        exact_core,
        cmp_match,
        fuzz_ratio,
        fuzz_partial,
        fuzz_tsort,
        fuzz_tset,
        core_fuzz_ratio,
        len_diff,
        len_ratio,
        tok_jaccard,
        gram_jaccard,
        s23_missing,
        both_present,
        addr_tsort,
        addr_tset,
        addr_tok_jaccard,
        postal_match,
        postal_mismatch,
        num_match,
        geom_mean,
        is_s2,
        is_s3
    ]

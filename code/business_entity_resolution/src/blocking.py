"""
Candidate Generation and Multi-Pass Blocking Module.
Processes business records partitioned by country using multi-key inverted indexes.
Generates candidate pairs with high recall ceiling and strict budget control.
"""

import re
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import pandas as pd
from .normalize import normalize_business_name, normalize_address

STOPWORDS = {
    "and", "the", "for", "with", "from", "center", "centre", "group", "services",
    "consulting", "solutions", "international", "national", "global", "holdings",
    "enterprises", "associates", "management", "products", "systems", "trading",
    "tech", "india", "american", "us", "de", "la", "le", "des", "du", "et", "en",
    "of", "co", "corp", "inc", "ltd", "llc", "sarl", "sas", "pvt", "limited"
}

def extract_blocking_keys(name_clean: str, name_core: str, addr_clean: str, postal: Optional[str]) -> Set[str]:
    """
    Extract multi-pass blocking keys from normalized name and address.
    """
    keys = set()
    
    # 1. Exact core name (weight: high)
    if len(name_core) >= 4:
        keys.add(f"nc:{name_core}")
        
    # 2. Compact / space-stripped core name (handles squashed URLs / domain names)
    compact = name_core.replace(" ", "")
    if len(compact) >= 5:
        keys.add(f"cmp:{compact[:25]}")
        
    # 3. Name tokens (length >= 4 and not in STOPWORDS)
    tokens = [t for t in name_core.split() if len(t) >= 4 and t not in STOPWORDS]
    for t in tokens:
        keys.add(f"tok:{t}")
        if len(t) >= 6:
            keys.add(f"p4:{t[:4]}")
        
    # 4. First 2 tokens joined (phrase prefix)
    all_tokens = [t for t in name_core.split() if t not in STOPWORDS]
    if len(all_tokens) >= 2:
        keys.add(f"pfx2:{all_tokens[0]}_{all_tokens[1]}")
    elif len(all_tokens) == 1 and len(all_tokens[0]) >= 4:
        keys.add(f"pfx1:{all_tokens[0][:5]}")
        
    # 5. Address keys (house/building number + street word or postal code)
    if addr_clean:
        nums = re.findall(r"\b\d+\b", addr_clean)
        addr_words = [w for w in addr_clean.split() if w.isalpha() and len(w) >= 4 and w not in STOPWORDS]
        if nums and addr_words:
            num = nums[0]
            for w in addr_words[:2]:
                keys.add(f"addr:{num}_{w}")
        if postal and nums:
            keys.add(f"post:{postal}_{nums[0]}")
        elif postal and addr_words:
            keys.add(f"post_str:{postal}_{addr_words[0]}")
            
    return keys


class CountryBlocker:
    """
    Inverted index blocker for a single country partition.
    Indexes Target entities (S2 and S3) and queries Reference entities (S1).
    """
    def __init__(self, country: str, max_candidates_per_entity: int = 20, max_postings_per_key: int = 300):
        self.country = country
        self.max_candidates = max_candidates_per_entity
        self.max_postings = max_postings_per_key
        self.index: Dict[str, List[str]] = defaultdict(list)
        self.target_meta: Dict[str, Tuple[str, str, str, Optional[str]]] = {}
        
    def add_target_records(self, df_records: pd.DataFrame):
        """
        Populate inverted index with S2 / S3 records.
        Expected columns: entity_id, business_name, business_address
        """
        for row in df_records.itertuples(index=False):
            eid = row.entity_id
            raw_name = str(row.business_name) if pd.notna(row.business_name) else ""
            raw_addr = str(row.business_address) if pd.notna(row.business_address) else ""
            
            nc, core = normalize_business_name(raw_name)
            ca, pc = normalize_address(raw_addr)
            
            self.target_meta[eid] = (nc, core, ca, pc)
            keys = extract_blocking_keys(nc, core, ca, pc)
            for k in keys:
                self.index[k].append(eid)
                
    def generate_candidates_for_s1(
        self,
        s1_df: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """
        Generate ranked candidate target IDs for each S1 entity in s1_df.
        Returns {s1_entity_id: [candidate_target_ids]}
        """
        results: Dict[str, List[str]] = {}
        
        for row in s1_df.itertuples(index=False):
            s1_id = row.entity_id
            raw_name = str(row.business_name) if pd.notna(row.business_name) else ""
            raw_addr = str(row.business_address) if pd.notna(row.business_address) else ""
            
            s1_nc, s1_core = normalize_business_name(raw_name)
            s1_ca, s1_pc = normalize_address(raw_addr)
            s1_keys = extract_blocking_keys(s1_nc, s1_core, s1_ca, s1_pc)
            
            # Tally candidate frequency across matching keys
            cand_scores: Dict[str, float] = defaultdict(float)
            for k in s1_keys:
                target_ids = self.index.get(k)
                if not target_ids:
                    continue
                # Skip overly broad keys that match too many records
                if len(target_ids) > self.max_postings:
                    continue
                    
                # Weight keys: exact core & compact get highest priority
                if k.startswith("nc:"):
                    weight = 6.0
                elif k.startswith("cmp:"):
                    weight = 5.0
                elif k.startswith("pfx2:"):
                    weight = 4.0
                elif k.startswith("post:") or k.startswith("addr:"):
                    weight = 3.0
                elif k.startswith("tok:"):
                    weight = 2.0
                else:
                    weight = 1.0
                    
                for tid in target_ids:
                    cand_scores[tid] += weight
                    
            if not cand_scores:
                results[s1_id] = []
            else:
                # Top K candidates sorted by cumulative score
                sorted_cands = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
                results[s1_id] = [c[0] for c in sorted_cands[:self.max_candidates]]
                
        return results

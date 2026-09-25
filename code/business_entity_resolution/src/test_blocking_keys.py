"""
Benchmarking blocking keys and candidate recall on training sample.
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pandas as pd
import re
from collections import defaultdict
from src.normalize import normalize_business_name, normalize_address

STOPWORDS = {
    "and", "the", "for", "with", "from", "center", "centre", "group", "services",
    "consulting", "solutions", "international", "national", "global", "holdings",
    "enterprises", "associates", "management", "products", "systems", "trading",
    "tech", "india", "american", "us", "de", "la", "le", "des", "du"
}

def extract_blocking_keys(name_clean: str, name_core: str, addr_clean: str, postal: str):
    keys = set()
    
    # 1. Exact core name (if >= 4 chars)
    if len(name_core) >= 4:
        keys.add(f"nc:{name_core}")
        
    # 2. Compact / space-stripped core name (handles websites like maurewilliamscolombier.com)
    compact = name_core.replace(" ", "")
    if len(compact) >= 5:
        keys.add(f"cmp:{compact[:25]}")
        
    # 3. Name tokens (length >= 4 and not stopword)
    tokens = [t for t in name_core.split() if len(t) >= 4 and t not in STOPWORDS]
    for t in tokens:
        keys.add(f"tok:{t}")
        
    # 4. First 2 tokens joined (e.g. 'maure_williams')
    all_tokens = name_core.split()
    if len(all_tokens) >= 2:
        keys.add(f"pfx2:{all_tokens[0]}_{all_tokens[1]}")
        
    # 5. Address keys (number + street token)
    if addr_clean:
        nums = re.findall(r"\b\d+\b", addr_clean)
        addr_words = [w for w in addr_clean.split() if w.isalpha() and len(w) >= 4 and w not in STOPWORDS]
        if nums and addr_words:
            # Pair first number with first 2 street words
            num = nums[0]
            for w in addr_words[:2]:
                keys.add(f"addr:{num}_{w}")
        if postal and nums:
            keys.add(f"post:{postal}_{nums[0]}")
            
    return keys

print("Testing blocking key extraction...")
k = extract_blocking_keys(
    "maure williams colombier inc",
    "maure williams colombier",
    "85 wayne avenue ticonderoga ny 12883",
    "12883"
)
for item in sorted(k):
    print(" ", item)

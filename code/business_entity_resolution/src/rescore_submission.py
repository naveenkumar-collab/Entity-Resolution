"""
Ultra-Fast Batched Re-Scoring Script for Business Entity Resolution Challenge.
Batch-scores candidate pairs using LightGBM Model V2.
Preserves strict subset compliance and generates matching_results.tsv in minutes.
"""

import sys
import os
import time
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if not (PROJECT_ROOT / "code").exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.normalize import normalize_business_name, normalize_address
from src.features import extract_pair_features
from src.model import EntityMatchingModel

def rescore_batched(
    test_dir: str = "dataset/test",
    candidate_path: str = "output/candidate_pairs.tsv",
    output_matching_path: str = "output/matching_results.tsv",
    model_path: str = "model_lgb.joblib",
    threshold: float = 0.55,
    singleton_threshold: float = 0.55,
    batch_size: int = 5000
):
    start_time = time.time()
    print("=" * 70, flush=True)
    print("ULTRA-FAST BATCHED RESCORE WITH MODEL V2", flush=True)
    print(f"Test Directory:      {test_dir}", flush=True)
    print(f"Candidate File:      {candidate_path}", flush=True)
    print(f"Output Matching:     {output_matching_path}", flush=True)
    print(f"Model Path:          {model_path}", flush=True)
    print(f"Decision Threshold:  {threshold:.2f}", flush=True)
    print(f"Singleton Threshold: {singleton_threshold:.2f}", flush=True)
    print("=" * 70, flush=True)
    
    # 1. Load Model
    print(f"Loading model from {model_path}...", flush=True)
    model = EntityMatchingModel.load(model_path)
    
    # 2. Load candidate pairs mapping
    print(f"Reading candidate pairs from {candidate_path}...", flush=True)
    cands_by_s1: Dict[str, List[str]] = {}
    with open(candidate_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            cands_by_s1[parts[0]] = parts[1].split(",") if len(parts) > 1 and parts[1] else []
            
    print(f"Loaded candidates for {len(cands_by_s1):,} S1 entities.", flush=True)
    
    # 3. Read test_source1 to preserve strict ordering and group by country
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    print(f"Reading {s1_path}...", flush=True)
    s1_df = pd.read_csv(s1_path, sep="\t")
    ordered_s1_ids = s1_df["entity_id"].tolist()
    
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")
    
    final_matches: Dict[str, List[str]] = {}
    test_countries = s1_df["country"].unique().tolist()
    
    for country in test_countries:
        c_t0 = time.time()
        print(f"\n{'='*40}", flush=True)
        print(f"PROCESSING COUNTRY: {country}", flush=True)
        print(f"{'='*40}", flush=True)
        
        country_s1 = s1_df[s1_df["country"] == country]
        country_s1_ids = country_s1["entity_id"].tolist()
        
        # Collect needed target candidate IDs for this country
        needed_target_ids = set()
        for sid in country_s1_ids:
            needed_target_ids.update(cands_by_s1.get(sid, []))
            
        print(f"  Country S1 count: {len(country_s1):,}", flush=True)
        print(f"  Needed target candidates: {len(needed_target_ids):,}", flush=True)
        
        # Stream S2 and S3 to load metadata only for needed candidates in this country
        target_meta: Dict[str, Tuple[str, str, str, Optional[str]]] = {}
        
        for t_path in [s2_path, s3_path]:
            print(f"  Streaming {os.path.basename(t_path)} for {country} candidates...", flush=True)
            with open(t_path, "r", encoding="utf-8") as f:
                header = [h.strip().lower() for h in next(f).split("\t")]
                eid_idx = header.index("entity_id")
                name_idx = header.index("business_name")
                addr_idx = header.index("business_address")
                
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) <= max(eid_idx, name_idx, addr_idx):
                        continue
                    eid = parts[eid_idx]
                    if eid in needed_target_ids:
                        nc, core = normalize_business_name(parts[name_idx])
                        ca, pc = normalize_address(parts[addr_idx])
                        target_meta[eid] = (nc, core, ca, pc)
                        
        print(f"  Loaded metadata for {len(target_meta):,} target records in {country}.", flush=True)
        
        # Pre-normalize S1 records
        print(f"  Pre-normalizing {len(country_s1):,} S1 records...", flush=True)
        s1_meta_list = []
        for row in country_s1.itertuples(index=False):
            sid = row.entity_id
            nc, core = normalize_business_name(row.business_name)
            ca, pc = normalize_address(row.business_address)
            s1_meta_list.append((sid, (nc, core, ca, pc)))
            
        # Batch inference
        print(f"  Running batched inference across {len(s1_meta_list):,} entities...", flush=True)
        n_matched = 0
        total_entities = len(s1_meta_list)
        
        for b_start in range(0, total_entities, batch_size):
            b_end = min(b_start + batch_size, total_entities)
            batch = s1_meta_list[b_start:b_end]
            
            batch_features = []
            entity_cand_slices = []  # [(sid, [cids], start_idx, end_idx)]
            
            for sid, s1_meta in batch:
                cands = cands_by_s1.get(sid, [])
                if not cands:
                    final_matches[sid] = []
                    continue
                    
                s1_nc, s1_core, s1_ca, s1_pc = s1_meta
                valid_cands = []
                cand_start = len(batch_features)
                
                for cid in cands:
                    if cid not in target_meta:
                        continue
                    m_nc, m_core, m_ca, m_pc = target_meta[cid]
                    feat = extract_pair_features(
                        s1_nc, s1_core, s1_ca, s1_pc,
                        m_nc, m_core, m_ca, m_pc,
                        cid
                    )
                    batch_features.append(feat)
                    valid_cands.append(cid)
                    
                cand_end = len(batch_features)
                entity_cand_slices.append((sid, valid_cands, cand_start, cand_end))
                
            if batch_features:
                X_batch = np.array(batch_features, dtype=np.float32)
                probs = model.predict_pair_probs(X_batch)
                
                for sid, valid_cands, c_start, c_end in entity_cand_slices:
                    if c_start == c_end:
                        final_matches[sid] = []
                        continue
                        
                    cand_probs = probs[c_start:c_end]
                    max_p = np.max(cand_probs)
                    
                    # Singleton Shield
                    if max_p < singleton_threshold:
                        final_matches[sid] = []
                    else:
                        selected = [cid for cid, p in zip(valid_cands, cand_probs) if p >= threshold]
                        final_matches[sid] = selected
                        if selected:
                            n_matched += 1
            else:
                for sid, _, _, _ in entity_cand_slices:
                    final_matches[sid] = []
                    
            if b_end % 50000 < batch_size or b_end == total_entities:
                elapsed = time.time() - c_t0
                rate = b_end / elapsed if elapsed > 0 else 0
                print(f"    Processed {b_end:,}/{total_entities:,} ({n_matched:,} matched, {rate:.0f} ent/s)...", flush=True)
                
        print(f"  Finished {country} in {time.time() - c_t0:.1f}s. Matched entities: {n_matched:,}/{len(country_s1):,}", flush=True)
        del target_meta, s1_meta_list
        gc.collect()
        
    # Write output matching_results.tsv
    print(f"\nWriting new matches to {output_matching_path}...", flush=True)
    with open(output_matching_path, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in ordered_s1_ids:
            m_str = ",".join(final_matches.get(sid, []))
            f_out.write(f"{sid}\t{m_str}\n")
            
    print(f"\nRe-scoring successfully finished in {time.time() - start_time:.1f}s!", flush=True)
    print(f"File size: {os.path.getsize(output_matching_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--candidate-path", default="output/candidate_pairs.tsv")
    parser.add_argument("--output-matching", default="output/matching_results.tsv")
    parser.add_argument("--model-path", default="model_lgb.joblib")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--singleton-threshold", type=float, default=0.55)
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()
    
    rescore_batched(
        test_dir=args.test_dir,
        candidate_path=args.candidate_path,
        output_matching_path=args.output_matching,
        model_path=args.model_path,
        threshold=args.threshold,
        singleton_threshold=args.singleton_threshold,
        batch_size=args.batch_size
    )

"""
End-to-End Inference Pipeline for Business Entity Resolution Challenge.
Generates matching_results.tsv and candidate_pairs.tsv for the test dataset.
Partitioned by country (France, US, India) to maintain memory efficiency and guarantee zero cross-country noise.
"""

import sys
import os
import argparse
import time
import gc
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.normalize import normalize_business_name, normalize_address
from src.blocking import CountryBlocker
from src.model import EntityMatchingModel


def stream_country_target_records(
    file_path: str,
    target_country: str,
    chunksize: int = 200000
) -> List[Tuple[str, str, str]]:
    """
    Read (entity_id, business_name, business_address) for records matching target_country.
    """
    print(f"    Scanning {os.path.basename(file_path)} for country '{target_country}'...", flush=True)
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        header_line = next(f)
        headers = [h.strip().lower() for h in header_line.split("\t")]
        eid_idx = headers.index("entity_id")
        name_idx = headers.index("business_name")
        addr_idx = headers.index("business_address")
        country_idx = headers.index("country")
        
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(eid_idx, name_idx, addr_idx, country_idx):
                continue
            if parts[country_idx] == target_country:
                records.append((parts[eid_idx], parts[name_idx], parts[addr_idx]))
                
    return records


def run_pipeline(
    test_dir: str = "dataset/test",
    output_dir: str = "output",
    model_path: str = "model_lgb.joblib",
    match_threshold: float = 0.60,
    singleton_threshold: float = 0.65
):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70, flush=True)
    print("BUSINESS ENTITY RESOLUTION — END-TO-END INFERENCE PIPELINE", flush=True)
    print(f"Test Directory:      {test_dir}", flush=True)
    print(f"Output Directory:    {output_dir}", flush=True)
    print(f"Match Threshold:     {match_threshold:.2f}", flush=True)
    print(f"Singleton Threshold: {singleton_threshold:.2f}", flush=True)
    print("="*70, flush=True)
    
    # 1. Load Model
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...", flush=True)
        model = EntityMatchingModel.load(model_path)
    else:
        # Fallback to repository root
        repo_root = Path(__file__).resolve().parents[3]
        fallback_path = repo_root / os.path.basename(model_path)
        if not fallback_path.exists():
            fallback_path = repo_root / "student_resource" / os.path.basename(model_path)
        print(f"Loading model from {fallback_path}...", flush=True)
        model = EntityMatchingModel.load(str(fallback_path))
        
    model.threshold = match_threshold
    model.singleton_threshold = singleton_threshold
    
    # 2. Read test_source1 to preserve strict entity ordering
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    print(f"\nReading {s1_path}...", flush=True)
    s1_df = pd.read_csv(s1_path, sep="\t")
    ordered_s1_ids = s1_df["entity_id"].tolist()
    print(f"Total S1 test records to resolve: {len(ordered_s1_ids):,}", flush=True)
    print("Country distribution in S1 test set:")
    print(s1_df["country"].value_counts())
    
    final_matches: Dict[str, List[str]] = {}
    final_candidates: Dict[str, List[str]] = {}
    
    # Identify unique countries in test set (e.g. France, US, India)
    test_countries = s1_df["country"].unique().tolist()
    print(f"\nProcessing countries: {test_countries}", flush=True)
    
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")
    
    for country in test_countries:
        c_t0 = time.time()
        print(f"\n{'='*40}", flush=True)
        print(f"STARTING COUNTRY PARTITION: {country}", flush=True)
        print(f"{'='*40}", flush=True)
        
        country_s1 = s1_df[s1_df["country"] == country]
        print(f"  S1 entities in {country}: {len(country_s1):,}", flush=True)
        
        # Stream S2 and S3 for this country
        s2_records = stream_country_target_records(s2_path, country)
        s3_records = stream_country_target_records(s3_path, country)
        total_targets = len(s2_records) + len(s3_records)
        print(f"  Target pool: {len(s2_records):,} S2 + {len(s3_records):,} S3 = {total_targets:,} total records.", flush=True)
        
        # Build CountryBlocker
        print(f"  Indexing target records into CountryBlocker...", flush=True)
        blocker = CountryBlocker(country=country, max_candidates_per_entity=15, max_postings_per_key=300)
        
        # Convert records to DataFrame for indexer
        target_df = pd.DataFrame(
            s2_records + s3_records,
            columns=["entity_id", "business_name", "business_address"]
        )
        del s2_records, s3_records
        gc.collect()
        
        blocker.add_target_records(target_df)
        print(f"  Target indexing complete. Total indexed keys: {len(blocker.index):,}", flush=True)
        
        # Candidate Generation
        print(f"  Generating candidates for {len(country_s1):,} S1 entities...", flush=True)
        cands_dict = blocker.generate_candidates_for_s1(country_s1)
        
        # Inference & Scoring
        print(f"  Scoring candidates and applying singleton filter...", flush=True)
        n_resolved = 0
        n_matched = 0
        
        for row in country_s1.itertuples(index=False):
            sid = row.entity_id
            cands = cands_dict.get(sid, [])
            final_candidates[sid] = cands
            
            if not cands:
                final_matches[sid] = []
                continue
                
            s1_nc, s1_core = normalize_business_name(row.business_name)
            s1_ca, s1_pc = normalize_address(row.business_address)
            s1_meta = (s1_nc, s1_core, s1_ca, s1_pc)
            
            scored = model.score_candidates_for_s1(s1_meta, cands, blocker.target_meta)
            preds = model.filter_matches(scored)
            
            final_matches[sid] = preds
            if preds:
                n_matched += 1
            n_resolved += 1
            
            if n_resolved % 100000 == 0:
                print(f"    Resolved {n_resolved:,}/{len(country_s1):,} ({n_matched:,} matched)...", flush=True)
                
        print(f"  Finished {country} in {time.time() - c_t0:.1f}s. Matched entities: {n_matched:,}/{len(country_s1):,}", flush=True)
        
        # Cleanup partition memory
        del blocker, target_df, cands_dict
        gc.collect()
        
    # 3. Write Output Files
    matching_out_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_out_path = os.path.join(output_dir, "candidate_pairs.tsv")
    
    print(f"\nWriting final outputs...", flush=True)
    print(f"  Writing {matching_out_path}...", flush=True)
    with open(matching_out_path, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in ordered_s1_ids:
            matches_str = ",".join(final_matches.get(sid, []))
            f_match.write(f"{sid}\t{matches_str}\n")
            
    print(f"  Writing {candidate_out_path}...", flush=True)
    with open(candidate_out_path, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in ordered_s1_ids:
            cands_str = ",".join(final_candidates.get(sid, []))
            f_cand.write(f"{sid}\t{cands_str}\n")
            
    print(f"\nPipeline successfully completed in {time.time() - start_time:.1f}s!", flush=True)
    print(f"  matching_results.tsv size: {os.path.getsize(matching_out_path) / (1024*1024):.2f} MB")
    print(f"  candidate_pairs.tsv size:  {os.path.getsize(candidate_out_path) / (1024*1024):.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Business Entity Resolution Inference Pipeline")
    parser.add_argument("--test-dir", type=str, default="dataset/test", help="Path to test directory")
    parser.add_argument("--output-dir", type=str, default="output", help="Path to output directory")
    parser.add_argument("--model-path", type=str, default="model_lgb.joblib", help="Path to model file")
    parser.add_argument("--threshold", type=float, default=0.60, help="Match decision threshold")
    parser.add_argument("--singleton-threshold", type=float, default=0.65, help="Singleton shield threshold")
    args = parser.parse_args()
    
    run_pipeline(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        match_threshold=args.threshold,
        singleton_threshold=args.singleton_threshold
    )

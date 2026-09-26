"""
Benchmark blocking recall ceiling and candidate reduction ratio on validation set.
"""

import sys
sys.path.append("D:/evalution/student_resource/code/business_entity_resolution")
import time
import pandas as pd
from src.blocking import CountryBlocker

def run_benchmark():
    print("Loading validation S1 (5,000 records sample)...")
    val_s1 = pd.read_csv("D:/evalution/student_resource/val_source1_30k.tsv", sep="\t").head(5000)
    val_gt = pd.read_csv("D:/evalution/student_resource/val_ground_truth_30k.tsv", sep="\t").head(5000)
    
    # Map true matches
    gt_map = {}
    total_true_matches = 0
    all_needed_mids = set()
    for _, r in val_gt.iterrows():
        sid = r["source1_entity_id"]
        raw_m = str(r["matched_entity_ids"]) if pd.notna(r["matched_entity_ids"]) else ""
        mids = set(raw_m.split(",")) if raw_m else set()
        gt_map[sid] = mids
        total_true_matches += len(mids)
        all_needed_mids.update(mids)
        
    print(f"Total S1 entities: {len(val_s1):,}")
    print(f"Total True Matches: {total_true_matches:,}")
    
    # Load corresponding target records from S2 and S3 + a sample of background distractors (100,000 records)
    print("Loading target records (true matches + distractors)...")
    s2_chunks = pd.read_csv("D:/evalution/student_resource/dataset/train/train_source2.tsv", sep="\t", nrows=200000)
    s3_chunks = pd.read_csv("D:/evalution/student_resource/dataset/train/train_source3.tsv", sep="\t", nrows=200000)
    
    # Also load the specific true match records if not already in chunks
    needed_missing = all_needed_mids - set(s2_chunks["entity_id"]) - set(s3_chunks["entity_id"])
    print(f"Loading {len(needed_missing):,} true matches that were outside initial 400k chunk...")
    
    # Read needed missing
    if needed_missing:
        # Load from s2 and s3 efficiently
        s2_extra = []
        with open("D:/evalution/student_resource/dataset/train/train_source2.tsv", "r", encoding="utf-8") as f:
            header = next(f).strip().split("\t")
            for line in f:
                parts = line.strip().split("\t")
                if parts[0] in needed_missing:
                    s2_extra.append(parts)
        if s2_extra:
            df_extra2 = pd.DataFrame(s2_extra, columns=header)
            s2_chunks = pd.concat([s2_chunks, df_extra2], ignore_index=True)
            
        s3_extra = []
        with open("D:/evalution/student_resource/dataset/train/train_source3.tsv", "r", encoding="utf-8") as f:
            header = next(f).strip().split("\t")
            for line in f:
                parts = line.strip().split("\t")
                if parts[0] in needed_missing:
                    s3_extra.append(parts)
        if s3_extra:
            df_extra3 = pd.DataFrame(s3_extra, columns=header)
            s3_chunks = pd.concat([s3_chunks, df_extra3], ignore_index=True)

    print(f"Combined target search pool: {len(s2_chunks) + len(s3_chunks):,} records.")
    
    # Run blocking per country
    total_candidates_generated = 0
    recalled_matches = 0
    t0 = time.time()
    
    for country in ["US", "India"]:
        c_s1 = val_s1[val_s1["country"] == country]
        c_s2 = s2_chunks[s2_chunks["country"] == country]
        c_s3 = s3_chunks[s3_chunks["country"] == country]
        c_target = pd.concat([c_s2, c_s3], ignore_index=True)
        
        print(f"\n--- Processing Country: {country} ---")
        print(f"  S1 count: {len(c_s1):,}, Target count: {len(c_target):,}")
        
        blocker = CountryBlocker(country=country, max_candidates_per_entity=15)
        blocker.add_target_records(c_target)
        cands = blocker.generate_candidates_for_s1(c_s1)
        
        # Evaluate recall
        for sid, cand_list in cands.items():
            total_candidates_generated += len(cand_list)
            true_set = gt_map.get(sid, set())
            recalled_matches += len(set(cand_list) & true_set)
            
    t1 = time.time()
    
    recall = recalled_matches / max(1, total_true_matches) * 100
    avg_cands = total_candidates_generated / len(val_s1)
    
    print("\n" + "="*50)
    print("BLOCKING BENCHMARK RESULTS")
    print("="*50)
    print(f"Time Taken:               {t1-t0:.2f} seconds")
    print(f"Total S1 Evaluated:       {len(val_s1):,}")
    print(f"Total True Matches:       {total_true_matches:,}")
    print(f"Recalled Matches:         {recalled_matches:,} ({recall:.2f}%)")
    print(f"Avg Candidates / S1:      {avg_cands:.2f}")
    print(f"Reduction Ratio:          {1.0 - (avg_cands / (len(s2_chunks)+len(s3_chunks))):.6f}")

if __name__ == "__main__":
    run_benchmark()

"""
Create a stratified held-out validation split (30,000 S1 entities).
Stratified by country (US, India) and match count (singleton vs non-singleton).
"""

import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if not (PROJECT_ROOT / "code").exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

def create_val_split(
    s1_path: str = str(PROJECT_ROOT / "dataset" / "train" / "train_source1.tsv"),
    gt_path: str = str(PROJECT_ROOT / "dataset" / "train" / "train_ground_truth.tsv"),
    val_out_path: str = str(PROJECT_ROOT / "val_ground_truth_30k.tsv"),
    val_s1_out_path: str = str(PROJECT_ROOT / "val_source1_30k.tsv"),
    n_samples: int = 30000,
    random_state: int = 42
):
    print("Loading S1 metadata...")
    s1_df = pd.read_csv(s1_path, sep="\t")
    print(f"Loaded {len(s1_df):,} S1 records.")
    
    print("Loading Ground Truth...")
    gt_df = pd.read_csv(gt_path, sep="\t")
    print(f"Loaded {len(gt_df):,} Ground Truth records.")
    
    merged = pd.merge(s1_df, gt_df, left_on="entity_id", right_on="source1_entity_id")
    merged["is_singleton"] = merged["matched_entity_ids"].isna() | (merged["matched_entity_ids"] == "")
    merged["strata"] = merged["country"].astype(str) + "_" + merged["is_singleton"].astype(str)
    
    print("Stratification distribution in full dataset:")
    print(merged["strata"].value_counts(normalize=True))
    
    # Train / validation split using scikit-learn
    _, val_df = train_test_split(
        merged,
        test_size=n_samples,
        stratify=merged["strata"],
        random_state=random_state
    )
    
    print(f"\nSampled {len(val_df):,} validation records.")
    print("Validation strata counts:")
    print(val_df["strata"].value_counts())
    
    # Save validation ground truth
    val_gt = val_df[["source1_entity_id", "matched_entity_ids"]]
    val_gt.to_csv(val_out_path, sep="\t", index=False)
    print(f"Saved validation ground truth to {val_out_path}")
    
    # Save validation S1
    val_s1 = val_df[["entity_id", "business_name", "business_address", "country"]]
    val_s1.to_csv(val_s1_out_path, sep="\t", index=False)
    print(f"Saved validation S1 to {val_s1_out_path}")

if __name__ == "__main__":
    create_val_split()

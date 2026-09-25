"""
Training and Validation Optimization Pipeline.
Generates training pairs (positives + hard negatives), trains LightGBM matcher,
and tunes decision threshold and singleton shield to maximize validation Macro F_0.5.
"""

import sys
import os
from pathlib import Path

# Dynamic path resolution
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if not (PROJECT_ROOT / "code").exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(Path(__file__).resolve().parents[1]))
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
from src.normalize import normalize_business_name, normalize_address
from src.blocking import CountryBlocker
from src.features import extract_pair_features, FEATURE_NAMES
from src.model import EntityMatchingModel
from src.metrics import evaluate_predictions


def build_training_pairs(
    n_train_s1: int = 8000,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build labeled pairwise training dataset (positives and hard negatives).
    """
    print(f"Building training set from {n_train_s1:,} S1 entities...", flush=True)
    
    # Load validation S1 IDs to exclude them from training
    val_s1_path = PROJECT_ROOT / "val_source1_30k.tsv"
    val_s1_ids = set(pd.read_csv(str(val_s1_path), sep="\t", usecols=["entity_id"])["entity_id"])
    
    # Load S1 train records excluding validation IDs
    train_s1_records = []
    train_s1_path = PROJECT_ROOT / "dataset" / "train" / "train_source1.tsv"
    with open(str(train_s1_path), "r", encoding="utf-8") as f:
        header = next(f).strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] not in val_s1_ids:
                train_s1_records.append(parts)
                if len(train_s1_records) >= n_train_s1:
                    break
                    
    df_s1 = pd.DataFrame(train_s1_records, columns=header)
    s1_ids_set = set(df_s1["entity_id"])
    
    # Load ground truth for these S1
    gt_map: Dict[str, Set[str]] = {}
    needed_target_ids: Set[str] = set()
    train_gt_path = PROJECT_ROOT / "dataset" / "train" / "train_ground_truth.tsv"
    with open(str(train_gt_path), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] in s1_ids_set:
                mids = set(parts[1].split(",")) if len(parts) > 1 and parts[1] else set()
                gt_map[parts[0]] = mids
                needed_target_ids.update(mids)
                
    print(f"Loaded {len(df_s1):,} training S1 entities with {sum(len(v) for v in gt_map.values()):,} true matches.", flush=True)
    
    # Load target records (true matches + distractors)
    print("Loading target pool for training...", flush=True)
    s2_path = PROJECT_ROOT / "dataset" / "train" / "train_source2.tsv"
    s3_path = PROJECT_ROOT / "dataset" / "train" / "train_source3.tsv"
    s2_df = pd.read_csv(str(s2_path), sep="\t", nrows=250000)
    s3_df = pd.read_csv(str(s3_path), sep="\t", nrows=250000)
    
    # Run blocking per country to produce candidate pairs
    X_rows = []
    y_rows = []
    
    for country in ["US", "India"]:
        c_s1 = df_s1[df_s1["country"] == country]
        c_target = pd.concat([
            s2_df[s2_df["country"] == country],
            s3_df[s3_df["country"] == country]
        ], ignore_index=True)
        
        print(f"  Blocking {country}: S1={len(c_s1):,}, Target={len(c_target):,}...", flush=True)
        blocker = CountryBlocker(country=country, max_candidates_per_entity=15, max_postings_per_key=300)
        blocker.add_target_records(c_target)
        cands_dict = blocker.generate_candidates_for_s1(c_s1)
        
        print(f"  Extracting features for {country} candidate pairs...", flush=True)
        for _, row in c_s1.iterrows():
            sid = row["entity_id"]
            true_matches = gt_map.get(sid, set())
            cands = set(cands_dict.get(sid, []))
            
            # Combine generated candidates with true matches (if present in target pool)
            all_cands = cands | (true_matches & set(blocker.target_meta.keys()))
            
            s1_nc, s1_core = normalize_business_name(row["business_name"])
            s1_ca, s1_pc = normalize_address(row["business_address"])
            
            for cid in all_cands:
                if cid not in blocker.target_meta:
                    continue
                m_nc, m_core, m_ca, m_pc = blocker.target_meta[cid]
                feat = extract_pair_features(
                    s1_nc, s1_core, s1_ca, s1_pc,
                    m_nc, m_core, m_ca, m_pc,
                    cid
                )
                label = 1 if cid in true_matches else 0
                X_rows.append(feat)
                y_rows.append(label)
                
    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    print(f"Built dataset: {len(y):,} pairs | Positives: {(y == 1).sum():,} ({np.mean(y)*100:.2f}%)", flush=True)
    return X, y


def train_and_optimize():
    t0 = time.time()
    X, y = build_training_pairs(n_train_s1=8000)
    
    # Train / Dev split for early stopping
    from sklearn.model_selection import train_test_split
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    
    print("\nTraining LightGBM model...", flush=True)
    matching_model = EntityMatchingModel(n_estimators=300, learning_rate=0.06, num_leaves=31)
    matching_model.fit(X_train, y_train, eval_set=(X_dev, y_dev))
    
    # Feature importances
    importances = matching_model.model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop 10 Feature Importances (Gain):", flush=True)
    for idx in sorted_idx[:10]:
        print(f"  {FEATURE_NAMES[idx]:25}: {importances[idx]:.1f}", flush=True)
        
    # Evaluate dev set ROC AUC and PR AUC
    from sklearn.metrics import roc_auc_score, average_precision_score
    dev_probs = matching_model.predict_pair_probs(X_dev)
    roc = roc_auc_score(y_dev, dev_probs)
    pr = average_precision_score(y_dev, dev_probs)
    print(f"\nDev Set ROC AUC: {roc:.4f} | PR AUC: {pr:.4f}", flush=True)
    
    # Save model
    model_save_path = str(PROJECT_ROOT / "model_lgb.joblib")
    matching_model.save(model_save_path)
    print(f"\nSaved trained model to {model_save_path}", flush=True)
    print(f"Total pipeline time: {time.time() - t0:.2f}s", flush=True)

if __name__ == "__main__":
    train_and_optimize()

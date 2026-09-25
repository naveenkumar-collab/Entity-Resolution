"""
Threshold Tuning Module for Macro F_0.5 Optimization.
Grid-searches decision threshold and singleton shield on validation split.
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
from src.features import extract_pair_features
from src.model import EntityMatchingModel
from src.metrics import evaluate_predictions

def run_threshold_tuning(n_val_eval: int = 2000):
    print(f"Loading trained LightGBM model...")
    model_path = PROJECT_ROOT / "model_lgb.joblib"
    model = EntityMatchingModel.load(str(model_path))
    
    print(f"Loading {n_val_eval:,} validation entities...")
    val_s1_path = PROJECT_ROOT / "val_source1_30k.tsv"
    val_gt_path = PROJECT_ROOT / "val_ground_truth_30k.tsv"
    val_s1 = pd.read_csv(str(val_s1_path), sep="\t").head(n_val_eval)
    val_gt = pd.read_csv(str(val_gt_path), sep="\t").head(n_val_eval)
    
    # Ground truth mapping
    gt_map: Dict[str, Set[str]] = {}
    for _, r in val_gt.iterrows():
        sid = r["source1_entity_id"]
        raw = str(r["matched_entity_ids"]) if pd.notna(r["matched_entity_ids"]) else ""
        gt_map[sid] = set(raw.split(",")) if raw else set()
        
    # Load target pool
    print("Loading target pool for validation scoring...")
    s2_path = PROJECT_ROOT / "dataset" / "train" / "train_source2.tsv"
    s3_path = PROJECT_ROOT / "dataset" / "train" / "train_source3.tsv"
    s2_df = pd.read_csv(str(s2_path), sep="\t", nrows=250000)
    s3_df = pd.read_csv(str(s3_path), sep="\t", nrows=250000)
    
    # Pre-score candidate pairs
    scored_candidates_by_s1: Dict[str, List[Tuple[str, float]]] = {}
    
    for country in ["US", "India"]:
        c_s1 = val_s1[val_s1["country"] == country]
        c_target = pd.concat([
            s2_df[s2_df["country"] == country],
            s3_df[s3_df["country"] == country]
        ], ignore_index=True)
        
        print(f"Blocking {country} ({len(c_s1):,} S1 entities)...", flush=True)
        blocker = CountryBlocker(country=country, max_candidates_per_entity=15, max_postings_per_key=300)
        blocker.add_target_records(c_target)
        cands_dict = blocker.generate_candidates_for_s1(c_s1)
        
        print(f"Scoring candidates for {country}...", flush=True)
        for _, row in c_s1.iterrows():
            sid = row["entity_id"]
            cands = cands_dict.get(sid, [])
            if not cands:
                scored_candidates_by_s1[sid] = []
                continue
            s1_nc, s1_core = normalize_business_name(row["business_name"])
            s1_ca, s1_pc = normalize_address(row["business_address"])
            s1_meta = (s1_nc, s1_core, s1_ca, s1_pc)
            
            scored = model.score_candidates_for_s1(s1_meta, cands, blocker.target_meta)
            scored_candidates_by_s1[sid] = scored

    print("\n--- Grid Searching Thresholds for Macro F_0.5 ---", flush=True)
    best_f05 = -1.0
    best_tau = 0.70
    best_sing = 0.75
    best_res = None
    
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    singleton_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    
    for sing_tau in singleton_thresholds:
        for tau in thresholds:
            if tau > sing_tau + 0.15 or tau < sing_tau - 0.20:
                continue
                
            predictions: Dict[str, Set[str]] = {}
            for sid, scored in scored_candidates_by_s1.items():
                if not scored:
                    predictions[sid] = set()
                    continue
                # Singleton Shield
                max_p = scored[0][1]
                if max_p < sing_tau:
                    predictions[sid] = set()
                else:
                    preds = {cid for cid, p in scored if p >= tau}
                    predictions[sid] = preds
                    
            res = evaluate_predictions(predictions, gt_map)
            f05 = res["macro_f05"]
            if f05 > best_f05:
                best_f05 = f05
                best_tau = tau
                best_sing = sing_tau
                best_res = res
                print(f"[*] New Best! Match Tau: {tau:.2f} | Sing Tau: {sing_tau:.2f} -> Macro F0.5: {f05:.4f} (Prec: {res['macro_precision']:.4f}, Rec: {res['macro_recall']:.4f}, SingAcc: {res['singleton_accuracy']:.4f})", flush=True)

    print("\n" + "="*60, flush=True)
    print("THRESHOLD OPTIMIZATION RESULTS", flush=True)
    print("="*60, flush=True)
    print(f"Optimal Match Threshold (tau):       {best_tau:.2f}", flush=True)
    print(f"Optimal Singleton Shield (sing_tau): {best_sing:.2f}", flush=True)
    print(f"Best Validation Macro F_0.5:         {best_f05:.4f}", flush=True)
    print(f"Precision:                           {best_res['macro_precision']:.4f}", flush=True)
    print(f"Recall:                              {best_res['macro_recall']:.4f}", flush=True)
    print(f"Singleton Accuracy:                  {best_res['singleton_accuracy']:.4f}", flush=True)
    
    # Save optimal settings
    opt_info = {
        "best_tau": best_tau,
        "best_sing": best_sing,
        "macro_f05": best_f05,
        "precision": best_res["macro_precision"],
        "recall": best_res["macro_recall"],
        "singleton_accuracy": best_res["singleton_accuracy"]
    }
    opt_path = PROJECT_ROOT / "optimal_thresholds.json"
    pd.Series(opt_info).to_json(str(opt_path))
    print(f"Saved optimal thresholds to {opt_path}", flush=True)

if __name__ == "__main__":
    run_threshold_tuning()

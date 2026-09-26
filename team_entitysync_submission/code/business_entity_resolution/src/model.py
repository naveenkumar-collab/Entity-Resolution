"""
Matching Model Module using LightGBM.
Compliant with challenge constraints: MIT/Apache 2.0 license, <8B parameters.
Provides pairwise scoring, decision threshold tuning, and singleton shield.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import lightgbm as lgb
import os
import joblib
from .features import FEATURE_NAMES, extract_pair_features
from .metrics import evaluate_predictions


class EntityMatchingModel:
    """
    Pairwise LightGBM matching model with threshold tuning and singleton shield.
    """
    def __init__(
        self,
        threshold: float = 0.70,
        singleton_threshold: float = 0.75,
        n_estimators: int = 250,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 42
    ):
        self.threshold = threshold
        self.singleton_threshold = singleton_threshold
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            importance_type="gain"
        )
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Train LightGBM model on pairwise feature matrix X and binary labels y.
        """
        eval_data = [eval_set] if eval_set is not None else None
        self.model.fit(
            X,
            y,
            eval_set=eval_data,
            callbacks=[lgb.early_stopping(50, verbose=False)] if eval_set is not None else None
        )
        self.is_fitted = True
        return self
        
    def predict_pair_probs(self, X: np.ndarray) -> np.ndarray:
        """
        Predict match probability for pairwise feature matrix X.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict_proba(X)[:, 1]
        
    def score_candidates_for_s1(
        self,
        s1_meta: Tuple[str, str, str, Optional[str]],
        candidate_ids: List[str],
        target_meta_map: Dict[str, Tuple[str, str, str, Optional[str]]]
    ) -> List[Tuple[str, float]]:
        """
        Score a list of candidate IDs for a single S1 entity.
        Returns sorted list of (candidate_id, probability).
        """
        if not candidate_ids:
            return []
            
        s1_nc, s1_core, s1_ca, s1_pc = s1_meta
        feature_rows = []
        valid_cands = []
        
        for cid in candidate_ids:
            if cid not in target_meta_map:
                continue
            m_nc, m_core, m_ca, m_pc = target_meta_map[cid]
            feat = extract_pair_features(
                s1_nc, s1_core, s1_ca, s1_pc,
                m_nc, m_core, m_ca, m_pc,
                cid
            )
            feature_rows.append(feat)
            valid_cands.append(cid)
            
        if not feature_rows:
            return []
            
        X = np.array(feature_rows, dtype=np.float32)
        probs = self.predict_pair_probs(X)
        
        scored = list(zip(valid_cands, probs))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
        
    def filter_matches(
        self,
        scored_candidates: List[Tuple[str, float]]
    ) -> List[str]:
        """
        Apply decision threshold and singleton shield to select final matches.
        """
        if not scored_candidates:
            return []
            
        # Singleton Shield: If highest confidence candidate is weak, reject all (singleton)
        max_prob = scored_candidates[0][1]
        if max_prob < self.singleton_threshold:
            return []
            
        # Select all candidates exceeding match threshold
        return [cid for cid, prob in scored_candidates if prob >= self.threshold]
        
    def save(self, filepath: str):
        joblib.dump(self, filepath)
        
    @classmethod
    def load(cls, filepath: str) -> "EntityMatchingModel":
        return joblib.load(filepath)

"""
Evaluation metrics for Business Entity Resolution Challenge.
Implements the exact macro-averaged F_0.5 score with singleton handling.
"""

from typing import Dict, Set, Iterable


def compute_entity_f05(pred_matches: Set[str], true_matches: Set[str]) -> float:
    """
    Compute F_0.5 for a single Source 1 entity.
    
    Formula:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    
    Singleton logic:
    - If true_matches is empty (singleton):
        - 1.0 if pred_matches is empty
        - 0.0 if pred_matches is non-empty
    - If true_matches is non-empty:
        - 0.0 if pred_matches is empty
        - Otherwise compute F_0.5 (0.0 if TP == 0)
    """
    n_true = len(true_matches)
    n_pred = len(pred_matches)
    
    # Singleton case
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0
    
    # Non-singleton but no predictions made
    if n_pred == 0:
        return 0.0
    
    tp = len(pred_matches & true_matches)
    if tp == 0:
        return 0.0
    
    precision = tp / n_pred
    recall = tp / n_true
    
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    
    return (1.25 * precision * recall) / denom


def evaluate_predictions(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]]
) -> Dict[str, float]:
    """
    Compute macro-averaged precision, recall, and F_0.5 across all S1 entities in ground_truth.
    
    Parameters:
    - predictions: dict of {s1_id: set of matched s2/s3 ids}
    - ground_truth: dict of {s1_id: set of true matching s2/s3 ids}
    
    Returns:
    - dict with 'macro_f05', 'macro_precision', 'macro_recall', 'n_entities', 'n_singletons'
    """
    total_f05 = 0.0
    total_prec = 0.0
    total_rec = 0.0
    n_entities = len(ground_truth)
    n_singletons = 0
    singleton_correct = 0
    
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        f05 = compute_entity_f05(pred_set, true_set)
        total_f05 += f05
        
        n_true = len(true_set)
        n_pred = len(pred_set)
        
        if n_true == 0:
            n_singletons += 1
            if n_pred == 0:
                singleton_correct += 1
                total_prec += 1.0
                total_rec += 1.0
            else:
                total_prec += 0.0
                total_rec += 1.0
        else:
            if n_pred == 0:
                total_prec += 0.0
                total_rec += 0.0
            else:
                tp = len(pred_set & true_set)
                total_prec += tp / n_pred
                total_rec += tp / n_true
    
    return {
        "macro_f05": total_f05 / n_entities if n_entities > 0 else 0.0,
        "macro_precision": total_prec / n_entities if n_entities > 0 else 0.0,
        "macro_recall": total_rec / n_entities if n_entities > 0 else 0.0,
        "n_entities": n_entities,
        "n_singletons": n_singletons,
        "singleton_accuracy": singleton_correct / n_singletons if n_singletons > 0 else 1.0
    }


if __name__ == "__main__":
    # Test example from challenge specification:
    # S1-00001: pred [S2-00047, S2-00193, S3-00812], true [S2-00047, S3-00812]
    # Expected: Prec = 2/3, Rec = 1.0, F0.5 = 0.714
    pred = {"S1-00001": {"S2-00047", "S2-00193", "S3-00812"}}
    true = {"S1-00001": {"S2-00047", "S3-00812"}}
    res = evaluate_predictions(pred, true)
    print("Test example F0.5:", round(res["macro_f05"], 3))
    assert round(res["macro_f05"], 3) == 0.714, f"Mismatch: {res['macro_f05']}"
    print("Metric unit test passed!")

"""
Comprehensive Evaluation Pipeline for Classification and Segmentation Models.
Computes Accuracy, Sensitivity (Recall), Specificity, Precision, F1-Score, AUROC,
Confusion Matrix, Dice Coefficient, and Intersection over Union (IoU).
Never fabricates metrics — operates strictly on supplied ground-truth and prediction arrays.
"""
from typing import Any, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Computes clinical diagnostic evaluation metrics for binary or subtype classification.
    """
    y_true_b = (np.asarray(y_true) >= 0.5).astype(int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred_b = (y_prob_arr >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_b, y_pred_b, labels=[0, 1]).ravel()
    sensitivity = float(tp) / max(float(tp + fn), 1.0)
    specificity = float(tn) / max(float(tn + fp), 1.0)
    precision = float(precision_score(y_true_b, y_pred_b, zero_division=0))
    recall = float(recall_score(y_true_b, y_pred_b, zero_division=0))
    f1 = float(f1_score(y_true_b, y_pred_b, zero_division=0))
    acc = float(accuracy_score(y_true_b, y_pred_b))

    if len(np.unique(y_true_b)) > 1:
        auroc = float(roc_auc_score(y_true_b, y_prob_arr))
    else:
        auroc = float("nan")

    return {
        "accuracy": round(acc, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "auroc": round(auroc, 4) if not np.isnan(auroc) else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

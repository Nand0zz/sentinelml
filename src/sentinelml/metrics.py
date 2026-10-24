"""Security-oriented evaluation: FPR and PR-AUC matter more than accuracy."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def false_positive_rate(y_true, y_pred) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (fp + tn)) if (fp + tn) else 0.0


def select_threshold(y_true, y_score, max_fpr: float = 0.05) -> float:
    """Highest-recall threshold whose false-positive rate stays <= max_fpr.

    Choose it on validation data, then apply it unchanged to the test set.
    """
    fpr, _, thresholds = roc_curve(y_true, y_score)
    idx = np.where(fpr <= max_fpr)[0][-1]  # fpr is non-decreasing
    thr = float(thresholds[idx])
    return thr if np.isfinite(thr) else 1.0


def evaluate(y_true, y_score, threshold: float) -> dict:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "fpr": false_positive_rate(y_true, y_pred),
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

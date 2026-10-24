import numpy as np

from sentinelml.metrics import evaluate, false_positive_rate, select_threshold


def test_false_positive_rate():
    y_true = [0, 0, 0, 0, 1, 1]
    y_pred = [1, 0, 0, 0, 1, 0]
    assert false_positive_rate(y_true, y_pred) == 0.25


def test_threshold_respects_fpr_budget():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    score = 0.4 * y + rng.random(5000) * 0.8
    thr = select_threshold(y, score, max_fpr=0.05)
    pred = (score >= thr).astype(int)
    assert false_positive_rate(y, pred) <= 0.05
    assert pred.sum() > 0  # and it still flags something


def test_evaluate_returns_expected_keys():
    m = evaluate([0, 1, 1, 0], [0.1, 0.9, 0.6, 0.4], threshold=0.5)
    assert {"precision", "recall", "f1", "pr_auc", "roc_auc", "fpr"} <= m.keys()
    assert m["recall"] == 1.0 and m["fpr"] == 0.0

"""How much recall do we give up at each false-positive budget?

Threshold is chosen on VALIDATION for each budget, then applied unchanged to TEST.

    python scripts/fpr_sensitivity.py                 # real data, champion model
    python scripts/fpr_sensitivity.py --synthetic     # smoke test
Does not retrain or touch the registry.
"""
from __future__ import annotations

import argparse

import pandas as pd
import plotly.graph_objects as go

from sentinelml import config as cfg
from sentinelml.data import load_splits, load_synthetic_splits, split_train_val
from sentinelml.metrics import evaluate, select_threshold
from sentinelml.registry import load_from_registry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tracking-uri", default=cfg.DEFAULT_TRACKING_URI)
    ap.add_argument("--model-name", default=cfg.DEFAULT_MODEL_NAME)
    ap.add_argument("--alias", default=cfg.ALIAS_CHAMPION)
    ap.add_argument("--data-dir", default=str(cfg.RAW_DIR))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.10])
    a = ap.parse_args()

    pipe, meta = load_from_registry(a.model_name, a.alias, a.tracking_uri)
    train_full, test = load_synthetic_splits(a.seed) if a.synthetic else load_splits(a.data_dir)
    _, val = split_train_val(train_full, a.seed)   # same split as training
    v_score = pipe.predict_proba(val[cfg.FEATURES])[:, 1]
    t_score = pipe.predict_proba(test[cfg.FEATURES])[:, 1]

    rows = []
    for b in a.budgets:
        thr = select_threshold(val[cfg.TARGET], v_score, b)
        m = evaluate(test[cfg.TARGET], t_score, thr)
        rows.append({"fpr_budget": b, "threshold": thr, "test_recall": m["recall"],
                     "test_fpr": m["fpr"], "test_precision": m["precision"],
                     "attacks_missed_pct": 100 * (1 - m["recall"])})
    out = pd.DataFrame(rows)

    reports = cfg.ROOT / "reports"
    reports.mkdir(exist_ok=True)
    out.to_csv(reports / "fpr_sensitivity.csv", index=False)
    fig = go.Figure(go.Scatter(x=out["fpr_budget"] * 100, y=out["test_recall"], mode="lines+markers"))
    fig.update_layout(title=f"Recall vs false-positive budget ({meta['model_name']} v{meta['version']})",
                      xaxis_title="FPR budget (%)", yaxis_title="Test recall")
    fig.write_html(reports / "fpr_sensitivity.html")
    print(f"\nModel: {meta['model_name']} v{meta['version']} ({a.alias})\n")
    print(out.round(4).to_string(index=False))
    print("\nWrote reports/fpr_sensitivity.csv and reports/fpr_sensitivity.html")


if __name__ == "__main__":
    main()

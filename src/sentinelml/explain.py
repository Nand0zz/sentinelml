"""Explanations: SHAP for tree models, exact coefficient contributions for linear models.

Contributions are in the model's raw-score (log-odds) space: positive pushes a flow toward
"suspicious", negative toward "benign". They explain the MODEL, not causality in the network.

    python -m sentinelml.explain        # global importance of the champion -> reports/shap_global.*
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sentinelml import config as cfg
from sentinelml.data import normalize_features


@dataclass
class Explainer:
    kind: str                       # "shap_tree" | "linear_contribution" | "none"
    names: list[str]
    impl: object = None
    coef: np.ndarray | None = field(default=None, repr=False)


def build_explainer(pipe) -> Explainer:
    est = pipe.named_steps["model"]
    names = [str(n) for n in pipe.named_steps["prep"].get_feature_names_out()]
    if hasattr(est, "feature_importances_"):
        import shap  # imported lazily: slow import, not needed for linear models

        return Explainer("shap_tree", names, impl=shap.TreeExplainer(est))
    if hasattr(est, "coef_"):
        return Explainer("linear_contribution", names, coef=np.asarray(est.coef_)[0])
    return Explainer("none", names)


def contributions(pipe, ex: Explainer, df: pd.DataFrame) -> np.ndarray | None:
    """(n_rows, n_transformed_features) contributions toward the 'suspicious' class."""
    if ex.kind == "none":
        return None
    Xt = pipe[:-1].transform(df)
    if ex.kind == "linear_contribution":
        # Inputs are already scaled, so coef * x is the exact additive decomposition.
        return Xt * ex.coef
    sv = ex.impl.shap_values(Xt, check_additivity=False)
    if isinstance(sv, list):                    # older shap: one array per class
        sv = sv[1] if len(sv) == 2 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:                            # newer shap: (rows, features, classes)
        sv = sv[:, :, 1]
    return sv


def global_importance(pipe, df: pd.DataFrame) -> pd.DataFrame:
    ex = build_explainer(pipe)
    vals = contributions(pipe, ex, df)
    if vals is None:
        raise ValueError("This model type has no supported explainer.")
    return (
        pd.DataFrame({"feature": ex.names, "mean_abs_contribution": np.abs(vals).mean(axis=0)})
        .sort_values("mean_abs_contribution", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    import plotly.express as px

    from sentinelml.registry import load_exported, load_from_registry

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tracking-uri", default=cfg.DEFAULT_TRACKING_URI)
    ap.add_argument("--model-name", default=cfg.DEFAULT_MODEL_NAME)
    ap.add_argument("--alias", default=cfg.ALIAS_CHAMPION)
    ap.add_argument("--model-dir", default=None, help="use an exported model folder instead")
    ap.add_argument("--reference", default=str(cfg.ROOT / "data" / "processed" / "reference_sample.csv.gz"))
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()

    pipe, meta = (load_exported(a.model_dir) if a.model_dir
                  else load_from_registry(a.model_name, a.alias, a.tracking_uri))
    ref = pd.read_csv(a.reference)
    df = normalize_features(ref.sample(min(a.n, len(ref)), random_state=0))
    imp = global_importance(pipe, df)

    out = cfg.ROOT / "reports"
    out.mkdir(exist_ok=True)
    imp.to_csv(out / "shap_global.csv", index=False)
    top = imp.head(15).iloc[::-1]
    fig = px.bar(top, x="mean_abs_contribution", y="feature", orientation="h",
                 title=f"Global feature importance, {meta['model_name']} v{meta['version']} (mean |contribution|)")
    fig.write_html(out / "shap_global.html")
    print(imp.head(10).round(4).to_string(index=False))
    print("\nWrote reports/shap_global.csv and reports/shap_global.html")


if __name__ == "__main__":
    main()

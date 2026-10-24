import mlflow
import numpy as np

from sentinelml import config as cfg
from sentinelml.data import normalize_features
from sentinelml.explain import build_explainer, contributions
from tests.conftest import build_registry


def test_linear_and_tree_contributions_are_additive(tmp_path):
    reg = build_registry(tmp_path)
    mlflow.set_tracking_uri(reg["uri"])
    df = normalize_features(reg["test"].head(50))

    # logistic regression: contributions + intercept == decision function (exact)
    lr = mlflow.sklearn.load_model(f"runs:/{reg['table'].set_index('model').loc['logreg', 'run_id']}/model")
    ex = build_explainer(lr)
    assert ex.kind == "linear_contribution"
    c = contributions(lr, ex, df)
    assert np.allclose(c.sum(axis=1) + lr.named_steps["model"].intercept_[0], lr.decision_function(df), atol=1e-6)

    # LightGBM: SHAP values + expected value == raw margin
    lg = mlflow.sklearn.load_model(f"runs:/{reg['lgbm_row']['run_id']}/model")
    ex = build_explainer(lg)
    assert ex.kind == "shap_tree"
    s = contributions(lg, ex, df)
    base = float(np.ravel(ex.impl.expected_value)[-1])
    raw = lg.named_steps["model"].predict(lg[:-1].transform(df), raw_score=True)
    assert s.shape == (50, len(ex.names))
    assert np.allclose(s.sum(axis=1) + base, raw, atol=1e-3)

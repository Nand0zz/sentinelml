from sentinelml import config as cfg
from sentinelml.data import clean
from sentinelml.features import add_engineered_features, build_pipeline
from sentinelml.synthetic import make_synthetic
from sklearn.linear_model import LogisticRegression


def _data(n=1500):
    df = clean(make_synthetic(n, seed=3))
    return df[cfg.FEATURES], df[cfg.TARGET]


def test_engineered_columns_added():
    X, _ = _data(200)
    out = add_engineered_features(X)
    assert set(cfg.ENGINEERED) <= set(out.columns)


def test_pipeline_fit_predict_and_unseen_category():
    X, y = _data()
    pipe = build_pipeline(LogisticRegression(max_iter=500), scale=True).fit(X, y)
    X_new = X.head(5).copy()
    X_new["proto"] = "never-seen-protocol"
    proba = pipe.predict_proba(X_new)
    assert proba.shape == (5, 2)


def test_pipeline_handles_nan_inputs():
    X, y = _data()
    pipe = build_pipeline(LogisticRegression(max_iter=500), scale=True).fit(X, y)
    X_new = X.head(3).copy()
    X_new.loc[:, "dur"] = float("nan")
    assert pipe.predict_proba(X_new).shape == (3, 2)

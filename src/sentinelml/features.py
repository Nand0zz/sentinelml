"""Feature engineering + preprocessing, packaged as one sklearn Pipeline.

Everything lives inside the pipeline so the logged MLflow model is
self-contained: the API sends raw columns and gets predictions back.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from sentinelml import config as cfg


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["total_bytes"] = X["sbytes"] + X["dbytes"]
    X["total_pkts"] = X["spkts"] + X["dpkts"]
    X["bytes_ratio"] = X["sbytes"] / (X["dbytes"] + 1)
    X["pkts_ratio"] = X["spkts"] / (X["dpkts"] + 1)
    X["src_bytes_per_pkt"] = X["sbytes"] / (X["spkts"] + 1)
    X["dst_bytes_per_pkt"] = X["dbytes"] / (X["dpkts"] + 1)
    return X


def signed_log1p(X):
    """Tames the heavy right tails typical of byte/rate/jitter features."""
    return np.sign(X) * np.log1p(np.abs(X))


def build_pipeline(estimator, scale: bool = False) -> Pipeline:
    numeric_steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
    ]
    if scale:  # needed for linear models only
        numeric_steps.append(("scale", StandardScaler()))

    prep = ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), cfg.NUMERIC + cfg.ENGINEERED),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore", min_frequency=0.005, sparse_output=False
                ),
                cfg.CATEGORICAL,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("engineer", FunctionTransformer(add_engineered_features)),
            ("prep", prep),
            ("model", estimator),
        ]
    )

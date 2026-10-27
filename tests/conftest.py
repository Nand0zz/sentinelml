"""Shared fixtures: a tiny synthetic model registry (fast mode), built once per module."""
import mlflow
import pandas as pd
import pytest

from sentinelml import config as cfg
from sentinelml.data import clean
from sentinelml.gate import GateConfig, promote
from sentinelml.registry import candidate_tags, register_candidate
from sentinelml.synthetic import make_synthetic
from sentinelml.train import run_training

MODEL = "test-ids"
RELAXED = GateConfig(min_pr_auc=0.80, fpr_slack=0.10, min_recall=0.50, max_latency_ms=1e6)


def build_registry(tmp_path):
    """Train 3 small models; make LightGBM the candidate so the SHAP path is exercised."""
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    train = clean(make_synthetic(3000, seed=0))
    test = clean(make_synthetic(1200, seed=1, shift=0.2), drop_duplicates=False)
    table, _ = run_training(train, test, tracking_uri=uri, model_name=MODEL, fast=True, out_dir=tmp_path)
    row = table[table["model"] == "lightgbm"].iloc[0]
    register_candidate(MODEL, row["run_id"], candidate_tags(row), uri)
    return {"uri": uri, "table": table, "lgbm_row": row, "test": test, "tmp": tmp_path}


def flow_dict(df: pd.DataFrame, i: int = 0) -> dict:
    """One row as plain JSON-safe python types (NaN -> None)."""
    row = df.iloc[i][cfg.FEATURES]
    return {k: (None if pd.isna(v) else (float(v) if k in cfg.NUMERIC else str(v))) for k, v in row.items()}


@pytest.fixture(scope="session", autouse=True)
def _isolated_root(tmp_path_factory):
    """Keep tests from overwriting the real reports/ (e.g. gate_report.json) in the repo."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cfg, "ROOT", tmp_path_factory.mktemp("root"))
        yield


@pytest.fixture(scope="module")
def promoted_registry(tmp_path_factory):
    reg = build_registry(tmp_path_factory.mktemp("api_reg"))
    assert promote(MODEL, reg["uri"], RELAXED).promoted
    mlflow.set_tracking_uri(reg["uri"])
    return reg

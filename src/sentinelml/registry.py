"""Thin helpers around the MLflow Model Registry.

Alias lifecycle:  candidate --(quality gate)--> champion ;  old champion -> previous_champion
"""
from __future__ import annotations

import importlib.metadata as md
import json
import platform
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from sentinelml import config as cfg


def get_client(tracking_uri: str | None = None) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri or cfg.DEFAULT_TRACKING_URI)
    return MlflowClient()


def candidate_tags(row) -> dict:
    """Metrics stored on the model version so the registry page is self-explanatory."""
    return {
        "threshold": row["val_threshold"],
        "val_pr_auc": row["val_pr_auc"],
        "test_pr_auc": row["test_pr_auc"],
        "test_fpr": row["test_fpr"],
        "test_recall": row["test_recall"],
    }


def register_candidate(model_name: str, run_id: str, tags: dict, tracking_uri: str | None = None) -> str:
    client = get_client(tracking_uri)
    mv = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    client.set_registered_model_alias(model_name, cfg.ALIAS_CANDIDATE, mv.version)
    for k, v in tags.items():
        client.set_model_version_tag(model_name, mv.version, k, f"{float(v):.6f}")
    return str(mv.version)


def alias_version(client: MlflowClient, model_name: str, alias: str):
    """ModelVersion behind an alias, or None if the alias (or model) does not exist."""
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None


def run_metrics(client: MlflowClient, mv) -> dict:
    return dict(client.get_run(mv.run_id).data.metrics)


def environment_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("scikit-learn", "lightgbm", "numpy", "pandas", "mlflow"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "not-installed"
    return out


def load_from_registry(model_name: str, alias: str, tracking_uri: str | None = None):
    """-> (sklearn Pipeline, metadata dict) for the model behind `alias`."""
    client = get_client(tracking_uri)
    mv = alias_version(client, model_name, alias)
    if mv is None:
        raise LookupError(
            f"No model version with alias '{alias}' for '{model_name}'. "
            "Train, then run:  python -m sentinelml.gate promote"
        )
    pipe = mlflow.sklearn.load_model(f"models:/{model_name}/{mv.version}")
    tags = {k: float(v) for k, v in mv.tags.items()}
    meta = {
        "model_name": model_name,
        "version": str(mv.version),
        "alias": alias,
        "threshold": tags.pop("threshold"),
        "metrics": tags,
        "source": "registry",
        "environment": environment_versions(),
    }
    return pipe, meta


def load_exported(path: str | Path):
    """-> (Pipeline, metadata) from a directory written by `sentinelml.export`."""
    path = Path(path)
    meta_file = path / "export_meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"{meta_file} not found. Create it with: python -m sentinelml.export")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["source"] = "exported"
    return mlflow.sklearn.load_model(str(path)), meta

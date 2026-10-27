"""Train, compare, track (MLflow) and register candidate intrusion-detection models.

Methodology (worth explaining in interviews):
  * models are fit on `train`
  * the operating threshold is chosen on `val` (max recall s.t. FPR <= target)
  * the best model is picked by VALIDATION PR-AUC, never by test
  * the official UNSW test set is reported once, at the validation-chosen threshold
The winner is registered with alias `candidate`; Day 2's quality gate decides promotion.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

from sentinelml import config as cfg
from sentinelml.data import load_processed, load_splits, load_synthetic_splits, split_train_val
from sentinelml.export import write_model_dir
from sentinelml.features import build_pipeline
from sentinelml.metrics import evaluate, select_threshold
from sentinelml.registry import candidate_tags, environment_versions, register_candidate

METRIC_KEYS = ["precision", "recall", "f1", "pr_auc", "roc_auc", "fpr", "threshold"]


def get_models(fast: bool = False, seed: int = cfg.DEFAULT_SEED) -> dict:
    """name -> (estimator, needs_scaling)"""
    n_rf, n_gb = (30, 60) if fast else (300, 400)
    return {
        "logreg": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            True,
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=n_rf, min_samples_leaf=2,
                class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
            ),
            False,
        ),
        "lightgbm": (
            LGBMClassifier(
                n_estimators=n_gb, learning_rate=0.05, num_leaves=63,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                # physical cores: LightGBM slows down badly when given hyper-threads (e.g. WSL: 250 s vs 3 s)
                random_state=seed, n_jobs=max(1, (os.cpu_count() or 2) // 2), verbose=-1,
            ),
            False,
        ),
    }


def _importance(estimator, names) -> pd.DataFrame:
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    else:
        values = np.abs(estimator.coef_[0])
    return (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _single_row_latency_ms(pipe, row: pd.DataFrame, n: int = 30) -> float:
    pipe.predict_proba(row)  # warm-up
    start = time.perf_counter()
    for _ in range(n):
        pipe.predict_proba(row)
    return (time.perf_counter() - start) / n * 1000


def run_training(
    train_full: pd.DataFrame,
    test: pd.DataFrame,
    *,
    tracking_uri: str,
    experiment: str = cfg.DEFAULT_EXPERIMENT,
    model_name: str = cfg.DEFAULT_MODEL_NAME,
    max_fpr: float = cfg.DEFAULT_MAX_FPR,
    fast: bool = False,
    seed: int = cfg.DEFAULT_SEED,
    out_dir: Path = cfg.ROOT,
    run_name: str | None = None,
    model_out: Path | None = None,
    data_source: str = "in-memory",
) -> tuple[pd.DataFrame, dict]:
    """Train all models under one MLflow parent run; register the winner as `candidate`.

    If `model_out` is given, the candidate is also written there as a self-contained folder
    (the DVC `train` stage output, read by `python -m sentinelml.gate promote`).
    """
    out_dir = Path(out_dir)
    reports = out_dir / "reports"
    processed = out_dir / "data" / "processed"
    reports.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    run_name = run_name or started.strftime(cfg.RUN_NAME_FORMAT)

    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(
            experiment, artifact_location=(out_dir / "mlartifacts").resolve().as_uri()
        )
    mlflow.set_experiment(experiment)

    train, val = split_train_val(train_full, seed)
    X_tr, y_tr = train[cfg.FEATURES], train[cfg.TARGET]
    X_val, y_val = val[cfg.FEATURES], val[cfg.TARGET]
    X_te, y_te = test[cfg.FEATURES], test[cfg.TARGET]
    params = {"max_fpr_target": max_fpr, "seed": seed, "fast": fast, "data_source": data_source,
              "n_train": len(X_tr), "n_val": len(X_val), "n_test": len(X_te)}

    with mlflow.start_run(run_name=run_name) as parent:
        mlflow.log_params({"experiment": experiment, **params})

        rows, curves, pipes, evals = [], {}, {}, {}
        for name, (estimator, scale) in get_models(fast, seed).items():
            pipe = build_pipeline(estimator, scale=scale)
            with mlflow.start_run(run_name=f"{run_name}-{name}", nested=True) as run:
                t0 = time.perf_counter()
                pipe.fit(X_tr, y_tr)
                fit_seconds = time.perf_counter() - t0

                val_score = pipe.predict_proba(X_val)[:, 1]
                threshold = select_threshold(y_val, val_score, max_fpr)
                val_m = evaluate(y_val, val_score, threshold)
                test_score = pipe.predict_proba(X_te)[:, 1]
                test_m = evaluate(y_te, test_score, threshold)
                latency = _single_row_latency_ms(pipe, X_val.iloc[[0]])

                mlflow.log_params(
                    {"model": name, "max_fpr_target": max_fpr, "seed": seed,
                     "n_train": len(X_tr), "n_val": len(X_val), "n_test": len(X_te),
                     **{f"model__{k}": v for k, v in estimator.get_params().items()}}
                )
                mlflow.log_metrics(
                    {f"val_{k}": v for k, v in val_m.items() if k in METRIC_KEYS}
                    | {f"test_{k}": v for k, v in test_m.items() if k in METRIC_KEYS}
                    | {"fit_seconds": fit_seconds, "single_row_latency_ms": latency}
                )
                mlflow.log_dict(
                    {"val": val_m, "test": test_m}, "evaluation/confusion_and_metrics.json"
                )

                names = pipe.named_steps["prep"].get_feature_names_out()
                imp = _importance(estimator, names)
                imp_path = reports / f"feature_importance_{name}.csv"
                imp.to_csv(imp_path, index=False)
                mlflow.log_artifact(str(imp_path), "explainability")

                mlflow.sklearn.log_model(
                    pipe, artifact_path="model",
                    signature=infer_signature(X_val.head(20), pipe.predict_proba(X_val.head(20))),
                    input_example=X_val.head(3),
                )

            p, r, _ = precision_recall_curve(y_te, test_score)
            keep = np.linspace(0, len(p) - 1, min(len(p), 500)).astype(int)
            curves[name] = (r[keep], p[keep])
            pipes[name], evals[name] = pipe, {"val": val_m, "test": test_m}
            rows.append(
                {"model": name, "run_id": run.info.run_id, "fit_seconds": fit_seconds,
                 "latency_ms": latency}
                | {f"val_{k}": val_m[k] for k in METRIC_KEYS}
                | {f"test_{k}": test_m[k] for k in METRIC_KEYS}
            )

        table = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
        table.to_csv(reports / "model_comparison.csv", index=False)

        fig = go.Figure()
        for name, (r, p) in curves.items():
            fig.add_trace(go.Scatter(x=r, y=p, mode="lines", name=name))
        fig.update_layout(
            title="Precision-Recall curves (official test set)",
            xaxis_title="Recall", yaxis_title="Precision",
        )
        pr_path = reports / "pr_curves_test.html"
        fig.write_html(pr_path)

        # Register the winner (chosen on validation) as the `candidate`.
        best = table.iloc[0].to_dict()
        with mlflow.start_run(run_id=best["run_id"], nested=True):
            mlflow.log_artifact(str(pr_path), "evaluation")
            ref_path = processed / "reference_sample.csv.gz"
            train[cfg.FEATURES].sample(min(20_000, len(train)), random_state=seed).to_csv(
                ref_path, index=False
            )
            mlflow.log_artifact(str(ref_path), "reference_data")
        best["registered_version"] = register_candidate(
            model_name, best["run_id"], candidate_tags(best), tracking_uri
        )
        # The training-session summary the brief asks for: model, version, PR-AUC, recall, FPR, threshold, time.
        mlflow.log_metrics({"best_test_pr_auc": best["test_pr_auc"], "best_test_recall": best["test_recall"],
                            "best_test_fpr": best["test_fpr"], "best_threshold": best["val_threshold"]})
        mlflow.set_tags({"best_model": best["model"], "model_name": model_name,
                         "model_version": best["registered_version"], "trained_at": started.isoformat()})
    best |= {"run_name": run_name, "parent_run_id": parent.info.run_id}

    # Evaluation summary for the DVC `train` stage (metrics + confusion matrix of the winner).
    ev = evals[best["model"]]
    keep_keys = ["precision", "recall", "f1", "pr_auc", "roc_auc", "fpr"]
    metrics = {"model": best["model"], "threshold": best["val_threshold"],
               "latency_ms": best["latency_ms"],
               "val": {k: ev["val"][k] for k in keep_keys}, "test": {k: ev["test"][k] for k in keep_keys}}
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cm = {split: {k: ev[split][k] for k in ("tn", "fp", "fn", "tp")} for split in ("val", "test")}
    (reports / "confusion_matrix.json").write_text(
        json.dumps({"model": best["model"], "threshold": best["val_threshold"], **cm}, indent=2),
        encoding="utf-8")

    if model_out is not None:
        pipe = pipes[best["model"]]
        tags = candidate_tags(best)
        meta = {
            "model_name": model_name,
            "version": str(best["registered_version"]),
            "alias": cfg.ALIAS_CANDIDATE,
            "model_type": best["model"],
            "threshold": float(tags.pop("threshold")),
            "metrics": {k: float(v) for k, v in tags.items()},
            "run_metrics": dict(mlflow.get_run(best["run_id"]).data.metrics),
            "run_id": best["run_id"],
            "run_name": run_name,
            "experiment": experiment,
            "params": params,
            "trained_at": started.isoformat(),
            "environment": environment_versions(),
        }
        signature = infer_signature(X_val.head(20), pipe.predict_proba(X_val.head(20)))
        write_model_dir(pipe, Path(model_out), meta, signature)
    return table, best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default=str(cfg.RAW_DIR), help="raw CSV folder")
    ap.add_argument("--processed-dir", default=None,
                    help="cleaned parquet folder from the preprocess stage (used by dvc repro)")
    ap.add_argument("--synthetic", action="store_true", help="use generated data (smoke test)")
    ap.add_argument("--fast", action="store_true", help="smaller models, quicker run")
    ap.add_argument("--model-out", default=str(cfg.MODELS_DIR / cfg.ALIAS_CANDIDATE),
                    help="where the candidate model folder is written")
    ap.add_argument("--run-id", default=None, help="training run name (default: run-YYYYMMDD-HHMMSS, UTC)")
    ap.add_argument("--tracking-uri", default=cfg.DEFAULT_TRACKING_URI)
    ap.add_argument("--experiment", default=cfg.DEFAULT_EXPERIMENT)
    ap.add_argument("--model-name", default=cfg.DEFAULT_MODEL_NAME)
    ap.add_argument("--max-fpr", type=float, default=cfg.DEFAULT_MAX_FPR)
    ap.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED)
    a = ap.parse_args()

    if a.synthetic:
        train_full, test = load_synthetic_splits(a.seed)
        source = "synthetic"
    elif a.processed_dir:
        train_full, test = load_processed(a.processed_dir)
        source = f"processed:{Path(a.processed_dir).as_posix()}"
    else:
        train_full, test = load_splits(a.data_dir)
        source = f"raw:{Path(a.data_dir).as_posix()}"

    table, best = run_training(
        train_full, test, tracking_uri=a.tracking_uri, experiment=a.experiment,
        model_name=a.model_name, max_fpr=a.max_fpr, fast=a.fast, seed=a.seed,
        run_name=a.run_id, model_out=Path(a.model_out), data_source=source,
    )
    cols = ["model", "val_pr_auc", "test_pr_auc", "test_precision", "test_recall",
            "test_f1", "test_fpr", "latency_ms"]
    print("\n=== Model comparison (threshold chosen on validation, FPR <= "
          f"{a.max_fpr:.0%}) ===")
    print(table[cols].round(4).to_string(index=False))
    print(f"\nBest by validation PR-AUC: {best['model']} -> registered as "
          f"'{a.model_name}' v{best['registered_version']} (alias: candidate), run {best['run_name']}")
    print(f"Candidate model folder: {a.model_out}")
    print("Next: python -m sentinelml.gate promote   # quality gate -> champion")
    print("Open the MLflow UI:  mlflow ui --backend-store-uri "
          f"{a.tracking_uri}")


if __name__ == "__main__":
    main()

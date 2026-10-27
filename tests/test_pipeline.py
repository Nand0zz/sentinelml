"""DVC pipeline pieces: preprocess stage, candidate folder from training, file-based gate."""
import hashlib
import json
import shutil

import pytest

from sentinelml import config as cfg
from sentinelml.data import clean, load_processed
from sentinelml.gate import GateConfig, promote_files, read_meta, rollback_files, status_files
from sentinelml.preprocess import run as preprocess
from sentinelml.registry import load_exported
from sentinelml.synthetic import make_synthetic
from sentinelml.train import run_training
from tests.conftest import MODEL, RELAXED


def _md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    out = tmp_path_factory.mktemp("processed")
    train = clean(make_synthetic(3000, seed=0))
    test = clean(make_synthetic(1200, seed=1, shift=0.2), drop_duplicates=False)
    preprocess(train, test, out, "synthetic")
    return out, train, test


def test_preprocess_outputs_and_schema(processed):
    out, train, test = processed
    tr, te = load_processed(out)
    assert list(tr.columns) == cfg.FEATURES + [cfg.TARGET]
    assert len(tr) == len(train) and len(te) == len(test)
    schema = json.loads((out / cfg.FEATURE_SCHEMA).read_text())
    assert schema["features"] == cfg.FEATURES
    assert set(schema["categorical"]) == set(cfg.CATEGORICAL)
    assert all(v == v.lower() for v in schema["categorical"]["proto"]["values"])


def test_preprocess_is_deterministic(processed, tmp_path):
    out, train, test = processed
    preprocess(train, test, tmp_path, "synthetic")
    for name in (cfg.PROCESSED_TRAIN, cfg.PROCESSED_TEST, cfg.FEATURE_SCHEMA):
        assert _md5(out / name) == _md5(tmp_path / name)


@pytest.fixture(scope="module")
def candidate(processed, tmp_path_factory):
    """Train once from the processed parquet and write models/candidate."""
    root = tmp_path_factory.mktemp("repo")
    uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    tr, te = load_processed(processed[0])
    _, best = run_training(tr, te, tracking_uri=uri, model_name=MODEL, fast=True, out_dir=root,
                           run_name="run-20260101-000000", model_out=root / "models" / "candidate")
    return root, uri, best


def test_training_writes_candidate_folder_and_reports(candidate, processed):
    root, _, best = candidate
    meta = read_meta(root / "models" / "candidate")
    assert meta["alias"] == "candidate" and meta["run_id"] == best["run_id"]
    assert meta["run_name"] == "run-20260101-000000" and meta["experiment"] == cfg.DEFAULT_EXPERIMENT
    assert {"test_pr_auc", "test_recall", "test_fpr", "single_row_latency_ms"} <= meta["run_metrics"].keys()
    assert {"model", "threshold", "val", "test"} <= json.loads((root / "reports" / "metrics.json").read_text()).keys()
    assert {"tn", "fp", "fn", "tp"} <= json.loads((root / "reports" / "confusion_matrix.json").read_text())["test"].keys()
    pipe, _ = load_exported(root / "models" / "candidate")  # same loader the API uses for MODEL_DIR
    assert pipe.predict_proba(processed[2][cfg.FEATURES].head(2)).shape == (2, 2)


def test_file_gate_promote_reject_and_rollback(candidate, tmp_path):
    root, uri, best = candidate
    models = tmp_path / "models"
    shutil.copytree(root / "models" / "candidate", models / "candidate")

    first = promote_files(models, RELAXED, report_path=tmp_path / "gate.json", tracking_uri=uri)
    assert first.promoted
    assert read_meta(models / "champion")["alias"] == "champion"
    manifest = json.loads((models / cfg.MODEL_MANIFEST).read_text())
    assert manifest["run_id"] == best["run_id"] and manifest["previous_champion"] is None
    assert {"promoted_at", "gate_report_path", "champion_version", "artifact_path"} <= manifest.keys()
    assert manifest["registry_synced"]  # MLflow alias mirrored

    # same model again -> nothing to do
    assert not promote_files(models, RELAXED, report_path=tmp_path / "gate.json").promoted

    # a "retrained" candidate (different run id) is promoted; the old champion becomes previous_champion
    meta = read_meta(models / "candidate")
    meta["run_id"], meta["run_name"] = "other-run", "run-20260102-000000"
    (models / "candidate" / cfg.MODEL_META).write_text(json.dumps(meta))
    assert promote_files(models, RELAXED, report_path=tmp_path / "gate.json").promoted
    assert read_meta(models / "previous_champion")["run_id"] == best["run_id"]
    assert json.loads((models / cfg.MODEL_MANIFEST).read_text())["previous_champion"] is not None

    # a strict gate rejects and leaves the champion untouched
    meta["run_id"] = "third-run"
    (models / "candidate" / cfg.MODEL_META).write_text(json.dumps(meta))
    res = promote_files(models, GateConfig(max_fpr=0.0001, fpr_slack=0.0), report_path=tmp_path / "gate.json")
    assert not res.passed and read_meta(models / "champion")["run_id"] == "other-run"

    # rollback swaps champion and previous_champion
    rollback_files(models)
    assert read_meta(models / "champion")["run_id"] == best["run_id"]
    assert read_meta(models / "previous_champion")["run_id"] == "other-run"
    assert status_files(models)["champion"].startswith(manifest["champion_version"])

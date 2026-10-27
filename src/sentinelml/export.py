"""Export the champion into a self-contained folder (MLflow model + export_meta.json).

The registry stores absolute artifact paths from the machine that trained the model, so a
Docker container cannot follow them. The exported folder has no such dependency:

    python -m sentinelml.export
    docker run -p 8000:8000 -v "$PWD/exported_model:/models/champion:ro" sentinelml-api
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from sentinelml import config as cfg
from sentinelml.registry import alias_version, environment_versions, get_client, run_metrics


def write_model_dir(pipe, out_dir: Path, meta: dict, signature=None) -> None:
    """Self-contained model folder (MLflow sklearn model + export_meta.json), loadable via MODEL_DIR.

    Used for exported_model/ and for the DVC-tracked models/{candidate,champion,previous_champion}/.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    mlflow.sklearn.save_model(pipe, str(out_dir), signature=signature)
    (out_dir / cfg.MODEL_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def export_model(model_name: str, alias: str, out_dir: Path, tracking_uri: str | None = None) -> dict:
    client = get_client(tracking_uri)
    mv = alias_version(client, model_name, alias)
    if mv is None:
        raise LookupError(f"No '{alias}' alias for '{model_name}'. Run: python -m sentinelml.gate promote")

    pipe = mlflow.sklearn.load_model(f"models:/{model_name}/{mv.version}")
    signature = mlflow.models.get_model_info(f"models:/{model_name}/{mv.version}").signature

    tags = {k: float(v) for k, v in mv.tags.items()}
    meta = {
        "model_name": model_name,
        "version": str(mv.version),
        "alias": alias,
        "threshold": tags.pop("threshold"),
        "metrics": tags,
        "run_metrics": run_metrics(client, mv),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment_versions(),
    }
    write_model_dir(pipe, out_dir, meta, signature)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tracking-uri", default=cfg.DEFAULT_TRACKING_URI)
    ap.add_argument("--model-name", default=cfg.DEFAULT_MODEL_NAME)
    ap.add_argument("--alias", default=cfg.ALIAS_CHAMPION)
    ap.add_argument("--out", default=str(cfg.EXPORT_DIR))
    a = ap.parse_args()
    try:
        meta = export_model(a.model_name, a.alias, Path(a.out), a.tracking_uri)
    except LookupError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(f"Exported {meta['model_name']} v{meta['version']} ({a.alias}) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

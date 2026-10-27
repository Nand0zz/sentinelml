"""Quality gate: decides whether the `candidate` model may replace the `champion`.

    python -m sentinelml.gate status
    python -m sentinelml.gate promote [--dry-run] [--push]
    python -m sentinelml.gate rollback

Rules (all must pass):
  Absolute  - FPR within the budget (+ slack), PR-AUC floor, recall floor, latency ceiling
  Relative  - no meaningful regression vs. the current champion (skipped if none exists)
Metrics come from the held-out test evaluation logged at training time, identical for both
models (same data, same threshold procedure). In production you would re-score both models
on a fresh labelled evaluation set instead.

Two backends, picked automatically:
  files    - models/{candidate,champion,previous_champion}/ (DVC pipeline). Promotion rotates the
             folders, `dvc add`s them, writes models/model_manifest.json and mirrors the MLflow
             aliases when the local registry has the candidate. Works from a fresh clone + dvc pull.
  registry - MLflow aliases only (used when models/ has no model folders).
Exit codes: 0 = pass, 1 = rejected, 2 = error.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from sentinelml import config as cfg
from sentinelml.registry import alias_version, get_client, run_metrics


@dataclass(frozen=True)
class GateConfig:
    max_fpr: float = cfg.DEFAULT_MAX_FPR
    fpr_slack: float = 0.01          # test FPR may exceed the budget by this much (val->test noise)
    min_pr_auc: float = 0.95
    min_recall: float = 0.80
    max_latency_ms: float = 100.0    # single-row prediction latency
    max_pr_auc_drop: float = 0.002   # allowed regression vs champion
    max_recall_drop: float = 0.01


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class GateResult:
    passed: bool
    promoted: bool
    candidate_version: str
    champion_version: str | None
    checks: list[Check]
    message: str


def _m(d: dict | None, key: str) -> float:
    return float(d.get(key, math.nan)) if d else math.nan


def evaluate_gate(cand: dict, champ: dict | None, c: GateConfig) -> list[Check]:
    """Pure function (no MLflow): easy to unit-test. A missing metric fails its check."""
    out = []
    fpr, limit = _m(cand, "test_fpr"), c.max_fpr + c.fpr_slack
    out.append(Check("fpr_within_budget", fpr <= limit, f"test FPR {fpr:.4f} <= {limit:.4f}"))
    pr = _m(cand, "test_pr_auc")
    out.append(Check("pr_auc_floor", pr >= c.min_pr_auc, f"test PR-AUC {pr:.4f} >= {c.min_pr_auc:.4f}"))
    rec = _m(cand, "test_recall")
    out.append(Check("recall_floor", rec >= c.min_recall, f"test recall {rec:.4f} >= {c.min_recall:.4f}"))
    lat = _m(cand, "single_row_latency_ms")
    out.append(Check("latency_ceiling", lat <= c.max_latency_ms,
                     f"latency {lat:.1f} ms <= {c.max_latency_ms:.1f} ms"))
    if champ is None:
        for name in ("no_pr_auc_regression", "no_recall_regression"):
            out.append(Check(name, True, "skipped (no champion yet)"))
    else:
        floor = _m(champ, "test_pr_auc") - c.max_pr_auc_drop
        out.append(Check("no_pr_auc_regression", pr >= floor,
                         f"PR-AUC {pr:.4f} >= champion {_m(champ, 'test_pr_auc'):.4f} - {c.max_pr_auc_drop}"))
        floor = _m(champ, "test_recall") - c.max_recall_drop
        out.append(Check("no_recall_regression", rec >= floor,
                         f"recall {rec:.4f} >= champion {_m(champ, 'test_recall'):.4f} - {c.max_recall_drop}"))
    return out


def promote(model_name: str = cfg.DEFAULT_MODEL_NAME, tracking_uri: str | None = None,
            config: GateConfig = GateConfig(), dry_run: bool = False,
            report_path: Path | None = None) -> GateResult:
    client = get_client(tracking_uri)
    cand = alias_version(client, model_name, cfg.ALIAS_CANDIDATE)
    if cand is None:
        raise LookupError(f"No '{cfg.ALIAS_CANDIDATE}' alias for '{model_name}'. Run training first.")
    champ = alias_version(client, model_name, cfg.ALIAS_CHAMPION)

    if champ is not None and str(champ.version) == str(cand.version):
        return GateResult(True, False, str(cand.version), str(champ.version), [],
                          f"v{cand.version} is already the champion; nothing to do.")

    checks = evaluate_gate(run_metrics(client, cand), run_metrics(client, champ) if champ else None, config)
    passed = all(ch.passed for ch in checks)
    promoted = False
    if passed and not dry_run:
        if champ is not None:
            client.set_registered_model_alias(model_name, cfg.ALIAS_PREVIOUS, champ.version)
        client.set_registered_model_alias(model_name, cfg.ALIAS_CHAMPION, cand.version)
        promoted = True

    if passed:
        msg = (f"PASS: v{cand.version} " + ("would be promoted (dry run)." if dry_run else "promoted to champion."))
    else:
        msg = f"REJECTED: v{cand.version} stays a candidate; champion is unchanged."
    result = GateResult(passed, promoted, str(cand.version), str(champ.version) if champ else None, checks, msg)
    _write_report(result, model_name, config, dry_run, report_path)
    return result


def _write_report(result: GateResult, model_name: str, config: GateConfig, dry_run: bool,
                  report_path: Path | None, extra: dict | None = None) -> Path:
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "dry_run": dry_run,
        "config": asdict(config),
        "result": {**asdict(result), "checks": [asdict(x) for x in result.checks]},
        **(extra or {}),
    }
    path = Path(report_path or cfg.ROOT / "reports" / "gate_report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------- files backend (DVC)

def read_meta(model_dir: Path) -> dict | None:
    f = Path(model_dir) / cfg.MODEL_META
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def has_model_dirs(models_dir: Path) -> bool:
    return any(read_meta(Path(models_dir) / a) for a in (cfg.ALIAS_CANDIDATE, cfg.ALIAS_CHAMPION))


def _set_alias(model_dir: Path, alias: str) -> None:
    meta = read_meta(model_dir)
    meta["alias"] = alias
    (model_dir / cfg.MODEL_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _copy(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp)
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)


def _dvc(models_dir: Path, *args: str) -> bool:
    """Run a DVC command from the repo that contains models_dir. False if DVC isn't set up there."""
    root = Path(models_dir).resolve().parent
    if not (root / ".dvc").is_dir() or importlib.util.find_spec("dvc") is None:
        return False
    subprocess.run([sys.executable, "-m", "dvc", *args], cwd=root, check=True)
    return True


def _dvc_md5(models_dir: Path, alias: str) -> str | None:
    f = Path(models_dir) / f"{alias}.dvc"
    return yaml.safe_load(f.read_text(encoding="utf-8"))["outs"][0]["md5"] if f.exists() else None


def _sync_registry(model_name: str, tracking_uri: str | None, cand_run_id: str, rollback_to: str | None = None) -> bool:
    """Best effort: mirror the folder promotion onto MLflow aliases if the local registry knows the model."""
    uri = tracking_uri or cfg.DEFAULT_TRACKING_URI
    if uri.startswith("sqlite:///") and not Path(uri[len("sqlite:///"):]).exists():
        return False
    try:
        client = get_client(uri)
        if rollback_to is not None:
            target = next((v for v in client.search_model_versions(f"name='{model_name}'")
                           if v.run_id == rollback_to), None)
        else:
            target = alias_version(client, model_name, cfg.ALIAS_CANDIDATE)
            target = target if target is not None and target.run_id == cand_run_id else None
        if target is None:
            return False
        champ = alias_version(client, model_name, cfg.ALIAS_CHAMPION)
        if champ is not None and str(champ.version) != str(target.version):
            client.set_registered_model_alias(model_name, cfg.ALIAS_PREVIOUS, champ.version)
        client.set_registered_model_alias(model_name, cfg.ALIAS_CHAMPION, target.version)
        return True
    except Exception:  # the registry is a mirror; the folders are the source of truth
        return False


def write_manifest(models_dir: Path, promotion: dict) -> dict:
    """models/model_manifest.json: what is deployed, where it lives, and its DVC content hash."""
    models_dir = Path(models_dir)
    champ = read_meta(models_dir / cfg.ALIAS_CHAMPION)
    prev = read_meta(models_dir / cfg.ALIAS_PREVIOUS)
    manifest = {
        "model_name": champ["model_name"],
        "champion_version": champ["version"],
        "run_id": champ.get("run_id"),
        "run_name": champ.get("run_name"),
        "model_type": champ.get("model_type"),
        "threshold": champ["threshold"],
        "metrics": champ.get("metrics", {}),
        "artifact_path": f"{models_dir.name}/{cfg.ALIAS_CHAMPION}",
        "dvc_md5": _dvc_md5(models_dir, cfg.ALIAS_CHAMPION),
        "previous_champion": prev["version"] if prev else None,
        "previous_champion_dvc_md5": _dvc_md5(models_dir, cfg.ALIAS_PREVIOUS) if prev else None,
        **promotion,
    }
    (models_dir / cfg.MODEL_MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _track(models_dir: Path, push: bool) -> bool:
    dirs = [f"{models_dir.name}/{a}" for a in (cfg.ALIAS_CHAMPION, cfg.ALIAS_PREVIOUS)
            if (models_dir / a).is_dir()]
    tracked = _dvc(models_dir, "add", *dirs)
    if tracked and push:
        _dvc(models_dir, "push", *[d + ".dvc" for d in dirs])
    return tracked


def promote_files(models_dir: Path = cfg.MODELS_DIR, config: GateConfig = GateConfig(),
                  dry_run: bool = False, report_path: Path | None = None,
                  tracking_uri: str | None = None, push: bool = False) -> GateResult:
    models_dir = Path(models_dir)
    cand_dir, champ_dir, prev_dir = (models_dir / a for a in
                                     (cfg.ALIAS_CANDIDATE, cfg.ALIAS_CHAMPION, cfg.ALIAS_PREVIOUS))
    cand, champ = read_meta(cand_dir), read_meta(champ_dir)
    if cand is None:
        raise LookupError(f"No candidate in {cand_dir}. Run: dvc repro  (or python -m sentinelml.train)")
    model_name = cand["model_name"]
    if champ is not None and champ.get("run_id") == cand.get("run_id"):
        return GateResult(True, False, cand["version"], champ["version"], [],
                          f"v{cand['version']} ({cand.get('run_name')}) is already the champion; nothing to do.")

    checks = evaluate_gate(cand.get("run_metrics"), champ.get("run_metrics") if champ else None, config)
    passed = all(ch.passed for ch in checks)
    promoted = passed and not dry_run
    if passed:
        msg = (f"PASS: v{cand['version']} ({cand.get('run_name')}) "
               + ("would be promoted (dry run)." if dry_run else "promoted to champion."))
    else:
        msg = f"REJECTED: v{cand['version']} stays a candidate; champion is unchanged."
    result = GateResult(passed, promoted, cand["version"], champ["version"] if champ else None, checks, msg)
    report = _write_report(result, model_name, config, dry_run, report_path, {"backend": "files"})

    if promoted:
        if champ is not None:
            _copy(champ_dir, prev_dir)
            _set_alias(prev_dir, cfg.ALIAS_PREVIOUS)
        _copy(cand_dir, champ_dir)
        _set_alias(champ_dir, cfg.ALIAS_CHAMPION)
        tracked = _track(models_dir, push)
        try:
            report_ref = report.resolve().relative_to(models_dir.resolve().parent).as_posix()
        except ValueError:
            report_ref = report.as_posix()
        write_manifest(models_dir, {
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "gate_report_path": report_ref,
            "dvc_tracked": tracked,
            "registry_synced": _sync_registry(model_name, tracking_uri, cand.get("run_id")),
        })
    return result


def rollback_files(models_dir: Path = cfg.MODELS_DIR, tracking_uri: str | None = None, push: bool = False) -> str:
    """Swap champion and previous_champion (running it twice rolls forward again)."""
    models_dir = Path(models_dir)
    champ_dir, prev_dir = models_dir / cfg.ALIAS_CHAMPION, models_dir / cfg.ALIAS_PREVIOUS
    prev = read_meta(prev_dir)
    if prev is None:
        raise LookupError(f"No previous champion in {prev_dir} to roll back to.")
    swap = models_dir / "_swap"
    _copy(champ_dir, swap)
    _copy(prev_dir, champ_dir)
    _copy(swap, prev_dir)
    shutil.rmtree(swap)
    _set_alias(champ_dir, cfg.ALIAS_CHAMPION)
    _set_alias(prev_dir, cfg.ALIAS_PREVIOUS)
    tracked = _track(models_dir, push)
    write_manifest(models_dir, {
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "gate_report_path": None,
        "rollback": True,
        "dvc_tracked": tracked,
        "registry_synced": _sync_registry(prev["model_name"], tracking_uri, prev.get("run_id"),
                                          rollback_to=prev.get("run_id")),
    })
    return prev["version"]


def status_files(models_dir: Path = cfg.MODELS_DIR) -> dict:
    out = {}
    for alias in (cfg.ALIAS_CANDIDATE, cfg.ALIAS_CHAMPION, cfg.ALIAS_PREVIOUS):
        m = read_meta(Path(models_dir) / alias)
        out[alias] = f"{m['version']} ({m.get('run_name')}, {m.get('model_type')})" if m else None
    return out


def rollback(model_name: str = cfg.DEFAULT_MODEL_NAME, tracking_uri: str | None = None) -> str:
    client = get_client(tracking_uri)
    prev = alias_version(client, model_name, cfg.ALIAS_PREVIOUS)
    if prev is None:
        raise LookupError("No previous champion to roll back to.")
    client.set_registered_model_alias(model_name, cfg.ALIAS_CHAMPION, prev.version)
    client.delete_registered_model_alias(model_name, cfg.ALIAS_PREVIOUS)
    return str(prev.version)


def status(model_name: str = cfg.DEFAULT_MODEL_NAME, tracking_uri: str | None = None) -> dict:
    client = get_client(tracking_uri)
    out = {}
    for alias in (cfg.ALIAS_CANDIDATE, cfg.ALIAS_CHAMPION, cfg.ALIAS_PREVIOUS):
        mv = alias_version(client, model_name, alias)
        out[alias] = str(mv.version) if mv else None
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SentinelML quality gate")
    ap.add_argument("--tracking-uri", default=cfg.DEFAULT_TRACKING_URI)
    ap.add_argument("--model-name", default=cfg.DEFAULT_MODEL_NAME)
    ap.add_argument("--models-dir", default=str(cfg.MODELS_DIR),
                    help="model folders (files backend, used whenever it contains a model)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("promote", help="run the gate; promote candidate -> champion if it passes")
    pr.add_argument("--dry-run", action="store_true")
    pr.add_argument("--push", action="store_true", help="dvc push the champion after promotion")
    d = GateConfig()
    pr.add_argument("--max-fpr", type=float, default=d.max_fpr)
    pr.add_argument("--fpr-slack", type=float, default=d.fpr_slack)
    pr.add_argument("--min-pr-auc", type=float, default=d.min_pr_auc)
    pr.add_argument("--min-recall", type=float, default=d.min_recall)
    pr.add_argument("--max-latency-ms", type=float, default=d.max_latency_ms)
    pr.add_argument("--max-pr-auc-drop", type=float, default=d.max_pr_auc_drop)
    pr.add_argument("--max-recall-drop", type=float, default=d.max_recall_drop)
    rb = sub.add_parser("rollback", help="restore the previous champion")
    rb.add_argument("--push", action="store_true", help="dvc push after rolling back")
    sub.add_parser("status", help="show which version holds each alias")
    a = ap.parse_args(argv)
    files = has_model_dirs(Path(a.models_dir))

    try:
        if a.cmd == "status":
            st = status_files(a.models_dir) if files else status(a.model_name, a.tracking_uri)
            print(f"(backend: {'files ' + a.models_dir if files else 'mlflow registry'})")
            for alias, ver in st.items():
                print(f"{alias:>18}: {'v' + ver if ver else '-'}")
            return 0
        if a.cmd == "rollback":
            ver = (rollback_files(Path(a.models_dir), a.tracking_uri, a.push) if files
                   else rollback(a.model_name, a.tracking_uri))
            print(f"Rolled back: champion is now v{ver}")
            return 0
        conf = GateConfig(a.max_fpr, a.fpr_slack, a.min_pr_auc, a.min_recall, a.max_latency_ms,
                          a.max_pr_auc_drop, a.max_recall_drop)
        if files:
            res = promote_files(Path(a.models_dir), conf, a.dry_run, tracking_uri=a.tracking_uri, push=a.push)
        else:
            res = promote(a.model_name, a.tracking_uri, conf, a.dry_run)
    except (LookupError, subprocess.CalledProcessError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(f"\nQuality gate: candidate v{res.candidate_version} vs champion "
          f"{'v' + res.champion_version if res.champion_version else '(none)'}")
    for ch in res.checks:
        print(f"  [{'PASS' if ch.passed else 'FAIL'}] {ch.name:<22} {ch.detail}")
    print(f"\n{res.message}")
    if files and res.promoted:
        print(f"Wrote {Path(a.models_dir).name}/{cfg.MODEL_MANIFEST}. "
              + ("Pushed to the DVC remote." if a.push else "Next: dvc push   (then commit the .dvc files + manifest)"))
    return 0 if res.passed else 1


if __name__ == "__main__":
    sys.exit(main())

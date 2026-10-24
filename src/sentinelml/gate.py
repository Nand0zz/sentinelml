"""Quality gate: decides whether the `candidate` model may replace the `champion`.

    python -m sentinelml.gate status
    python -m sentinelml.gate promote [--dry-run]
    python -m sentinelml.gate rollback

Rules (all must pass):
  Absolute  - FPR within the budget (+ slack), PR-AUC floor, recall floor, latency ceiling
  Relative  - no meaningful regression vs. the current champion (skipped if none exists)
Metrics come from the held-out test evaluation logged at training time, identical for both
models (same data, same threshold procedure). In production you would re-score both models
on a fresh labelled evaluation set instead.
Exit codes: 0 = pass, 1 = rejected, 2 = error.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

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

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "dry_run": dry_run,
        "config": asdict(config),
        "result": {**asdict(result), "checks": [asdict(x) for x in checks]},
    }
    path = Path(report_path or cfg.ROOT / "reports" / "gate_report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return result


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
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("promote", help="run the gate; promote candidate -> champion if it passes")
    pr.add_argument("--dry-run", action="store_true")
    d = GateConfig()
    pr.add_argument("--max-fpr", type=float, default=d.max_fpr)
    pr.add_argument("--fpr-slack", type=float, default=d.fpr_slack)
    pr.add_argument("--min-pr-auc", type=float, default=d.min_pr_auc)
    pr.add_argument("--min-recall", type=float, default=d.min_recall)
    pr.add_argument("--max-latency-ms", type=float, default=d.max_latency_ms)
    pr.add_argument("--max-pr-auc-drop", type=float, default=d.max_pr_auc_drop)
    pr.add_argument("--max-recall-drop", type=float, default=d.max_recall_drop)
    sub.add_parser("rollback", help="restore the previous champion")
    sub.add_parser("status", help="show which version holds each alias")
    a = ap.parse_args(argv)

    try:
        if a.cmd == "status":
            for alias, ver in status(a.model_name, a.tracking_uri).items():
                print(f"{alias:>18}: {'v' + ver if ver else '-'}")
            return 0
        if a.cmd == "rollback":
            print(f"Rolled back: champion is now v{rollback(a.model_name, a.tracking_uri)}")
            return 0
        conf = GateConfig(a.max_fpr, a.fpr_slack, a.min_pr_auc, a.min_recall, a.max_latency_ms,
                          a.max_pr_auc_drop, a.max_recall_drop)
        res = promote(a.model_name, a.tracking_uri, conf, a.dry_run)
    except LookupError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(f"\nQuality gate: candidate v{res.candidate_version} vs champion "
          f"{'v' + res.champion_version if res.champion_version else '(none)'}")
    for ch in res.checks:
        print(f"  [{'PASS' if ch.passed else 'FAIL'}] {ch.name:<22} {ch.detail}")
    print(f"\n{res.message}")
    return 0 if res.passed else 1


if __name__ == "__main__":
    sys.exit(main())

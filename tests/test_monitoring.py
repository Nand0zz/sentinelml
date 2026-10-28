import json

import numpy as np
import pandas as pd

from sentinelml import config as cfg
from sentinelml.data import clean
from sentinelml.monitoring import drift_report, load_prediction_log, psi, psi_categorical
from sentinelml.synthetic import make_synthetic


def test_psi_is_near_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    ref = rng.lognormal(0, 1, 20_000)
    assert psi(ref, rng.lognormal(0, 1, 5_000)) < 0.02
    assert psi(ref, rng.lognormal(1.0, 1, 5_000)) > 0.25


def test_psi_categorical():
    a = pd.Series(["tcp"] * 80 + ["udp"] * 20)
    assert psi_categorical(a, a) < 1e-9
    assert psi_categorical(a, pd.Series(["tcp"] * 20 + ["udp"] * 80)) > 0.25


def test_drift_report_flags_shifted_traffic():
    ref = clean(make_synthetic(4000, seed=0))
    same = drift_report(ref, clean(make_synthetic(2000, seed=1)))
    shifted = drift_report(ref, clean(make_synthetic(2000, seed=1, shift=1.0)))
    assert set(same["feature"]) == set(cfg.NUMERIC + cfg.CATEGORICAL)
    assert (shifted["status"] == "significant").sum() > (same["status"] == "significant").sum()
    assert shifted["psi"].is_monotonic_decreasing


def test_load_prediction_log_from_file_and_records(tmp_path):
    rec = {"ts": "2026-10-28T10:00:00+00:00", "model_version": "1", "score": 0.9, "is_suspicious": True,
           "request_latency_ms": 12.0, "batch_size": 1, "features": {"proto": "tcp", "sbytes": 10.0}}
    p = tmp_path / "predictions.jsonl"
    p.write_text(json.dumps(rec) + "\n" + json.dumps({**rec, "score": 0.1}) + "\n{broken", encoding="utf-8")
    df = load_prediction_log(p)
    assert len(df) == 2 and {"score", "proto", "sbytes"} <= set(df.columns)
    assert len(load_prediction_log(p, limit=2)) == 1  # the broken last line is skipped
    assert load_prediction_log(tmp_path / "missing.jsonl").empty
    assert len(load_prediction_log([rec])) == 1

"""Prediction-log monitoring: load the log, measure feature drift against the training data.

Drift per feature: PSI (population stability index, quantile bins of the reference) and the
two-sample KS test for numerics; PSI on category shares for categoricals.
Rule of thumb for PSI: < 0.10 stable, 0.10-0.25 moderate, > 0.25 significant.
Used by the dashboard (dashboard/app.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from sentinelml import config as cfg

PSI_MODERATE, PSI_SIGNIFICANT = 0.10, 0.25
_EPS = 1e-4  # floor for empty bins, keeps log() finite


def load_prediction_log(source, limit: int | None = None) -> pd.DataFrame:
    """JSONL path or list of log records -> one row per scored flow (log fields + features)."""
    if isinstance(source, (str, Path)):
        p = Path(source)
        lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
        records = []
        for line in lines[-limit:] if limit else lines:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:  # a partially written last line
                continue
    else:
        records = list(source)
    if not records:
        return pd.DataFrame()
    meta = pd.DataFrame([{k: v for k, v in r.items() if k != "features"} for r in records])
    feats = pd.DataFrame([r.get("features") or {} for r in records])
    out = pd.concat([meta, feats], axis=1)
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce", format="ISO8601")
    return out


def psi(expected, actual, bins: int = 10) -> float:
    """PSI of `actual` against `expected`, using quantile cut points of `expected` (NaNs ignored)."""
    e = pd.to_numeric(pd.Series(expected), errors="coerce").dropna().to_numpy(dtype=float)
    a = pd.to_numeric(pd.Series(actual), errors="coerce").dropna().to_numpy(dtype=float)
    if len(e) == 0 or len(a) == 0:
        return float("nan")
    cuts = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)[1:-1]))
    n_bins = len(cuts) + 1
    pe = np.bincount(np.searchsorted(cuts, e, side="right"), minlength=n_bins) / len(e)
    pa = np.bincount(np.searchsorted(cuts, a, side="right"), minlength=n_bins) / len(a)
    pe, pa = np.clip(pe, _EPS, None), np.clip(pa, _EPS, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_categorical(expected, actual) -> float:
    e, a = pd.Series(expected).dropna().astype(str), pd.Series(actual).dropna().astype(str)
    if e.empty or a.empty:
        return float("nan")
    cats = sorted(set(e) | set(a))
    pe = e.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=_EPS)
    pa = a.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=_EPS)
    return float(((pa - pe) * np.log(pa / pe)).sum())


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """One row per model input feature: PSI, KS statistic/p-value (numerics) and a status label."""
    rows = []
    for c in cfg.NUMERIC:
        if c not in current:
            continue
        ref = pd.to_numeric(reference[c], errors="coerce").dropna()
        cur = pd.to_numeric(current[c], errors="coerce").dropna()
        ks = ks_2samp(ref, cur) if len(ref) > 1 and len(cur) > 1 else None
        rows.append({"feature": c, "type": "numeric", "psi": psi(ref, cur),
                     "ks_stat": float(ks.statistic) if ks else np.nan,
                     "ks_pvalue": float(ks.pvalue) if ks else np.nan})
    for c in cfg.CATEGORICAL:
        if c in current:
            rows.append({"feature": c, "type": "categorical", "psi": psi_categorical(reference[c], current[c]),
                         "ks_stat": np.nan, "ks_pvalue": np.nan})
    out = pd.DataFrame(rows, columns=["feature", "type", "psi", "ks_stat", "ks_pvalue"])
    out["status"] = np.select([out["psi"] > PSI_SIGNIFICANT, out["psi"] > PSI_MODERATE],
                              ["significant", "moderate"], "stable")
    return out.sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)

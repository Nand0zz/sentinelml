"""Exploratory analysis -> reports/eda_summary.md + reports/eda_plots.html

    python scripts/eda.py                # real UNSW-NB15 in data/raw
    python scripts/eda.py --synthetic    # smoke test
"""
from __future__ import annotations

import argparse

import pandas as pd
import plotly.express as px
from scipy.stats import ks_2samp

from sentinelml import config as cfg
from sentinelml.data import clean, load_raw
from sentinelml.synthetic import make_synthetic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(cfg.RAW_DIR))
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    if a.synthetic:
        raw_tr, raw_te = make_synthetic(20_000, 42), make_synthetic(8_000, 43, shift=0.3)
    else:
        raw_tr = load_raw(f"{a.data_dir}/{cfg.TRAIN_FILE}")
        raw_te = load_raw(f"{a.data_dir}/{cfg.TEST_FILE}")

    num = raw_tr[cfg.NUMERIC].apply(pd.to_numeric, errors="coerce")
    n_inf = int(((num == float("inf")) | (num == float("-inf"))).sum().sum())
    feats = cfg.FEATURES + [cfg.TARGET]
    n_dup = int(raw_tr[feats].duplicated().sum())
    tr, te = clean(raw_tr), clean(raw_te, drop_duplicates=False)

    corr = (
        tr[cfg.NUMERIC].corrwith(tr[cfg.TARGET], method="spearman")
        .abs().sort_values(ascending=False).head(10)
    )
    ks = {}
    s_tr, s_te = tr.sample(min(20_000, len(tr)), random_state=0), te.sample(min(20_000, len(te)), random_state=0)
    for c in cfg.NUMERIC:
        ks[c] = ks_2samp(s_tr[c].dropna(), s_te[c].dropna()).statistic
    ks = pd.Series(ks).sort_values(ascending=False).head(10)

    lines = [
        "# EDA summary", "",
        f"- Train rows (raw): {len(raw_tr):,} | Test rows (raw): {len(raw_te):,}",
        f"- Exact duplicate rows in train: {n_dup:,} (removed in `clean`)",
        f"- +-inf values in numeric train columns: {n_inf:,} (-> NaN -> median-imputed)",
        f"- Attack share, train: {tr[cfg.TARGET].mean():.1%} | test: {te[cfg.TARGET].mean():.1%}",
        "  (attack-heavy on purpose; real networks are overwhelmingly benign, so",
        "  precision here is optimistic vs. production prevalence)", "",
        "## Top-10 features by |Spearman| with label", corr.round(3).to_frame("abs_corr").to_markdown(), "",
        "## Top-10 train-vs-test shift (KS statistic)",
        ks.round(3).to_frame("ks_stat").to_markdown(), "",
        "Large KS values show the official split is already distribution-shifted,",
        "a natural motivation for the drift monitoring built on Day 3.",
    ]
    (cfg.ROOT / "reports").mkdir(exist_ok=True)
    (cfg.ROOT / "reports" / "eda_summary.md").write_text("\n".join(lines))

    figs = [
        px.bar(tr[cfg.TARGET].value_counts().rename({0: "benign", 1: "malicious"}),
               title="Class balance (train)"),
        px.bar(corr.sort_values(), orientation="h", title="Top |Spearman| correlation with label"),
        px.bar(ks.sort_values(), orientation="h", title="Train vs test shift (KS)"),
    ]
    if "attack_cat" in raw_tr:
        figs.insert(1, px.bar(raw_tr["attack_cat"].value_counts(), title="Attack categories (train)"))
    html = "".join(f.to_html(full_html=False, include_plotlyjs="cdn") for f in figs)
    (cfg.ROOT / "reports" / "eda_plots.html").write_text(html)
    print("Wrote reports/eda_summary.md and reports/eda_plots.html")


if __name__ == "__main__":
    main()

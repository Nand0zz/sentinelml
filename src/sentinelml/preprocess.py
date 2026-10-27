"""Preprocessing stage: raw UNSW-NB15 CSVs -> cleaned parquet splits + feature schema.

    python -m sentinelml.preprocess                # data/raw -> data/processed (DVC stage `preprocess`)
    python -m sentinelml.preprocess --synthetic    # generated data, no download (CI / smoke tests)

"Normalising" here is the same `clean()` used by training and the API: typed numerics, +-inf -> NaN,
lower-cased categoricals, leakage/artifact columns dropped, train de-duplicated. Imputation, scaling
and one-hot encoding deliberately stay inside the sklearn Pipeline, so serving can never diverge
from training. Outputs contain no timestamps: the same input always gives identical files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sentinelml import config as cfg
from sentinelml.data import load_splits, load_synthetic_splits


def _num(v) -> float | None:
    return None if pd.isna(v) else float(v)


def feature_schema(train: pd.DataFrame, test: pd.DataFrame, source: str) -> dict:
    """Everything inference needs to know about the model input, derived from the TRAIN split."""
    return {
        "source": source,
        "target": cfg.TARGET,
        "features": cfg.FEATURES,
        "dropped_columns": cfg.LEAKAGE_COLS + cfg.ARTIFACT_COLS,
        "engineered_in_pipeline": cfg.ENGINEERED,
        "categorical": {
            c: {"dtype": "string", "values": sorted(train[c].dropna().unique().tolist())}
            for c in cfg.CATEGORICAL
        },
        "numeric": {
            c: {"dtype": "float64", "min": _num(train[c].min()), "median": _num(train[c].median()),
                "max": _num(train[c].max()), "missing": int(train[c].isna().sum())}
            for c in cfg.NUMERIC
        },
        "rows": {"train": len(train), "test": len(test)},
        "positive_rate": {"train": float(train[cfg.TARGET].mean()), "test": float(test[cfg.TARGET].mean())},
    }


def run(train: pd.DataFrame, test: pd.DataFrame, out_dir: Path, source: str) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / cfg.PROCESSED_TRAIN, index=False)
    test.to_parquet(out_dir / cfg.PROCESSED_TEST, index=False)
    schema = feature_schema(train, test, source)
    (out_dir / cfg.FEATURE_SCHEMA).write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return schema


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-dir", default=str(cfg.RAW_DIR))
    ap.add_argument("--out-dir", default=str(cfg.PROCESSED_DIR))
    ap.add_argument("--synthetic", action="store_true", help="use generated data (smoke test)")
    ap.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED, help="synthetic data only")
    a = ap.parse_args()

    if a.synthetic:
        train, test = load_synthetic_splits(a.seed)
    else:
        train, test = load_splits(a.raw_dir)
    schema = run(train, test, Path(a.out_dir), "synthetic" if a.synthetic else "unsw-nb15")
    print(f"Wrote {a.out_dir}/{{{cfg.PROCESSED_TRAIN},{cfg.PROCESSED_TEST},{cfg.FEATURE_SCHEMA}}}  "
          f"rows: train={schema['rows']['train']:,} test={schema['rows']['test']:,}")


if __name__ == "__main__":
    main()

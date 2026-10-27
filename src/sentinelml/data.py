"""Loading and cleaning UNSW-NB15."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sentinelml import config as cfg


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def normalize_features(X: pd.DataFrame) -> pd.DataFrame:
    """Single source of truth for feature typing, used by training AND the API.

    Keeping this in one place prevents training/serving skew (e.g. 'TCP' vs 'tcp').
    """
    out = X[cfg.FEATURES].copy()
    out[cfg.NUMERIC] = out[cfg.NUMERIC].apply(pd.to_numeric, errors="coerce").astype("float64")
    out[cfg.NUMERIC] = out[cfg.NUMERIC].replace([np.inf, -np.inf], np.nan)
    for col in cfg.CATEGORICAL:
        out[col] = out[col].astype(str).str.strip().str.lower()
    return out


def clean(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Return a model-ready frame with columns FEATURES + [TARGET].

    - drops leakage / artifact columns
    - coerces numerics to float64, replaces +-inf with NaN (imputed in the pipeline)
    - normalises categorical strings
    - optionally drops exact duplicate rows (do this for training data only)
    """
    missing = set(cfg.FEATURES + [cfg.TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    out = normalize_features(df[cfg.FEATURES])
    out[cfg.TARGET] = df[cfg.TARGET].astype(int).to_numpy()
    if drop_duplicates:
        out = out.drop_duplicates()
    return out.reset_index(drop=True)


def split_train_val(train_full: pd.DataFrame, seed: int = cfg.DEFAULT_SEED, val_size: float = 0.2):
    """The one place the train/validation split is defined (stratified on the label)."""
    return train_test_split(
        train_full, test_size=val_size, stratify=train_full[cfg.TARGET], random_state=seed
    )


def load_synthetic_splits(seed: int = cfg.DEFAULT_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generated stand-in for the official splits (smoke tests / CI only)."""
    from sentinelml.synthetic import make_synthetic

    train = clean(make_synthetic(20_000, seed))
    test = clean(make_synthetic(8_000, seed + 1, shift=0.3), drop_duplicates=False)
    return train, test


def load_splits(data_dir: str | Path = cfg.RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Official UNSW-NB15 train/test CSVs, cleaned. Test rows are never deduplicated."""
    data_dir = Path(data_dir)
    train_path, test_path = data_dir / cfg.TRAIN_FILE, data_dir / cfg.TEST_FILE
    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Download UNSW-NB15 and place "
                f"{cfg.TRAIN_FILE} and {cfg.TEST_FILE} in {data_dir} "
                "(or run with --synthetic to smoke-test)."
            )
    train = clean(load_raw(train_path), drop_duplicates=True)
    test = clean(load_raw(test_path), drop_duplicates=False)
    return train, test


def load_processed(processed_dir: str | Path = cfg.PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cleaned train/test written by the `preprocess` stage (python -m sentinelml.preprocess)."""
    processed_dir = Path(processed_dir)
    out = []
    for name in (cfg.PROCESSED_TRAIN, cfg.PROCESSED_TEST):
        p = processed_dir / name
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run: dvc repro preprocess  (or python -m sentinelml.preprocess)")
        df = pd.read_parquet(p)
        missing = set(cfg.FEATURES + [cfg.TARGET]) - set(df.columns)
        if missing:
            raise ValueError(f"{p} is missing columns: {sorted(missing)}. Re-run preprocessing.")
        out.append(df[cfg.FEATURES + [cfg.TARGET]])
    return out[0], out[1]

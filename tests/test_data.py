import numpy as np
import pytest

from sentinelml import config as cfg
from sentinelml.data import clean
from sentinelml.synthetic import make_synthetic


def test_clean_output_contract():
    raw = make_synthetic(2000, seed=1)
    out = clean(raw)
    assert list(out.columns) == cfg.FEATURES + [cfg.TARGET]
    assert not out.isin([np.inf, -np.inf]).any().any()
    assert not out.duplicated().any()
    assert all(out[c].dtype == "float64" for c in cfg.NUMERIC)
    for leaky in cfg.LEAKAGE_COLS + cfg.ARTIFACT_COLS:
        assert leaky not in out.columns


def test_clean_can_keep_duplicates_for_test_split():
    raw = make_synthetic(2000, seed=1)
    assert len(clean(raw, drop_duplicates=False)) == len(raw)


def test_clean_rejects_missing_columns():
    with pytest.raises(ValueError):
        clean(make_synthetic(100).drop(columns=["proto"]))

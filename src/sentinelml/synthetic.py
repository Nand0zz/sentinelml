"""Synthetic UNSW-like data for smoke tests and CI (NOT for reporting results)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sentinelml import config as cfg

_ATTACKS = ["Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance", "Analysis"]


def make_synthetic(n: int = 5000, seed: int = 0, shift: float = 0.0) -> pd.DataFrame:
    """Raw-format frame (incl. id, attack_cat, stcpb, dtcpb, a few inf/NaN/dups).

    `shift` moves feature distributions to emulate drift / the train-test gap.
    """
    param_rng = np.random.default_rng(12345)  # fixed so train and test share structure
    rng = np.random.default_rng(seed)
    label = (rng.random(n) < 0.6).astype(int)

    data = {}
    for col in cfg.NUMERIC:
        mu = param_rng.uniform(0.0, 4.0)
        delta = param_rng.choice([0.0, 0.0, 1.2, -1.0])  # ~half of features carry signal
        data[col] = rng.lognormal(mu + delta * label + shift, 1.0, n)

    def pick(options, p_benign, p_attack):
        return np.where(
            label == 1,
            rng.choice(options, n, p=p_attack),
            rng.choice(options, n, p=p_benign),
        )

    data["proto"] = pick(["tcp", "udp", "icmp", "arp"], [0.5, 0.35, 0.1, 0.05], [0.75, 0.15, 0.08, 0.02])
    data["service"] = pick(["-", "http", "dns", "ftp"], [0.3, 0.3, 0.35, 0.05], [0.6, 0.25, 0.1, 0.05])
    data["state"] = pick(["FIN", "INT", "CON", "REQ"], [0.6, 0.1, 0.25, 0.05], [0.3, 0.45, 0.2, 0.05])
    data["stcpb"] = rng.integers(0, 2**32, n).astype(float)
    data["dtcpb"] = rng.integers(0, 2**32, n).astype(float)

    noisy = label.copy()
    flip = rng.random(n) < 0.03  # irreducible label noise
    noisy[flip] = 1 - noisy[flip]
    data["label"] = noisy
    data["attack_cat"] = np.where(noisy == 1, rng.choice(_ATTACKS, n), "Normal")
    data["id"] = np.arange(1, n + 1)

    df = pd.DataFrame(data)
    k = max(1, n // 200)
    df.loc[rng.choice(n, k, replace=False), "sload"] = np.inf  # dirty values
    df.loc[rng.choice(n, k, replace=False), "dur"] = np.nan
    return pd.concat([df, df.sample(k, random_state=seed)], ignore_index=True)  # duplicates

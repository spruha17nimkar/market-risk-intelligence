"""VaR backtesting: Kupiec unconditional coverage (POF), Christoffersen independence and the Basel
traffic light, run on the full sample and separately within stable and stress regimes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import binom, chi2

from .risk_engine import METHODS, RiskEngineResult


@dataclass
class KupiecResult:
    n_obs: int
    exceptions: int
    expected: float
    exception_rate: float
    lr_stat: float
    p_value: float
    reject: bool


def kupiec_pof(exceptions: int, n_obs: int, p: float, alpha: float = 0.05) -> KupiecResult:
    """Kupiec (1995) proportion-of-failures likelihood-ratio test.

    H0: the true exception probability equals ``p`` (= 1 - VaR confidence). LR ~ chi2(1) under H0.
    """
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    if not 0 <= exceptions <= n_obs:
        raise ValueError("exceptions must lie in [0, n_obs]")
    x, n = exceptions, n_obs
    phat = x / n
    ll_null = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll_alt = xlogy(n - x, 1 - phat) + xlogy(x, phat)
    lr = float(max(-2.0 * (ll_null - ll_alt), 0.0))
    pval = float(chi2.sf(lr, df=1))
    return KupiecResult(n, x, p * n, phat, lr, pval, pval < alpha)


def christoffersen_independence(hits: np.ndarray) -> tuple[float, float]:
    """Christoffersen (1998) test that exceptions do not cluster. Returns (LR, p-value)."""
    h = np.asarray(hits, dtype=int)
    prev, curr = h[:-1], h[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_alt = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = float(max(-2.0 * (ll_null - ll_alt), 0.0))
    return lr, float(chi2.sf(lr, df=1))


def traffic_light(exceptions: int, n_obs: int, p: float = 0.01) -> str:
    """Basel zone from P(X <= exceptions) under Binomial(n, p): green < 95%, yellow < 99.99%, red otherwise.
    For 250 days at 99% this reproduces the familiar 0-4 green / 5-9 yellow / 10+ red bands."""
    cum = binom.cdf(exceptions, n_obs, p)
    if cum < 0.95:
        return "green"
    if cum < 0.9999:
        return "yellow"
    return "red"


def backtest_by_regime(
    risk: RiskEngineResult, stress: pd.Series, alpha: float = 0.05
) -> pd.DataFrame:
    """Kupiec test for every method and VaR level on all days, stable days and stress days."""
    table = risk.table
    regime = stress.reindex(table.index).fillna(0).astype(int)
    masks = {"all": np.ones(len(table), bool), "stable": regime.to_numpy() == 0, "stress": regime.to_numpy() == 1}

    rows = []
    for method in METHODS:
        for level in risk.var_levels:
            hits = (table["loss"] > table[risk.var_col(method, level)]).to_numpy()
            p = 1 - level
            for name, mask in masks.items():
                n = int(mask.sum())
                if n == 0:
                    continue
                res = kupiec_pof(int(hits[mask].sum()), n, p, alpha)
                row = {"method": method, "var_level": level, "regime": name, **asdict(res)}
                row["result"] = "FAIL" if res.reject else "PASS"
                row["exceedance_ratio"] = res.exceptions / res.expected if res.expected else np.nan
                if name == "all":
                    row["christoffersen_p"] = christoffersen_independence(hits)[1]
                    last = hits[-250:]
                    row["traffic_light_last_250"] = traffic_light(int(last.sum()), len(last), p)
                rows.append(row)
    return pd.DataFrame(rows)


def exception_flags(risk: RiskEngineResult, method: str, level: float) -> pd.Series:
    return (risk.table["loss"] > risk.table[risk.var_col(method, level)]).astype(int)

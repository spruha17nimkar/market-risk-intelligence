"""Rolling 252-day risk engine: portfolio volatility, cross-asset covariance, VaR and Expected Shortfall.

Every reading on date t is a forecast built only from returns up to t-1, so it can be backtested
against the loss realised on t without look-ahead.

Three estimators are run side by side:
  * ``hs``    historical simulation over the window (the classic, slow-to-react model)
  * ``param`` variance-covariance (normal) using the window's sample covariance
  * ``ewma``  variance-covariance with a RiskMetrics EWMA covariance (reacts to volatility clustering)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

METHODS = ("hs", "param", "ewma")
TRADING_DAYS = 252


def level_tag(level: float) -> str:
    """0.99 -> '99', 0.975 -> '975'."""
    return f"{level * 100:g}".replace(".", "")


@dataclass
class RiskEngineResult:
    table: pd.DataFrame          # one row per forecast date
    cov: np.ndarray              # (n_dates, k, k) daily window covariance used on each date
    assets: list[str]
    var_levels: list[float]
    es_level: float

    def var_col(self, method: str, level: float) -> str:
        return f"var_{level_tag(level)}_{method}"

    def es_col(self, method: str) -> str:
        return f"es_{level_tag(self.es_level)}_{method}"

    def corr_at(self, i: int) -> np.ndarray:
        c = self.cov[i]
        sd = np.sqrt(np.diag(c))
        return c / np.outer(sd, sd)


def historical_var_es(pnl: np.ndarray, var_level: float, es_level: float) -> tuple[float, float]:
    """VaR and ES as positive loss numbers from a sample of portfolio returns."""
    var = -np.quantile(pnl, 1 - var_level)
    cutoff = np.quantile(pnl, 1 - es_level)
    es = -pnl[pnl <= cutoff].mean()
    return float(var), float(es)


def normal_var(sigma: float, level: float) -> float:
    return float(norm.ppf(level) * sigma)


def normal_es(sigma: float, level: float) -> float:
    return float(sigma * norm.pdf(norm.ppf(level)) / (1 - level))


def run_risk_engine(
    returns: pd.DataFrame,
    weights: np.ndarray | list[float],
    window: int = 252,
    var_levels: list[float] | tuple[float, ...] = (0.99, 0.975),
    es_level: float = 0.975,
    ewma_lambda: float = 0.94,
) -> RiskEngineResult:
    R = returns.to_numpy(dtype=float)
    w = np.asarray(weights, dtype=float)
    T, k = R.shape
    if T <= window:
        raise ValueError(f"need more than {window} return observations, got {T}")
    port = R @ w
    n = T - window
    iu = np.triu_indices(k, 1)

    covs = np.empty((n, k, k))
    cols: dict[str, np.ndarray] = {
        name: np.empty(n) for name in ("port_ret", "vol_ann", "ewma_vol_ann", "avg_corr")
    }
    for m in METHODS:
        for lvl in var_levels:
            cols[f"var_{level_tag(lvl)}_{m}"] = np.empty(n)
        cols[f"es_{level_tag(es_level)}_{m}"] = np.empty(n)

    ewma_cov = None
    for i, t in enumerate(range(window, T)):
        win = R[t - window : t]
        pnl = port[t - window : t]
        cov = np.cov(win, rowvar=False)
        covs[i] = cov
        sd = np.sqrt(np.diag(cov))
        cols["avg_corr"][i] = (cov / np.outer(sd, sd))[iu].mean()

        # Seed EWMA with the first window, then update with each newly observed return.
        if ewma_cov is None:
            ewma_cov = cov.copy()
        else:
            r_prev = R[t - 1]
            ewma_cov = ewma_lambda * ewma_cov + (1 - ewma_lambda) * np.outer(r_prev, r_prev)

        sig_param = float(np.sqrt(w @ cov @ w))
        sig_ewma = float(np.sqrt(w @ ewma_cov @ w))
        cols["port_ret"][i] = port[t]
        cols["vol_ann"][i] = sig_param * np.sqrt(TRADING_DAYS)
        cols["ewma_vol_ann"][i] = sig_ewma * np.sqrt(TRADING_DAYS)

        es_tag = level_tag(es_level)
        for lvl in var_levels:
            tag = level_tag(lvl)
            cols[f"var_{tag}_hs"][i], _ = historical_var_es(pnl, lvl, es_level)
            cols[f"var_{tag}_param"][i] = normal_var(sig_param, lvl)
            cols[f"var_{tag}_ewma"][i] = normal_var(sig_ewma, lvl)
        cols[f"es_{es_tag}_hs"][i] = historical_var_es(pnl, es_level, es_level)[1]
        cols[f"es_{es_tag}_param"][i] = normal_es(sig_param, es_level)
        cols[f"es_{es_tag}_ewma"][i] = normal_es(sig_ewma, es_level)

    table = pd.DataFrame(cols, index=returns.index[window:])
    table.insert(1, "loss", -table["port_ret"])
    return RiskEngineResult(
        table=table, cov=covs, assets=list(returns.columns), var_levels=list(var_levels), es_level=es_level
    )

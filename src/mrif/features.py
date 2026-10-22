"""Stress-regime features. Each value on date t uses only returns observed up to and including t."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig

# feature -> (driver group used for attribution, plain-English definition for the model card)
FEATURES: dict[str, tuple[str, str]] = {
    "vol_level": (
        "volatility_surge",
        "log of annualised portfolio volatility over the last vol_window days",
    ),
    "vol_persistence": (
        "volatility_surge",
        "corr_window-day autocorrelation of |portfolio return| at lag 1 (volatility clustering)",
    ),
    "drawdown": (
        "drawdown",
        "portfolio wealth relative to its running drawdown_window-day peak, minus 1 (<= 0)",
    ),
    "avg_corr": (
        "correlation_spike",
        "average pairwise correlation across the asset classes over the last corr_window days",
    ),
    "duration_corr": (
        "duration_sensitivity",
        "corr_window-day correlation between the duration_pair returns (equity-rates co-movement)",
    ),
}
FEATURE_NAMES = list(FEATURES)
DRIVER_GROUPS = list(dict.fromkeys(g for g, _ in FEATURES.values()))


def rolling_avg_pairwise_corr(returns: pd.DataFrame, window: int, absolute: bool = False) -> pd.Series:
    R = returns.to_numpy(dtype=float)
    T, k = R.shape
    iu = np.triu_indices(k, 1)
    out = np.full(T, np.nan)
    for t in range(window - 1, T):
        pairs = np.corrcoef(R[t - window + 1 : t + 1], rowvar=False)[iu]
        out[t] = (np.abs(pairs) if absolute else pairs).mean()
    return pd.Series(out, index=returns.index, name="avg_corr")


def correlation_diagnostics(returns: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Reported alongside the model features but not used by the classifier.

    The signed five-asset average nets hedges (rates, USD) against risk assets, so two complementary views
    are kept: the average absolute correlation, and the average correlation among the risk assets only.
    """
    w = cfg.corr_window
    return pd.DataFrame(
        {
            "avg_corr_all": rolling_avg_pairwise_corr(returns, w),
            "avg_abs_corr_all": rolling_avg_pairwise_corr(returns, w, absolute=True),
            "avg_corr_risk_assets": rolling_avg_pairwise_corr(returns[cfg.risk_assets], w),
        }
    ).dropna()


def build_features(returns: pd.DataFrame, weights: np.ndarray | list[float], cfg: FeatureConfig) -> pd.DataFrame:
    port = pd.Series(returns.to_numpy() @ np.asarray(weights, dtype=float), index=returns.index)
    abs_ret = port.abs()
    wealth = (1 + port).cumprod()
    a, b = cfg.duration_pair

    feats = pd.DataFrame(
        {
            "vol_level": np.log(port.rolling(cfg.vol_window).std() * np.sqrt(252)),
            "vol_persistence": abs_ret.rolling(cfg.corr_window).corr(abs_ret.shift(1)),
            "drawdown": wealth / wealth.rolling(cfg.drawdown_window, min_periods=1).max() - 1,
            "avg_corr": rolling_avg_pairwise_corr(returns, cfg.corr_window),
            "duration_corr": returns[a].rolling(cfg.corr_window).corr(returns[b]),
        },
        index=returns.index,
    )
    # Drawdown needs a full lookback to be comparable across dates.
    feats.iloc[: cfg.drawdown_window - 1] = np.nan
    return feats[FEATURE_NAMES].dropna()

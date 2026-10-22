"""Market data: Yahoo Finance download with an on-disk cache, and a synthetic regime-switching generator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config


def load_prices(cfg: Config, refresh: bool = False) -> tuple[pd.DataFrame, Path]:
    """Return adjusted close prices with columns named by asset class, and the cache file used.

    Rows where any proxy is missing are dropped so every date has all five risk factors.
    """
    tickers = list(cfg.assets.values())
    key = json.dumps({"t": sorted(tickers), "s": cfg.data.start, "e": cfg.data.end})
    cache = Path(cfg.data.cache_dir) / f"prices_{hashlib.sha256(key.encode()).hexdigest()[:10]}.csv"

    if cache.exists() and not refresh:
        raw = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        raw = download_yahoo(tickers, cfg.data.start, cfg.data.end)
        cache.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(cache)

    by_class = {ticker: asset for asset, ticker in cfg.assets.items()}
    prices = raw[tickers].rename(columns=by_class)[cfg.asset_classes].dropna()
    prices.index.name = "date"
    return prices, cache


def download_yahoo(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {tickers}")
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    missing = [t for t in tickers if t not in close.columns]
    if missing:
        raise RuntimeError(f"Yahoo Finance returned no prices for {missing}")
    return close[tickers]


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns."""
    return prices.pct_change().dropna(how="any")


def synthetic_prices(
    asset_classes: list[str],
    n_days: int = 3500,
    seed: int = 7,
    p_enter_stress: float = 0.008,
    p_exit_stress: float = 0.03,
) -> tuple[pd.DataFrame, pd.Series]:
    """Two-state Markov regime-switching returns: calm (low vol, low correlation) vs stress
    (vol x2.5, correlations converge to 0.75). Returns prices and the true stress regime (1 = stress).

    Used for offline demos and tests where the true regime is known.
    """
    rng = np.random.default_rng(seed)
    k = len(asset_classes)
    calm_vol = np.resize(np.array([0.16, 0.12, 0.07, 0.07, 0.20]), k) / np.sqrt(252)
    stress_vol = calm_vol * 2.5
    calm_corr = np.full((k, k), 0.10) + 0.90 * np.eye(k)
    stress_corr = np.full((k, k), 0.75) + 0.25 * np.eye(k)
    calm_cov = np.outer(calm_vol, calm_vol) * calm_corr
    stress_cov = np.outer(stress_vol, stress_vol) * stress_corr
    drift = np.full(k, 0.04 / 252)

    regime = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        if regime[t - 1] == 0:
            regime[t] = int(rng.random() < p_enter_stress)
        else:
            regime[t] = int(rng.random() >= p_exit_stress)

    rets = np.where(
        regime[:, None] == 1,
        rng.multivariate_normal(-drift, stress_cov, n_days),
        rng.multivariate_normal(drift, calm_cov, n_days),
    )
    dates = pd.bdate_range("2010-01-01", periods=n_days + 1, name="date")
    prices = pd.DataFrame(
        100 * np.vstack([np.ones(k), np.cumprod(1 + rets, axis=0)]), index=dates, columns=asset_classes
    )
    # regime[t] drives the return realised on dates[t + 1]
    labels = pd.Series(regime, index=dates[1:], name="stress")
    return prices, labels

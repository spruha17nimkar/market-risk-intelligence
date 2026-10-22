import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from mrif.risk_engine import historical_var_es, normal_es, normal_var, run_risk_engine


@pytest.fixture
def returns():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=400)
    return pd.DataFrame(rng.normal(0, 0.01, (400, 3)), index=idx, columns=["a", "b", "c"])


def test_normal_var_and_es_closed_form():
    assert normal_var(0.01, 0.99) == pytest.approx(0.01 * norm.ppf(0.99))
    # 97.5% normal ES is about 2.338 sigma
    assert normal_es(1.0, 0.975) == pytest.approx(2.3378, abs=1e-3)


def test_historical_es_is_at_least_var():
    pnl = np.random.default_rng(1).standard_t(4, 5000) * 0.01
    var, es = historical_var_es(pnl, 0.975, 0.975)
    assert es >= var > 0


def test_engine_uses_only_past_data(returns):
    w = np.array([0.5, 0.3, 0.2])
    res = run_risk_engine(returns, w, window=252)
    assert res.table.index[0] == returns.index[252]
    # Reading on day t is computed from rows t-252 .. t-1, never from day t itself.
    t = 300
    win = returns.iloc[t - 252 : t].to_numpy()
    sigma = np.sqrt(w @ np.cov(win, rowvar=False) @ w)
    row = res.table.loc[returns.index[t]]
    assert row["var_99_param"] == pytest.approx(norm.ppf(0.99) * sigma)
    assert row["port_ret"] == pytest.approx(returns.iloc[t].to_numpy() @ w)

    shocked = returns.copy()
    shocked.iloc[t] = -0.5
    assert run_risk_engine(shocked, w, window=252).table.loc[returns.index[t], "var_99_hs"] == pytest.approx(row["var_99_hs"])


def test_engine_shapes_and_correlation(returns):
    res = run_risk_engine(returns, [1 / 3] * 3, window=252)
    assert res.cov.shape == (len(returns) - 252, 3, 3)
    corr = res.corr_at(0)
    assert np.allclose(np.diag(corr), 1)
    assert abs(res.table["avg_corr"].iloc[0]) < 0.2  # independent draws


def test_ewma_reacts_faster_than_window(returns):
    shocked = returns.copy()
    shocked.iloc[300:320] *= 6
    res = run_risk_engine(shocked, [1 / 3] * 3, window=252).table
    day = shocked.index[320]
    assert res.loc[day, "var_99_ewma"] > res.loc[day, "var_99_param"]

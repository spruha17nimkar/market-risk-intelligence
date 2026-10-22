import numpy as np
import pytest

from mrif.backtest import christoffersen_independence, kupiec_pof, traffic_light


def test_kupiec_exact_expected_count_has_zero_lr():
    res = kupiec_pof(exceptions=10, n_obs=1000, p=0.01)
    assert res.lr_stat == pytest.approx(0.0, abs=1e-10)
    assert res.p_value == pytest.approx(1.0)
    assert not res.reject


def test_kupiec_matches_hand_computation():
    # 20 exceptions in 1000 days at p = 1%
    x, n, p = 20, 1000, 0.01
    phat = x / n
    lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - phat) - x * np.log(phat))
    res = kupiec_pof(x, n, p)
    assert res.lr_stat == pytest.approx(lr)
    assert res.reject


def test_kupiec_handles_zero_exceptions():
    res = kupiec_pof(0, 500, 0.01)
    assert np.isfinite(res.lr_stat)
    assert res.lr_stat == pytest.approx(-2 * 500 * np.log(0.99))


def test_kupiec_rejects_too_few_exceptions_too():
    assert kupiec_pof(0, 2000, 0.01).reject


@pytest.mark.parametrize("x,zone", [(0, "green"), (4, "green"), (5, "yellow"), (9, "yellow"), (10, "red")])
def test_basel_traffic_light_bands(x, zone):
    assert traffic_light(x, 250, 0.01) == zone


def test_christoffersen_detects_clustering():
    clustered = np.zeros(1000, int)
    clustered[100:110] = 1
    spread = np.zeros(1000, int)
    spread[::100] = 1
    assert christoffersen_independence(clustered)[1] < 0.01
    assert christoffersen_independence(spread)[1] > 0.05

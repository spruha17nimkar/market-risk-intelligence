"""End-to-end tests on the synthetic regime-switching data set (no network needed)."""

import numpy as np
import pandas as pd
import pytest

from mrif.config import Config
from mrif.explain import attribute
from mrif.features import FEATURE_NAMES, build_features
from mrif.governance import read_run, trace
from mrif.pipeline import run_pipeline
from mrif.regimes import episode_labels


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    cfg = Config.default()
    cfg.governance.runs_dir = str(tmp_path_factory.mktemp("runs"))
    cfg.classifier.first_oos_year = 2014
    return run_pipeline(cfg, synthetic=True)


def test_config_validation():
    raw = Config.default().to_dict()
    raw["weights"]["equities"] = 0.9
    with pytest.raises(ValueError, match="sum to 1"):
        Config.from_dict(raw)


def test_episode_labels_are_inclusive():
    idx = pd.bdate_range("2020-01-01", "2020-01-31")
    lab = episode_labels(idx, [["2020-01-06", "2020-01-10", "x"]])
    assert lab.sum() == 5
    assert lab.loc["2020-01-06"] == 1 and lab.loc["2020-01-10"] == 1


def test_features_have_no_lookahead(result):
    r = result.returns
    w = Config.default().weight_vector()
    cfg = Config.default().features
    full = build_features(r, w, cfg)
    cut = r.index[1500]
    trunc = build_features(r.loc[:cut], w, cfg)
    pd.testing.assert_series_equal(full.loc[cut], trunc.loc[cut], check_exact=False, rtol=1e-9)


def test_classifier_separates_known_regimes(result):
    # On synthetic data the true regime is known, so the out-of-sample AUC should be high.
    assert result.classifier.metrics["oos_roc_auc"] > 0.9


def test_attribution_sums_to_logit(result):
    att = attribute(result.features, result.classifier)
    contrib = att[[f"c_{f}" for f in FEATURE_NAMES]].sum(axis=1) + att["base_logit"]
    prob = 1 / (1 + np.exp(-contrib))
    np.testing.assert_allclose(prob, result.classifier.predictions["prob"].loc[att.index], atol=1e-10)


def test_walk_forward_models_never_see_the_future(result):
    for mid, m in result.classifier.models.items():
        if mid.startswith("wf-"):
            assert pd.Timestamp(m.train_end) < pd.Timestamp(f"{mid[3:]}-01-01")


def test_stress_correlation_exceeds_stable_in_synthetic(result):
    m = result.findings["correlation"]["measures"]["avg_corr_all"]
    assert m["mean_stress"] > m["mean_stable"] + 0.3


def test_backtest_has_all_regimes(result):
    bt = result.backtest
    assert set(bt["regime"]) == {"all", "stable", "stress"}
    for (method, level), g in bt.groupby(["method", "var_level"]):
        parts = g.set_index("regime")
        assert parts.loc["stable", "n_obs"] + parts.loc["stress", "n_obs"] == parts.loc["all", "n_obs"]
        assert parts.loc["stable", "exceptions"] + parts.loc["stress", "exceptions"] == parts.loc["all", "exceptions"]


def test_run_is_stored_and_traceable(result):
    run = read_run(result.run_dir)
    assert run["manifest"]["config"]["risk"]["window"] == 252
    for date in [run["stress"].index[10], run["stress"].index[-1]]:
        report = trace(result.run_dir, str(date.date()))
        assert all(report["checks"].values()), report["checks"]


def test_trace_detects_tampered_inputs(result, tmp_path):
    import shutil

    copy = tmp_path / "tampered"
    shutil.copytree(result.run_dir, copy)
    returns = pd.read_csv(copy / "returns.csv", index_col=0, parse_dates=True)
    day = pd.read_csv(copy / "stress.csv", index_col=0, parse_dates=True).index[-1]
    returns.loc[day, returns.columns[0]] += 0.01
    returns.to_csv(copy / "returns.csv")
    report = trace(copy, str(day.date()))
    assert not report["checks"]["input_hash_matches"]

"""End-to-end pipeline: data -> rolling risk engine -> regime backtest -> stress classifier -> attribution
-> governance store and findings report."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .backtest import backtest_by_regime
from .classifier import ClassifierResult, walk_forward
from .config import Config
from .data import load_prices, synthetic_prices, to_returns
from .explain import attribute, coefficient_table
from .features import FEATURES, build_features, correlation_diagnostics
from .governance import RunStore, environment, git_revision, sha256_array, sha256_file, window_hashes
from .regimes import episode_labels, episode_name_series
from .risk_engine import RiskEngineResult, normal_es, run_risk_engine

CORR_BREAKDOWN = 0.70


@dataclass
class PipelineResult:
    run_dir: Path | None
    returns: pd.DataFrame
    risk: RiskEngineResult
    labels: pd.Series
    features: pd.DataFrame
    classifier: ClassifierResult
    attribution: pd.DataFrame
    backtest: pd.DataFrame
    findings: dict


def correlation_findings(returns: pd.DataFrame, diagnostics: pd.DataFrame, labels: pd.Series, weights: np.ndarray, es_level: float) -> dict:
    """How far diversification breaks down in stress, and what that does to capital."""
    y = labels.reindex(diagnostics.index).fillna(0).astype(int)
    measures = {}
    for col in diagnostics.columns:
        corr = diagnostics[col]
        stress, stable = corr[y == 1], corr[y == 0]
        measures[col] = {
            "mean_stable": float(stable.mean()),
            "mean_stress": float(stress.mean()),
            "p90_stress": float(stress.quantile(0.9)),
            "max": float(corr.max()),
            "max_date": str(corr.idxmax().date()),
            "share_stress_days_above_0_70": float((stress > CORR_BREAKDOWN).mean()),
            "share_stable_days_above_0_70": float((stable > CORR_BREAKDOWN).mean()),
        }

    # Same volatilities, different correlation assumptions: how much ES moves when stress correlations apply.
    r = returns.reindex(diagnostics.index)
    sd = r.std().to_numpy()
    es = {}
    for name, mask in {"stable": y == 0, "stress": y == 1, "full_sample": y >= 0}.items():
        c = r[mask].corr().to_numpy()
        es[name] = normal_es(float(np.sqrt(weights @ (np.outer(sd, sd) * c) @ weights)), es_level)

    return {
        "window_days": None,  # filled by caller
        "measures": measures,
        "stress_correlation_matrix": r[y == 1].corr().round(4).to_dict(),
        "stable_correlation_matrix": r[y == 0].corr().round(4).to_dict(),
        "es_with_stable_correlations": es["stable"],
        "es_with_full_sample_correlations": es["full_sample"],
        "es_with_stress_correlations": es["stress"],
        "es_uplift_stress_vs_full_sample": es["stress"] / es["full_sample"] - 1,
    }


def episode_table(diagnostics: pd.DataFrame, risk: RiskEngineResult, preds: pd.DataFrame, episodes: list[list[str]]) -> pd.DataFrame:
    rows = []
    for start, end, *rest in episodes:
        sl = slice(pd.Timestamp(start), pd.Timestamp(end))
        d, rk, p = diagnostics.loc[sl], risk.table.loc[sl], preds.loc[sl]
        if d.empty:
            continue
        rows.append(
            {
                "episode": rest[0] if rest else f"{start}..{end}",
                "start": start,
                "end": end,
                "days": len(d),
                "mean_avg_corr": d["avg_corr_all"].mean(),
                "max_risk_asset_corr": d["avg_corr_risk_assets"].max(),
                "mean_stress_prob": p["prob"].mean() if not p.empty else np.nan,
                "hs_99_exceptions": int((rk["loss"] > rk["var_99_hs"]).sum()) if "var_99_hs" in rk else None,
                "ewma_99_exceptions": int((rk["loss"] > rk["var_99_ewma"]).sum()) if "var_99_ewma" in rk else None,
            }
        )
    return pd.DataFrame(rows)


def run_pipeline(cfg: Config, synthetic: bool = False, refresh: bool = False, persist: bool = True) -> PipelineResult:
    synthetic = synthetic or cfg.data.source == "synthetic"
    weights = np.asarray(cfg.weight_vector())

    if synthetic:
        prices, true_regime = synthetic_prices(cfg.asset_classes)
        data_file, data_info = None, {"source": "synthetic", "generator": "mrif.data.synthetic_prices(seed=7)"}
    else:
        prices, data_file = load_prices(cfg, refresh=refresh)
        data_info = {"source": "yahoo", "tickers": cfg.assets, "file": str(data_file), "file_sha256": sha256_file(data_file)}

    returns = to_returns(prices)
    labels = true_regime.reindex(returns.index).fillna(0).astype(int) if synthetic else episode_labels(returns.index, cfg.regime.episodes)

    risk = run_risk_engine(returns, weights, cfg.risk.window, cfg.risk.var_levels, cfg.risk.es_level, cfg.risk.ewma_lambda)
    backtest = backtest_by_regime(risk, labels)

    features = build_features(returns, weights, cfg.features).loc[lambda d: d.index >= risk.table.index[0]]
    diagnostics = correlation_diagnostics(returns, cfg.features).reindex(features.index)
    clf = walk_forward(features, labels, cfg.classifier)
    attribution = attribute(features, clf)

    lookback = cfg.risk.window
    hashes = window_hashes(returns, lookback)
    stress = clf.predictions.join(attribution).join(hashes)
    stress["episode"] = "" if synthetic else episode_name_series(stress.index, cfg.regime.episodes)

    findings = {
        "classifier": clf.metrics,
        "correlation": correlation_findings(returns, diagnostics, labels, weights, cfg.risk.es_level),
        "backtest_summary": _backtest_summary(backtest),
        "sample": {
            "returns_start": str(returns.index[0].date()),
            "returns_end": str(returns.index[-1].date()),
            "risk_start": str(risk.table.index[0].date()),
            "n_risk_days": len(risk.table),
            "n_stress_days": int(labels.reindex(risk.table.index).sum()),
        },
    }
    findings["correlation"]["window_days"] = cfg.features.corr_window
    episodes = pd.DataFrame() if synthetic else episode_table(diagnostics, risk, clf.predictions, cfg.regime.episodes)

    run_dir = None
    if persist:
        store = RunStore.create(cfg.governance.runs_dir, cfg.to_dict())
        run_dir = store.path
        manifest = {
            "run_id": run_dir.name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "package_version": __version__,
            "code": git_revision(),
            "environment": environment(),
            "config": cfg.to_dict(),
            "data": {**data_info, "returns_sha256": sha256_array(returns.to_numpy()), "n_returns": len(returns)},
            "input_hash_lookback": lookback,
            "outputs": ["returns.csv", "risk.csv", "covariances.npy", "features.csv", "diagnostics.csv", "stress.csv",
                        "backtest.csv", "episodes.csv", "model_card.json", "findings.json", "report.md"],
        }
        store.write_json("manifest.json", manifest)
        store.write_frame("returns.csv", returns)
        store.write_frame("risk.csv", risk.table)
        store.write_array("covariances.npy", risk.cov)
        store.write_frame("features.csv", features)
        store.write_frame("diagnostics.csv", diagnostics)
        store.write_frame("stress.csv", stress)
        backtest.to_csv(run_dir / "backtest.csv", index=False)
        episodes.to_csv(run_dir / "episodes.csv", index=False)
        store.write_json("model_card.json", model_card(cfg, clf, synthetic))
        store.write_json("findings.json", findings)
        store.write_text("report.md", render_report(manifest, findings, backtest, coefficient_table(clf), episodes))

    return PipelineResult(run_dir, returns, risk, labels, features, clf, attribution, backtest, findings)


def _backtest_summary(bt: pd.DataFrame) -> list[dict]:
    keep = ["method", "var_level", "regime", "n_obs", "exceptions", "expected", "p_value", "result"]
    return bt[keep].to_dict(orient="records")


def model_card(cfg: Config, clf: ClassifierResult, synthetic: bool) -> dict:
    return {
        "model": "Stress regime classifier (logistic regression on standardised features)",
        "purpose": "Estimate the probability that cross-asset markets are in a stress regime, as an early-warning "
                   "input to VaR/ES model monitoring and capital buffer discussions.",
        "intended_use": "Monitoring and challenge of the VaR/ES model. Not a trading signal and not a capital model.",
        "ground_truth": "true simulated regime" if synthetic else "documented stress episodes (see config.regime.episodes)",
        "episodes": cfg.regime.episodes,
        "features": {f: {"driver_group": g, "definition": d} for f, (g, d) in FEATURES.items()},
        "hyperparameters": {"C": cfg.classifier.C, "class_weight": cfg.classifier.class_weight,
                            "threshold": cfg.classifier.threshold, "first_oos_year": cfg.classifier.first_oos_year},
        "validation": "Walk-forward: model wf-YYYY is trained on all days before 1 Jan YYYY and scores only YYYY.",
        "metrics": clf.metrics,
        "limitations": [
            "Stress labels are a judgemental list of episodes; changing the dates changes the target.",
            "Features are contemporaneous: the model nowcasts the regime, it does not forecast it.",
            "Five ETF proxies stand in for the risk factors; they carry fees, tracking error and, for HYG, rate duration.",
            "Logistic regression assumes the log-odds are linear in the standardised features.",
        ],
        "models": {mid: m.to_dict() for mid, m in clf.models.items()},
    }


def _fmt_pct(x: float) -> str:
    return f"{x:.1%}"


def render_report(manifest: dict, findings: dict, bt: pd.DataFrame, coefs: pd.DataFrame, episodes: pd.DataFrame) -> str:
    c, k, s = findings["correlation"], findings["classifier"], findings["sample"]
    lines = [
        f"# Run report `{manifest['run_id']}`",
        "",
        f"Data: {manifest['data']['source']} | returns {s['returns_start']} to {s['returns_end']} | "
        f"risk readings from {s['risk_start']} ({s['n_risk_days']} days, {s['n_stress_days']} labelled stress)",
        "",
        "## Stress classifier",
        f"- Out-of-sample ROC-AUC (walk-forward {k['oos_start']} to {k['oos_end']}): **{k['oos_roc_auc']:.3f}**",
        f"- In-sample ROC-AUC (full-sample model): {k['in_sample_roc_auc_full_model']:.3f}",
        f"- Precision / recall at p >= {k['threshold']}: {k['oos_precision']:.2f} / {k['oos_recall']:.2f}; Brier {k['oos_brier']:.3f}",
        "",
        "| feature | driver | std coef | odds ratio / 1 sd |",
        "|---|---|---:|---:|",
        *[f"| {r.feature} | {r.driver_group} | {r.std_coef:+.3f} | {r.odds_ratio_per_1sd:.2f} |" for r in coefs.itertuples()],
        "",
        f"## Correlation convergence ({c['window_days']}-day rolling average pairwise correlation)",
        "| measure | mean stable | mean stress | p90 stress | max (date) | stress days > 0.70 |",
        "|---|---:|---:|---:|---:|---:|",
        *[f"| {name} | {m['mean_stable']:.2f} | {m['mean_stress']:.2f} | {m['p90_stress']:.2f} | {m['max']:.2f} ({m['max_date']}) | "
          f"{_fmt_pct(m['share_stress_days_above_0_70'])} |" for name, m in c["measures"].items()],
        "",
        f"- Parametric 97.5% ES with full-sample correlations {c['es_with_full_sample_correlations']:.4%} vs stress correlations "
        f"{c['es_with_stress_correlations']:.4%} (change {c['es_uplift_stress_vs_full_sample']:+.1%}), same volatilities",
        "",
        "## Kupiec POF backtest (99% VaR)",
        "| method | regime | days | exceptions | expected | p-value | result |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in bt[bt["var_level"] == 0.99].itertuples():
        lines.append(f"| {r.method} | {r.regime} | {r.n_obs} | {r.exceptions} | {r.expected:.1f} | {r.p_value:.4f} | {r.result} |")
    if not episodes.empty:
        lines += ["", "## Stress episodes",
                  "| episode | days | mean corr (all 5) | max corr (risk assets) | mean stress prob | HS 99% exc. | EWMA 99% exc. |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for r in episodes.itertuples():
            lines.append(f"| {r.episode} | {r.days} | {r.mean_avg_corr:.2f} | {r.max_risk_asset_corr:.2f} | {r.mean_stress_prob:.2f} | "
                         f"{r.hs_99_exceptions} | {r.ewma_99_exceptions} |")
    return "\n".join(lines) + "\n"

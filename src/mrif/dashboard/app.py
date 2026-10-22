"""Streamlit dashboard over a stored pipeline run. Launch with ``mrif dashboard``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mrif.features import DRIVER_GROUPS
from mrif.governance import RunStore, read_run, trace

STRESS_FILL = "rgba(220, 53, 69, 0.10)"
DRIVER_COLORS = {
    "volatility_surge": "#d62728",
    "drawdown": "#9467bd",
    "correlation_spike": "#ff7f0e",
    "duration_sensitivity": "#1f77b4",
}

st.set_page_config(page_title="Market Risk Intelligence", layout="wide")


def _runs_dir() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    return parser.parse_known_args(sys.argv[1:])[0].runs_dir


@st.cache_data(show_spinner="Loading run...")
def load(run_dir: str) -> dict:
    return read_run(run_dir)


def stress_spans(labels: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    lab = labels.fillna(0).astype(int)
    starts = lab.index[(lab == 1) & (lab.shift(1, fill_value=0) == 0)]
    ends = lab.index[(lab == 1) & (lab.shift(-1, fill_value=0) == 0)]
    return list(zip(starts, ends))


def shade(fig: go.Figure, spans) -> go.Figure:
    for s, e in spans:
        fig.add_vrect(x0=s, x1=e, fillcolor=STRESS_FILL, line_width=0, layer="below")
    return fig


def layout(fig: go.Figure, title: str, height: int = 380, **kw) -> go.Figure:
    kw.setdefault("hovermode", "x unified")
    fig.update_layout(title=dict(text=title, y=0.98, yanchor="top"), height=height, margin=dict(l=10, r=10, t=80, b=10),
                      legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"), **kw)
    return fig


runs_dir = _runs_dir()
available = sorted((p for p in Path(runs_dir).glob("*") if (p / "manifest.json").exists()), reverse=True)
if not available:
    st.error(f"No runs found in `{runs_dir}`. Run `mrif run` first.")
    st.stop()

with st.sidebar:
    st.header("Run")
    run_path = st.selectbox("Stored run", available, format_func=lambda p: p.name)
    run = load(str(run_path))
    risk, stress, diag, feats = run["risk"], run["stress"], run["diagnostics"], run["features"]
    lo, hi = risk.index.min().date(), risk.index.max().date()
    start, end = st.slider("Date range", min_value=lo, max_value=hi, value=(lo, hi), format="YYYY-MM")
    method = st.radio("VaR / ES model", ["hs", "param", "ewma"],
                      format_func={"hs": "Historical simulation", "param": "Parametric (normal)", "ewma": "EWMA (RiskMetrics)"}.get)
    st.caption("Shaded bands mark labelled stress episodes.")

sl = slice(pd.Timestamp(start), pd.Timestamp(end))
spans = stress_spans(stress["label"].loc[sl])
findings, card, manifest = run["findings"], run["model_card"], run["manifest"]

st.title("Cross-Asset Market Risk Intelligence")
st.caption(f"Run `{manifest['run_id']}` | data: {manifest['data']['source']} | "
           f"code {str(manifest['code'].get('commit'))[:8]}{' (dirty)' if manifest['code'].get('dirty') else ''}")

last = risk.index[-1]
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(f"99% VaR ({method})", f"{risk.loc[last, f'var_99_{method}']:.2%}")
k2.metric(f"97.5% ES ({method})", f"{risk.loc[last, f'es_975_{method}']:.2%}")
k3.metric("Stress probability", f"{stress['prob'].iloc[-1]:.0%}", help=f"top driver: {stress['top_driver'].iloc[-1]}")
k4.metric("Avg corr, risk assets (63d)", f"{diag['avg_corr_risk_assets'].iloc[-1]:.2f}")
k5.metric("Classifier OOS ROC-AUC", f"{findings['classifier']['oos_roc_auc']:.3f}")
st.caption(f"Latest reading: {last.date()}")

tab_risk, tab_corr, tab_stress, tab_gov = st.tabs(["VaR & ES", "Correlations", "Stress monitor", "Model governance"])

with tab_risk:
    r = risk.loc[sl]
    var_col, es_col = f"var_99_{method}", f"es_975_{method}"
    exc = r[r["loss"] > r[var_col]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=r.index, y=r["loss"], name="Realised loss", marker_color="rgba(120,120,120,0.45)"))
    fig.add_trace(go.Scatter(x=r.index, y=r[var_col], name="99% VaR", line=dict(color="#1f77b4", width=1.6)))
    fig.add_trace(go.Scatter(x=r.index, y=r[es_col], name="97.5% ES", line=dict(color="#d62728", width=1.2, dash="dot")))
    fig.add_trace(go.Scatter(x=exc.index, y=exc["loss"], mode="markers", name=f"VaR exceptions ({len(exc)})",
                             marker=dict(color="#d62728", size=6, symbol="x")))
    st.plotly_chart(shade(layout(fig, "Daily loss vs one-day 99% VaR and 97.5% ES", 460, yaxis_tickformat=".1%"), spans),
                    width="stretch")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r.index, y=r["vol_ann"], name="252-day window vol"))
    fig.add_trace(go.Scatter(x=r.index, y=r["ewma_vol_ann"], name="EWMA vol (lambda 0.94)"))
    st.plotly_chart(shade(layout(fig, "Annualised portfolio volatility", yaxis_tickformat=".0%"), spans), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for m, name in [("hs", "HS"), ("param", "Param"), ("ewma", "EWMA")]:
            fig.add_trace(go.Scatter(x=r.index, y=r[f"var_99_{m}"], name=name))
        st.plotly_chart(shade(layout(fig, "99% VaR by model", yaxis_tickformat=".1%"), spans), width="stretch")
    with c2:
        fig = go.Figure(go.Scatter(x=r.index, y=r[es_col] / r[var_col], name="ES / VaR"))
        st.plotly_chart(shade(layout(fig, "Tail severity: 97.5% ES / 99% VaR"), spans), width="stretch")

with tab_corr:
    assets = list(run["returns"].columns)
    dates = risk.index
    pick = st.select_slider("Correlation matrix date (252-day window)", options=list(dates.date), value=dates[-1].date())
    i = dates.get_loc(pd.Timestamp(pick))
    cov = run["cov"][i]
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)

    def heat(m: np.ndarray, title: str) -> go.Figure:
        fig = go.Figure(go.Heatmap(z=m, x=assets, y=assets, zmin=-1, zmax=1, colorscale="RdBu_r",
                                   text=np.round(m, 2), texttemplate="%{text}", hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>"))
        return layout(fig, title, 380, yaxis_autorange="reversed", hovermode="closest")

    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(heat(corr, f"Rolling 252-day correlation, {pick}"), width="stretch")
    cm = findings["correlation"]
    to_mat = lambda d: pd.DataFrame(d).loc[assets, assets].to_numpy()  # noqa: E731
    c2.plotly_chart(heat(to_mat(cm["stable_correlation_matrix"]), "Pooled correlation: stable days"), width="stretch")
    c3.plotly_chart(heat(to_mat(cm["stress_correlation_matrix"]), "Pooled correlation: stress days"), width="stretch")

    d = diag.loc[sl]
    fig = go.Figure()
    names = {"avg_corr_all": "All 5 asset classes (signed)", "avg_abs_corr_all": "All 5 (absolute)",
             "avg_corr_risk_assets": "Risk assets: equities, credit, commodities"}
    for col, name in names.items():
        fig.add_trace(go.Scatter(x=d.index, y=d[col], name=name))
    fig.add_hline(y=0.70, line_dash="dash", line_color="#d62728", annotation_text="0.70")
    st.plotly_chart(shade(layout(fig, "63-day average pairwise correlation", 420), spans), width="stretch")
    st.caption("Treasuries and the US dollar usually rally in sell-offs, which pulls the signed five-asset average down. "
               "Convergence shows up most clearly among the risk assets.")

with tab_stress:
    s = stress.loc[sl]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s["prob"], name="Stress probability", line=dict(color="#d62728")))
    ins = s[s["in_sample"].astype(bool)]
    if len(ins):
        fig.add_trace(go.Scatter(x=ins.index, y=ins["prob"], name="in-sample (pre walk-forward)", line=dict(color="#aaaaaa")))
    fig.add_hline(y=findings["classifier"]["threshold"], line_dash="dot")
    st.plotly_chart(shade(layout(fig, "Stress regime probability (walk-forward, out of sample)", 380, yaxis_range=[0, 1]), spans),
                    width="stretch")

    monthly = s[[f"g_{g}" for g in DRIVER_GROUPS]].resample("ME").mean()
    fig = go.Figure()
    for g in DRIVER_GROUPS:
        fig.add_trace(go.Bar(x=monthly.index, y=monthly[f"g_{g}"], name=g.replace("_", " "), marker_color=DRIVER_COLORS[g]))
    st.plotly_chart(shade(layout(fig, "What drives the reading: monthly average log-odds contribution by driver", 380, barmode="relative"), spans),
                    width="stretch")

    st.subheader("Explain a single reading")
    day = st.select_slider("Reading date", options=list(s.index.date), value=s.index[-1].date(), key="explain_day")
    row = stress.loc[pd.Timestamp(day)]
    contrib = {f.removeprefix("c_"): row[f] for f in stress.columns if f.startswith("c_")}
    fig = go.Figure(go.Waterfall(
        x=["baseline", *contrib], y=[row["base_logit"], *contrib.values()],
        measure=["absolute"] + ["relative"] * len(contrib), connector=dict(line=dict(color="#888")),
    ))
    st.plotly_chart(layout(fig, f"{day}: log-odds build-up -> p(stress) = {row['prob']:.1%} (model {row['model_id']})", 380, hovermode="closest"),
                    width="stretch")
    st.dataframe(feats.loc[[pd.Timestamp(day)]].T.rename(columns=lambda c: "feature value"), width="stretch")

with tab_gov:
    st.subheader("Kupiec POF backtest by regime")
    bt = run["backtest"]
    lvl = st.radio("VaR level", sorted(bt["var_level"].unique(), reverse=True), horizontal=True, format_func=lambda v: f"{v:.1%}")
    show = bt[bt["var_level"] == lvl][["method", "regime", "n_obs", "exceptions", "expected", "exception_rate", "lr_stat", "p_value", "result"]]
    st.dataframe(show.style.format({"expected": "{:.1f}", "exception_rate": "{:.2%}", "lr_stat": "{:.2f}", "p_value": "{:.4f}"})
                 .map(lambda v: "color: #d62728; font-weight: 600" if v == "FAIL" else "color: #2ca02c", subset=["result"]),
                 width="stretch", hide_index=True)

    if not run["episodes"].empty:
        st.subheader("Stress episodes")
        st.dataframe(run["episodes"], width="stretch", hide_index=True)

    st.subheader("Classifier parameters")
    models = card["models"]
    mid = st.selectbox("Model version", list(models), index=len(models) - 1)
    m = models[mid]
    st.caption(f"Trained {m['train_start']} to {m['train_end']} on {m['n_train']} days ({m['n_stress']} stress). Intercept {m['intercept']:+.3f}")
    st.dataframe(pd.DataFrame({"feature": m["features"], "std coef": m["coef"], "odds ratio / 1 sd": np.exp(m["coef"]),
                               "train mean": m["mean"], "train sd": m["scale"],
                               "definition": [card["features"][f]["definition"] for f in m["features"]]}),
                 width="stretch", hide_index=True)
    with st.expander("Model card"):
        st.json({k: v for k, v in card.items() if k != "models"})
    with st.expander("Run manifest"):
        st.json(manifest)

    st.subheader("Audit trace")
    tday = st.date_input("Trace a reading", value=stress.index[-1].date(), min_value=stress.index[0].date(), max_value=stress.index[-1].date())
    if st.button("Trace and verify"):
        report = trace(run_path, str(tday))
        ok = all(report["checks"].values())
        (st.success if ok else st.error)("All checks pass: inputs, features and probability reproduce." if ok else "Verification failed")
        st.code(json.dumps(report, indent=2), language="json")

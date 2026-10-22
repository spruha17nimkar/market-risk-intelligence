# Cross-Asset Market Risk Intelligence Framework

Standard Value-at-Risk models assume volatility and correlations are fairly stable. In real stress events
two things happen at once: **volatility clusters** (large moves follow large moves) and **correlations
converge** (assets that normally offset each other fall together). A VaR model calibrated in calm markets
then understates tail risk exactly when it matters most.

This framework measures that risk, tests whether the risk model holds up in each regime, and flags when
markets are moving into stress, with every reading auditable back to its inputs.

```
Yahoo Finance ──► 252-day rolling risk engine ──► Kupiec backtest by regime ──► governance run store
  (5 proxies)      vol · covariance · VaR · ES           │                          │
                         │                                ▼                          ▼
                         └──► regime features ──► logistic stress classifier ──► XAI attribution ──► Streamlit dashboard
```

## Coverage

| Risk factor | Proxy | Notes |
|---|---|---|
| Equities | `SPY` | S&P 500 |
| Duration / rates | `TLT` | 20+ year US Treasuries |
| Credit spreads | `HYG` | US high yield (also carries some rate duration) |
| FX | `UUP` | US Dollar Index proxy (DXY) |
| Commodities | `DBC` | Diversified commodity index (GSCI-style) |

- **Data source:** Yahoo Finance adjusted closes via `yfinance`, cached locally.
- **Sample:** 2007-04-11 (HYG inception) to 2025-12-30. Risk readings start 2008-04-11, after the first full 252-day window (4,459 days).
- **Portfolio:** static long-only weights of 40% equities, 30% rates, 15% credit, 5% FX and 10% commodities (configurable).
- **Confidence levels:** 99% and 97.5% VaR, 97.5% ES (the FRTB calibration), one-day horizon.

## Components

### 1. Rolling risk engine ([`risk_engine.py`](src/mrif/risk_engine.py))
Each day, a 252-day window recomputes portfolio returns and volatility, the full 5×5 cross-asset covariance
matrix, VaR and Expected Shortfall. The reading for day *t* uses only returns up to *t−1*, so it can be
backtested against the loss realised on day *t* without look-ahead. Three estimators run side by side:

| Model | Description |
|---|---|
| `hs` | Historical simulation over the window |
| `param` | Variance-covariance (normal) using the window's sample covariance |
| `ewma` | Variance-covariance using a RiskMetrics EWMA covariance (λ = 0.94), which reacts to volatility clustering |

FRTB moved regulatory market risk capital from 99% VaR to 97.5% ES because ES captures how severe losses in
the tail are, not only where the tail begins. The engine reports both.

### 2. VaR backtesting and model validation ([`backtest.py`](src/mrif/backtest.py))
- **Kupiec POF:** a likelihood-ratio test of whether the observed exception count matches the count the model predicts.
- **Christoffersen independence:** tests whether exceptions cluster in time.
- **Basel traffic light:** applied to the last 250 days.

All tests run on the full sample and separately on **stable** and **stress** days, so the results show where
the model is well calibrated and where it breaks. This is the core question in SR 11-7 model validation.

### 3. Stress regime classifier ([`classifier.py`](src/mrif/classifier.py), [`features.py`](src/mrif/features.py))
A scikit-learn logistic regression on standardised features outputs a probability that markets are in a
stress regime:

| Driver | Feature | Definition |
|---|---|---|
| Volatility clustering | `vol_level` | log of 21-day annualised portfolio volatility |
| | `vol_persistence` | 63-day lag-1 autocorrelation of absolute portfolio returns |
| Drawdown | `drawdown` | distance from the 252-day running peak |
| Correlation stress | `avg_corr` | 63-day average pairwise correlation across the five asset classes |
| Duration sensitivity | `duration_corr` | 63-day equity-rates correlation |

Ground truth is a documented list of 11 stress episodes (GFC, the euro crises, China 2015, Volmageddon,
Q4 2018, COVID-19, the 2022 rate shock, SVB and the 2025 tariff shock; see
[`default.toml`](src/mrif/default.toml)). Validation is **walk-forward**: model `wf-YYYY` is trained only on
days before 1 January YYYY and scores only YYYY. The reported ROC-AUC is therefore genuinely out of sample.

Logistic regression is a deliberate choice over a black-box model. Its coefficients can be read directly and
explained to a risk committee or a model validator.

### 4. Explainability and governance ([`explain.py`](src/mrif/explain.py), [`governance.py`](src/mrif/governance.py))
- **Attribution.** For a logistic model on standardised features, `coef × z` is the exact SHAP value in
  log-odds. Each reading therefore splits *exactly* into volatility surge, drawdown, correlation spike and
  duration sensitivity contributions, using the walk-forward model that produced it.
- **Run store.** Every `mrif run` writes an immutable `runs/<run_id>/` folder containing:
  - the config, the git commit, library versions and the data file's SHA-256
  - every fitted model's parameters and a model card (purpose, limitations, metrics)
  - all outputs
- **Reading-level traceability.** Each daily reading stores a SHA-256 hash of the exact returns window it
  consumed. `mrif trace <date>` recomputes the hash, the features and the probability from the stored model
  parameters, then reports whether each one reproduces.

```text
$ mrif trace 2020-03-16
  "model_id": "wf-2020",  "model_trained_on": "2008-04-11 to 2019-12-31",
  "input_window": "2019-03-15 to 2020-03-16 (253 rows)",
  "stored_probability": 0.99998, "recomputed_probability": 0.99998, "top_driver": "volatility_surge",
  "checks": {"input_hash_matches": true, "features_reproduce": true, "probability_reproduces": true}
```

### 5. Risk dashboard ([`dashboard/app.py`](src/mrif/dashboard/app.py))
A Streamlit and Plotly dashboard with four tabs:

- **VaR & ES:** daily loss against VaR and ES with exceptions marked, and a model comparison.
- **Correlations:** rolling correlation heatmaps across the five asset classes, plus pooled stable-day and stress-day matrices.
- **Stress monitor:** the classifier's probability over time, driver attribution over time, and a waterfall for any single reading.
- **Model governance:** Kupiec results by regime, per-episode statistics, parameters for every model version, the model card, the manifest and an interactive audit trace.

Labelled stress episodes are shaded on every time series.

## Results (run of 22 Oct 2026, Yahoo data to 2025-12-30)

Every number here comes from the run's `report.md`. Re-running reproduces them from the same data.

### Stress classifier
- **Out-of-sample ROC-AUC: 0.870** (walk-forward, 2012–2025). The full-sample model's in-sample AUC is 0.935.
- At p ≥ 0.5, precision is 0.44 and recall is 0.53 (Brier score 0.101).
- Volatility level is the dominant driver: about a 6× increase in the odds of stress per standard deviation.
- **Weak spot:** in **2022** the out-of-sample AUC fell to **0.45**. Earlier crises taught the model that
  stress comes with *negative* equity-rates correlation (a flight to quality). The 2022 inflation shock broke
  that pattern, because stocks and bonds fell together. This is the kind of structural break a validator
  should flag.

### VaR backtest (Kupiec POF, 99% VaR)

| Model | Stable days (3,749) | Stress days (710) |
|---|---|---|
| Historical simulation | 19 exceptions vs 37.5 expected, **FAIL** (too conservative) | 45 vs 7.1, **FAIL** (6.3×) |
| Parametric normal | 27 vs 37.5, **PASS** (p = 0.07) | 62 vs 7.1, **FAIL** (8.7×) |
| EWMA | 55 vs 37.5, **FAIL** | 33 vs 7.1, **FAIL** (4.6×) |

The window-based models pass, or err on the conservative side, in calm markets. Every model fails badly in
stress, where exceptions run 4.6 to 8.7 times the expected rate. Historical simulation is too conservative in
calm periods, because crisis losses stay in its 252-day window long after the crisis (the "ghost effect").
EWMA reacts fastest to volatility clustering and cuts stress-period exceptions roughly in half compared with
the parametric model, but it still fails.

### Correlation convergence

| 63-day average pairwise correlation | Stable | Stress (mean) | Stress (p90) | Max |
|---|---:|---:|---:|---:|
| All 5 asset classes (signed) | 0.01 | 0.02 | 0.13 | 0.36 |
| All 5 asset classes (absolute) | 0.36 | 0.37 | 0.57 | 0.75 |
| Risk assets: equities, credit, commodities | 0.48 | 0.51 | **0.73** | **0.86** (16 Mar 2020) |

The risk assets converge in stress. Their average correlation tops 0.70 on 15% of stress days, peaking at
0.86 in the COVID crash and reaching 0.74–0.76 in the two euro crises. The **five-asset average does not**. Treasuries
and the US dollar usually rally in sell-offs, which offsets the risk-asset convergence. As a result, applying
stress-period correlations to this particular portfolio *lowers* its parametric ES slightly (−2.3%). The
diversification that breaks down is among the risk assets, while the duration hedge does most of the work.
The 2022 episode is the exception, and it is the scenario a stressed-correlation capital buffer must cover.

## Quick start

```bash
git clone https://github.com/spruha17nimkar/market-risk-intelligence.git
cd market-risk-intelligence
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

mrif run                 # download data, run the pipeline, store runs/<run_id>/ and print the report
mrif trace 2020-03-16    # audit one reading end to end
mrif dashboard           # open the Streamlit dashboard
mrif run --synthetic     # offline demo on regime-switching simulated data
pytest                   # 25 tests, no network needed
```

Use `mrif --config my.toml run` to change proxies, weights, windows, confidence levels or stress episodes.
Start from [`src/mrif/default.toml`](src/mrif/default.toml).

## Project layout

```
src/mrif/
  default.toml       configuration: proxies, weights, windows, episodes, classifier
  data.py            Yahoo Finance loader with cache; synthetic regime-switching generator
  risk_engine.py     rolling 252-day vol, covariance, VaR and ES (HS / parametric / EWMA)
  backtest.py        Kupiec POF, Christoffersen, Basel traffic light, by-regime backtest
  regimes.py         stress-episode labels
  features.py        classifier features and correlation diagnostics
  classifier.py      walk-forward logistic regression with stored parameters
  explain.py         exact log-odds attribution by driver
  governance.py      run store, hashing, manifest, trace and verify
  pipeline.py        end-to-end orchestration and report
  cli.py             mrif run | trace | dashboard
  dashboard/app.py   Streamlit + Plotly dashboard
tests/               unit and end-to-end tests (synthetic data)
```

## Limitations
- The stress labels are a judgement call. Moving the episode dates changes the target and the AUC.
- The classifier *nowcasts* the current regime from contemporaneous features. It does not forecast stress.
- The ETF proxies carry fees and tracking error. HYG mixes credit spread with rate duration.
- The normal VaR models ignore fat tails, which is part of why they fail in stress. Filtered historical simulation and Student-t models are natural extensions.

## License
MIT

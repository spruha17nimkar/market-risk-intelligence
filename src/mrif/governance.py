"""Model governance (SR 11-7 style): versioned run store, input hashing and reading-level traceability.

Each pipeline run writes an immutable folder ``runs/<run_id>/`` holding the configuration, data
fingerprints, library versions, every fitted model's parameters and all outputs. Each daily reading
carries ``input_hash``: a SHA-256 of the exact returns window it consumed, so ``trace`` can prove which
inputs produced it and recompute the stress probability from the stored model parameters.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HASH_DECIMALS = 12
LIBRARIES = ("numpy", "pandas", "scipy", "scikit-learn", "yfinance")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.round(a, HASH_DECIMALS)).tobytes()).hexdigest()


def window_hashes(returns: pd.DataFrame, lookback: int) -> pd.Series:
    """Hash of returns rows [t - lookback, t] for every t with a full lookback."""
    R = returns.to_numpy(dtype=float)
    idx = range(lookback, len(R))
    return pd.Series([sha256_array(R[t - lookback : t + 1]) for t in idx], index=returns.index[lookback:], name="input_hash")


def git_revision() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout.strip())
        return {"commit": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def environment() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for lib in LIBRARIES:
        try:
            versions[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            versions[lib] = "not installed"
    return versions


class RunStore:
    def __init__(self, path: Path):
        self.path = path

    @classmethod
    def create(cls, runs_dir: str | Path, config: dict) -> "RunStore":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        cfg_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
        path = Path(runs_dir) / f"{stamp}-{cfg_hash}"
        path.mkdir(parents=True, exist_ok=False)
        return cls(path)

    @staticmethod
    def latest(runs_dir: str | Path) -> Path:
        runs = sorted(p for p in Path(runs_dir).glob("*") if (p / "manifest.json").exists())
        if not runs:
            raise FileNotFoundError(f"no runs found in {runs_dir}; run `mrif run` first")
        return runs[-1]

    def write_json(self, name: str, obj: Any) -> None:
        (self.path / name).write_text(json.dumps(obj, indent=2, default=_json_default))

    def write_frame(self, name: str, df: pd.DataFrame) -> None:
        df.to_csv(self.path / name)

    def write_array(self, name: str, a: np.ndarray) -> None:
        np.save(self.path / name, a)

    def write_text(self, name: str, text: str) -> None:
        (self.path / name).write_text(text)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def read_run(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    load = lambda n: pd.read_csv(run / n, index_col=0, parse_dates=True, float_precision="round_trip")  # noqa: E731
    return {
        "path": run,
        "manifest": json.loads((run / "manifest.json").read_text()),
        "model_card": json.loads((run / "model_card.json").read_text()),
        "findings": json.loads((run / "findings.json").read_text()),
        "returns": load("returns.csv"),
        "risk": load("risk.csv"),
        "features": load("features.csv"),
        "diagnostics": load("diagnostics.csv"),
        "stress": load("stress.csv"),
        "episodes": pd.read_csv(run / "episodes.csv") if (run / "episodes.csv").stat().st_size > 1 else pd.DataFrame(),
        "backtest": pd.read_csv(run / "backtest.csv"),
        "cov": np.load(run / "covariances.npy"),
    }


def trace(run_dir: str | Path, date: str) -> dict[str, Any]:
    """Reconstruct one stress reading from stored inputs and verify it end to end."""
    from .classifier import FittedModel
    from .config import FeatureConfig
    from .features import build_features

    run = read_run(run_dir)
    ts = pd.Timestamp(date)
    stress = run["stress"]
    if ts not in stress.index:
        later = stress.index[stress.index >= ts]
        if len(later) == 0:
            raise KeyError(f"{date} is after the last reading in this run")
        ts = later[0]
    row = stress.loc[ts]
    manifest = run["manifest"]
    cfg = manifest["config"]
    returns = run["returns"]
    lookback = manifest["input_hash_lookback"]

    t = returns.index.get_loc(ts)
    window = returns.iloc[t - lookback : t + 1]
    recomputed_hash = sha256_array(window.to_numpy(dtype=float))

    feats = build_features(window, [cfg["weights"][a] for a in cfg["assets"]], FeatureConfig(**cfg["features"]))
    model = FittedModel.from_dict(run["model_card"]["models"][row["model_id"]])
    x_stored = run["features"].loc[ts, model.features].to_numpy(dtype=float)
    x_recomputed = feats.loc[ts, model.features].to_numpy(dtype=float)
    prob = float(model.predict_proba(x_stored[None, :])[0])

    return {
        "run_id": manifest["run_id"],
        "date": str(ts.date()),
        "model_id": row["model_id"],
        "model_trained_on": f"{model.train_start} to {model.train_end}",
        "input_window": f"{window.index[0].date()} to {window.index[-1].date()} ({len(window)} rows)",
        "input_hash": row["input_hash"],
        "features": dict(zip(model.features, x_stored.round(6).tolist())),
        "contributions_log_odds": {f: round(float(row[f"c_{f}"]), 6) for f in model.features},
        "intercept": model.intercept,
        "stored_probability": float(row["prob"]),
        "recomputed_probability": prob,
        "top_driver": row["top_driver"],
        "checks": {
            "input_hash_matches": recomputed_hash == row["input_hash"],
            "features_reproduce": bool(np.allclose(x_stored, x_recomputed, atol=1e-9)),
            "probability_reproduces": bool(abs(prob - float(row["prob"])) < 1e-9),
        },
    }

"""Command line: ``mrif run``, ``mrif trace`` and ``mrif dashboard``."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .config import Config


def _load_config(path: str | None) -> Config:
    return Config.from_toml(path) if path else Config.default()


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run_pipeline

    cfg = _load_config(args.config)
    if args.runs_dir:
        cfg.governance.runs_dir = args.runs_dir
    result = run_pipeline(cfg, synthetic=args.synthetic, refresh=args.refresh)
    print((result.run_dir / "report.md").read_text())
    print(f"Run stored in {result.run_dir}")
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    from .governance import RunStore, trace

    run_dir = args.run or RunStore.latest(_load_config(args.config).governance.runs_dir)
    report = trace(run_dir, args.date)
    print(json.dumps(report, indent=2))
    return 0 if all(report["checks"].values()) else 1


def cmd_dashboard(args: argparse.Namespace) -> int:
    app = Path(__file__).parent / "dashboard" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--", "--runs-dir", args.runs_dir]
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mrif", description="Cross-Asset Market Risk Intelligence Framework")
    parser.add_argument("--config", help="TOML config (defaults to the packaged default.toml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the full pipeline and store an auditable run")
    p_run.add_argument("--synthetic", action="store_true", help="use the synthetic regime-switching data set")
    p_run.add_argument("--refresh", action="store_true", help="re-download market data instead of using the cache")
    p_run.add_argument("--runs-dir", help="override governance.runs_dir")
    p_run.set_defaults(func=cmd_run)

    p_trace = sub.add_parser("trace", help="trace one stress reading back to its inputs and verify it")
    p_trace.add_argument("date", help="reading date, e.g. 2020-03-16")
    p_trace.add_argument("--run", help="run folder (defaults to the latest run)")
    p_trace.set_defaults(func=cmd_trace)

    p_dash = sub.add_parser("dashboard", help="launch the Streamlit dashboard")
    p_dash.add_argument("--runs-dir", default="runs")
    p_dash.set_defaults(func=cmd_dashboard)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

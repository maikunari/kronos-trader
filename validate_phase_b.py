"""
validate_phase_b.py
CLI wrapper for the Phase B signal validation report.

Runs the divergence reversal detector across the Krillin watchlist
for a historical window and writes a markdown report.

Example:
    ./venv/bin/python validate_phase_b.py \
        --timeframe 1h \
        --start 2025-04-01 --end 2026-04-01 \
        --out tasks/phase_b_validation.md

Writes a human-readable markdown report alongside a JSON snapshot of
the same data for downstream analysis.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from setups.divergence import DivergenceReversalDetector
from validation.report import run_validation


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")
    logging.getLogger("risk_manager").setLevel(logging.ERROR)   # quiet per-trade noise

    parser = argparse.ArgumentParser(description="Phase B signal validation")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start", default="2025-04-01")
    parser.add_argument("--end", default="2026-04-01")
    parser.add_argument("--threshold-pct", type=float, default=0.20)
    parser.add_argument("--window-hours", type=float, default=72.0)
    parser.add_argument("--out", default="tasks/phase_b_validation.md")
    parser.add_argument("--tickers", nargs="*",
                        help="Override ticker list (default: full watchlist)")
    parser.add_argument("--max-lead-hours", type=float, default=24.0)
    parser.add_argument("--max-lag-hours", type=float, default=4.0)
    args = parser.parse_args()

    if args.tickers:
        tickers = args.tickers
    else:
        wl = yaml.safe_load(Path("markets/krillin_watchlist.yaml").read_text())
        tickers = wl["tickers"]

    print(f"Running Phase B validation on {len(tickers)} tickers, {args.timeframe}, "
          f"{args.start} -> {args.end}", file=sys.stderr)

    report = run_validation(
        detectors=[DivergenceReversalDetector()],
        tickers=tickers,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        threshold_pct=args.threshold_pct,
        window_hours=args.window_hours,
        max_lead=pd.Timedelta(hours=args.max_lead_hours),
        max_lag=pd.Timedelta(hours=args.max_lag_hours),
        progress_cb=lambda msg: print(f"  {msg}", file=sys.stderr),
    )

    md = report.format_markdown()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"\nReport written to {out_path}", file=sys.stderr)

    # JSON snapshot alongside
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "stats": [asdict(s) for s in report.stats],
        "per_ticker_pops": report.per_ticker_pops,
        "total_pops": report.total_pops,
        "total_triggers": report.total_triggers,
        "config": report.config,
    }, indent=2, default=str))
    print(f"JSON snapshot at {json_path}", file=sys.stderr)

    print("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
validate_cbs_replication.py
CLI wrapper for the CBS replication-fidelity check.

Runs the registered detector library against every trade in
validation/cbs_trades.yaml and reports per-trade match status.

Example:
    ./venv/bin/python validate_cbs_replication.py \\
        --out tasks/cbs_replication.md
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from setups.divergence import DivergenceReversalDetector
from validation.cbs_replication import (
    load_cbs_trades,
    run_replication,
)


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="CBS replication-fidelity check")
    parser.add_argument("--trades-file", default="validation/cbs_trades.yaml")
    parser.add_argument("--match-window-hours", type=float, default=24.0)
    parser.add_argument("--lookback-days", type=int, default=45)
    parser.add_argument("--post-days", type=int, default=2)
    parser.add_argument("--out", default="tasks/cbs_replication.md")
    args = parser.parse_args()

    trades = load_cbs_trades(args.trades_file)
    print(f"Loaded {len(trades)} CBS trades from {args.trades_file}", file=sys.stderr)

    detectors = [DivergenceReversalDetector()]
    report = run_replication(
        trades, detectors,
        match_window=pd.Timedelta(hours=args.match_window_hours),
        lookback_days=args.lookback_days,
        post_days=args.post_days,
        progress_cb=lambda msg: print(f"  {msg}", file=sys.stderr),
    )

    md = report.format_markdown()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"\nReport written to {out_path}", file=sys.stderr)

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "results": [_serialize_result(r) for r in report.results],
        "config": report.config,
        "match_rate": report.match_rate,
    }, indent=2, default=str))
    print(f"JSON snapshot at {json_path}", file=sys.stderr)

    print("\n" + md)
    return 0


def _serialize_result(r) -> dict:
    return {
        "trade_id": r.trade.id,
        "ticker": r.trade.ticker,
        "direction": r.trade.direction,
        "entry_date": str(r.trade.entry_date),
        "matched": r.matched,
        "best_trigger": asdict(r.best_trigger) if r.best_trigger else None,
        "same_direction_trigger_count": len(r.same_direction_triggers),
        "opposite_direction_trigger_count": len(r.opposite_direction_triggers),
        "gap_reason": r.gap_reason,
        "fetch_error": r.fetch_error,
        "lead_time_hours": (r.lead_time.total_seconds() / 3600
                            if r.lead_time is not None else None),
        "bars_scanned": r.bars_scanned,
    }


if __name__ == "__main__":
    sys.exit(main())

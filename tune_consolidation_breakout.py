"""
tune_consolidation_breakout.py
Parameter sweep for setups/consolidation_breakout.py.

Runs a grid of (max_range_pct, required_confirm_closes, breakout_buffer_pct)
against:
  * Phase B validation on a fixed 8-ticker probe set
  * CBS replication on the seed corpus

Outputs a summary markdown table ranking configs by Phase B recall on
the breakout setup, with CBS match count as a secondary column.

Intent: pick the best defaults before baking into the detector. The
current defaults (3%, 2, 10bp) yield 0% recall on the probe — we know
they're too conservative; this script measures how much each lever
moves the needle.
"""
from __future__ import annotations

import itertools
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from setups.consolidation_breakout import ConsolidationBreakoutDetector
from validation.cbs_replication import load_cbs_trades, run_replication
from validation.report import run_validation


PROBE_TICKERS = ["HYPE", "TAO", "ENA", "NEAR", "LINK", "AVAX", "ZRO", "WLD"]

# Grid: deliberately small. 3 × 2 × 2 = 12 configs.
MAX_RANGE_PCTS = [0.03, 0.04, 0.05]
CONFIRM_CLOSES = [1, 2]
BREAKOUT_BUFFERS = [0.001, 0.002]


def _breakout_stats(report):
    """Extract the consolidation_breakout row from a ValidationReport."""
    long_stats = next(
        (s for s in report.stats
         if s.setup == "consolidation_breakout" and s.direction == "long"),
        None,
    )
    short_stats = next(
        (s for s in report.stats
         if s.setup == "consolidation_breakout" and s.direction == "short"),
        None,
    )
    return long_stats, short_stats


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")

    trades = load_cbs_trades("validation/cbs_trades.yaml")
    rows: list[dict] = []

    configs = list(itertools.product(MAX_RANGE_PCTS, CONFIRM_CLOSES, BREAKOUT_BUFFERS))
    print(f"Running {len(configs)} configs × (Phase B + CBS replication)",
          file=sys.stderr)

    for i, (max_range, confirm, buffer) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] max_range={max_range:.2%} "
              f"N={confirm} buf={buffer:.3%}", file=sys.stderr)

        detector = ConsolidationBreakoutDetector(
            max_range_pct=max_range,
            required_confirm_closes=confirm,
            breakout_buffer_pct=buffer,
        )

        # Phase B probe
        pb_report = run_validation(
            detectors=[detector],
            tickers=PROBE_TICKERS,
            timeframe="1h",
            start=pd.Timestamp("2025-04-01", tz="UTC"),
            end=pd.Timestamp("2026-04-01", tz="UTC"),
        )
        long_stats, short_stats = _breakout_stats(pb_report)

        # CBS replication
        cbs_report = run_replication(trades, [detector])
        cbs_matches = sum(1 for r in cbs_report.assessed if r.matched)

        rows.append({
            "max_range_pct": max_range,
            "required_confirm_closes": confirm,
            "breakout_buffer_pct": buffer,
            "pb_triggers_long": long_stats.triggers if long_stats else 0,
            "pb_recall_long": long_stats.recall if long_stats else 0.0,
            "pb_precision_long": long_stats.precision if long_stats else 0.0,
            "pb_capture_long": (long_stats.median_capture_ratio
                                if long_stats else 0.0),
            "pb_triggers_short": short_stats.triggers if short_stats else 0,
            "pb_recall_short": short_stats.recall if short_stats else 0.0,
            "cbs_matches": cbs_matches,
            "cbs_assessed": len(cbs_report.assessed),
        })

    # Write markdown + json
    out_md = Path("tasks/tune_consolidation_breakout.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Consolidation-breakout parameter sweep")
    lines.append("")
    lines.append(f"- Probe tickers: {', '.join(PROBE_TICKERS)}")
    lines.append("- Timeframe: 1h, 2025-04-01 → 2026-04-01")
    lines.append(f"- Configs: {len(rows)}")
    lines.append(f"- CBS seed trades: {len(trades)}")
    lines.append("")
    lines.append("Ranked by Phase B recall (long), CBS matches as tiebreak.")
    lines.append("")
    lines.append("| range% | N | buf% | trigs L | recall L | prec L | cap% L "
                 "| trigs S | recall S | CBS |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    ranked = sorted(rows, key=lambda r: (-r["pb_recall_long"], -r["cbs_matches"]))
    for r in ranked:
        lines.append(
            f"| {r['max_range_pct']*100:.0f}% | {r['required_confirm_closes']} "
            f"| {r['breakout_buffer_pct']*100:.2f}% "
            f"| {r['pb_triggers_long']} | {r['pb_recall_long']*100:.1f}% "
            f"| {r['pb_precision_long']*100:.1f}% "
            f"| {r['pb_capture_long']*100:.1f}% "
            f"| {r['pb_triggers_short']} | {r['pb_recall_short']*100:.1f}% "
            f"| {r['cbs_matches']}/{r['cbs_assessed']} |"
        )
    lines.append("")

    out_md.write_text("\n".join(lines))
    (out_md.with_suffix(".json")).write_text(json.dumps(rows, indent=2))
    print(f"\nReport written to {out_md}", file=sys.stderr)
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

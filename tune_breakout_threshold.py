"""
tune_breakout_threshold.py
Re-run Phase B validation at a lower pop threshold (10%) for a few
representative consolidation-breakout configs.

Hypothesis: the detector fires correctly but the default 20% pop
labeler filters out the 5-15% moves breakouts actually produce. If
the 10%-threshold run shows non-zero recall, validation framing was
the gap. If still 0% across configs, §4.3 is structurally the wrong
setup shape for this universe.

Pins to the same 8-ticker probe + 1h window as the previous sweep so
the diagnosis is apples-to-apples.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

from setups.consolidation_breakout import ConsolidationBreakoutDetector
from validation.report import run_validation


PROBE_TICKERS = ["HYPE", "TAO", "ENA", "NEAR", "LINK", "AVAX", "ZRO", "WLD"]

# Three representative configs from the earlier sweep:
#   baseline current default (3% / N=2)
#   moderate loosening (4% / N=1)
#   loosest (5% / N=1 / wider buffer)
CONFIGS = [
    {"max_range_pct": 0.03, "required_confirm_closes": 2, "breakout_buffer_pct": 0.001},
    {"max_range_pct": 0.04, "required_confirm_closes": 1, "breakout_buffer_pct": 0.001},
    {"max_range_pct": 0.05, "required_confirm_closes": 1, "breakout_buffer_pct": 0.001},
]

THRESHOLDS = [0.10, 0.20]


def _breakout_stats(report, direction):
    return next(
        (s for s in report.stats
         if s.setup == "consolidation_breakout" and s.direction == direction),
        None,
    )


def main() -> int:
    logging.basicConfig(level=logging.ERROR)

    rows: list[dict] = []
    total = len(CONFIGS) * len(THRESHOLDS)
    i = 0
    for cfg in CONFIGS:
        detector = ConsolidationBreakoutDetector(**cfg)
        for threshold in THRESHOLDS:
            i += 1
            print(f"[{i}/{total}] range={cfg['max_range_pct']:.0%} "
                  f"N={cfg['required_confirm_closes']} "
                  f"threshold={threshold:.0%}", file=sys.stderr)

            report = run_validation(
                detectors=[detector],
                tickers=PROBE_TICKERS,
                timeframe="1h",
                start=pd.Timestamp("2025-04-01", tz="UTC"),
                end=pd.Timestamp("2026-04-01", tz="UTC"),
                threshold_pct=threshold,
            )
            long_s = _breakout_stats(report, "long")
            short_s = _breakout_stats(report, "short")
            rows.append({
                **cfg,
                "threshold": threshold,
                "total_pops": report.total_pops,
                "long_triggers": long_s.triggers if long_s else 0,
                "long_tp": long_s.true_positives if long_s else 0,
                "long_recall": long_s.recall if long_s else 0.0,
                "long_precision": long_s.precision if long_s else 0.0,
                "long_capture": long_s.median_capture_ratio if long_s else 0.0,
                "long_mean_ret": long_s.mean_realized_return if long_s else 0.0,
                "short_triggers": short_s.triggers if short_s else 0,
                "short_tp": short_s.true_positives if short_s else 0,
                "short_recall": short_s.recall if short_s else 0.0,
            })

    out_md = Path("tasks/tune_breakout_threshold.md")
    lines: list[str] = []
    lines.append("# Consolidation-breakout: threshold diagnostic")
    lines.append("")
    lines.append(f"- Probe tickers: {', '.join(PROBE_TICKERS)}")
    lines.append("- Same 8-ticker / 1h / 1-year window as the earlier sweep.")
    lines.append(
        "- Question: does loosening the pop threshold from 20% → 10% "
        "reveal recall that was hidden by move-magnitude filtering?"
    )
    lines.append("")
    lines.append(
        "| range% | N | thr | pops | trigs L | TP L | recall L | prec L "
        "| cap% L | mean_ret L | trigs S | TP S | recall S |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['max_range_pct']*100:.0f}% | {r['required_confirm_closes']} "
            f"| {r['threshold']*100:.0f}% | {r['total_pops']} "
            f"| {r['long_triggers']} | {r['long_tp']} "
            f"| {r['long_recall']*100:.1f}% | {r['long_precision']*100:.1f}% "
            f"| {r['long_capture']*100:.1f}% "
            f"| {r['long_mean_ret']*100:+.2f}% "
            f"| {r['short_triggers']} | {r['short_tp']} "
            f"| {r['short_recall']*100:.1f}% |"
        )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append(
        "Compare each (range, N) pair's 10% vs 20% rows. If the 10% row "
        "has non-zero TP/recall while the 20% row is zero, the detector "
        "was catching real moves — just smaller than the 20% label cut-off."
    )

    out_md.write_text("\n".join(lines))
    out_md.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    print(f"\nReport written to {out_md}", file=sys.stderr)
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

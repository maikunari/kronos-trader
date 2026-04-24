"""
tune_v_reversal.py
Parameter sweep for setups/v_reversal.py.

Grid over (min_move_pct_long, min_bounce_pct, max_bars_since_hl) against the
same 8-ticker / 1h / 1-year probe as phase_b_validation_v5.md. Aim: tighten
precision above 10% without killing recall.

Decision rule (from the plan):
  * If any config hits precision >= 10% and recall >= 5%, bake those as new
    defaults.
  * If nothing crosses that bar, the deterministic grid is tapped out and
    the agentic loop becomes the right next move.
"""
from __future__ import annotations

import itertools
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from setups.v_reversal import VReversalDetector
from validation.report import run_validation


PROBE_TICKERS = ["HYPE", "TAO", "ENA", "NEAR", "LINK", "AVAX", "ZRO", "WLD"]

MIN_MOVE_PCTS = [0.10, 0.15, 0.20]      # long-side; raise to reject shallow dips
MIN_BOUNCES = [0.03, 0.05]              # demand a more meaningful bounce
MAX_BARS_SINCE_HL = [2, 5]              # tighter freshness window


def _row(report, direction: str):
    for s in report.stats:
        if s.setup == "v_reversal" and s.direction == direction:
            return s
    return None


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")

    configs = list(itertools.product(MIN_MOVE_PCTS, MIN_BOUNCES, MAX_BARS_SINCE_HL))
    print(f"Running {len(configs)} configs on {len(PROBE_TICKERS)} tickers",
          file=sys.stderr)

    rows: list[dict] = []
    for i, (min_move, min_bounce, max_hl) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] min_move={min_move:.0%} "
              f"min_bounce={min_bounce:.0%} max_hl={max_hl}", file=sys.stderr)

        detector = VReversalDetector(
            min_move_pct_long=min_move,
            # Short threshold scales proportionally (arch §4.4: 10% long -> 15% short)
            min_move_pct_short=min_move + 0.05,
            min_bounce_pct=min_bounce,
            max_bars_since_hl=max_hl,
        )
        report = run_validation(
            detectors=[detector],
            tickers=PROBE_TICKERS,
            timeframe="1h",
            start=pd.Timestamp("2025-04-01", tz="UTC"),
            end=pd.Timestamp("2026-04-01", tz="UTC"),
        )
        long_s = _row(report, "long")
        short_s = _row(report, "short")
        rows.append({
            "min_move_pct_long": min_move,
            "min_bounce_pct": min_bounce,
            "max_bars_since_hl": max_hl,
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

    # Rank by long precision, recall as tiebreak
    ranked = sorted(rows, key=lambda r: (-r["long_precision"], -r["long_recall"]))

    lines: list[str] = []
    lines.append("# v_reversal parameter sweep")
    lines.append("")
    lines.append(f"- Probe tickers: {', '.join(PROBE_TICKERS)}")
    lines.append("- 1h, 2025-04-01 → 2026-04-01 (same as phase_b_validation_v5.md)")
    lines.append("- Ranked by long precision; recall as tiebreak.")
    lines.append("")
    lines.append("| move% | bounce% | hl | trigs L | TP L | recall L | prec L | cap% L "
                 "| mean_ret L | trigs S | TP S | recall S |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ranked:
        lines.append(
            f"| {r['min_move_pct_long']*100:.0f}% | {r['min_bounce_pct']*100:.0f}% "
            f"| {r['max_bars_since_hl']} | {r['long_triggers']} | {r['long_tp']} "
            f"| {r['long_recall']*100:.1f}% | {r['long_precision']*100:.1f}% "
            f"| {r['long_capture']*100:.1f}% "
            f"| {r['long_mean_ret']*100:+.2f}% "
            f"| {r['short_triggers']} | {r['short_tp']} "
            f"| {r['short_recall']*100:.1f}% |"
        )
    lines.append("")
    lines.append("## Decision gate")
    lines.append("")
    lines.append(
        "Any config with **precision ≥ 10% AND recall ≥ 5%** qualifies as a "
        "new default. If the top row is below that bar, deterministic "
        "tuning is exhausted and the agentic loop is the path forward."
    )

    out_md = Path("tasks/tune_v_reversal.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    out_md.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    print(f"\nReport: {out_md}", file=sys.stderr)
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

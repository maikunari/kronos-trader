"""
sr_cli.py
Phase A deliverable: command-line tool that outputs S/R zones for a
ticker as of a given timestamp.

Pulls candles through the cache layer, runs detect_sr_zones, prints a
formatted zone table. Useful as:
  * Manual S/R sanity check before trusting the auto-detector
  * Starting point for the Phase D LLM curation tool
  * Scratchpad for tuning pivot_window / merge_pct per-ticker

Example:
    ./venv/bin/python sr_cli.py --symbol ENA --timeframe 1h \
        --as-of 2026-04-17T20:00:00Z --lookback-days 180
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from data_cache import get_candles
from support_resistance import SRZone, detect_sr_zones


def _parse_ts(s: str) -> pd.Timestamp:
    """Parse an ISO-like timestamp into UTC."""
    ts = pd.Timestamp(s)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _slice_as_of(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Return only bars with timestamp strictly before `as_of` (no lookahead)."""
    return df[df["timestamp"] < as_of].reset_index(drop=True)


def _format_zone_table(
    zones: list[SRZone],
    current_price: float,
    limit: int = 20,
) -> str:
    if not zones:
        return "No zones found (try increasing lookback or loosening merge_pct)."
    header = (
        f"{'rank':>4}  {'mid':>12}  {'band':>18}  "
        f"{'side':<11}  {'str':>5}  {'t':>3}  "
        f"{'dist%':>7}  {'last_touch':<25}"
    )
    rows = [header, "-" * len(header)]
    for i, z in enumerate(zones[:limit], 1):
        dist_pct = z.distance_pct(current_price) * 100
        band = f"{z.price_low:.4f}–{z.price_high:.4f}"
        last = z.touches[-1].timestamp if z.touches and z.touches[-1].timestamp else ""
        rows.append(
            f"{i:>4}  {z.price_mid:>12.4f}  {band:>18}  "
            f"{z.side:<11}  {z.strength:>5.2f}  {z.touch_count:>3}  "
            f"{dist_pct:>+7.2f}  {str(last):<25}"
        )
    if len(zones) > limit:
        rows.append(f"... ({len(zones) - limit} more)")
    return "\n".join(rows)


def run(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    as_of = _parse_ts(args.as_of) if args.as_of else pd.Timestamp.now(tz="UTC")
    start = as_of - timedelta(days=args.lookback_days)

    print(f"\n{'='*70}")
    print(f"  {args.symbol} / {args.timeframe}    as-of: {as_of}")
    print(f"  lookback: {args.lookback_days} days    pivot_window: {args.pivot_window}    merge_pct: {args.merge_pct}")
    print(f"{'='*70}\n")

    df = get_candles(args.symbol, args.timeframe, start, as_of + timedelta(hours=1))
    if df.empty:
        print(f"No candles fetched for {args.symbol}/{args.timeframe}.")
        return 1

    df = _slice_as_of(df, as_of)
    if len(df) < 2 * args.pivot_window + 2:
        print(f"Not enough bars after slicing ({len(df)}). Need at least {2 * args.pivot_window + 2}.")
        return 1

    current_price = float(df["close"].iloc[-1])
    print(f"  Candles loaded:  {len(df)}")
    print(f"  First bar:       {df['timestamp'].iloc[0]}")
    print(f"  Last bar:        {df['timestamp'].iloc[-1]}")
    print(f"  Current price:   {current_price:.4f}")

    zones = detect_sr_zones(
        df,
        pivot_window=args.pivot_window,
        merge_pct=args.merge_pct,
        min_touches=args.min_touches,
        reference_price=current_price,
    )
    print(f"  Zones detected:  {len(zones)}\n")
    print(_format_zone_table(zones, current_price, limit=args.limit))
    print()
    return 0


def run_universe(args) -> int:
    """Run the detector across the full Krillin watchlist. Useful first-pass."""
    wl_path = Path("markets/krillin_watchlist.yaml")
    if not wl_path.exists():
        print(f"Watchlist not found at {wl_path}", file=sys.stderr)
        return 1
    tickers = yaml.safe_load(wl_path.read_text())["tickers"]
    as_of = _parse_ts(args.as_of) if args.as_of else pd.Timestamp.now(tz="UTC")

    print(f"\nScanning {len(tickers)} tickers at {args.timeframe} as of {as_of}\n")
    print(f"{'ticker':<8}  {'zones':>5}  {'supp>0.5':>8}  {'res>0.5':>8}  {'current':>12}")
    print("-" * 60)

    for sym in tickers:
        start = as_of - timedelta(days=args.lookback_days)
        try:
            df = get_candles(sym, args.timeframe, start, as_of + timedelta(hours=1))
        except Exception as exc:
            print(f"{sym:<8}  ERROR {exc}")
            continue
        df = _slice_as_of(df, as_of)
        if len(df) < 2 * args.pivot_window + 2:
            print(f"{sym:<8}  insufficient bars ({len(df)})")
            continue
        current_price = float(df["close"].iloc[-1])
        zones = detect_sr_zones(
            df, pivot_window=args.pivot_window,
            merge_pct=args.merge_pct, min_touches=args.min_touches,
            reference_price=current_price,
        )
        supp = sum(1 for z in zones if z.side == "support" and z.strength >= 0.5)
        res = sum(1 for z in zones if z.side == "resistance" and z.strength >= 0.5)
        print(f"{sym:<8}  {len(zones):>5}  {supp:>8}  {res:>8}  {current_price:>12.4f}")
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Output S/R zones for a ticker (or the full watchlist)."
    )
    parser.add_argument("--symbol", help="Single ticker (e.g., ENA). Omit for full-universe scan.")
    parser.add_argument("--timeframe", default="1h", help="15m / 1h / 4h (default 1h)")
    parser.add_argument("--as-of", help="ISO timestamp to evaluate at (default now)")
    parser.add_argument("--lookback-days", type=int, default=180, help="Historical window")
    parser.add_argument("--pivot-window", type=int, default=5, help="Fractal window (bars each side)")
    parser.add_argument("--merge-pct", type=float, default=0.01, help="Cluster merge band (0.01 = 1%)")
    parser.add_argument("--min-touches", type=int, default=2, help="Minimum pivots per zone")
    parser.add_argument("--limit", type=int, default=20, help="Rows in zone table")
    args = parser.parse_args()

    if args.symbol:
        return run(args)
    return run_universe(args)


if __name__ == "__main__":
    sys.exit(main())

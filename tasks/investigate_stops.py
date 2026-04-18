"""
investigate_stops.py
One-off diagnostic: pull per-trade detail on the 5 true positives from the
Phase B validation to understand why stops are eating TPs.

Replays the detector across our universe, matches to pops, and dumps
the full capture lifecycle for each true positive: entry/stop/TPs,
every tier fill with bar index and price, bars held.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from data_cache import get_candles
from setups.base import MarketContext
from setups.divergence import DivergenceReversalDetector
from validation.capture import simulate_capture
from validation.labeler import label_pops
from validation.matcher import match_triggers_to_pops
from validation.report import _scan_ticker


def main():
    logging.basicConfig(level=logging.WARNING)

    tickers = yaml.safe_load(Path("markets/krillin_watchlist.yaml").read_text())["tickers"]
    timeframe = "1h"
    start = pd.Timestamp("2025-04-01", tz="UTC")
    end = pd.Timestamp("2026-04-01", tz="UTC")

    detector = DivergenceReversalDetector()

    all_triggers = []
    all_pops = []
    ticker_candles = {}

    print(f"Scanning {len(tickers)} tickers...", file=sys.stderr)
    for ticker in tickers:
        try:
            candles = get_candles(ticker, timeframe, start, end)
        except Exception as e:
            print(f"  skip {ticker}: {e}", file=sys.stderr)
            continue
        if len(candles) < 310:
            continue
        ticker_candles[ticker] = candles
        pops = label_pops(candles, ticker, threshold_pct=0.20, timeframe="1h", window_hours=72)
        all_pops.extend(pops)
        triggers = _scan_ticker(detector, candles, ticker, timeframe, 300, 500)
        all_triggers.extend(triggers)

    print(f"\nTotal: {len(all_pops)} pops, {len(all_triggers)} triggers\n", file=sys.stderr)

    matches = match_triggers_to_pops(
        all_triggers, all_pops,
        max_lead=pd.Timedelta(hours=24), max_lag=pd.Timedelta(hours=4),
    )
    tps = [m for m in matches if m.outcome == "true_positive"]

    print(f"{'='*90}")
    print(f"  {len(tps)} TRUE POSITIVES — PER-TRADE DIAGNOSTIC")
    print(f"{'='*90}\n")

    for i, m in enumerate(tps, 1):
        trig = m.trigger
        pop = m.pop
        candles = ticker_candles.get(trig.ticker)
        if candles is None:
            continue

        cap = simulate_capture(trig, pop, candles, max_hold_bars=120)

        risk_abs = abs(trig.entry_price - trig.stop_price)
        risk_pct = risk_abs / trig.entry_price
        tp_distances = [(tp.price, (tp.price - trig.entry_price) / trig.entry_price) for tp in trig.tp_ladder]

        print(f"--- #{i}: {trig.ticker} {trig.direction.upper()} @ {trig.timestamp} ---")
        print(f"  Pop: magnitude={pop.magnitude*100:+.1f}% starting {pop.timestamp} peak {pop.peak_timestamp}")
        print(f"  Entry: {trig.entry_price:.4f}")
        print(f"  Stop:  {trig.stop_price:.4f}  (risk: {risk_abs:.4f}, {risk_pct*100:.2f}% of entry)")
        print(f"  TPs:")
        for tp_price, tp_dist in tp_distances:
            print(f"    {tp_price:.4f}  ({tp_dist*100:+.2f}% from entry, R:R {abs(tp_dist)/risk_pct:.1f})")

        print(f"  Confidence: {trig.confidence:.2f}  |  size_fraction: {trig.size_fraction}")
        print(f"  Components: {trig.components}")

        print(f"\n  Simulated lifecycle:")
        print(f"    Exit reason: {cap.exit_reason}")
        print(f"    Bars held:   {cap.bars_held}")
        print(f"    Realized:    {cap.realized_return_pct*100:+.2f}%")
        print(f"    Capture ratio: {cap.capture_ratio*100:+.1f}% of pop magnitude ({pop.magnitude*100:.1f}%)")
        print(f"    Fills:")
        for f in cap.tier_fills:
            print(f"      bar {f.bar_index:5d}  {f.level:8s}  price {f.price:.4f}  frac {f.fraction:.2f}")

        # What's the best a perfect exit would have captured?
        entry_idx_series = candles["timestamp"].apply(lambda t: t == trig.timestamp)
        if entry_idx_series.any():
            entry_idx = int(entry_idx_series.idxmax())
            # Look at what happened in the 120 bars after entry
            future = candles.iloc[entry_idx+1:entry_idx+121]
            if trig.direction == "long":
                max_high = future["high"].max()
                max_gain_pct = (max_high - trig.entry_price) / trig.entry_price
                # How many bars until max was reached?
                bars_to_max = int(future["high"].idxmax() - entry_idx)
                print(f"    Best-case (perfect exit at peak):")
                print(f"      Max high in next 120 bars: {max_high:.4f} (+{max_gain_pct*100:.1f}%)")
                print(f"      Reached at bar +{bars_to_max}")
                # Where was the lowest low BEFORE the max?
                if bars_to_max > 1:
                    window_before_max = future.iloc[:bars_to_max]
                    min_low = window_before_max["low"].min()
                    max_pullback_pct = 1 - min_low / trig.entry_price
                    print(f"      Deepest pullback before peak: {min_low:.4f} ({max_pullback_pct*100:.1f}% below entry)")
            else:  # short
                min_low = future["low"].min()
                max_gain_pct = (trig.entry_price - min_low) / trig.entry_price
                bars_to_max = int(future["low"].idxmin() - entry_idx)
                print(f"    Best-case (perfect exit at trough):")
                print(f"      Min low in next 120 bars: {min_low:.4f} (+{max_gain_pct*100:.1f}% short gain)")
                print(f"      Reached at bar +{bars_to_max}")
                if bars_to_max > 1:
                    window_before_min = future.iloc[:bars_to_max]
                    max_high = window_before_min["high"].max()
                    max_adverse_pct = (max_high - trig.entry_price) / trig.entry_price
                    print(f"      Deepest adverse move before trough: {max_high:.4f} ({max_adverse_pct*100:+.1f}%)")
        print()


if __name__ == "__main__":
    main()

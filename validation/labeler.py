"""
validation/labeler.py
Identify historical "pops" — significant directional moves — as ground truth.

A PopEvent represents: "at bar T on ticker X, price moved >= N% within
W hours in direction D." These events are what we hope the bot's setup
detectors would have fired on before they happened.

Implementation:
  * Scan bars forward. At each candidate start bar, look ahead window_bars
    and find the peak (for long pops) or trough (for short pops).
  * If max forward move meets threshold: record a PopEvent.
  * Skip past the peak before looking for the next event. This prevents
    every bar in a run-up from being separately labeled.
  * If both long AND short moves hit threshold in the window (rare, wild
    swings), label the one with the larger magnitude.

Dedup via skip-past-peak is intentionally simple. It can still produce
adjacent events (e.g., a pop followed by a deep reversal) which is
correct: both are real separate moves that the bot might want to catch
in opposite directions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Direction = Literal["long", "short"]

# Bars-per-hour for each supported timeframe
_BARS_PER_HOUR = {
    "1m": 60,
    "5m": 12,
    "15m": 4,
    "1h": 1,
    "4h": 0.25,
    "1d": 1 / 24,
}


def window_bars_for(timeframe: str, window_hours: float) -> int:
    """Convert a window in hours to bars for the given timeframe."""
    bars_per_hour = _BARS_PER_HOUR.get(timeframe)
    if bars_per_hour is None:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")
    return max(1, int(round(window_hours * bars_per_hour)))


@dataclass(frozen=True)
class PopEvent:
    """A labeled pop.

    timestamp:      the "start" bar — price at this bar is the reference
                    against which the move is measured.
    peak_timestamp: the bar at which the peak (up) or trough (down) was hit.
    magnitude:      fractional move (0.25 = 25%).
    """
    ticker: str
    timestamp: pd.Timestamp
    direction: Direction
    magnitude: float
    peak_timestamp: pd.Timestamp
    start_bar_index: int
    peak_bar_index: int
    threshold: float
    start_price: float
    peak_price: float


def label_pops(
    candles: pd.DataFrame,
    ticker: str,
    *,
    threshold_pct: float = 0.20,
    timeframe: str = "1h",
    window_hours: float = 72.0,
    min_bars: int = 3,
) -> list[PopEvent]:
    """
    Scan `candles` for significant forward moves and return them as PopEvents.

    Args:
        candles:       DataFrame with [timestamp, high, low, close].
                       Timestamp must be UTC.
        ticker:        string label attached to each event (for cross-ticker
                       aggregation in the report).
        threshold_pct: fractional move required (0.20 = 20%).
        timeframe:     used to convert window_hours → window_bars.
        window_hours:  forward-looking window in which the move must occur.
        min_bars:      minimum bars from start to peak — filters out
                       ultra-fast spikes that likely can't be traded.

    Returns: list of PopEvent in chronological order.
    """
    required = {"timestamp", "high", "low", "close"}
    missing = required - set(candles.columns)
    if missing:
        raise ValueError(f"candles missing required columns: {missing}")
    if not (0 < threshold_pct < 10):
        raise ValueError(f"threshold_pct must be in (0, 10), got {threshold_pct}")

    window_bars = window_bars_for(timeframe, window_hours)
    if len(candles) < min_bars + 1:
        return []

    closes = candles["close"].to_numpy(dtype=float)
    highs = candles["high"].to_numpy(dtype=float)
    lows = candles["low"].to_numpy(dtype=float)
    # Keep timestamps as a tz-aware pandas Series — converting via to_numpy
    # strips tz and then re-localizing errors in pandas 3.0+
    timestamps = pd.to_datetime(candles["timestamp"], utc=True).reset_index(drop=True)

    events: list[PopEvent] = []
    i = 0
    n = len(closes)
    while i < n - min_bars:
        end = min(i + 1 + window_bars, n)
        fwd_high = highs[i + 1 : end]
        fwd_low = lows[i + 1 : end]
        if len(fwd_high) < min_bars:
            break

        start_price = closes[i]
        if start_price <= 0:
            i += 1
            continue

        peak_hi_rel = int(np.argmax(fwd_high))
        peak_lo_rel = int(np.argmin(fwd_low))
        up_move = float(fwd_high[peak_hi_rel]) / start_price - 1.0
        down_move = 1.0 - float(fwd_low[peak_lo_rel]) / start_price

        long_qualified = up_move >= threshold_pct and peak_hi_rel + 1 >= min_bars
        short_qualified = down_move >= threshold_pct and peak_lo_rel + 1 >= min_bars

        if long_qualified and (up_move >= down_move or not short_qualified):
            peak_idx = i + 1 + peak_hi_rel
            events.append(PopEvent(
                ticker=ticker,
                timestamp=timestamps.iloc[i],
                direction="long",
                magnitude=up_move,
                peak_timestamp=timestamps.iloc[peak_idx],
                start_bar_index=i,
                peak_bar_index=peak_idx,
                threshold=threshold_pct,
                start_price=float(start_price),
                peak_price=float(highs[peak_idx]),
            ))
            i = peak_idx + 1
            continue

        if short_qualified:
            peak_idx = i + 1 + peak_lo_rel
            events.append(PopEvent(
                ticker=ticker,
                timestamp=timestamps.iloc[i],
                direction="short",
                magnitude=down_move,
                peak_timestamp=timestamps.iloc[peak_idx],
                start_bar_index=i,
                peak_bar_index=peak_idx,
                threshold=threshold_pct,
                start_price=float(start_price),
                peak_price=float(lows[peak_idx]),
            ))
            i = peak_idx + 1
            continue

        i += 1

    return events


def pop_stats(events: list[PopEvent]) -> dict:
    """Summary stats for a list of pops."""
    if not events:
        return {
            "total": 0, "long": 0, "short": 0,
            "avg_magnitude": 0.0, "median_magnitude": 0.0,
            "median_time_to_peak_bars": 0.0,
        }
    mags = np.array([e.magnitude for e in events])
    times = np.array([e.peak_bar_index - e.start_bar_index for e in events])
    longs = sum(1 for e in events if e.direction == "long")
    shorts = sum(1 for e in events if e.direction == "short")
    return {
        "total": len(events),
        "long": longs,
        "short": shorts,
        "avg_magnitude": float(mags.mean()),
        "median_magnitude": float(np.median(mags)),
        "median_time_to_peak_bars": float(np.median(times)),
    }

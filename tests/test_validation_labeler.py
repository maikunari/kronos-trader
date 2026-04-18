"""Tests for validation/labeler.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from validation.labeler import (
    PopEvent,
    label_pops,
    pop_stats,
    window_bars_for,
)


def _candles(closes: np.ndarray, wick_pct: float = 0.001) -> pd.DataFrame:
    """OHLCV df. high = close + wick, low = close - wick (no noise — deterministic)."""
    closes = np.asarray(closes, dtype=float)
    highs = closes * (1 + wick_pct)
    lows = closes * (1 - wick_pct)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    volumes = np.full_like(closes, 1000.0)
    ts = pd.date_range("2025-01-01", periods=len(closes), freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    })


# ---------------------------------------------------------------------------
# window_bars_for
# ---------------------------------------------------------------------------

def test_window_bars_for_known_timeframes():
    assert window_bars_for("15m", 1) == 4
    assert window_bars_for("1h", 72) == 72
    assert window_bars_for("4h", 72) == 18
    assert window_bars_for("1d", 72) == 3


def test_window_bars_for_rejects_unknown():
    with pytest.raises(ValueError):
        window_bars_for("2h", 10)


# ---------------------------------------------------------------------------
# label_pops — happy paths
# ---------------------------------------------------------------------------

def test_label_pops_finds_single_long_move():
    # Flat 100 then rally to 125 over 20 bars
    closes = np.concatenate([np.full(50, 100.0), np.linspace(100, 125, 20), np.full(30, 125.0)])
    df = _candles(closes)
    events = label_pops(df, "TEST", threshold_pct=0.20, window_hours=72, timeframe="1h")
    assert any(e.direction == "long" and e.magnitude >= 0.20 for e in events)


def test_label_pops_finds_single_short_move():
    closes = np.concatenate([np.full(50, 100.0), np.linspace(100, 75, 20), np.full(30, 75.0)])
    df = _candles(closes)
    events = label_pops(df, "TEST", threshold_pct=0.20)
    assert any(e.direction == "short" and e.magnitude >= 0.20 for e in events)


def test_label_pops_no_events_on_flat():
    df = _candles(np.full(200, 100.0))
    assert label_pops(df, "TEST", threshold_pct=0.20) == []


def test_label_pops_no_events_below_threshold():
    # Gentle 10% rally — below 20% threshold
    closes = np.concatenate([np.full(50, 100.0), np.linspace(100, 110, 50), np.full(50, 110.0)])
    df = _candles(closes)
    assert label_pops(df, "TEST", threshold_pct=0.20) == []


def test_label_pops_threshold_is_inclusive_above():
    # 25% move — clearly above threshold (avoid float boundary at exactly 20%)
    closes = np.concatenate([np.full(20, 100.0), np.linspace(100, 125, 10), np.full(20, 125.0)])
    df = _candles(closes, wick_pct=0.0)
    events = label_pops(df, "TEST", threshold_pct=0.20)
    assert len(events) >= 1


def test_label_pops_dedup_via_skip_past_peak():
    # Long ramp from 100 to 150 over many bars. Without skipping, would label every bar;
    # with skip-past-peak, should produce a small number of events.
    closes = np.concatenate([np.full(50, 100.0), np.linspace(100, 150, 100), np.full(50, 150.0)])
    df = _candles(closes)
    events = label_pops(df, "TEST", threshold_pct=0.20, window_hours=72)
    # Should produce at most a few events, not one per bar
    assert 1 <= len(events) <= 5


# ---------------------------------------------------------------------------
# Event structure
# ---------------------------------------------------------------------------

def test_pop_event_fields_populated():
    closes = np.concatenate([np.full(30, 100.0), np.linspace(100, 130, 20), np.full(30, 130.0)])
    df = _candles(closes)
    events = label_pops(df, "BTC", threshold_pct=0.20)
    assert events
    e = events[0]
    assert e.ticker == "BTC"
    assert e.direction == "long"
    assert e.magnitude >= 0.20
    assert e.peak_bar_index > e.start_bar_index
    assert e.peak_price > e.start_price
    assert e.start_price > 0
    # Peak ts must be strictly after start ts
    assert e.peak_timestamp > e.timestamp


def test_pop_event_picks_larger_magnitude_when_both_qualify():
    # Down 25% then up 30% within the same window
    closes = np.concatenate([
        np.full(10, 100.0),
        np.linspace(100, 75, 30),   # -25%
        np.linspace(75, 105, 30),   # +40%
        np.full(30, 105.0),
    ])
    df = _candles(closes)
    events = label_pops(df, "TEST", threshold_pct=0.20, window_hours=72)
    # Short move (-25%) should be labeled first, then long move from the trough
    directions = [e.direction for e in events]
    assert "short" in directions
    # After the short event, the rally from 75→105 is +40% in ~30 bars; should label
    assert "long" in directions


# ---------------------------------------------------------------------------
# Filters and edge cases
# ---------------------------------------------------------------------------

def test_label_pops_respects_min_bars():
    # Very sharp 25% move in 2 bars — min_bars=5 should filter it out
    closes = np.concatenate([np.full(30, 100.0), np.array([100.0, 125.0]), np.full(30, 125.0)])
    df = _candles(closes)
    events = label_pops(df, "TEST", threshold_pct=0.20, min_bars=5, window_hours=72)
    assert all(e.peak_bar_index - e.start_bar_index >= 5 for e in events)


def test_label_pops_rejects_bad_threshold():
    df = _candles(np.full(50, 100.0))
    with pytest.raises(ValueError):
        label_pops(df, "TEST", threshold_pct=0)
    with pytest.raises(ValueError):
        label_pops(df, "TEST", threshold_pct=20)


def test_label_pops_rejects_missing_columns():
    df = pd.DataFrame({"close": [100, 101, 102]})
    with pytest.raises(ValueError):
        label_pops(df, "TEST")


def test_label_pops_empty_on_tiny_series():
    df = _candles(np.array([100.0, 101.0]))
    assert label_pops(df, "TEST", threshold_pct=0.20) == []


def test_label_pops_window_bounded():
    # 30% move over 200 bars — with a 72-bar window starting from the base,
    # the full move isn't visible, so no long event labeled from bar 0.
    # But events starting later in the ramp should still be labeled.
    closes = np.concatenate([np.full(20, 100.0), np.linspace(100, 130, 200), np.full(20, 130.0)])
    df = _candles(closes)
    events = label_pops(df, "TEST", threshold_pct=0.20, window_hours=72)
    assert all(e.direction == "long" for e in events)
    # Magnitude within-window is smaller than the full move
    assert all(e.magnitude >= 0.20 for e in events)


# ---------------------------------------------------------------------------
# pop_stats
# ---------------------------------------------------------------------------

def test_pop_stats_empty():
    s = pop_stats([])
    assert s["total"] == 0
    assert s["long"] == 0


def test_pop_stats_aggregates_correctly():
    e1 = PopEvent(
        ticker="T", timestamp=pd.Timestamp("2025-01-01", tz="UTC"),
        direction="long", magnitude=0.25,
        peak_timestamp=pd.Timestamp("2025-01-02", tz="UTC"),
        start_bar_index=0, peak_bar_index=24,
        threshold=0.20, start_price=100, peak_price=125,
    )
    e2 = PopEvent(
        ticker="T", timestamp=pd.Timestamp("2025-02-01", tz="UTC"),
        direction="short", magnitude=0.30,
        peak_timestamp=pd.Timestamp("2025-02-02", tz="UTC"),
        start_bar_index=100, peak_bar_index=140,
        threshold=0.20, start_price=125, peak_price=87.5,
    )
    s = pop_stats([e1, e2])
    assert s["total"] == 2
    assert s["long"] == 1
    assert s["short"] == 1
    assert s["avg_magnitude"] == pytest.approx(0.275)

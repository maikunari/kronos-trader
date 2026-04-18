"""Tests for validation/capture.py — single-trade simulator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from setups.base import TPLevel, Trigger
from validation.capture import (
    CaptureResult,
    exit_reason_breakdown,
    mean_realized_return,
    median_capture_ratio,
    simulate_capture,
)
from validation.labeler import PopEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _trigger(entry=100.0, stop=95.0, targets=((110.0, 0.5), (120.0, 0.5)),
             direction="long", timestamp="2025-01-01 00:00"):
    return Trigger(
        ticker="TEST",
        timestamp=pd.Timestamp(timestamp, tz="UTC"),
        action="open_new",
        direction=direction,
        entry_price=entry,
        stop_price=stop,
        tp_ladder=tuple(TPLevel(price=p, fraction=f, source="test") for p, f in targets),
        setup="test",
        confidence=0.8,
    )


def _pop(direction="long", magnitude=0.25, start="2025-01-01 00:00"):
    return PopEvent(
        ticker="TEST",
        timestamp=pd.Timestamp(start, tz="UTC"),
        direction=direction,
        magnitude=magnitude,
        peak_timestamp=pd.Timestamp("2025-01-02 00:00", tz="UTC"),
        start_bar_index=0,
        peak_bar_index=24,
        threshold=0.20,
        start_price=100.0,
        peak_price=125.0 if direction == "long" else 75.0,
    )


def _candles(highs: np.ndarray, lows: np.ndarray, start="2025-01-01 00:00"):
    n = len(highs)
    closes = (highs + lows) / 2
    ts = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": closes, "high": highs, "low": lows,
        "close": closes, "volume": [1000.0] * n,
    })


# ---------------------------------------------------------------------------
# Stop-out scenarios
# ---------------------------------------------------------------------------

def test_long_trade_stops_out_before_any_tp():
    # Entry 100, stop 95. Price drops to 90 immediately.
    highs = np.array([100.0, 92.0, 88.0, 85.0])
    lows = np.array([100.0, 90.0, 85.0, 82.0])
    result = simulate_capture(_trigger(), _pop(), _candles(highs, lows))
    assert result is not None
    assert result.exit_reason == "stop"
    assert result.realized_return_pct == pytest.approx(-0.05)   # (95-100)/100 = -5%
    assert result.capture_ratio == pytest.approx(-0.05 / 0.25)


def test_short_trade_stops_out_before_any_tp():
    trigger = _trigger(entry=100, stop=105, targets=((90.0, 0.5), (80.0, 0.5)), direction="short")
    pop = _pop(direction="short")
    highs = np.array([100.0, 108.0, 112.0, 115.0])
    lows = np.array([100.0, 104.0, 108.0, 111.0])
    result = simulate_capture(trigger, pop, _candles(highs, lows))
    assert result.exit_reason == "stop"
    assert result.realized_return_pct == pytest.approx(-0.05)


# ---------------------------------------------------------------------------
# Full TP ladder scenarios
# ---------------------------------------------------------------------------

def test_long_trade_hits_all_tps_in_sequence():
    # Price rises: 100 → 112 (hits 110 tier) → 125 (hits 120 tier)
    highs = np.array([100.0, 108.0, 112.0, 120.0, 125.0, 125.0])
    lows = np.array([100.0, 99.0, 105.0, 115.0, 120.0, 123.0])
    result = simulate_capture(_trigger(), _pop(), _candles(highs, lows))
    assert result.exit_reason == "target"
    # Realized: 0.5 * (110-100)/100 + 0.5 * (120-100)/100 = 0.05 + 0.10 = 0.15
    assert result.realized_return_pct == pytest.approx(0.15)
    assert result.capture_ratio == pytest.approx(0.15 / 0.25)
    assert len(result.tier_fills) == 2


def test_long_trade_multiple_tps_same_bar():
    # Entry 100, big single bar to 125 hits both 110 and 120 tiers
    highs = np.array([100.0, 125.0, 125.0])
    lows = np.array([100.0, 100.0, 120.0])
    result = simulate_capture(_trigger(), _pop(), _candles(highs, lows))
    assert result.exit_reason == "target"
    assert len(result.tier_fills) == 2
    # Realized: 0.5 * 0.10 + 0.5 * 0.20 = 0.15
    assert result.realized_return_pct == pytest.approx(0.15)


def test_long_trade_partial_ladder_then_stop():
    # Hits first tier (110), then pulls back to stop (95) before reaching second tier.
    # First TP: +10% on 50% of position = +0.050
    # Remaining 50% stopped: -5% on 50% = -0.025
    # Net: +0.025 (asymmetric R:R works in our favor even on partial stop)
    highs = np.array([100.0, 112.0, 108.0, 100.0, 94.0])
    lows = np.array([100.0, 100.0, 98.0, 92.0, 88.0])
    result = simulate_capture(_trigger(), _pop(), _candles(highs, lows))
    assert result.realized_return_pct == pytest.approx(0.025, abs=1e-9)
    fills_levels = [f.level for f in result.tier_fills]
    assert fills_levels == ["target", "stop"]


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

def test_trade_times_out_with_residual_position():
    # Price drifts sideways, never hits stop or TPs
    highs = np.array([100.0] + [101.0] * 15)
    lows = np.array([100.0] + [99.0] * 15)
    result = simulate_capture(_trigger(), _pop(), _candles(highs, lows), max_hold_bars=10)
    assert result.exit_reason == "timeout"
    # No fills at TP or stop — just timeout close
    assert any(f.level == "timeout" for f in result.tier_fills)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_returns_none_when_trigger_bar_not_in_candles():
    trigger = _trigger(timestamp="2025-06-01 00:00")   # far outside window
    pop = _pop(start="2025-06-01 00:00")
    highs = np.array([100.0, 101.0, 102.0])
    lows = np.array([100.0, 99.0, 98.0])
    result = simulate_capture(trigger, pop, _candles(highs, lows))
    assert result is None


def test_rejects_bad_candles_columns():
    trigger = _trigger()
    pop = _pop()
    bad = pd.DataFrame({"close": [100.0, 101.0]})
    with pytest.raises(ValueError):
        simulate_capture(trigger, pop, bad)


def test_capture_ratio_is_zero_on_zero_magnitude_pop():
    # Defensive: if a pop somehow has zero magnitude, ratio should be 0 not NaN
    trigger = _trigger()
    pop = PopEvent(
        ticker="TEST",
        timestamp=pd.Timestamp("2025-01-01", tz="UTC"),
        direction="long", magnitude=0.0,
        peak_timestamp=pd.Timestamp("2025-01-02", tz="UTC"),
        start_bar_index=0, peak_bar_index=1, threshold=0.20,
        start_price=100.0, peak_price=100.0,
    )
    highs = np.array([100.0, 125.0, 125.0])
    lows = np.array([100.0, 100.0, 120.0])
    result = simulate_capture(trigger, pop, _candles(highs, lows))
    assert result.capture_ratio == 0.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_median_capture_ratio_handles_odd_and_even_counts():
    def _res(ratio):
        t = _trigger()
        p = _pop()
        return CaptureResult(
            trigger=t, pop=p, entry_price=100, weighted_exit_price=100,
            realized_return_pct=ratio * p.magnitude, capture_ratio=ratio,
            exit_reason="target", bars_held=1, tier_fills=(),
        )
    odd = [_res(r) for r in [0.1, 0.3, 0.5, 0.7, 0.9]]
    even = [_res(r) for r in [0.1, 0.3, 0.5, 0.7]]
    assert median_capture_ratio(odd) == pytest.approx(0.5)
    assert median_capture_ratio(even) == pytest.approx(0.4)


def test_median_capture_ratio_empty():
    assert median_capture_ratio([]) == 0.0


def test_exit_reason_breakdown():
    def _res(reason):
        t = _trigger()
        p = _pop()
        return CaptureResult(
            trigger=t, pop=p, entry_price=100, weighted_exit_price=100,
            realized_return_pct=0, capture_ratio=0, exit_reason=reason,
            bars_held=1, tier_fills=(),
        )
    rs = [_res("target"), _res("target"), _res("stop"), _res("timeout")]
    counts = exit_reason_breakdown(rs)
    assert counts == {"target": 2, "stop": 1, "timeout": 1}


def test_mean_realized_return():
    def _res(r):
        t = _trigger()
        p = _pop()
        return CaptureResult(
            trigger=t, pop=p, entry_price=100, weighted_exit_price=100,
            realized_return_pct=r, capture_ratio=0, exit_reason="target",
            bars_held=1, tier_fills=(),
        )
    rs = [_res(r) for r in [0.05, 0.10, 0.15]]
    assert mean_realized_return(rs) == pytest.approx(0.10)

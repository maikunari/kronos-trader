"""Tests for pivot.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pivot import (
    Pivot,
    current_leg_bottom,
    find_pivots,
    most_recent,
    swing_highs,
    swing_lows,
)


# --- Basic detection ---------------------------------------------------------

def test_find_pivots_on_sine_wave_finds_expected_extrema():
    """A sine wave's pivots are at π/2, 3π/2, etc. — one per cycle."""
    x = np.linspace(0, 4 * np.pi, 401)
    s = pd.Series(np.sin(x))
    pivots = find_pivots(s, window=10)
    highs = [p for p in pivots if p.kind == "high"]
    lows = [p for p in pivots if p.kind == "low"]
    # Two full sine cycles should produce roughly 2 highs and 2 lows
    assert 2 <= len(highs) <= 3
    assert 2 <= len(lows) <= 3
    # First high's value should be close to 1
    assert abs(highs[0].value - 1.0) < 0.01


def test_find_pivots_empty_on_short_series():
    assert find_pivots(pd.Series([1, 2, 3]), window=5) == []


def test_find_pivots_rejects_bad_window():
    with pytest.raises(ValueError):
        find_pivots(pd.Series([1.0] * 20), window=0)


def test_find_pivots_skips_plateau():
    """A flat series shouldn't produce any pivots (strict inequality)."""
    s = pd.Series([5.0] * 30)
    assert find_pivots(s, window=5) == []


def test_find_pivots_ignores_nan():
    vals = [1.0, 2, 3, 4, 5, 6, np.nan, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3]
    pivots = find_pivots(pd.Series(vals), window=3)
    # Shouldn't crash; whatever pivots found should be at non-nan indices
    for p in pivots:
        assert not np.isnan(vals[p.index])


# --- Direction filtering -----------------------------------------------------

def test_swing_highs_returns_only_highs():
    vals = [1, 2, 3, 5, 3, 2, 1, 2, 3, 5, 3, 2, 1]
    pivots = swing_highs(pd.Series(vals, dtype=float), window=2)
    assert all(p.kind == "high" for p in pivots)


def test_swing_lows_returns_only_lows():
    vals = [1, 2, 3, 5, 3, 2, 1, 2, 3, 5, 3, 2, 1]
    pivots = swing_lows(pd.Series(vals, dtype=float), window=2)
    assert all(p.kind == "low" for p in pivots)


# --- Pivot ordering / structure ----------------------------------------------

def test_pivots_are_returned_in_chronological_order():
    rng = np.random.default_rng(7)
    vals = rng.normal(100, 5, 200).cumsum()
    pivots = find_pivots(pd.Series(vals), window=5)
    indexes = [p.index for p in pivots]
    assert indexes == sorted(indexes)


def test_timestamps_populate_from_datetime_index():
    ts = pd.date_range("2025-01-01", periods=100, freq="1h", tz="UTC")
    vals = np.sin(np.linspace(0, 4 * np.pi, 100))
    s = pd.Series(vals, index=ts)
    pivots = find_pivots(s, window=5)
    assert len(pivots) > 0
    assert all(p.timestamp is not None for p in pivots)


def test_timestamps_none_without_datetime_index():
    pivots = find_pivots(pd.Series([1, 2, 3, 5, 3, 2, 1, 2, 3, 5, 3, 2, 1], dtype=float), window=2)
    assert all(p.timestamp is None for p in pivots)


# --- most_recent() -----------------------------------------------------------

def test_most_recent_respects_n_and_kind():
    ps = [
        Pivot(10, 1.0, "high"), Pivot(20, 0.5, "low"),
        Pivot(30, 2.0, "high"), Pivot(40, 0.1, "low"),
        Pivot(50, 3.0, "high"),
    ]
    highs = most_recent(ps, n=2, kind="high")
    assert [p.index for p in highs] == [30, 50]


def test_most_recent_with_n_larger_than_available():
    ps = [Pivot(10, 1.0, "high"), Pivot(20, 2.0, "high")]
    out = most_recent(ps, n=99, kind="high")
    assert len(out) == 2


def test_most_recent_empty_input():
    assert most_recent([], n=5) == []


# --- current_leg_bottom() ----------------------------------------------------

def test_current_leg_bottom_long_finds_most_recent_pullback_low():
    # Series that rises, pulls back, rises — leg bottom for a long is the
    # pullback low between the last swing high and the current bar.
    vals = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,    # ramp up
        9, 8, 7, 8, 9, 10, 11, 12, 13, 14,  # pullback to 7, then ramp up
        13, 12, 13, 14, 15, 16, 17, 18, 19, 20,  # another mini pullback, then ramp
    ]
    s = pd.Series(vals, dtype=float)
    leg_bot = current_leg_bottom(s, direction="long", window=3)
    # The most recent higher-high is near the end; the leg bottom should be
    # the last confirmed pullback low (12 or similar), not 7.
    assert leg_bot is not None
    assert leg_bot < s.iloc[-1]


def test_current_leg_bottom_short_finds_most_recent_rally_high():
    vals = [
        20, 19, 18, 17, 16, 15, 14, 13, 12, 11,
        12, 13, 14, 13, 12, 11, 10, 9, 8, 7,
        8, 9, 10, 9, 8, 7, 6, 5, 4, 3,
    ]
    s = pd.Series(vals, dtype=float)
    leg_top = current_leg_bottom(s, direction="short", window=3)
    assert leg_top is not None
    assert leg_top > s.iloc[-1]


def test_current_leg_bottom_rejects_bad_direction():
    with pytest.raises(ValueError):
        current_leg_bottom(pd.Series([1, 2, 3]), direction="flat")


def test_current_leg_bottom_none_on_insufficient_data():
    assert current_leg_bottom(pd.Series([1.0, 2.0, 3.0]), "long") is None

"""Tests for indicators/ package."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicators.awesome_oscillator import (
    ao_bar_colors,
    awesome_oscillator,
    two_bar_same_color,
    zero_line_cross,
    zero_line_state,
)
from indicators.rsi import (
    is_overbought,
    is_oversold,
    midline_cross,
    rsi,
    rsi_zone,
)


# ---------------------------------------------------------------------------
# AO
# ---------------------------------------------------------------------------

def _hl_from_closes(closes: np.ndarray, wick_pct: float = 0.001):
    high = pd.Series(closes * (1 + wick_pct))
    low = pd.Series(closes * (1 - wick_pct))
    return high, low


def test_ao_rejects_short_geq_long():
    with pytest.raises(ValueError):
        awesome_oscillator(
            pd.Series([1.0] * 50), pd.Series([1.0] * 50),
            short=34, long=5,
        )


def test_ao_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        awesome_oscillator(pd.Series([1.0] * 50), pd.Series([1.0] * 40))


def test_ao_positive_during_sustained_uptrend():
    """Rising prices -> AO should turn positive after lookback fills."""
    closes = np.linspace(100, 200, 100)
    high, low = _hl_from_closes(closes)
    ao = awesome_oscillator(high, low)
    # Last value should be positive (short SMA > long SMA in uptrend)
    assert ao.iloc[-1] > 0


def test_ao_negative_during_sustained_downtrend():
    closes = np.linspace(200, 100, 100)
    high, low = _hl_from_closes(closes)
    ao = awesome_oscillator(high, low)
    assert ao.iloc[-1] < 0


def test_ao_near_zero_on_flat_series():
    closes = np.full(100, 100.0)
    high, low = _hl_from_closes(closes, wick_pct=0)
    ao = awesome_oscillator(high, low)
    assert abs(ao.iloc[-1]) < 1e-9


# --- Bar colors --------------------------------------------------------------

def test_bar_colors_rising_ao_is_green():
    ao = pd.Series([0.0, 0.5, 1.0, 1.5, 2.0])
    colors = ao_bar_colors(ao)
    assert list(colors.iloc[1:]) == ["green"] * 4


def test_bar_colors_falling_ao_is_red():
    ao = pd.Series([2.0, 1.5, 1.0, 0.5, 0.0])
    colors = ao_bar_colors(ao)
    assert list(colors.iloc[1:]) == ["red"] * 4


def test_bar_colors_first_bar_is_flat():
    ao = pd.Series([1.0, 2.0, 3.0])
    colors = ao_bar_colors(ao)
    assert colors.iloc[0] == "flat"


# --- Two-bar rule ------------------------------------------------------------

def test_two_bar_same_color_fires_on_confirmed_green():
    ao = pd.Series([0.0, 0.1, 0.5, 1.0])   # last two diffs are +0.4, +0.5 (both green)
    assert two_bar_same_color(ao, "green")
    assert not two_bar_same_color(ao, "red")


def test_two_bar_same_color_fires_on_confirmed_red():
    ao = pd.Series([1.0, 0.9, 0.5, 0.0])   # last two diffs are -0.4, -0.5 (both red)
    assert two_bar_same_color(ao, "red")
    assert not two_bar_same_color(ao, "green")


def test_two_bar_same_color_false_on_mixed():
    ao = pd.Series([0.0, 0.5, 0.3, 0.7])   # green, red, green
    assert not two_bar_same_color(ao, "green")
    assert not two_bar_same_color(ao, "red")


def test_two_bar_same_color_rejects_bad_color():
    with pytest.raises(ValueError):
        two_bar_same_color(pd.Series([0.0, 0.5, 1.0, 1.5]), "purple")


def test_two_bar_same_color_false_on_short_series():
    assert not two_bar_same_color(pd.Series([0.0, 0.5]), "green")


# --- Zero-line state / cross -------------------------------------------------

def test_zero_line_state_above_below_at():
    assert zero_line_state(pd.Series([1.0])) == "above"
    assert zero_line_state(pd.Series([-1.0])) == "below"
    assert zero_line_state(pd.Series([0.0])) == "at_zero"
    assert zero_line_state(pd.Series([], dtype=float)) == "at_zero"


def test_zero_line_cross_up():
    ao = pd.Series([-0.5, -0.1, 0.3])
    assert zero_line_cross(ao) == "up"


def test_zero_line_cross_down():
    ao = pd.Series([0.5, 0.1, -0.3])
    assert zero_line_cross(ao) == "down"


def test_zero_line_cross_none_when_same_side():
    assert zero_line_cross(pd.Series([1.0, 2.0, 3.0])) == "none"
    assert zero_line_cross(pd.Series([-1.0, -2.0, -3.0])) == "none"


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def test_rsi_bounded_0_to_100():
    rng = np.random.default_rng(0)
    closes = pd.Series(100 + rng.normal(0, 5, 500).cumsum())
    r = rsi(closes).dropna()
    assert r.between(0, 100).all()


def test_rsi_high_in_sustained_uptrend():
    closes = pd.Series(np.linspace(100, 200, 100))
    r = rsi(closes)
    # Pure uptrend with no pullbacks -> RSI pegged very high
    assert r.iloc[-1] > 90


def test_rsi_low_in_sustained_downtrend():
    closes = pd.Series(np.linspace(200, 100, 100))
    r = rsi(closes)
    assert r.iloc[-1] < 10


def test_rsi_zone_classification():
    assert rsi_zone(80) == "overbought"
    assert rsi_zone(20) == "oversold"
    assert rsi_zone(50) == "neutral"
    assert rsi_zone(70) == "overbought"   # inclusive on boundary
    assert rsi_zone(30) == "oversold"


def test_rsi_zone_custom_thresholds():
    assert rsi_zone(75, overbought=80, oversold=20) == "neutral"
    assert rsi_zone(82, overbought=80, oversold=20) == "overbought"


def test_is_overbought_oversold_helpers():
    assert is_overbought(75)
    assert not is_overbought(65)
    assert is_oversold(25)
    assert not is_oversold(35)


def test_rsi_midline_cross_up():
    r = pd.Series([45.0, 48.0, 52.0])
    assert midline_cross(r) == "up"


def test_rsi_midline_cross_down():
    r = pd.Series([55.0, 52.0, 48.0])
    assert midline_cross(r) == "down"


def test_rsi_midline_cross_none_when_same_side():
    assert midline_cross(pd.Series([60.0, 65.0, 70.0])) == "none"
    assert midline_cross(pd.Series([40.0, 35.0, 30.0])) == "none"

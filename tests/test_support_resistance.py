"""Tests for support_resistance.py core detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from support_resistance import (
    SRZone,
    TouchEvent,
    _cluster_by_price,
    age_factor,
    confirmed_breakout,
    detect_sr_zones,
    is_close_above,
    is_close_below,
    nearest_in_direction,
    touch_strength,
    volume_weight,
    zones_in_direction,
)


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------

def test_touch_strength_peaks_at_3_and_decays():
    """CBS's sweet-spot rule. This is the counter-intuitive design decision."""
    assert touch_strength(0) == 0.0
    assert touch_strength(1) == 0.0
    assert touch_strength(2) == pytest.approx(0.6)
    assert touch_strength(3) == pytest.approx(1.0)     # peak
    assert touch_strength(4) == pytest.approx(0.85)    # decaying
    assert touch_strength(5) == pytest.approx(0.7)
    assert touch_strength(10) >= 0.3                   # floor
    assert touch_strength(100) >= 0.3


def test_touch_strength_3_is_strictly_greater_than_10():
    """Load-bearing invariant: more touches does NOT mean stronger after 3."""
    assert touch_strength(3) > touch_strength(5)
    assert touch_strength(3) > touch_strength(10)
    assert touch_strength(3) > touch_strength(20)


def test_age_factor_deadband_and_decay():
    assert age_factor(last_touch_bar=100, current_bar=100) == 1.0
    assert age_factor(last_touch_bar=0, current_bar=500, start_decay_age=500) == 1.0
    assert age_factor(last_touch_bar=0, current_bar=1250,
                      start_decay_age=500, full_decay_age=2000) == pytest.approx(0.5)
    assert age_factor(last_touch_bar=0, current_bar=2500,
                      start_decay_age=500, full_decay_age=2000) == 0.0


def test_volume_weight_respects_bounds():
    median_vol = 1000.0
    low_vol_touches = [TouchEvent(0, 100.0, 100.0, "high")]    # 10% of median
    high_vol_touches = [TouchEvent(0, 100.0, 5000.0, "high")]  # 5x median
    # Low-volume touches clipped to 0.5
    assert volume_weight(low_vol_touches, median_vol) == pytest.approx(0.5)
    # High-volume touches clipped to 1.5
    assert volume_weight(high_vol_touches, median_vol) == pytest.approx(1.5)


def test_volume_weight_no_data_returns_neutral():
    assert volume_weight([], 1000.0) == 1.0
    assert volume_weight([TouchEvent(0, 100.0, 100.0, "high")], 0) == 1.0


# ---------------------------------------------------------------------------
# Cluster helper
# ---------------------------------------------------------------------------

def _ev(price, bar=0, volume=1000.0, kind="high"):
    return TouchEvent(bar_index=bar, price=price, volume=volume, pivot_kind=kind)


def test_cluster_by_price_merges_nearby():
    events = [_ev(100.0), _ev(100.4), _ev(100.6), _ev(110.0), _ev(110.2)]
    clusters = _cluster_by_price(events, merge_pct=0.01)   # 1% band
    assert len(clusters) == 2
    assert len(clusters[0]) == 3    # near 100
    assert len(clusters[1]) == 2    # near 110


def test_cluster_by_price_splits_when_too_far():
    events = [_ev(100.0), _ev(105.0)]
    clusters = _cluster_by_price(events, merge_pct=0.01)
    assert len(clusters) == 2


def test_cluster_by_price_empty():
    assert _cluster_by_price([], 0.01) == []


# ---------------------------------------------------------------------------
# detect_sr_zones — main API
# ---------------------------------------------------------------------------

def _candles_with_sr(
    n_bars: int = 200,
    low_level: float = 100.0,
    high_level: float = 110.0,
    cycles: int = 6,
    noise_pct: float = 0.002,
):
    """Synthetic OHLCV: sine wave between two levels + small noise.

    Peaks cluster at `high_level`, troughs at `low_level`. Provides clean
    swing highs and swing lows for the S/R detector to find.
    """
    rng = np.random.default_rng(42)
    mid = (low_level + high_level) / 2
    half_range = (high_level - low_level) / 2
    x = np.linspace(0, cycles * 2 * np.pi, n_bars)
    closes = mid + half_range * np.sin(x) + rng.normal(0, mid * noise_pct, n_bars)
    highs = closes * (1 + np.abs(rng.normal(0, noise_pct, n_bars)))
    lows = closes * (1 - np.abs(rng.normal(0, noise_pct, n_bars)))
    opens = np.concatenate([[closes[0]], closes[:-1]])
    volumes = rng.uniform(800, 1200, n_bars)
    ts = pd.date_range("2025-01-01", periods=n_bars, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    })


def test_detect_sr_zones_requires_close_and_volume_columns():
    df = pd.DataFrame({"close": [100, 101, 102]})
    with pytest.raises(ValueError):
        detect_sr_zones(df)


def test_detect_sr_zones_on_short_series_returns_empty():
    df = pd.DataFrame({"close": [100.0] * 5, "volume": [1000.0] * 5})
    assert detect_sr_zones(df, pivot_window=5) == []


def test_detect_sr_zones_identifies_two_levels_in_oscillating_series():
    """A zigzag between 100 and 110 should produce zones at both levels."""
    df = _candles_with_sr(n_bars=200, low_level=100.0, high_level=110.0)
    zones = detect_sr_zones(df, pivot_window=3, merge_pct=0.015, min_touches=2)
    # Should find at least two distinct zones
    assert len(zones) >= 2
    # One near 100, one near 110
    mids = [z.price_mid for z in zones]
    assert any(abs(m - 100) < 2 for m in mids)
    assert any(abs(m - 110) < 2 for m in mids)


def test_detect_sr_zones_classifies_side_relative_to_current_price():
    df = _candles_with_sr(n_bars=200, low_level=100.0, high_level=110.0)
    # Pretend current price is 105 (middle of range)
    zones = detect_sr_zones(df, pivot_window=3, merge_pct=0.015, reference_price=105.0)
    below = [z for z in zones if z.price_mid < 105]
    above = [z for z in zones if z.price_mid > 105]
    assert all(z.side == "support" for z in below)
    assert all(z.side == "resistance" for z in above)


def test_detect_sr_zones_sorted_by_strength_desc():
    df = _candles_with_sr(n_bars=200)
    zones = detect_sr_zones(df, pivot_window=3, merge_pct=0.015)
    for a, b in zip(zones, zones[1:]):
        assert a.strength >= b.strength


def test_detect_sr_zones_min_touches_filters_out_weak():
    df = _candles_with_sr(n_bars=200)
    zones_min2 = detect_sr_zones(df, pivot_window=3, merge_pct=0.015, min_touches=2)
    zones_min3 = detect_sr_zones(df, pivot_window=3, merge_pct=0.015, min_touches=3)
    # Tightening requirement can only shrink the set
    assert len(zones_min3) <= len(zones_min2)


def test_detect_sr_zones_populates_touch_events_with_volume():
    df = _candles_with_sr(n_bars=200)
    zones = detect_sr_zones(df, pivot_window=3, merge_pct=0.015)
    for z in zones:
        for t in z.touches:
            assert t.volume > 0
            assert t.price == pytest.approx(t.price)


# ---------------------------------------------------------------------------
# Breakout detection
# ---------------------------------------------------------------------------

def _zone(price_mid=100.0, band=1.0, side="resistance", strength=0.8, n_touches=3):
    price_low = price_mid - band
    price_high = price_mid + band
    touches = tuple(TouchEvent(i, price_mid, 1000.0, "high") for i in range(n_touches))
    return SRZone(
        price_mid=price_mid, price_low=price_low, price_high=price_high,
        side=side, strength=strength, touches=touches,
    )


def test_is_close_above_strict():
    z = _zone(100.0, band=1.0)
    assert is_close_above(z, 101.5)
    assert not is_close_above(z, 101.0)    # equal to band is NOT above
    assert not is_close_above(z, 99.0)


def test_is_close_below_strict():
    z = _zone(100.0, band=1.0)
    assert is_close_below(z, 98.5)
    assert not is_close_below(z, 99.0)     # equal to lower band is NOT below
    assert not is_close_below(z, 101.0)


def test_confirmed_breakout_requires_consecutive_closes():
    z = _zone(100.0, band=1.0)
    # Only one close above -> provisional, not confirmed
    closes = pd.Series([99.0, 101.5])
    assert confirmed_breakout(z, closes, required_closes=2) == "none"
    # Two consecutive closes above -> confirmed up
    closes = pd.Series([99.0, 101.5, 102.0])
    assert confirmed_breakout(z, closes, required_closes=2) == "up"
    # Two consecutive closes below -> confirmed down
    closes = pd.Series([101.0, 98.5, 98.0])
    assert confirmed_breakout(z, closes, required_closes=2) == "down"


def test_confirmed_breakout_none_when_not_through():
    z = _zone(100.0, band=1.0)
    closes = pd.Series([99.5, 100.0, 100.5])   # inside the band
    assert confirmed_breakout(z, closes) == "none"


# ---------------------------------------------------------------------------
# Direction helpers (for TP ladder)
# ---------------------------------------------------------------------------

def test_nearest_in_direction_up():
    zones = [
        _zone(105.0, strength=0.5, side="resistance"),
        _zone(110.0, strength=0.8, side="resistance"),
        _zone(95.0, strength=0.7, side="support"),
    ]
    z = nearest_in_direction(zones, reference=100.0, direction="up")
    assert z is not None
    assert z.price_mid == 105.0


def test_nearest_in_direction_respects_min_strength():
    zones = [
        _zone(105.0, strength=0.3, side="resistance"),
        _zone(110.0, strength=0.8, side="resistance"),
    ]
    z = nearest_in_direction(zones, reference=100.0, direction="up", min_strength=0.5)
    assert z is not None
    assert z.price_mid == 110.0   # skipped the weak 105 zone


def test_nearest_in_direction_respects_max_distance():
    zones = [_zone(200.0, strength=0.8, side="resistance")]
    # 200 is 100% above; max_distance_pct=0.1 should filter it out
    z = nearest_in_direction(zones, reference=100.0, direction="up",
                              max_distance_pct=0.1)
    assert z is None


def test_zones_in_direction_returns_sorted_by_proximity():
    zones = [
        _zone(120.0, strength=0.8, side="resistance"),
        _zone(105.0, strength=0.7, side="resistance"),
        _zone(110.0, strength=0.6, side="resistance"),
    ]
    result = zones_in_direction(zones, reference=100.0, direction="up")
    assert [z.price_mid for z in result] == [105.0, 110.0, 120.0]


def test_zones_in_direction_empty_when_none_match():
    zones = [_zone(105.0, strength=0.8, side="resistance")]
    result = zones_in_direction(zones, reference=100.0, direction="down")
    assert result == []

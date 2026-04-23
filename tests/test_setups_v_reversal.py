"""Tests for setups/v_reversal.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from setups.base import MarketContext, TPLevel, Trigger
from setups.v_reversal import (
    VReversalDetector,
    VStructure,
    _extended_move_long,
    _extended_move_short,
    _find_v_structure,
)


# ---------------------------------------------------------------------------
# Candle factory
# ---------------------------------------------------------------------------

def _candles_df(closes: np.ndarray, noise_pct: float = 0.001, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = np.asarray(closes, dtype=float)
    highs = closes * (1 + np.abs(rng.normal(0, noise_pct, len(closes))))
    lows = closes * (1 - np.abs(rng.normal(0, noise_pct, len(closes))))
    opens = np.concatenate([[closes[0]], closes[:-1]])
    vols = rng.uniform(800, 1200, len(closes))
    ts = pd.date_range("2025-01-01", periods=len(closes), freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": vols,
    })


def _v_series_long(
    *,
    n_leadup: int = 40,
    n_decline: int = 60,
    n_bounce: int = 12,
    n_pullback: int = 4,
    n_continuation: int = 2,
    prior_high: float = 100.0,
    v_low: float = 85.0,           # 15% decline
    bounce_peak: float = 95.0,
    hl: float = 90.0,
    final: float = 94.0,
) -> np.ndarray:
    """Build a synthetic close series: leadup → decline → V → bounce → HL → up."""
    leadup = np.linspace(prior_high * 0.95, prior_high, n_leadup)
    decline = np.linspace(prior_high, v_low, n_decline)
    bounce = np.linspace(v_low, bounce_peak, n_bounce)
    pullback = np.linspace(bounce_peak, hl, n_pullback)
    cont = np.linspace(hl, final, n_continuation)
    return np.concatenate([leadup, decline, bounce, pullback, cont])


def _v_series_short(**kwargs) -> np.ndarray:
    """Mirrored Λ-shape for shorts."""
    return _v_series_long(
        prior_high=85.0, v_low=100.0, bounce_peak=90.0, hl=95.0, final=91.0,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# _extended_move_long
# ---------------------------------------------------------------------------

def test_extended_move_long_detects_decline_above_threshold():
    closes = pd.Series(np.concatenate([
        np.linspace(100, 100, 10),
        np.linspace(100, 85, 20),   # 15% decline
        np.linspace(85, 90, 10),
    ]))
    ext = _extended_move_long(closes, lookback_bars=40, min_move_pct=0.10)
    assert ext is not None
    prior_idx, v_idx, mag = ext
    assert mag == pytest.approx(0.15, abs=0.01)
    assert v_idx > prior_idx


def test_extended_move_long_rejects_when_move_too_small():
    closes = pd.Series(np.linspace(100, 97, 40))   # only 3% decline
    assert _extended_move_long(closes, lookback_bars=40, min_move_pct=0.10) is None


def test_extended_move_long_rejects_on_short_history():
    closes = pd.Series(np.linspace(100, 80, 10))
    assert _extended_move_long(closes, lookback_bars=40, min_move_pct=0.10) is None


def test_extended_move_short_detects_rally_above_threshold():
    closes = pd.Series(np.concatenate([
        np.linspace(100, 100, 10),
        np.linspace(100, 120, 20),   # 20% rally
        np.linspace(120, 115, 10),
    ]))
    ext = _extended_move_short(closes, lookback_bars=40, min_move_pct=0.15)
    assert ext is not None
    _, v_idx, mag = ext
    assert mag == pytest.approx(0.20, abs=0.01)


# ---------------------------------------------------------------------------
# _find_v_structure
# ---------------------------------------------------------------------------

def test_find_v_structure_long_happy_path():
    closes = _v_series_long()
    df = _candles_df(closes)
    v = _find_v_structure(
        df, "long",
        lookback_bars=100, min_move_pct=0.10,
        min_bounce_pct=0.03, max_bars_since_hl=5,
    )
    assert v is not None
    assert v.direction == "long"
    assert v.v_extreme_close == pytest.approx(85.0, rel=0.02)
    assert v.hl_extreme_close > v.v_extreme_close
    assert v.bounce_peak_idx > v.v_idx
    assert v.hl_idx > v.bounce_peak_idx


def test_find_v_structure_long_rejects_when_no_higher_low():
    # Construct: decline then straight up (never pulls back)
    closes = np.concatenate([
        np.linspace(100, 100, 40),
        np.linspace(100, 85, 60),
        np.linspace(85, 100, 10),    # straight back up, no pullback
    ])
    df = _candles_df(closes)
    v = _find_v_structure(
        df, "long",
        lookback_bars=100, min_move_pct=0.10,
        min_bounce_pct=0.03, max_bars_since_hl=5,
    )
    # Post-bounce "min" equals bounce-peak itself → HL not strictly above V
    # but argmin will pick the last bar which IS above V. Test that it at
    # least verifies structure: hl must be STRICTLY above v.
    # In a monotonic bounce, the min after bounce-peak = bounce-peak itself
    # (no bars after), so v should be None.
    assert v is None


def test_find_v_structure_long_rejects_stale_hl():
    # Series where HL is old (past max_bars_since_hl from current)
    closes = _v_series_long(
        n_pullback=4, n_continuation=20,   # 20 bars after HL
    )
    df = _candles_df(closes)
    v = _find_v_structure(
        df, "long",
        lookback_bars=140, min_move_pct=0.10,
        min_bounce_pct=0.03, max_bars_since_hl=5,
    )
    # Bars since HL = 20+ > 5 → reject
    assert v is None


def test_find_v_structure_long_rejects_insufficient_bounce():
    # Decline then tiny 1% bounce
    closes = np.concatenate([
        np.linspace(100, 100, 40),
        np.linspace(100, 85, 60),
        np.linspace(85, 85.3, 8),   # ~0.4% bounce
        np.linspace(85.3, 85.1, 4),
        np.linspace(85.1, 85.2, 2),
    ])
    df = _candles_df(closes)
    v = _find_v_structure(
        df, "long",
        lookback_bars=100, min_move_pct=0.10,
        min_bounce_pct=0.03, max_bars_since_hl=5,
    )
    assert v is None


def test_find_v_structure_short_happy_path():
    closes = _v_series_short()
    df = _candles_df(closes)
    v = _find_v_structure(
        df, "short",
        lookback_bars=100, min_move_pct=0.15,
        min_bounce_pct=0.03, max_bars_since_hl=5,
    )
    assert v is not None
    assert v.direction == "short"
    assert v.hl_extreme_close < v.v_extreme_close


# ---------------------------------------------------------------------------
# Detector end-to-end
# ---------------------------------------------------------------------------

def _build_ctx(closes: np.ndarray) -> MarketContext:
    return MarketContext.build(
        ticker="TEST", timeframe="1h",
        candles=_candles_df(closes), compute_sr=True,
    )


def test_detector_fires_long_on_v_reversal():
    closes = _v_series_long()
    ctx = _build_ctx(closes)
    det = VReversalDetector(
        lookback_bars=100, min_move_pct_long=0.10,
        min_target_distance_pct=0.05,
    )
    trig = det.detect(ctx)
    assert trig is not None
    assert trig.direction == "long"
    assert trig.setup == "v_reversal"
    # Stop should be near/below the V-wick
    assert trig.stop_price < ctx.current_price
    assert trig.entry_price == pytest.approx(ctx.current_price)
    # Components recorded
    assert trig.components["move_magnitude"] >= 0.10
    assert trig.components["bars_since_hl"] <= 5


def test_detector_fires_short_on_inverted_v():
    closes = _v_series_short()
    ctx = _build_ctx(closes)
    det = VReversalDetector(
        lookback_bars=100, min_move_pct_long=0.10, min_move_pct_short=0.15,
        min_target_distance_pct=0.05,
    )
    trig = det.detect(ctx)
    assert trig is not None
    assert trig.direction == "short"
    assert trig.stop_price > ctx.current_price


def test_detector_no_fire_when_move_not_extended_enough():
    # Only 5% decline
    closes = np.concatenate([
        np.linspace(100, 100, 40),
        np.linspace(100, 95, 60),
        np.linspace(95, 97, 12),
        np.linspace(97, 96, 4),
        np.linspace(96, 96.5, 2),
    ])
    ctx = _build_ctx(closes)
    det = VReversalDetector(
        lookback_bars=100, min_move_pct_long=0.10,
        min_target_distance_pct=0.05,
    )
    assert det.detect(ctx) is None


def test_detector_no_fire_when_hl_stale():
    closes = _v_series_long(n_continuation=20)
    ctx = _build_ctx(closes)
    det = VReversalDetector(
        lookback_bars=140, min_move_pct_long=0.10,
        max_bars_since_hl=5, min_target_distance_pct=0.05,
    )
    assert det.detect(ctx) is None


def test_detector_no_fire_when_still_declining():
    # Straight down — no V, no bounce, no HL
    closes = np.linspace(100, 80, 130)
    ctx = _build_ctx(closes)
    det = VReversalDetector(
        lookback_bars=100, min_move_pct_long=0.10,
        min_target_distance_pct=0.05,
    )
    assert det.detect(ctx) is None


def test_detector_rejects_when_no_runway_for_tp():
    closes = _v_series_long(final=90.5)   # tight continuation
    ctx = _build_ctx(closes)
    det = VReversalDetector(
        lookback_bars=100, min_move_pct_long=0.10,
        min_target_distance_pct=0.80,   # demand 80% TP runway
    )
    assert det.detect(ctx) is None


def test_detector_confidence_and_size_defaults():
    closes = _v_series_long()
    ctx = _build_ctx(closes)
    det = VReversalDetector(
        lookback_bars=100, min_move_pct_long=0.10,
        min_target_distance_pct=0.05,
    )
    trig = det.detect(ctx)
    assert trig is not None
    assert trig.confidence == pytest.approx(0.70)
    assert trig.size_fraction == pytest.approx(0.4)


def test_detector_returns_none_on_insufficient_bars():
    closes = np.full(30, 100.0)
    ctx = _build_ctx(closes)
    det = VReversalDetector(lookback_bars=100)
    assert det.detect(ctx) is None

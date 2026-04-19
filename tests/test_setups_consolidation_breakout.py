"""Tests for setups/consolidation_breakout.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from setups.base import MarketContext, TPLevel, Trigger
from setups.consolidation_breakout import (
    ConsolidationBreakoutDetector,
    _confirmation_closes_ok,
    _consolidation_range,
    _is_breakout_close,
    _prior_bar_inside_range,
)


# ---------------------------------------------------------------------------
# Synthetic candle factory
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


def _tight_range_then_breakout(
    *,
    n_leadup: int = 100,
    n_range: int = 20,
    n_breakout: int = 2,
    range_mid: float = 100.0,
    range_width_pct: float = 0.02,
    breakout_pct: float = 0.02,
    direction: str = "long",
) -> np.ndarray:
    """Build a close series: trend leadup + tight oscillating range + breakout.

    Generates exactly `n_breakout` bars past the range boundary at the end
    — matching `required_confirm_closes` in the detector under test.
    """
    leadup = np.linspace(range_mid * 0.9, range_mid, n_leadup)
    half = range_width_pct * range_mid / 2
    # Oscillate between [mid - half, mid + half]
    rng_vals = np.array([
        range_mid + (half if i % 2 == 0 else -half) * 0.5
        for i in range(n_range)
    ])
    if direction == "long":
        breakout = np.array([range_mid + half + range_mid * breakout_pct] * n_breakout)
    else:
        breakout = np.array([range_mid - half - range_mid * breakout_pct] * n_breakout)
    return np.concatenate([leadup, rng_vals, breakout])


# ---------------------------------------------------------------------------
# _consolidation_range
# ---------------------------------------------------------------------------

def test_consolidation_range_excludes_current_bar():
    # 20 bars at 100 + current bar spike to 110; confirm_closes=1 default
    closes = np.concatenate([np.full(20, 100.0), np.array([110.0])])
    df = _candles_df(closes, noise_pct=0)
    rng = _consolidation_range(df, window_bars=20, confirm_closes=1)
    assert rng is not None
    low, high, mean_close = rng
    assert high == pytest.approx(100.0, rel=0.01)
    assert low == pytest.approx(100.0, rel=0.01)
    assert mean_close == pytest.approx(100.0, rel=0.01)


def test_consolidation_range_excludes_full_confirmation_sequence():
    # 20 range bars at 100 + 2 breakout bars at 110; window should still
    # be pure range, not include any breakout bars.
    closes = np.concatenate([np.full(20, 100.0), np.full(2, 110.0)])
    df = _candles_df(closes, noise_pct=0)
    rng = _consolidation_range(df, window_bars=20, confirm_closes=2)
    assert rng is not None
    low, high, _ = rng
    assert high == pytest.approx(100.0, rel=0.01)
    assert low == pytest.approx(100.0, rel=0.01)


def test_consolidation_range_returns_none_when_insufficient():
    df = _candles_df(np.array([100.0] * 10))
    assert _consolidation_range(df, window_bars=20) is None


# ---------------------------------------------------------------------------
# _is_breakout_close
# ---------------------------------------------------------------------------

def test_breakout_close_long_requires_above_high_plus_buffer():
    # range_high=100, buffer=1% → must be > 101
    assert _is_breakout_close(102.0, 95.0, 100.0, "long", breakout_buffer_pct=0.01)
    assert not _is_breakout_close(100.5, 95.0, 100.0, "long", breakout_buffer_pct=0.01)
    # Exactly at boundary — not decisive
    assert not _is_breakout_close(100.0, 95.0, 100.0, "long", breakout_buffer_pct=0.01)


def test_breakout_close_short_requires_below_low_minus_buffer():
    # range_low=95, buffer=1% → must be < 94.05
    assert _is_breakout_close(93.0, 95.0, 100.0, "short", breakout_buffer_pct=0.01)
    assert not _is_breakout_close(94.5, 95.0, 100.0, "short", breakout_buffer_pct=0.01)


# ---------------------------------------------------------------------------
# _confirmation_closes_ok
# ---------------------------------------------------------------------------

def test_confirmation_closes_two_required_both_above():
    closes = pd.Series([99.0, 99.5, 101.0, 102.0])
    assert _confirmation_closes_ok(closes, 95.0, 100.0, "long", required_closes=2)


def test_confirmation_closes_single_close_fails_when_two_required():
    # Only the final close is above; the penultimate is still inside range
    closes = pd.Series([99.0, 99.5, 100.0, 102.0])
    assert not _confirmation_closes_ok(closes, 95.0, 100.0, "long", required_closes=2)


def test_confirmation_closes_one_required_passes_on_single_break():
    closes = pd.Series([99.0, 99.5, 100.0, 102.0])
    assert _confirmation_closes_ok(closes, 95.0, 100.0, "long", required_closes=1)


def test_confirmation_closes_short_direction():
    closes = pd.Series([96.0, 95.5, 93.0, 92.0])
    assert _confirmation_closes_ok(closes, 95.0, 100.0, "short", required_closes=2)


# ---------------------------------------------------------------------------
# _prior_bar_inside_range
# ---------------------------------------------------------------------------

def test_prior_bar_inside_range_true_when_bar_before_sequence_in_band():
    # N=1 → offset=2 → check candles[-2]; 99.5 is inside range
    df = _candles_df(np.array([99.0, 99.5, 102.0]))
    assert _prior_bar_inside_range(
        df, range_low=95.0, range_high=100.0, required_confirm_closes=1,
    )


def test_prior_bar_inside_range_false_when_stale_breakout():
    # N=1 → candles[-2]=101 is already above range → stale
    df = _candles_df(np.array([99.0, 101.0, 102.0]))
    assert not _prior_bar_inside_range(
        df, range_low=95.0, range_high=100.0, required_confirm_closes=1,
    )


def test_prior_bar_inside_range_accounts_for_confirmation_closes():
    # N=2 → offset=3 → check candles[-3]=99.0, still in range ✓
    df = _candles_df(np.array([99.0, 101.0, 102.0]))
    assert _prior_bar_inside_range(
        df, range_low=95.0, range_high=100.0, required_confirm_closes=2,
    )


# ---------------------------------------------------------------------------
# Detector end-to-end
# ---------------------------------------------------------------------------

def _build_ctx(closes: np.ndarray) -> MarketContext:
    return MarketContext.build(
        ticker="TEST", timeframe="1h",
        candles=_candles_df(closes), compute_sr=True,
    )


def test_detector_fires_long_on_clean_breakout():
    closes = _tight_range_then_breakout(direction="long", breakout_pct=0.12)
    ctx = _build_ctx(closes)
    det = ConsolidationBreakoutDetector(
        required_confirm_closes=2, min_target_distance_pct=0.05,
    )
    trig = det.detect(ctx)
    assert trig is not None
    assert trig.direction == "long"
    assert trig.setup == "consolidation_breakout"
    assert trig.entry_price == pytest.approx(ctx.current_price)
    # Stop sits below the range low, using wick
    assert trig.stop_price < min(ctx.candles["low"].iloc[-23:-3])
    # Components recorded
    assert "range_low" in trig.components
    assert "range_high" in trig.components


def test_detector_fires_short_on_clean_breakdown():
    closes = _tight_range_then_breakout(direction="short", breakout_pct=0.12)
    ctx = _build_ctx(closes)
    det = ConsolidationBreakoutDetector(
        required_confirm_closes=2, min_target_distance_pct=0.05,
    )
    trig = det.detect(ctx)
    assert trig is not None
    assert trig.direction == "short"
    # Short stop above range high
    assert trig.stop_price > max(ctx.candles["high"].iloc[-23:-3])


def test_detector_no_fire_when_range_too_wide():
    # Oscillation width 10% > default 3% ceiling
    closes = _tight_range_then_breakout(
        direction="long", range_width_pct=0.10, breakout_pct=0.12,
    )
    ctx = _build_ctx(closes)
    det = ConsolidationBreakoutDetector(required_confirm_closes=2)
    assert det.detect(ctx) is None


def test_detector_no_fire_when_current_close_inside_range():
    # Construct: tight range + a final bar whose close is INSIDE the range
    leadup = np.linspace(90.0, 100.0, 100)
    rng = np.array([
        100.0 + (1.0 if i % 2 == 0 else -1.0) * 0.5
        for i in range(20)
    ])
    final = np.array([100.1])   # still inside range
    closes = np.concatenate([leadup, rng, final])
    ctx = _build_ctx(closes)
    det = ConsolidationBreakoutDetector(required_confirm_closes=2)
    assert det.detect(ctx) is None


def test_detector_no_fire_with_stale_breakout_guard():
    # Tight range, then 3 bars well above — the prior bar is outside range,
    # so the fresh-breakout guard should reject.
    closes = _tight_range_then_breakout(direction="long", breakout_pct=0.15)
    # The helper already appends 3 breakout bars; the 2nd-to-last close is
    # above range. Detector should reject on fresh-breakout guard.
    ctx = _build_ctx(closes)
    det = ConsolidationBreakoutDetector(required_confirm_closes=1)
    # required_confirm_closes=1 removes the multi-close filter, so the only
    # remaining gate is the fresh-breakout guard → expect None because the
    # prior bar is already past the range.
    assert det.detect(ctx) is None


def test_detector_single_close_insufficient_when_n_required_is_2():
    # Tight range then a single breakout bar (not two consecutive)
    leadup = np.linspace(90.0, 100.0, 100)
    rng = np.array([
        100.0 + (1.0 if i % 2 == 0 else -1.0) * 0.5
        for i in range(20)
    ])
    single_break = np.array([112.0])
    closes = np.concatenate([leadup, rng, single_break])
    ctx = _build_ctx(closes)
    det = ConsolidationBreakoutDetector(required_confirm_closes=2)
    # Only ONE close above the range (the final bar); second-most-recent
    # is still inside the range → should NOT fire with N=2.
    assert det.detect(ctx) is None


def test_detector_single_close_fires_when_n_required_is_1():
    leadup = np.linspace(90.0, 100.0, 100)
    rng = np.array([
        100.0 + (1.0 if i % 2 == 0 else -1.0) * 0.5
        for i in range(20)
    ])
    single_break = np.array([112.0])
    closes = np.concatenate([leadup, rng, single_break])
    ctx = _build_ctx(closes)
    det = ConsolidationBreakoutDetector(
        required_confirm_closes=1, min_target_distance_pct=0.05,
    )
    # With N=1, a single decisive close above the range is enough.
    trig = det.detect(ctx)
    assert trig is not None
    assert trig.direction == "long"


def test_detector_rejects_when_no_runway_for_tp():
    # Tight range breakout, but the ATR + S/R landscape gives no TP ≥ 10%
    # away. min_target_distance_pct default = 10%; use a VERY narrow
    # breakout with no SR far above so the ATR fallback lands too close.
    closes = _tight_range_then_breakout(
        direction="long", breakout_pct=0.005,  # tiny breakout
    )
    ctx = _build_ctx(closes)
    det = ConsolidationBreakoutDetector(
        required_confirm_closes=1,
        min_target_distance_pct=0.50,    # demand 50% runway; ATR won't reach
    )
    assert det.detect(ctx) is None


def test_detector_returns_none_on_insufficient_bars():
    closes = np.full(10, 100.0)   # less than window_bars + buffer
    ctx = _build_ctx(closes)
    det = ConsolidationBreakoutDetector(required_confirm_closes=2)
    assert det.detect(ctx) is None


def test_detector_confidence_and_size_fraction_defaults_present():
    closes = _tight_range_then_breakout(direction="long", breakout_pct=0.12)
    ctx = _build_ctx(closes)
    det = ConsolidationBreakoutDetector(
        required_confirm_closes=2, min_target_distance_pct=0.05,
    )
    trig = det.detect(ctx)
    assert trig is not None
    assert 0.0 <= trig.confidence <= 1.0
    assert trig.confidence == pytest.approx(0.65)
    assert trig.size_fraction == pytest.approx(0.5)

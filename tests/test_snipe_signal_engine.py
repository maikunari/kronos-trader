"""Tests for snipe_signal_engine.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime import RegimeDetector
from snipe_signal_engine import (
    MarketContext,
    SnipeSignalEngine,
    supertrend,
)


# --- Helpers ------------------------------------------------------------------

def _ohlc(closes: np.ndarray, noise: float = 0.0005, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    highs = closes * (1 + np.abs(rng.normal(0, noise, len(closes))))
    lows = closes * (1 - np.abs(rng.normal(0, noise, len(closes))))
    opens = np.concatenate([[closes[0]], closes[:-1]])
    vols = rng.uniform(800, 1200, len(closes))
    ts = pd.date_range("2025-01-01", periods=len(closes), freq="15min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": vols,
    })


def _trending_closes(n: int, drift: float = 0.002, sigma: float = 0.002,
                     phi: float = 0.5, seed: int = 1) -> np.ndarray:
    """AR(1) positive-drift returns -> trending prices."""
    rng = np.random.default_rng(seed)
    ret = np.zeros(n)
    ret[0] = rng.normal(drift, sigma)
    for i in range(1, n):
        ret[i] = drift + phi * (ret[i - 1] - drift) + rng.normal(0, sigma)
    return 100 * np.exp(np.cumsum(ret))


def _lenient_detector() -> RegimeDetector:
    """Detector with small rv_hist_window so test frames don't need to be huge."""
    return RegimeDetector(adx_period=14, adx_threshold=20, hurst_window=300,
                          hurst_threshold=0.50, rv_window=48, rv_hist_window=200)


def _engine(**overrides) -> SnipeSignalEngine:
    defaults = dict(regime=_lenient_detector())
    defaults.update(overrides)
    return SnipeSignalEngine(**defaults)


# --- SuperTrend utility -------------------------------------------------------

def test_supertrend_detects_uptrend_end():
    closes = pd.Series(_trending_closes(500, drift=0.002, sigma=0.001))
    highs = closes * 1.001
    lows = closes * 0.999
    direction = supertrend(highs, lows, closes, period=10, multiplier=3.0)
    assert direction.iloc[-1] == 1.0


def test_supertrend_detects_downtrend_end():
    closes = pd.Series(_trending_closes(500, drift=-0.002, sigma=0.001))
    highs = closes * 1.001
    lows = closes * 0.999
    direction = supertrend(highs, lows, closes, period=10, multiplier=3.0)
    assert direction.iloc[-1] == -1.0


# --- Engine: gates ------------------------------------------------------------

def test_insufficient_bars_returns_flat():
    engine = _engine()
    df = _ohlc(np.full(10, 100.0))
    sig = engine.evaluate(MarketContext(candles_15m=df))
    assert sig.action == "flat"
    assert sig.skip_reason.startswith("insufficient")


def test_range_regime_returns_flat_with_regime_reason():
    engine = _engine()
    # Choppy no-drift series -> ADX low
    rng = np.random.default_rng(0)
    closes = 100 + np.cumsum(rng.normal(0, 0.05, 500)) * 0.1
    sig = engine.evaluate(MarketContext(candles_15m=_ohlc(closes)))
    assert sig.action == "flat"
    assert sig.skip_reason.startswith("regime:")
    assert sig.regime is not None


# --- Engine: full path --------------------------------------------------------

def test_long_breakout_fires_under_trending_regime():
    """Engineer a regime-trending series with a forced Donchian break on the final bar."""
    engine = _engine(donchian_period=20, composite_threshold=-1.0)  # allow neutral composite
    closes = _trending_closes(800, drift=0.0015, sigma=0.001)
    # Force last bar to close above the prior 20-bar high by a clear margin
    prior_high = float(np.max(closes[-21:-1]))
    closes = closes.copy()
    closes[-2] = prior_high * 0.999      # previous close below
    closes[-1] = prior_high * 1.003      # current close above breakout
    sig = engine.evaluate(MarketContext(candles_15m=_ohlc(closes)))
    assert sig.action == "long", f"expected long, got {sig.action} ({sig.skip_reason})"
    assert sig.breakout_channel in {"donchian", "keltner"}
    # R:R invariant
    stop_dist = sig.entry_price - sig.stop_price
    target_dist = sig.target_price - sig.entry_price
    assert abs(target_dist / stop_dist - 2.0) < 1e-6


def test_funding_veto_rejects_long_when_funding_too_positive():
    engine = _engine(donchian_period=20, composite_threshold=-1.0,
                     funding_veto_pct_hourly=0.0003)
    closes = _trending_closes(800, drift=0.0015, sigma=0.001)
    prior_high = float(np.max(closes[-21:-1]))
    closes = closes.copy()
    closes[-2] = prior_high * 0.999
    closes[-1] = prior_high * 1.003
    ctx = MarketContext(candles_15m=_ohlc(closes), funding_rate_hourly=0.001)
    sig = engine.evaluate(ctx)
    assert sig.action == "flat"
    assert sig.skip_reason.startswith("funding_too_positive")


def test_composite_score_below_threshold_rejects():
    """OI delta opposite the entry direction should push composite negative."""
    engine = _engine(donchian_period=20, composite_threshold=0.5,
                     oi_confirm_pct=0.01)
    closes = _trending_closes(800, drift=0.0015, sigma=0.001)
    prior_high = float(np.max(closes[-21:-1]))
    closes = closes.copy()
    closes[-2] = prior_high * 0.999
    closes[-1] = prior_high * 1.003
    # OI contracting -> -1 for long -> composite negative
    oi = pd.Series([100, 99, 97, 94, 90], dtype=float)
    ctx = MarketContext(candles_15m=_ohlc(closes), oi_series=oi)
    sig = engine.evaluate(ctx)
    assert sig.action == "flat"
    assert "composite_below_threshold" in sig.skip_reason


def test_rr_ratio_always_reported_even_when_flat():
    engine = _engine(stop_atr_mult=1.5, target_atr_mult=3.0)
    df = _ohlc(np.full(10, 100.0))
    sig = engine.evaluate(MarketContext(candles_15m=df))
    assert sig.action == "flat"
    assert sig.rr_ratio == pytest.approx(2.0)

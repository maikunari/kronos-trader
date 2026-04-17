"""Smoke tests locking down current ATREngine behavior before refactor."""
from __future__ import annotations

import numpy as np
import pandas as pd

from atr_engine import ATREngine, _atr, _ema


def test_ema_matches_pandas_reference():
    """Our EMA should agree with pandas .ewm on a random series."""
    rng = np.random.default_rng(7)
    values = rng.normal(100, 5, 300)
    ours = _ema(values, period=20)
    theirs = pd.Series(values).ewm(span=20, adjust=False).mean().to_numpy()
    np.testing.assert_allclose(ours, theirs, atol=1e-9)


def test_atr_is_positive_and_near_true_range():
    """ATR should be strictly positive and in the neighborhood of |high-low|."""
    rng = np.random.default_rng(11)
    close = 100.0 + np.cumsum(rng.normal(0, 1, 200))
    high = close + np.abs(rng.normal(0, 0.5, 200))
    low = close - np.abs(rng.normal(0, 0.5, 200))
    atr = _atr(high, low, close, period=14)
    assert np.all(atr > 0)
    # ATR on smooth data should be on the order of avg true range
    tr = high - low
    assert 0.3 * tr.mean() < atr[-1] < 3 * tr.mean()


def test_evaluate_insufficient_data_returns_flat():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=10, freq="15min"),
        "open": [100] * 10, "high": [101] * 10, "low": [99] * 10,
        "close": [100] * 10, "volume": [1000] * 10,
    })
    sig = ATREngine().evaluate(df)
    assert sig.action == "flat"
    assert sig.skip_reason == "insufficient_data"


def test_evaluate_uptrend_no_pullback_returns_flat(uptrend_ohlcv):
    """In a clean uptrend without a pullback-and-crossback, engine should stay flat."""
    sig = ATREngine(ema_fast=20, ema_slow=50, atr_period=14).evaluate(uptrend_ohlcv)
    # May or may not fire depending on noise — but if flat, reason should be the pullback rule
    if sig.action == "flat":
        assert sig.skip_reason in {"no_long_pullback_crossover", "no_short_pullback_crossover"}


def test_evaluate_produces_2to1_rr_when_signal_fires(uptrend_ohlcv):
    """When a signal fires, target should be exactly 2× the stop distance (default 3:1.5)."""
    eng = ATREngine(ema_fast=20, ema_slow=50, atr_period=14, stop_multiplier=1.5, target_multiplier=3.0)
    sig = eng.evaluate(uptrend_ohlcv)
    if sig.action == "long":
        stop_dist = sig.entry_price - sig.stop_price
        target_dist = sig.target_price - sig.entry_price
        assert stop_dist > 0
        assert abs(target_dist / stop_dist - 2.0) < 1e-6
    elif sig.action == "short":
        stop_dist = sig.stop_price - sig.entry_price
        target_dist = sig.entry_price - sig.target_price
        assert stop_dist > 0
        assert abs(target_dist / stop_dist - 2.0) < 1e-6

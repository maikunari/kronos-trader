"""Smoke tests locking down current MTFFilter behavior before refactor to veto+confirm.

These lock in two load-bearing properties:
  1. _compute_bias correctly classifies trend/range EMA regimes.
  2. get_bias_at enforces no-lookahead (uses only candles strictly before ts).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mtf_filter import MTFFilter


def _make_hourly(closes: np.ndarray) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=len(closes), freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": np.full_like(closes, 1000.0),
    })


def test_compute_bias_detects_uptrend_on_both_tfs():
    mtf = MTFFilter(symbol="BTC", ema_fast=20, ema_slow=50, require_both=True)
    # Strong uptrend: 200 bars rising linearly
    closes = np.linspace(100, 200, 200)
    mtf._df_1h = _make_hourly(closes)
    mtf._df_4h = _make_hourly(closes)
    sig = mtf.get_bias_at(mtf._df_1h["timestamp"].iloc[-1] + pd.Timedelta(hours=1))
    assert sig.tf_1h == "long"
    assert sig.tf_4h == "long"
    assert sig.bias == "long"


def test_compute_bias_neutral_when_timeframes_disagree_under_require_both():
    mtf = MTFFilter(symbol="BTC", ema_fast=20, ema_slow=50, require_both=True)
    up = np.linspace(100, 200, 200)
    down = np.linspace(200, 100, 200)
    mtf._df_1h = _make_hourly(up)
    mtf._df_4h = _make_hourly(down)
    sig = mtf.get_bias_at(mtf._df_1h["timestamp"].iloc[-1] + pd.Timedelta(hours=1))
    assert sig.bias == "neutral"


def test_get_bias_at_uses_only_prior_candles():
    """No lookahead: candles at or after the query timestamp must be excluded."""
    mtf = MTFFilter(symbol="BTC", ema_fast=20, ema_slow=50, require_both=True)
    # First 100 bars: downtrend. Last 100: uptrend.
    closes = np.concatenate([np.linspace(200, 100, 100), np.linspace(100, 200, 100)])
    mtf._df_1h = _make_hourly(closes)
    mtf._df_4h = _make_hourly(closes)

    # Querying in the middle of the downtrend half — should see only downtrend data.
    mid_ts = mtf._df_1h["timestamp"].iloc[90]
    sig_mid = mtf.get_bias_at(mid_ts)
    assert sig_mid.tf_1h in {"short", "neutral"}

    # Querying at the very end — uptrend is now visible.
    end_ts = mtf._df_1h["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
    sig_end = mtf.get_bias_at(end_ts)
    assert sig_end.tf_1h == "long"


def test_empty_data_returns_neutral():
    mtf = MTFFilter(symbol="BTC")
    sig = mtf.get_bias_at(pd.Timestamp("2025-01-01", tz="UTC"))
    assert sig.bias == "neutral"
    assert sig.tf_1h == "neutral"
    assert sig.tf_4h == "neutral"

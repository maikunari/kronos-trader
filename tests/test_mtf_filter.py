"""Tests for MTFFilter — covers the new veto+confirm API plus the legacy
.bias property kept for backwards compatibility."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mtf_filter import MTFFilter, MTFSignal


def _make_hourly(closes: np.ndarray) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=len(closes), freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": np.full_like(closes, 1000.0),
    })


# --- Per-timeframe detection --------------------------------------------------

def test_detects_uptrend_on_both_timeframes():
    mtf = MTFFilter(symbol="BTC", ema_fast=20, ema_slow=50)
    closes = np.linspace(100, 200, 200)
    mtf._df_1h = _make_hourly(closes)
    mtf._df_4h = _make_hourly(closes)
    sig = mtf.get_bias_at(mtf._df_1h["timestamp"].iloc[-1] + pd.Timedelta(hours=1))
    assert sig.tf_1h == "long"
    assert sig.tf_4h == "long"


def test_detects_downtrend_on_both_timeframes():
    mtf = MTFFilter(symbol="BTC")
    closes = np.linspace(200, 100, 200)
    mtf._df_1h = _make_hourly(closes)
    mtf._df_4h = _make_hourly(closes)
    sig = mtf.get_bias_at(mtf._df_1h["timestamp"].iloc[-1] + pd.Timedelta(hours=1))
    assert sig.tf_1h == "short"
    assert sig.tf_4h == "short"


def test_empty_data_returns_neutral():
    mtf = MTFFilter(symbol="BTC")
    sig = mtf.get_bias_at(pd.Timestamp("2025-01-01", tz="UTC"))
    assert sig.tf_1h == "neutral" and sig.tf_4h == "neutral"


# --- No-lookahead invariant ---------------------------------------------------

def test_get_bias_at_uses_only_prior_candles():
    mtf = MTFFilter(symbol="BTC", ema_fast=20, ema_slow=50)
    # First 100 bars down, next 100 bars up.
    closes = np.concatenate([np.linspace(200, 100, 100), np.linspace(100, 200, 100)])
    mtf._df_1h = _make_hourly(closes)
    mtf._df_4h = _make_hourly(closes)

    sig_mid = mtf.get_bias_at(mtf._df_1h["timestamp"].iloc[90])
    assert sig_mid.tf_1h in {"short", "neutral"}

    sig_end = mtf.get_bias_at(mtf._df_1h["timestamp"].iloc[-1] + pd.Timedelta(hours=1))
    assert sig_end.tf_1h == "long"


# --- vetoes() -----------------------------------------------------------------

def test_vetoes_long_when_4h_is_short():
    sig = MTFSignal(
        tf_1h="long", tf_4h="short",
        ema_fast_1h=1, ema_slow_1h=1, ema_fast_4h=1, ema_slow_4h=1,
    )
    assert sig.vetoes("long") is True
    assert sig.vetoes("short") is False   # 4H agrees with short


def test_vetoes_neutral_does_not_veto():
    sig = MTFSignal(
        tf_1h="neutral", tf_4h="neutral",
        ema_fast_1h=0, ema_slow_1h=0, ema_fast_4h=0, ema_slow_4h=0,
    )
    assert sig.vetoes("long") is False
    assert sig.vetoes("short") is False


def test_vetoes_accepts_which_kwarg_to_check_1h():
    sig = MTFSignal(
        tf_1h="short", tf_4h="long",
        ema_fast_1h=1, ema_slow_1h=1, ema_fast_4h=1, ema_slow_4h=1,
    )
    # 4H agrees with long, but 1H opposes
    assert sig.vetoes("long", which="4h") is False
    assert sig.vetoes("long", which="1h") is True


# --- confirms() ---------------------------------------------------------------

def test_confirms_requires_both_agree():
    both_long = MTFSignal(
        tf_1h="long", tf_4h="long",
        ema_fast_1h=1, ema_slow_1h=1, ema_fast_4h=1, ema_slow_4h=1,
    )
    mixed = MTFSignal(
        tf_1h="long", tf_4h="neutral",
        ema_fast_1h=1, ema_slow_1h=1, ema_fast_4h=1, ema_slow_4h=1,
    )
    assert both_long.confirms("long") is True
    assert mixed.confirms("long") is False


# --- Legacy .bias property ----------------------------------------------------

def test_legacy_bias_long_when_both_agree_long():
    sig = MTFSignal(
        tf_1h="long", tf_4h="long",
        ema_fast_1h=1, ema_slow_1h=1, ema_fast_4h=1, ema_slow_4h=1,
    )
    assert sig.bias == "long"


def test_legacy_bias_neutral_when_mixed():
    sig = MTFSignal(
        tf_1h="long", tf_4h="short",
        ema_fast_1h=1, ema_slow_1h=1, ema_fast_4h=1, ema_slow_4h=1,
    )
    assert sig.bias == "neutral"


def test_require_both_kwarg_is_accepted_but_ignored():
    mtf_new = MTFFilter(symbol="BTC")
    mtf_legacy = MTFFilter(symbol="BTC", require_both=False)
    # Should not raise and should classify identically on the same data.
    closes = np.linspace(100, 200, 200)
    for m in (mtf_new, mtf_legacy):
        m._df_1h = _make_hourly(closes)
        m._df_4h = _make_hourly(closes)
    ts = mtf_new._df_1h["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
    assert mtf_new.get_bias_at(ts).tf_1h == mtf_legacy.get_bias_at(ts).tf_1h

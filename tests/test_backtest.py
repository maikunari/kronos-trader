"""Tests for backtest.py — uses synthetic data, deterministic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import BacktestResult, Trade, run_snipe_backtest, walk_forward
from regime import RegimeDetector
from snipe_signal_engine import SnipeSignalEngine


# --- Helpers ------------------------------------------------------------------

def _synth_15m(n_bars: int, drift: float = 0.0008, sigma: float = 0.002,
               phi: float = 0.3, seed: int = 0) -> pd.DataFrame:
    """AR(1) drift + noise in log-return space."""
    rng = np.random.default_rng(seed)
    ret = np.zeros(n_bars)
    ret[0] = rng.normal(drift, sigma)
    for i in range(1, n_bars):
        ret[i] = drift + phi * (ret[i - 1] - drift) + rng.normal(0, sigma)
    close = 100.0 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, sigma, n_bars)))
    low = close * (1 - np.abs(rng.normal(0, sigma, n_bars)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = rng.uniform(800, 1200, n_bars)
    ts = pd.date_range("2025-01-01", periods=n_bars, freq="15min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low,
        "close": close, "volume": vol,
    })


def _engine_factory():
    def make():
        return SnipeSignalEngine(
            regime=RegimeDetector(
                adx_period=14, adx_threshold=20, hurst_window=300,
                hurst_threshold=0.50, rv_window=48, rv_hist_window=200,
            ),
            donchian_period=20,
            composite_threshold=-1.0,          # accept neutral composite in tests
        )
    return make


# --- Core backtest ------------------------------------------------------------

def test_backtest_runs_on_synthetic_uptrend():
    df = _synth_15m(n_bars=2_000, drift=0.0012, sigma=0.0015, phi=0.5, seed=1)
    result = run_snipe_backtest(
        df, engine=_engine_factory()(), initial_capital=10_000.0,
        fee_rate=0.00035, slippage_base_bps=2.0, slippage_atr_frac=0.05,
        max_hold_bars=200,
    )
    assert isinstance(result, BacktestResult)
    # Basic shape invariants
    assert len(result.equity_curve) >= 1
    assert result.equity_curve[0] == pytest.approx(result.initial_capital)
    # Fees + slippage should be non-negative
    assert result.fees_total >= 0
    assert result.slippage_total >= 0


def test_missing_columns_raises():
    df = pd.DataFrame({"close": [100, 101]})
    with pytest.raises(ValueError):
        run_snipe_backtest(df, engine=_engine_factory()())


def test_no_trades_on_flat_noise():
    rng = np.random.default_rng(9)
    closes = 100 + np.cumsum(rng.normal(0, 0.05, 1_500)) * 0.05
    highs = closes + 0.1
    lows = closes - 0.1
    opens = np.concatenate([[closes[0]], closes[:-1]])
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=1_500, freq="15min", tz="UTC"),
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": np.full_like(closes, 1000.0),
    })
    result = run_snipe_backtest(df, engine=_engine_factory()(), initial_capital=10_000.0)
    # With no drift, regime gate should reject almost everything
    assert result.trades_count <= 3


# --- Summary stats ------------------------------------------------------------

def test_summary_contains_expected_fields():
    df = _synth_15m(n_bars=1_500, drift=0.001, sigma=0.0015, seed=2)
    result = run_snipe_backtest(df, engine=_engine_factory()(), initial_capital=10_000.0)
    summary = result.summary()
    for key in ("return_pct", "sharpe", "max_dd", "trades", "win_rate", "profit_factor", "cost_drag"):
        assert key in summary


def test_final_equity_equals_initial_plus_pnls():
    df = _synth_15m(n_bars=1_500, drift=0.0012, sigma=0.0015, seed=3)
    result = run_snipe_backtest(df, engine=_engine_factory()(), initial_capital=10_000.0)
    pnl_sum = sum(t.pnl_usd for t in result.trades)
    assert abs(result.final_equity - (result.initial_capital + pnl_sum)) < 1e-6


# --- Funding cost accrual ----------------------------------------------------

def test_funding_accrues_for_long_positions_with_positive_rate():
    df = _synth_15m(n_bars=1_500, drift=0.0012, sigma=0.0015, seed=4)
    result = run_snipe_backtest(
        df, engine=_engine_factory()(), initial_capital=10_000.0,
        funding_rate_hourly=0.001,     # 0.1%/hr — very high, easy to detect
    )
    # If any long trades were taken, expect some funding to be recorded.
    long_trades = [t for t in result.trades if t.direction == "long"]
    if long_trades:
        assert any(t.funding_usd > 0 for t in long_trades)


# --- Purged walk-forward -----------------------------------------------------

def test_walk_forward_produces_folds():
    df = _synth_15m(n_bars=3 * 96 * 90, drift=0.0008, sigma=0.0015, seed=5)  # ~90 days
    folds = walk_forward(
        df, engine_factory=_engine_factory(),
        in_sample_days=30, out_of_sample_days=14, embargo_days=1, step_days=14,
    )
    assert len(folds) >= 1
    for f in folds:
        assert f.in_sample_start < f.in_sample_end <= f.oos_start < f.oos_end
        # Embargo at least matches config
        assert (f.oos_start - f.in_sample_end) >= pd.Timedelta(days=1)

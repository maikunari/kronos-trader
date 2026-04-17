"""End-to-end integration test: deterministic historical replay.

Loads a frozen 30-day slice of real BTC 15m data and runs the full
snipe_signal_engine backtest with the shipped config. Asserts exact
baseline metrics so any silent regression in the pipeline (engine,
risk, sizing, fill modeling, ATR math) surfaces immediately.

The fixture is checked in (tests/fixtures/btc_15m_2025_03.csv) so this
test runs without network access.

If the pipeline intentionally changes semantics and baselines must
update, re-run tests/_print_integration_baseline.py (trivial helper
below) and paste the new values in. Do not blindly match — investigate
the change first.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml

from backtest import run_snipe_backtest
from main import build_engine


FIXTURE = Path(__file__).parent / "fixtures" / "btc_15m_2025_03.csv"


@pytest.fixture(scope="module")
def frozen_fixture():
    df = pd.read_csv(FIXTURE, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@pytest.fixture(scope="module")
def shipped_config():
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    return yaml.safe_load(cfg_path.read_text())


def _run(df, config):
    engine = build_engine(config)
    return run_snipe_backtest(
        df, engine=engine,
        initial_capital=10_000.0,
        fee_rate=0.00035,
        slippage_base_bps=2.0,
        slippage_atr_frac=0.05,
        funding_rate_hourly=0.0,
        use_chandelier_trail=True,
        chandelier_atr_mult=3.0,
        max_hold_bars=0,
    )


def test_backtest_baseline_metrics(frozen_fixture, shipped_config, caplog):
    """Golden values for March 2025 BTC 15m slice + shipped config.

    Update deliberately (not casually) after intentional pipeline changes.
    """
    caplog.set_level(logging.WARNING, logger="risk_manager")
    result = _run(frozen_fixture, shipped_config)

    # Exact invariants — these are deterministic.
    assert result.trades_count == 80
    assert result.final_equity == pytest.approx(9048.2897, abs=1e-3)
    assert result.sharpe_ratio == pytest.approx(-13.3595, abs=1e-3)
    assert result.max_drawdown_pct == pytest.approx(0.0955, abs=1e-3)
    assert result.fees_total == pytest.approx(217.5834, abs=1e-3)
    assert result.slippage_total == pytest.approx(240.4511, abs=1e-3)


def test_determinism_two_runs_identical(frozen_fixture, shipped_config, caplog):
    """Two runs on the same input must produce identical results."""
    caplog.set_level(logging.WARNING, logger="risk_manager")
    a = _run(frozen_fixture, shipped_config)
    b = _run(frozen_fixture, shipped_config)
    assert a.final_equity == b.final_equity
    assert a.trades_count == b.trades_count
    assert a.sharpe_ratio == b.sharpe_ratio
    assert a.max_drawdown_pct == b.max_drawdown_pct


def test_cost_breakdown_rational(frozen_fixture, shipped_config, caplog):
    """Fees and slippage per trade should be positive and in sensible ranges."""
    caplog.set_level(logging.WARNING, logger="risk_manager")
    result = _run(frozen_fixture, shipped_config)
    assert result.fees_total > 0
    assert result.slippage_total > 0
    # Average per-trade cost should be a small fraction of the average position size
    avg_cost = (result.fees_total + result.slippage_total) / max(1, result.trades_count)
    # Rough bound: < 1% of a ~$3000 average size => < $30
    assert avg_cost < 30.0

"""Tests for the rewritten RiskManager."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from risk_manager import RiskManager


# --- Approve happy path -------------------------------------------------------

def test_approve_fallback_to_hard_cap_without_instrument_vol():
    rm = RiskManager(initial_equity=10_000, max_position_pct=0.25)
    approval = rm.approve_trade("long", current_price=100.0)
    assert approval.approved
    assert approval.position_size_usd == pytest.approx(2_500.0)   # 25% of 10k
    assert approval.sizing_breakdown["vol_target_size_usd"] is None


def test_vol_targeting_shrinks_in_high_vol():
    rm = RiskManager(initial_equity=10_000, target_annual_vol=0.20, max_position_pct=1.0)
    # Instrument vol = 80% annualized -> size = 10k * 0.20 / 0.80 = 2_500
    approval = rm.approve_trade("long", current_price=100.0, instrument_annual_vol=0.80)
    assert approval.position_size_usd == pytest.approx(2_500.0)


def test_vol_targeting_grows_in_low_vol_but_clipped_by_hard_cap():
    rm = RiskManager(initial_equity=10_000, target_annual_vol=0.20, max_position_pct=0.30)
    # Very low vol -> unbounded would go to 20_000; hard cap clips at 3_000
    approval = rm.approve_trade("long", current_price=100.0, instrument_annual_vol=0.05)
    assert approval.position_size_usd == pytest.approx(3_000.0)


# --- Rejection paths ----------------------------------------------------------

def test_flat_signal_rejected():
    rm = RiskManager(initial_equity=10_000)
    assert not rm.approve_trade("flat", current_price=100.0).approved


def test_invalid_price_rejected():
    rm = RiskManager(initial_equity=10_000)
    assert not rm.approve_trade("long", current_price=0.0).approved


def test_max_concurrent_positions_blocks_second_entry():
    rm = RiskManager(initial_equity=10_000, max_concurrent_positions=1)
    rm.approve_trade("long", current_price=100.0)
    rm.on_trade_open()
    approval = rm.approve_trade("long", current_price=100.0)
    assert not approval.approved
    assert "max_concurrent" in approval.reason


# --- Daily loss halt ----------------------------------------------------------

def test_daily_loss_limit_halts_trading():
    rm = RiskManager(initial_equity=10_000, daily_loss_limit_pct=0.03)
    rm.approve_trade("long", current_price=100.0)
    rm.on_trade_open()
    rm.on_trade_close(pnl=-400.0, won=False)   # -4%, past limit
    approval = rm.approve_trade("long", current_price=100.0)
    assert not approval.approved
    assert "daily_loss_limit" in approval.reason


def test_win_does_not_halt():
    rm = RiskManager(initial_equity=10_000, daily_loss_limit_pct=0.03)
    rm.approve_trade("long", current_price=100.0)
    rm.on_trade_open()
    rm.on_trade_close(pnl=500.0, won=True)
    assert rm.approve_trade("long", current_price=100.0).approved


# --- Consecutive losses -------------------------------------------------------

def test_three_consecutive_losses_trigger_cooloff():
    t0 = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    rm = RiskManager(
        initial_equity=100_000,
        daily_loss_limit_pct=0.50,                # disable daily-loss halt to isolate
        consecutive_loss_limit=3,
        consecutive_loss_cooloff_seconds=600,
    )
    for i in range(3):
        rm.on_trade_close(pnl=-50.0, won=False, now=t0 + timedelta(minutes=i))
    # Within cooloff
    approval = rm.approve_trade("long", current_price=100.0, now=t0 + timedelta(minutes=5))
    assert not approval.approved
    assert "consecutive_loss_cooloff" in approval.reason
    # After cooloff
    approval = rm.approve_trade("long", current_price=100.0, now=t0 + timedelta(minutes=15))
    assert approval.approved


def test_winning_trade_resets_consecutive_loss_counter():
    t0 = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    rm = RiskManager(
        initial_equity=100_000,
        daily_loss_limit_pct=0.50,
        consecutive_loss_limit=3,
    )
    rm.on_trade_close(pnl=-50.0, won=False, now=t0)
    rm.on_trade_close(pnl=-50.0, won=False, now=t0 + timedelta(minutes=1))
    rm.on_trade_close(pnl=+150.0, won=True,  now=t0 + timedelta(minutes=2))
    rm.on_trade_close(pnl=-50.0, won=False, now=t0 + timedelta(minutes=3))
    rm.on_trade_close(pnl=-50.0, won=False, now=t0 + timedelta(minutes=4))
    # Only 2 consecutive at this point -> no cooloff
    assert rm.approve_trade("long", current_price=100.0, now=t0 + timedelta(minutes=5)).approved


# --- Drawdown halving ---------------------------------------------------------

def test_rolling_drawdown_halves_sizing():
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rm = RiskManager(
        initial_equity=10_000,
        max_position_pct=0.50,
        weekly_drawdown_halve_pct=0.08,
        consecutive_loss_limit=999,   # don't cooloff
        daily_loss_limit_pct=0.99,
    )
    # Push equity up to establish a peak
    rm.on_trade_close(pnl=+500.0, won=True,  now=t0)
    rm.on_trade_close(pnl=+500.0, won=True,  now=t0 + timedelta(hours=1))
    # 11_000 peak. Now lose 10% (dd = 9.1%) -> triggers halving
    rm.on_trade_close(pnl=-1_000.0, won=False, now=t0 + timedelta(hours=2))
    approval = rm.approve_trade("long", current_price=100.0,
                                  now=t0 + timedelta(hours=3))
    # Without halving: 10_000 * 0.50 = 5_000; with halving: 2_500
    assert approval.approved
    assert approval.position_size_usd == pytest.approx(2_500.0)
    assert approval.sizing_breakdown.get("drawdown_halve_factor") == 0.5


# --- Manual halt + tripwire ---------------------------------------------------

def test_manual_halt_and_resume():
    rm = RiskManager(initial_equity=10_000)
    rm.halt()
    assert rm.is_halted()
    assert not rm.approve_trade("long", current_price=100.0).approved
    rm.resume()
    assert rm.approve_trade("long", current_price=100.0).approved


def test_tripwire_file_blocks_entries(tmp_path: Path):
    wire = tmp_path / "halt.flag"
    rm = RiskManager(initial_equity=10_000, tripwire_file=str(wire))
    assert rm.approve_trade("long", current_price=100.0).approved
    wire.write_text("stop")
    assert not rm.approve_trade("long", current_price=100.0).approved
    wire.unlink()
    assert rm.approve_trade("long", current_price=100.0).approved


# --- Fractional Kelly --------------------------------------------------------

def test_kelly_overlay_clips_size_when_edge_is_small():
    rm = RiskManager(
        initial_equity=10_000, max_position_pct=1.0,
        target_annual_vol=1.0,                # make vol target large
        kelly_fraction=0.25,
    )
    # win_rate=0.55, payoff=2.0 -> Kelly f* = (2*0.55 - 0.45) / 2 = 0.325
    # quarter-kelly = 0.0813, size = 813
    approval = rm.approve_trade(
        "long", current_price=100.0,
        instrument_annual_vol=1.0,
        win_rate=0.55, payoff_ratio=2.0,
    )
    assert approval.approved
    assert approval.sizing_breakdown["kelly_f"] == pytest.approx(0.0813, abs=1e-3)
    assert approval.position_size_usd <= 900   # well below equity


def test_kelly_zero_edge_produces_no_trade_when_enabled():
    rm = RiskManager(initial_equity=10_000, max_position_pct=1.0,
                     target_annual_vol=1.0, kelly_fraction=0.25)
    # win_rate=0.4, payoff=1 -> Kelly negative, clipped to 0 -> size 0
    approval = rm.approve_trade(
        "long", current_price=100.0, instrument_annual_vol=1.0,
        win_rate=0.40, payoff_ratio=1.0,
    )
    assert approval.approved  # approved but size 0
    assert approval.position_size_usd == 0.0

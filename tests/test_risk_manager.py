"""Smoke tests locking down current RiskManager behavior before rewrite."""
from __future__ import annotations

from risk_manager import RiskManager


def test_approve_happy_path():
    rm = RiskManager(initial_equity=10_000, max_position_pct=0.10)
    approval = rm.approve_trade("long", current_price=100.0)
    assert approval.approved
    assert approval.position_size_usd == 1000.0
    assert approval.position_size_contracts == 10.0


def test_flat_signal_rejected():
    rm = RiskManager(initial_equity=10_000)
    approval = rm.approve_trade("flat", current_price=100.0)
    assert not approval.approved
    assert "no signal" in approval.reason.lower()


def test_max_concurrent_positions_blocks_second_entry():
    rm = RiskManager(initial_equity=10_000, max_concurrent_positions=1)
    rm.approve_trade("long", current_price=100.0)
    rm.on_trade_open()
    approval = rm.approve_trade("long", current_price=100.0)
    assert not approval.approved
    assert "max concurrent" in approval.reason.lower()


def test_daily_loss_limit_halts_trading():
    rm = RiskManager(initial_equity=10_000, daily_loss_limit_pct=0.03)
    rm.approve_trade("long", current_price=100.0)
    rm.on_trade_open()
    # Lose 4% (past the 3% limit)
    rm.on_trade_close(pnl=-400.0, won=False)
    approval = rm.approve_trade("long", current_price=100.0)
    assert not approval.approved
    assert "halted" in approval.reason.lower()


def test_equity_tracks_pnl():
    rm = RiskManager(initial_equity=10_000)
    rm.approve_trade("long", current_price=100.0)
    rm.on_trade_open()
    rm.on_trade_close(pnl=150.0, won=True)
    assert rm.get_equity() == 10_150.0
    stats = rm.get_daily_stats()
    assert stats.trades_taken == 1
    assert stats.trades_won == 1
    assert stats.realized_pnl == 150.0


def test_winning_trade_does_not_halt_even_if_large_loss_limit_configured():
    """A big win should never trip the daily-loss halt."""
    rm = RiskManager(initial_equity=10_000, daily_loss_limit_pct=0.03)
    rm.approve_trade("long", current_price=100.0)
    rm.on_trade_open()
    rm.on_trade_close(pnl=500.0, won=True)
    approval = rm.approve_trade("long", current_price=100.0)
    assert approval.approved

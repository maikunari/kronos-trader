"""Unit tests for the pure helpers in dashboard.py (no streamlit required)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from backtest import BacktestResult, Trade
from dashboard import (
    cost_attribution,
    exit_reason_breakdown,
    load_optimizer_results,
    optimizer_leaderboard,
    trades_to_frame,
)


def _trade(pnl: float = 10.0, reason: str = "target") -> Trade:
    return Trade(
        entry_ts=pd.Timestamp("2025-01-01", tz="UTC"),
        exit_ts=pd.Timestamp("2025-01-01 00:30", tz="UTC"),
        direction="long",
        entry_price=100.0, exit_price=101.0, size_usd=1000.0,
        pnl_usd=pnl, exit_reason=reason,
        fees_usd=0.70, slippage_usd=0.40, funding_usd=0.10,
    )


def _result(trades: list[Trade]) -> BacktestResult:
    return BacktestResult(
        trades=trades, equity_curve=[10_000.0, 10_100.0],
        initial_capital=10_000.0, final_equity=10_100.0,
        total_return_pct=0.01, sharpe_ratio=1.5, max_drawdown_pct=0.02,
        win_rate=0.6, profit_factor=2.0, avg_win_pct=0.01, avg_loss_pct=-0.005,
        trades_count=len(trades),
        fees_total=sum(t.fees_usd for t in trades),
        slippage_total=sum(t.slippage_usd for t in trades),
        funding_total=sum(t.funding_usd for t in trades),
        start_date="2025-01-01", end_date="2025-01-02",
    )


def test_trades_to_frame_empty():
    df = trades_to_frame([])
    assert list(df.columns) == [
        "entry_ts", "exit_ts", "direction", "entry_price", "exit_price",
        "size_usd", "pnl_usd", "exit_reason", "fees_usd", "slippage_usd",
        "funding_usd",
    ]


def test_trades_to_frame_populates_rows():
    df = trades_to_frame([_trade(10), _trade(-5, reason="stop")])
    assert len(df) == 2
    assert set(df["exit_reason"]) == {"target", "stop"}


def test_cost_attribution_rows_and_signs():
    trades = [_trade(10, "target"), _trade(-5, "stop")]
    res = _result(trades)
    ca = cost_attribution(res)
    gross_row = ca[ca["component"] == "Gross PnL"].iloc[0]
    assert gross_row["usd"] > 0
    fees_row = ca[ca["component"] == "Fees"].iloc[0]
    assert fees_row["usd"] < 0


def test_exit_reason_breakdown():
    trades = [_trade(10, "target"), _trade(-5, "stop"), _trade(8, "target")]
    br = exit_reason_breakdown(trades)
    target_row = br[br["exit_reason"] == "target"].iloc[0]
    assert target_row["count"] == 2
    assert target_row["total_pnl_usd"] == pytest.approx(18)


def test_exit_reason_breakdown_empty():
    assert exit_reason_breakdown([]).empty


def test_load_optimizer_results_missing_returns_none(tmp_path: Path):
    p = tmp_path / "nope.json"
    assert load_optimizer_results(str(p)) is None


def test_load_optimizer_results_bad_json_returns_none(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid")
    assert load_optimizer_results(str(p)) is None


def test_optimizer_leaderboard_flattens_params(tmp_path: Path):
    payload = {
        "symbol": "BTC", "timeframe": "15m",
        "start": "2024-01-01", "end": "2024-02-01",
        "results": [
            {
                "params": {"donchian_period": 20, "stop_atr_mult": 1.5},
                "median_oos_sharpe": 1.2, "mean_oos_return": 0.03,
                "median_oos_max_dd": 0.05, "median_oos_win_rate": 0.55,
                "median_oos_profit_factor": 1.8, "folds": 4, "total_trades": 80,
            },
        ],
    }
    lb = optimizer_leaderboard(payload)
    assert lb.iloc[0]["donchian_period"] == 20
    assert lb.iloc[0]["median_oos_sharpe"] == 1.2

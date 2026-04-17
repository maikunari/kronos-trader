"""
backtest.py
Backtest engine: simulate trades on historical OHLCV data.

Usage:
    python backtest.py --config config.yaml
    python backtest.py --config config.yaml --symbol ETH --timeframe 5m
    python backtest.py --config config.yaml --start 2024-06-01 --end 2024-12-31
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from hyperliquid_feed import fetch_historical
from atr_engine import ATREngine, ATRSignal
from risk_manager import RiskManager
from mtf_filter import MTFFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    side: str
    entry_price: float
    exit_price: float
    size_usd: float
    gross_pnl: float
    fees: float
    slippage: float
    net_pnl: float
    won: bool
    exit_reason: str  # "target" | "stop" | "end_of_data"


@dataclass
class BacktestResult:
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_backtest(
    config: dict,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> BacktestResult:
    """Run full backtest and return results."""
    t = config["trading"]
    b = config["backtest"]
    r = config["risk"]

    symbol = symbol or t["symbol"]
    timeframe = timeframe or t["timeframe"]
    start_date = start_date or b["start_date"]
    end_date = end_date or b["end_date"]
    initial_capital = b["initial_capital"]
    fee_rate = b["fee_rate"]
    slippage_rate = b["slippage_rate"]
    data_source = b.get("data_source", "hyperliquid")

    logger.info(f"Backtest: {symbol}/{timeframe} | {start_date} → {end_date} | capital=${initial_capital:,.0f}")

    # Fetch data
    df = fetch_historical(symbol, timeframe, start_date, end_date, source=data_source)
    logger.info(f"Data: {len(df)} candles")

    # MTF filter
    mtf_cfg = config.get("mtf", {})
    mtf = None
    if mtf_cfg.get("enabled", False):
        mtf = MTFFilter(
            symbol=symbol,
            ema_fast=mtf_cfg.get("ema_fast", 20),
            ema_slow=mtf_cfg.get("ema_slow", 50),
            require_both=mtf_cfg.get("require_both", True),
            data_source=b.get("data_source", "hyperliquid"),
        )
        mtf.load_backtest_data(start_date, end_date)

    engine = ATREngine(
        ema_fast=mtf_cfg.get("ema_fast", 20),
        ema_slow=mtf_cfg.get("ema_slow", 50),
        atr_period=14,
        stop_multiplier=1.5,
        target_multiplier=3.0,
        mtf_filter=mtf,
    )
    risk = RiskManager(
        initial_equity=initial_capital,
        max_position_pct=r["max_position_pct"],
        daily_loss_limit_pct=r["daily_loss_limit_pct"],
        max_concurrent_positions=r["max_concurrent_positions"],
    )

    lookback = max(mtf_cfg.get("ema_slow", 50), 14) + 10  # enough for EMA50 + ATR14
    trades: List[Trade] = []
    equity_curve = [initial_capital]

    open_trade: Optional[dict] = None

    logger.info("Running simulation...")
    for i in range(lookback, len(df)):
        candle = df.iloc[i]
        candles_window = df.iloc[i - lookback: i]

        # --- Manage open trade ---
        if open_trade is not None:
            side = open_trade["side"]
            stop = open_trade["stop"]
            target = open_trade["target"]
            entry_price = open_trade["entry_price"]
            size_usd = open_trade["size_usd"]
            entry_time = open_trade["entry_time"]

            exit_price = None
            exit_reason = None

            if side == "long":
                if candle["low"] <= stop:
                    exit_price = stop
                    exit_reason = "stop"
                elif candle["high"] >= target:
                    exit_price = target
                    exit_reason = "target"
            else:  # short
                if candle["high"] >= stop:
                    exit_price = stop
                    exit_reason = "stop"
                elif candle["low"] <= target:
                    exit_price = target
                    exit_reason = "target"

            if exit_price is not None:
                # Apply slippage to exit
                if exit_reason == "stop":
                    if side == "long":
                        exit_price *= (1 - slippage_rate)
                    else:
                        exit_price *= (1 + slippage_rate)

                gross_pnl = _calc_pnl(side, entry_price, exit_price, size_usd)
                fees = size_usd * fee_rate * 2  # entry + exit
                slip_cost = size_usd * slippage_rate * 2
                net_pnl = gross_pnl - fees - slip_cost
                won = net_pnl > 0

                trade = Trade(
                    entry_time=entry_time,
                    exit_time=candle["timestamp"],
                    side=side,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    size_usd=size_usd,
                    gross_pnl=gross_pnl,
                    fees=fees,
                    slippage=slip_cost,
                    net_pnl=net_pnl,
                    won=won,
                    exit_reason=exit_reason,
                )
                trades.append(trade)
                risk.on_trade_close(net_pnl, won)
                equity_curve.append(risk.get_equity())
                open_trade = None

        # --- Check for new signal (only if no open trade) ---
        if open_trade is None:
            signal: ATRSignal = engine.evaluate(candles_window, timestamp=candle["timestamp"])
            approval = risk.approve_trade(signal.action, float(candle["close"]))

            if approval.approved and signal.action in ("long", "short"):
                # Entry on next candle open (simulate realistic entry)
                if i + 1 < len(df):
                    next_open = float(df.iloc[i + 1]["open"])
                    # Apply entry slippage
                    if signal.action == "long":
                        entry_price = next_open * (1 + slippage_rate)
                        stop = entry_price * (1 - t["stop_pct"])
                        target = entry_price * (1 + t["target_pct"])
                    else:
                        entry_price = next_open * (1 - slippage_rate)
                        stop = entry_price * (1 + t["stop_pct"])
                        target = entry_price * (1 - t["target_pct"])

                    open_trade = {
                        "side": signal.action,
                        "entry_price": entry_price,
                        "stop": stop,
                        "target": target,
                        "size_usd": approval.position_size_usd,
                        "entry_time": df.iloc[i + 1]["timestamp"],
                    }
                    risk.on_trade_open()

    # Close any remaining open trade at last price
    if open_trade is not None:
        exit_price = float(df.iloc[-1]["close"])
        gross_pnl = _calc_pnl(open_trade["side"], open_trade["entry_price"], exit_price, open_trade["size_usd"])
        fees = open_trade["size_usd"] * fee_rate * 2
        net_pnl = gross_pnl - fees
        won = net_pnl > 0
        trades.append(Trade(
            entry_time=open_trade["entry_time"],
            exit_time=df.iloc[-1]["timestamp"],
            side=open_trade["side"],
            entry_price=open_trade["entry_price"],
            exit_price=exit_price,
            size_usd=open_trade["size_usd"],
            gross_pnl=gross_pnl,
            fees=fees,
            slippage=0.0,
            net_pnl=net_pnl,
            won=won,
            exit_reason="end_of_data",
        ))
        risk.on_trade_close(net_pnl, won)
        equity_curve.append(risk.get_equity())

    return _compute_result(
        symbol, timeframe, start_date, end_date, initial_capital,
        risk.get_equity(), trades, equity_curve
    )


def _calc_pnl(side: str, entry: float, exit_p: float, size_usd: float) -> float:
    if side == "long":
        return (exit_p - entry) / entry * size_usd
    else:
        return (entry - exit_p) / entry * size_usd


def _compute_result(symbol, timeframe, start_date, end_date, initial_capital,
                    final_equity, trades, equity_curve) -> BacktestResult:
    total = len(trades)
    if total == 0:
        logger.warning("No trades taken in backtest period")
        return BacktestResult(
            symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date,
            initial_capital=initial_capital, final_equity=initial_capital,
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
            avg_win_pct=0, avg_loss_pct=0, profit_factor=0, total_return_pct=0,
            max_drawdown_pct=0, sharpe_ratio=0, trades=[], equity_curve=equity_curve
        )

    winners = [t for t in trades if t.won]
    losers = [t for t in trades if not t.won]
    win_rate = len(winners) / total

    wins = [t.net_pnl / t.size_usd for t in winners] if winners else [0]
    losses = [t.net_pnl / t.size_usd for t in losers] if losers else [0]
    avg_win_pct = float(np.mean(wins))
    avg_loss_pct = float(np.mean(losses))

    gross_wins = sum(t.net_pnl for t in winners) if winners else 0
    gross_losses = abs(sum(t.net_pnl for t in losers)) if losers else 1
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    total_return = (final_equity - initial_capital) / initial_capital

    # Max drawdown
    eq = np.array(equity_curve)
    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq - running_max) / running_max
    max_dd = float(drawdowns.min())

    # Sharpe (daily, annualized)
    trade_returns = [t.net_pnl / initial_capital for t in trades]
    if len(trade_returns) > 1:
        sharpe = float(np.mean(trade_returns) / (np.std(trade_returns) + 1e-10) * np.sqrt(252))
    else:
        sharpe = 0.0

    return BacktestResult(
        symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date,
        initial_capital=initial_capital, final_equity=final_equity,
        total_trades=total, winning_trades=len(winners), losing_trades=len(losers),
        win_rate=win_rate, avg_win_pct=avg_win_pct, avg_loss_pct=avg_loss_pct,
        profit_factor=profit_factor, total_return_pct=total_return,
        max_drawdown_pct=max_dd, sharpe_ratio=sharpe,
        trades=trades, equity_curve=equity_curve,
    )


def print_report(result: BacktestResult):
    print("\n" + "=" * 60)
    print(f"  BACKTEST REPORT: {result.symbol}/{result.timeframe}")
    print(f"  {result.start_date} → {result.end_date}")
    print("=" * 60)
    print(f"  Initial capital:   ${result.initial_capital:>12,.2f}")
    print(f"  Final equity:      ${result.final_equity:>12,.2f}")
    print(f"  Total return:       {result.total_return_pct:>11.2%}")
    print(f"  Max drawdown:       {result.max_drawdown_pct:>11.2%}")
    print(f"  Sharpe ratio:       {result.sharpe_ratio:>11.2f}")
    print("-" * 60)
    print(f"  Total trades:       {result.total_trades:>11}")
    print(f"  Winning trades:     {result.winning_trades:>11}")
    print(f"  Losing trades:      {result.losing_trades:>11}")
    print(f"  Win rate:           {result.win_rate:>11.1%}")
    print(f"  Avg win:            {result.avg_win_pct:>11.2%}")
    print(f"  Avg loss:           {result.avg_loss_pct:>11.2%}")
    print(f"  Profit factor:      {result.profit_factor:>11.2f}")
    print("=" * 60 + "\n")


def plot_equity_curve(result: BacktestResult, output_path: str = "equity_curve.png"):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Equity curve
    ax1 = axes[0]
    ax1.plot(result.equity_curve, color="#00c896", linewidth=1.5, label="Equity")
    ax1.axhline(result.initial_capital, color="#888", linestyle="--", linewidth=0.8, label="Initial capital")
    ax1.set_title(f"Equity Curve — {result.symbol}/{result.timeframe} ({result.start_date} → {result.end_date})")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2 = axes[1]
    eq = np.array(result.equity_curve)
    running_max = np.maximum.accumulate(eq)
    drawdown = (eq - running_max) / running_max * 100
    ax2.fill_between(range(len(drawdown)), drawdown, 0, color="#ff4444", alpha=0.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Trade #")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Equity curve saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Kronos + TimesFM Backtester")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--symbol", help="Override symbol (e.g. ETH)")
    parser.add_argument("--timeframe", help="Override timeframe (1m/5m/15m)")
    parser.add_argument("--start", dest="start_date", help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end", dest="end_date", help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--plot", default="equity_curve.png", help="Output path for equity curve PNG")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    result = run_backtest(
        config,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print_report(result)
    plot_equity_curve(result, output_path=args.plot)


if __name__ == "__main__":
    main()

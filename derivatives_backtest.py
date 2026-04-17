"""
derivatives_backtest.py
Standalone backtest for the Derivatives (Funding Rate) strategy.
No ML model dependencies — runs fast.

Strategy: Counter-leverage squeeze plays using Hyperliquid funding rates.
  - Extreme positive funding + D1 downtrend → SHORT
  - Extreme negative funding + D1 uptrend  → LONG

Usage:
    python derivatives_backtest.py
    python derivatives_backtest.py --start 2026-01-01 --end 2026-04-14
    python derivatives_backtest.py --threshold 0.0003 --timeframe 4h
    python derivatives_backtest.py --threshold 0.0005 --no-mtf
    python derivatives_backtest.py --sweep   # test multiple thresholds
"""

import argparse
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from hyperliquid_feed import fetch_historical
from derivatives_feed import fetch_funding_for_backtest
from derivatives_signal_engine import DerivativesSignalEngine
from risk_manager import RiskManager
from mtf_filter import MTFFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes (self-contained, no dependency on backtest.py)
# ---------------------------------------------------------------------------

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
    exit_reason: str
    funding_rate: float = 0.0   # funding rate at signal time
    mtf_bias: str = ""


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
    funding_threshold: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def run_backtest(
    config: dict,
    symbol: str = "SOL",
    timeframe: str = "4h",
    start_date: str = "2026-01-01",
    end_date: str = "2026-04-14",
    funding_threshold: float = 0.0005,
    use_mtf: bool = True,
) -> BacktestResult:

    b = config["backtest"]
    r = config["risk"]
    t = config["trading"]
    initial_capital = b["initial_capital"]
    fee_rate = b["fee_rate"]
    slippage_rate = b["slippage_rate"]
    data_source = b.get("data_source", "hyperliquid")

    logger.info(
        f"=== Derivatives Backtest: {symbol}/{timeframe} | {start_date} → {end_date} ==="
    )
    logger.info(
        f"Funding threshold: ±{funding_threshold:.4%}/hr | "
        f"MTF: {'on' if use_mtf else 'off'} | "
        f"Capital: ${initial_capital:,.0f}"
    )

    # --- Price data ---
    df = fetch_historical(symbol, timeframe, start_date, end_date, source=data_source)
    if df.empty:
        raise RuntimeError(f"No OHLCV data for {symbol}/{timeframe}")
    logger.info(f"OHLCV: {len(df)} candles")

    # --- Funding rate data ---
    funding_df = fetch_funding_for_backtest(symbol, start_date, end_date)
    if funding_df.empty:
        raise RuntimeError("No funding history fetched — check Hyperliquid API")
    logger.info(f"Funding: {len(funding_df)} records | "
                f"range: {float(funding_df['funding_rate'].min()):.6f} to "
                f"{float(funding_df['funding_rate'].max()):.6f}")

    # How many extreme readings?
    extreme = (funding_df["funding_rate"].abs() >= funding_threshold).sum()
    logger.info(f"Extreme readings (>= threshold): {extreme} / {len(funding_df)} "
                f"({extreme/len(funding_df):.1%})")

    # --- MTF filter ---
    mtf = None
    if use_mtf:
        mtf_cfg = config.get("mtf", {})
        mtf = MTFFilter(
            symbol=symbol,
            ema_fast=mtf_cfg.get("ema_fast", 20),
            ema_slow=mtf_cfg.get("ema_slow", 50),
            require_both=False,
            use_d1=True,
            data_source=data_source,
        )
        mtf.load_backtest_data(start_date, end_date)

    # --- Signal engine ---
    engine = DerivativesSignalEngine(
        funding_df=funding_df,
        funding_threshold=funding_threshold,
        stop_pct=0.005,     # 0.5% stop — wider for 4H candles
        target_pct=0.010,   # 1.0% target → 2:1 R:R
        mtf_filter=mtf,
    )

    # --- Risk manager ---
    risk = RiskManager(
        initial_equity=initial_capital,
        max_position_pct=r["max_position_pct"],
        daily_loss_limit_pct=r["daily_loss_limit_pct"],
        max_concurrent_positions=r["max_concurrent_positions"],
    )

    # --- Simulation loop ---
    lookback = 5
    trades: List[Trade] = []
    equity_curve = [initial_capital]
    open_trade: Optional[dict] = None

    for i in range(lookback, len(df)):
        candle = df.iloc[i]
        candles_window = df.iloc[i - lookback: i + 1]

        # Manage open trade
        if open_trade is not None:
            side = open_trade["side"]
            stop = open_trade["stop"]
            target = open_trade["target"]
            ep = open_trade["entry_price"]
            size = open_trade["size_usd"]
            et = open_trade["entry_time"]

            exit_price = None
            exit_reason = None

            if side == "long":
                if candle["low"] <= stop:
                    exit_price = stop
                    exit_reason = "stop"
                elif candle["high"] >= target:
                    exit_price = target
                    exit_reason = "target"
            else:
                if candle["high"] >= stop:
                    exit_price = stop
                    exit_reason = "stop"
                elif candle["low"] <= target:
                    exit_price = target
                    exit_reason = "target"

            if exit_price is not None:
                if exit_reason == "stop":
                    exit_price *= (1 - slippage_rate) if side == "long" else (1 + slippage_rate)

                gross = _pnl(side, ep, exit_price, size)
                fees = size * fee_rate * 2
                slip = size * slippage_rate * 2
                net = gross - fees - slip
                won = net > 0

                trades.append(Trade(
                    entry_time=et,
                    exit_time=candle["timestamp"],
                    side=side,
                    entry_price=ep,
                    exit_price=exit_price,
                    size_usd=size,
                    gross_pnl=gross,
                    fees=fees,
                    slippage=slip,
                    net_pnl=net,
                    won=won,
                    exit_reason=exit_reason,
                    funding_rate=open_trade.get("funding_rate", 0.0),
                    mtf_bias=open_trade.get("mtf_bias", ""),
                ))
                risk.on_trade_close(net, won)
                equity_curve.append(risk.get_equity())
                open_trade = None

        # New signal
        if open_trade is None:
            signal = engine.evaluate(candles_window, timestamp=candle["timestamp"])
            approval = risk.approve_trade(signal.action, float(candle["close"]))

            if approval.approved and signal.action in ("long", "short"):
                if i + 1 < len(df):
                    next_open = float(df.iloc[i + 1]["open"])
                    if signal.action == "long":
                        ep = next_open * (1 + slippage_rate)
                        stop = ep * (1 - engine.stop_pct)
                        target = ep * (1 + engine.target_pct)
                    else:
                        ep = next_open * (1 - slippage_rate)
                        stop = ep * (1 + engine.stop_pct)
                        target = ep * (1 - engine.target_pct)

                    open_trade = {
                        "side": signal.action,
                        "entry_price": ep,
                        "stop": stop,
                        "target": target,
                        "size_usd": approval.position_size_usd,
                        "entry_time": df.iloc[i + 1]["timestamp"],
                        "funding_rate": signal.funding_rate,
                        "mtf_bias": signal.mtf_bias,
                    }
                    risk.on_trade_open()
                    logger.info(
                        f"ENTRY {signal.action.upper()} @ {ep:.2f} | "
                        f"stop={stop:.2f} target={target:.2f} | "
                        f"funding={signal.funding_rate:.6f} mtf={signal.mtf_bias}"
                    )

    # Close any remaining open trade
    if open_trade is not None:
        exit_price = float(df.iloc[-1]["close"])
        gross = _pnl(open_trade["side"], open_trade["entry_price"], exit_price, open_trade["size_usd"])
        fees = open_trade["size_usd"] * fee_rate * 2
        net = gross - fees
        won = net > 0
        trades.append(Trade(
            entry_time=open_trade["entry_time"],
            exit_time=df.iloc[-1]["timestamp"],
            side=open_trade["side"],
            entry_price=open_trade["entry_price"],
            exit_price=exit_price,
            size_usd=open_trade["size_usd"],
            gross_pnl=gross,
            fees=fees,
            slippage=0.0,
            net_pnl=net,
            won=won,
            exit_reason="end_of_data",
            funding_rate=open_trade.get("funding_rate", 0.0),
            mtf_bias=open_trade.get("mtf_bias", ""),
        ))
        risk.on_trade_close(net, won)
        equity_curve.append(risk.get_equity())

    return _compute_result(
        symbol, timeframe, start_date, end_date, initial_capital,
        risk.get_equity(), trades, equity_curve, funding_threshold,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pnl(side: str, entry: float, exit_p: float, size: float) -> float:
    if side == "long":
        return (exit_p - entry) / entry * size
    return (entry - exit_p) / entry * size


def _compute_result(
    symbol, timeframe, start_date, end_date, initial_capital,
    final_equity, trades, equity_curve, funding_threshold=0.0,
) -> BacktestResult:
    total = len(trades)
    if total == 0:
        logger.warning("No trades taken")
        return BacktestResult(
            symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date,
            initial_capital=initial_capital, final_equity=initial_capital,
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
            avg_win_pct=0, avg_loss_pct=0, profit_factor=0, total_return_pct=0,
            max_drawdown_pct=0, sharpe_ratio=0, funding_threshold=funding_threshold,
            trades=[], equity_curve=equity_curve,
        )

    winners = [t for t in trades if t.won]
    losers = [t for t in trades if not t.won]
    win_rate = len(winners) / total

    wins = [t.net_pnl / t.size_usd for t in winners] if winners else [0.0]
    losses = [t.net_pnl / t.size_usd for t in losers] if losers else [0.0]
    avg_win = float(np.mean(wins))
    avg_loss = float(np.mean(losses))

    gross_wins = sum(t.net_pnl for t in winners) if winners else 0.0
    gross_losses = abs(sum(t.net_pnl for t in losers)) if losers else 1.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    total_return = (final_equity - initial_capital) / initial_capital

    eq = np.array(equity_curve)
    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq - running_max) / running_max
    max_dd = float(drawdowns.min())

    # Daily Sharpe
    daily_pnl: dict = defaultdict(float)
    for t in trades:
        day = t.entry_time.date() if hasattr(t.entry_time, "date") else pd.Timestamp(t.entry_time).date()
        daily_pnl[day] += t.net_pnl
    if len(daily_pnl) > 1:
        daily_rets = np.array([v / initial_capital for v in daily_pnl.values()])
        sharpe = float(np.mean(daily_rets) / (np.std(daily_rets) + 1e-10) * np.sqrt(252))
    else:
        sharpe = 0.0

    return BacktestResult(
        symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date,
        initial_capital=initial_capital, final_equity=final_equity,
        total_trades=total, winning_trades=len(winners), losing_trades=len(losers),
        win_rate=win_rate, avg_win_pct=avg_win, avg_loss_pct=avg_loss,
        profit_factor=profit_factor, total_return_pct=total_return,
        max_drawdown_pct=max_dd, sharpe_ratio=sharpe,
        funding_threshold=funding_threshold,
        trades=trades, equity_curve=equity_curve,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(result: BacktestResult):
    print("\n" + "=" * 65)
    print(f"  DERIVATIVES BACKTEST: {result.symbol}/{result.timeframe}")
    print(f"  {result.start_date} → {result.end_date}")
    print(f"  Funding threshold: ±{result.funding_threshold:.4%}/hr")
    print("=" * 65)
    print(f"  Initial capital:    ${result.initial_capital:>12,.2f}")
    print(f"  Final equity:       ${result.final_equity:>12,.2f}")
    print(f"  Total return:        {result.total_return_pct:>11.2%}")
    print(f"  Max drawdown:        {result.max_drawdown_pct:>11.2%}")
    print(f"  Sharpe ratio:        {result.sharpe_ratio:>11.2f}")
    print("-" * 65)
    print(f"  Total trades:        {result.total_trades:>11}")
    print(f"  Winning trades:      {result.winning_trades:>11}")
    print(f"  Losing trades:       {result.losing_trades:>11}")
    print(f"  Win rate:            {result.win_rate:>11.1%}")
    print(f"  Avg win:             {result.avg_win_pct:>11.2%}")
    print(f"  Avg loss:            {result.avg_loss_pct:>11.2%}")
    print(f"  Profit factor:       {result.profit_factor:>11.2f}")
    print("=" * 65 + "\n")

    if result.trades:
        long_trades = [t for t in result.trades if t.side == "long"]
        short_trades = [t for t in result.trades if t.side == "short"]
        print(f"  Long trades:  {len(long_trades)} | Short trades: {len(short_trades)}")
        if long_trades:
            lw = sum(1 for t in long_trades if t.won)
            print(f"  Long WR:  {lw}/{len(long_trades)} ({lw/len(long_trades):.1%})")
        if short_trades:
            sw = sum(1 for t in short_trades if t.won)
            print(f"  Short WR: {sw}/{len(short_trades)} ({sw/len(short_trades):.1%})")
        print()


def plot_results(result: BacktestResult, output_path: str = "results_derivatives_2026.png"):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1, 1]})

    # Equity curve
    ax1 = axes[0]
    ax1.plot(result.equity_curve, color="#00c896", linewidth=1.5, label="Equity")
    ax1.axhline(result.initial_capital, color="#888", linestyle="--", linewidth=0.8, label="Initial capital")
    ax1.set_title(
        f"Derivatives Strategy — {result.symbol}/{result.timeframe} "
        f"({result.start_date} → {result.end_date})\n"
        f"Funding threshold: ±{result.funding_threshold:.4%}/hr | "
        f"Return: {result.total_return_pct:.2%} | WR: {result.win_rate:.1%} | "
        f"PF: {result.profit_factor:.2f}"
    )
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
    ax2.grid(True, alpha=0.3)

    # Trade P&L bar chart
    ax3 = axes[2]
    if result.trades:
        pnls = [t.net_pnl for t in result.trades]
        colors = ["#00c896" if p > 0 else "#ff4444" for p in pnls]
        ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
        ax3.axhline(0, color="#888", linewidth=0.5)
        ax3.set_ylabel("Trade P&L ($)")
        ax3.set_xlabel("Trade #")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Chart saved: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def run_sweep(config: dict, symbol: str, timeframe: str, start: str, end: str):
    """Test multiple funding thresholds and print comparison table."""
    thresholds = [0.00001, 0.00003, 0.00005, 0.00008, 0.0001, 0.00015, 0.0002]
    print("\n" + "=" * 90)
    print(f"  THRESHOLD SWEEP: {symbol}/{timeframe} | {start} → {end}")
    print("=" * 90)
    print(f"  {'Threshold':>12} {'Trades':>7} {'WR':>7} {'Return':>9} {'MaxDD':>8} {'PF':>8} {'Sharpe':>8}")
    print("-" * 90)

    results = []
    for thresh in thresholds:
        try:
            r = run_backtest(config, symbol, timeframe, start, end, thresh, use_mtf=True)
            results.append(r)
            print(
                f"  {thresh:>11.4%}  {r.total_trades:>7}  {r.win_rate:>6.1%}  "
                f"{r.total_return_pct:>8.2%}  {r.max_drawdown_pct:>7.2%}  "
                f"{r.profit_factor:>7.2f}  {r.sharpe_ratio:>7.2f}"
            )
        except Exception as e:
            print(f"  {thresh:>11.4%}  ERROR: {e}")
    print("=" * 90 + "\n")

    # Find best by profit factor (with min 5 trades)
    valid = [r for r in results if r.total_trades >= 5 and r.profit_factor != float("inf")]
    if valid:
        best = max(valid, key=lambda r: r.profit_factor)
        print(f"  Best by profit factor: threshold={best.funding_threshold:.4%}/hr "
              f"(PF={best.profit_factor:.2f}, WR={best.win_rate:.1%}, "
              f"return={best.total_return_pct:.2%})\n")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Derivatives (Funding Rate) Strategy Backtester")
    parser.add_argument("--symbol", default="SOL")
    parser.add_argument("--timeframe", default="4h", help="Candle timeframe (4h recommended)")
    parser.add_argument("--start", default="2026-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-04-14", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--threshold", type=float, default=0.0005,
        help="Funding rate extreme threshold per hour (e.g. 0.0005 = 0.05%%/hr)"
    )
    parser.add_argument("--no-mtf", action="store_true", help="Disable D1 MTF trend filter")
    parser.add_argument("--sweep", action="store_true", help="Sweep multiple thresholds")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--plot", default="results_derivatives_2026.png")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.sweep:
        run_sweep(config, args.symbol, args.timeframe, args.start, args.end)
        return

    result = run_backtest(
        config,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        funding_threshold=args.threshold,
        use_mtf=not args.no_mtf,
    )

    print_report(result)
    plot_results(result, output_path=args.plot)


if __name__ == "__main__":
    main()

"""
trend_backtest.py
Backtest the simple EMA crossover trend-following strategy.
No ML dependencies. Runs in seconds.

2:1 R:R always enforced via ATR-based stops/targets.
Win rate target: >33% (break-even), aiming for 40%+.

Usage:
    python trend_backtest.py
    python trend_backtest.py --timeframe 1h --ema-fast 9 --ema-slow 21
    python trend_backtest.py --sweep          # test all timeframe/EMA combos
    python trend_backtest.py --no-mtf         # disable D1 filter
    python trend_backtest.py --start 2026-01-01 --end 2026-04-14
"""

import argparse
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from hyperliquid_feed import fetch_historical
from trend_signal_engine import TrendSignalEngine, TradeSignal
from risk_manager import RiskManager
from mtf_filter import MTFFilter

logging.basicConfig(
    level=logging.WARNING,   # suppress noise; use --verbose for debug
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    side: str
    entry_price: float
    exit_price: float
    size_usd: float
    net_pnl: float
    won: bool
    exit_reason: str


@dataclass
class BacktestResult:
    symbol: str
    timeframe: str
    ema_fast: int
    ema_slow: int
    atr_stop_mult: float
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
    use_mtf: bool
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def run_backtest(
    config: dict,
    symbol: str = "SOL",
    timeframe: str = "1h",
    start_date: str = "2026-01-01",
    end_date: str = "2026-04-14",
    ema_fast: int = 9,
    ema_slow: int = 21,
    atr_period: int = 14,
    atr_stop_mult: float = 1.5,
    atr_target_mult: float = 3.0,
    use_mtf: bool = True,
    verbose: bool = False,
) -> BacktestResult:

    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    b = config["backtest"]
    r = config["risk"]
    initial_capital = b["initial_capital"]
    fee_rate = b["fee_rate"]
    slippage_rate = b["slippage_rate"]
    data_source = b.get("data_source", "hyperliquid")

    # --- Price data ---
    df = fetch_historical(symbol, timeframe, start_date, end_date, source=data_source)
    if df.empty:
        raise RuntimeError(f"No data for {symbol}/{timeframe}")

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
    engine = TrendSignalEngine(
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        atr_period=atr_period,
        atr_stop_mult=atr_stop_mult,
        atr_target_mult=atr_target_mult,
        mtf_filter=mtf,
        require_mtf=use_mtf,
    )

    # --- Risk manager ---
    risk = RiskManager(
        initial_equity=initial_capital,
        max_position_pct=r["max_position_pct"],
        daily_loss_limit_pct=r["daily_loss_limit_pct"],
        max_concurrent_positions=r["max_concurrent_positions"],
    )

    lookback = ema_slow + atr_period + 5
    trades: List[Trade] = []
    equity_curve = [initial_capital]
    open_trade: Optional[dict] = None

    for i in range(lookback, len(df)):
        candle = df.iloc[i]
        candles_window = df.iloc[i - lookback: i + 1]

        # --- Manage open trade ---
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

                if side == "long":
                    gross = (exit_price - ep) / ep * size
                else:
                    gross = (ep - exit_price) / ep * size
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
                    net_pnl=net,
                    won=won,
                    exit_reason=exit_reason,
                ))
                risk.on_trade_close(net, won)
                equity_curve.append(risk.get_equity())
                open_trade = None

        # --- New signal ---
        if open_trade is None:
            signal: TradeSignal = engine.evaluate(candles_window, timestamp=candle["timestamp"])
            approval = risk.approve_trade(signal.action, float(candle["close"]))

            if approval.approved and signal.action in ("long", "short"):
                if i + 1 < len(df):
                    next_open = float(df.iloc[i + 1]["open"])
                    # Recalculate stops from actual entry price (next bar open)
                    stop_dist = engine.atr_stop_mult * signal.atr
                    target_dist = engine.atr_target_mult * signal.atr

                    if signal.action == "long":
                        ep = next_open * (1 + slippage_rate)
                        stop = ep - stop_dist
                        target = ep + target_dist
                    else:
                        ep = next_open * (1 - slippage_rate)
                        stop = ep + stop_dist
                        target = ep - target_dist

                    open_trade = {
                        "side": signal.action,
                        "entry_price": ep,
                        "stop": stop,
                        "target": target,
                        "size_usd": approval.position_size_usd,
                        "entry_time": df.iloc[i + 1]["timestamp"],
                    }
                    risk.on_trade_open()
                    if verbose:
                        rr = target_dist / stop_dist
                        logger.info(
                            f"ENTRY {signal.action.upper()} @ {ep:.2f} | "
                            f"stop={stop:.2f} target={target:.2f} | "
                            f"R:R={rr:.2f} MTF={signal.mtf_bias}"
                        )

    # Close remaining trade at last price
    if open_trade is not None:
        exit_price = float(df.iloc[-1]["close"])
        ep = open_trade["entry_price"]
        size = open_trade["size_usd"]
        side = open_trade["side"]
        if side == "long":
            gross = (exit_price - ep) / ep * size
        else:
            gross = (ep - exit_price) / ep * size
        fees = size * fee_rate * 2
        net = gross - fees
        won = net > 0
        trades.append(Trade(
            entry_time=open_trade["entry_time"],
            exit_time=df.iloc[-1]["timestamp"],
            side=side,
            entry_price=ep,
            exit_price=exit_price,
            size_usd=size,
            net_pnl=net,
            won=won,
            exit_reason="end_of_data",
        ))
        risk.on_trade_close(net, won)
        equity_curve.append(risk.get_equity())

    return _compute_result(
        symbol, timeframe, ema_fast, ema_slow, atr_stop_mult,
        start_date, end_date, initial_capital,
        risk.get_equity(), trades, equity_curve, use_mtf,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_result(
    symbol, timeframe, ema_fast, ema_slow, atr_stop_mult,
    start_date, end_date, initial_capital,
    final_equity, trades, equity_curve, use_mtf,
) -> BacktestResult:
    total = len(trades)
    if total == 0:
        return BacktestResult(
            symbol=symbol, timeframe=timeframe, ema_fast=ema_fast, ema_slow=ema_slow,
            atr_stop_mult=atr_stop_mult, start_date=start_date, end_date=end_date,
            initial_capital=initial_capital, final_equity=initial_capital,
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
            avg_win_pct=0, avg_loss_pct=0, profit_factor=0, total_return_pct=0,
            max_drawdown_pct=0, sharpe_ratio=0, use_mtf=use_mtf,
            trades=[], equity_curve=equity_curve,
        )

    winners = [t for t in trades if t.won]
    losers = [t for t in trades if not t.won]
    win_rate = len(winners) / total

    wins = [t.net_pnl / t.size_usd for t in winners] if winners else [0.0]
    losses = [t.net_pnl / t.size_usd for t in losers] if losers else [0.0]

    gross_wins = sum(t.net_pnl for t in winners) if winners else 0.0
    gross_losses = abs(sum(t.net_pnl for t in losers)) if losers else 1.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    total_return = (final_equity - initial_capital) / initial_capital

    eq = np.array(equity_curve)
    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq - running_max) / running_max
    max_dd = float(drawdowns.min())

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
        symbol=symbol, timeframe=timeframe, ema_fast=ema_fast, ema_slow=ema_slow,
        atr_stop_mult=atr_stop_mult, start_date=start_date, end_date=end_date,
        initial_capital=initial_capital, final_equity=final_equity,
        total_trades=total, winning_trades=len(winners), losing_trades=len(losers),
        win_rate=win_rate, avg_win_pct=float(np.mean(wins)), avg_loss_pct=float(np.mean(losses)),
        profit_factor=profit_factor, total_return_pct=total_return,
        max_drawdown_pct=max_dd, sharpe_ratio=sharpe, use_mtf=use_mtf,
        trades=trades, equity_curve=equity_curve,
    )


def print_report(result: BacktestResult):
    rr = 3.0 / result.atr_stop_mult  # target_mult / stop_mult
    print("\n" + "=" * 65)
    print(f"  TREND BACKTEST: {result.symbol}/{result.timeframe}")
    print(f"  {result.start_date} → {result.end_date}")
    print(f"  EMA({result.ema_fast}/{result.ema_slow}) | ATR stop={result.atr_stop_mult}x → {rr:.1f}:1 R:R")
    print(f"  D1 MTF filter: {'ON' if result.use_mtf else 'OFF'}")
    print("=" * 65)
    print(f"  Initial capital:    ${result.initial_capital:>12,.2f}")
    print(f"  Final equity:       ${result.final_equity:>12,.2f}")
    print(f"  Total return:        {result.total_return_pct:>11.2%}")
    print(f"  Max drawdown:        {result.max_drawdown_pct:>11.2%}")
    print(f"  Sharpe ratio:        {result.sharpe_ratio:>11.2f}")
    print("-" * 65)
    print(f"  Total trades:        {result.total_trades:>11}")
    print(f"  Win rate:            {result.win_rate:>11.1%}   (need >33% at 2:1)")
    print(f"  Profit factor:       {result.profit_factor:>11.2f}")
    print(f"  Avg win:             {result.avg_win_pct:>11.2%}")
    print(f"  Avg loss:            {result.avg_loss_pct:>11.2%}")
    print("=" * 65)
    if result.trades:
        longs = [t for t in result.trades if t.side == "long"]
        shorts = [t for t in result.trades if t.side == "short"]
        if longs:
            lw = sum(1 for t in longs if t.won)
            print(f"  Long:  {lw}/{len(longs)} ({lw/len(longs):.0%})", end="  ")
        if shorts:
            sw = sum(1 for t in shorts if t.won)
            print(f"Short: {sw}/{len(shorts)} ({sw/len(shorts):.0%})", end="")
        print()
    print()


def plot_results(result: BacktestResult, output_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
    rr = 3.0 / result.atr_stop_mult

    ax1 = axes[0]
    ax1.plot(result.equity_curve, color="#00c896", linewidth=1.5)
    ax1.axhline(result.initial_capital, color="#666", linestyle="--", linewidth=0.8)
    ax1.set_title(
        f"EMA({result.ema_fast}/{result.ema_slow}) Trend — {result.symbol}/{result.timeframe} "
        f"({result.start_date} → {result.end_date})\n"
        f"R:R={rr:.1f}:1 | Return={result.total_return_pct:.2%} | "
        f"WR={result.win_rate:.1%} | PF={result.profit_factor:.2f} | "
        f"Trades={result.total_trades}"
    )
    ax1.set_ylabel("Portfolio ($)")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    eq = np.array(result.equity_curve)
    rmax = np.maximum.accumulate(eq)
    dd = (eq - rmax) / rmax * 100
    ax2.fill_between(range(len(dd)), dd, 0, color="#ff4444", alpha=0.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Trade #")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Chart saved: {output_path}")


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

def run_sweep(config: dict, symbol: str, start: str, end: str, use_mtf: bool):
    """Sweep timeframes and EMA combos, find best by profit factor."""
    combos = [
        # (timeframe, ema_fast, ema_slow, atr_stop_mult)
        ("15m", 5,  13, 1.5),
        ("15m", 9,  21, 1.5),
        ("15m", 9,  21, 2.0),
        ("15m", 12, 26, 1.5),
        ("1h",  5,  13, 1.5),
        ("1h",  9,  21, 1.5),
        ("1h",  9,  21, 2.0),
        ("1h",  12, 26, 1.5),
        ("1h",  20, 50, 1.5),
        ("4h",  5,  13, 1.5),
        ("4h",  9,  21, 1.5),
        ("4h",  9,  21, 2.0),
        ("4h",  12, 26, 1.5),
    ]

    mtf_label = "MTF=ON" if use_mtf else "MTF=OFF"
    print("\n" + "=" * 100)
    print(f"  SWEEP: {symbol} | {start} → {end} | {mtf_label}")
    print("=" * 100)
    print(f"  {'TF':>4} {'EMA':>8} {'Stop':>5} {'Trades':>7} {'WR':>7} {'Return':>9} {'MaxDD':>8} {'PF':>8} {'Sharpe':>8}")
    print("-" * 100)

    results = []
    for tf, ef, es, sm in combos:
        try:
            r = run_backtest(config, symbol, tf, start, end, ef, es, 14, sm, sm * 2.0, use_mtf)
            results.append(r)
            wr_flag = " ✓" if r.win_rate > 0.33 and r.total_trades >= 5 else "  "
            print(
                f"  {tf:>4} {ef}/{es:>5}  {sm:.1f}x  "
                f"{r.total_trades:>7}  {r.win_rate:>6.1%}{wr_flag}  "
                f"{r.total_return_pct:>8.2%}  {r.max_drawdown_pct:>7.2%}  "
                f"{r.profit_factor:>7.2f}  {r.sharpe_ratio:>7.2f}"
            )
        except Exception as e:
            print(f"  {tf:>4} {ef}/{es:>5}  {sm:.1f}x  ERROR: {e}")

    print("=" * 100)

    # Best by profit factor (min 5 trades, WR > 33%)
    valid = [r for r in results if r.total_trades >= 5 and r.win_rate > 0.33]
    if valid:
        best = max(valid, key=lambda r: r.profit_factor)
        print(f"\n  ★ Best: {best.timeframe} EMA({best.ema_fast}/{best.ema_slow}) "
              f"stop={best.atr_stop_mult}x → WR={best.win_rate:.1%} "
              f"PF={best.profit_factor:.2f} return={best.total_return_pct:.2%}\n")
        return best
    else:
        valid_any = [r for r in results if r.total_trades >= 5]
        if valid_any:
            best = max(valid_any, key=lambda r: r.win_rate)
            print(f"\n  Best WR (no 33% combo found): {best.timeframe} EMA({best.ema_fast}/{best.ema_slow}) "
                  f"→ WR={best.win_rate:.1%} trades={best.total_trades}\n")
        else:
            print("\n  No combos with ≥5 trades found.\n")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EMA Trend Following Backtester (2:1 R:R)")
    parser.add_argument("--symbol", default="SOL")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-04-14")
    parser.add_argument("--ema-fast", type=int, default=9, dest="ema_fast")
    parser.add_argument("--ema-slow", type=int, default=21, dest="ema_slow")
    parser.add_argument("--atr-stop", type=float, default=1.5, dest="atr_stop",
                        help="ATR multiplier for stop (target = 2x stop → 2:1 R:R)")
    parser.add_argument("--no-mtf", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="Sweep all TF/EMA combos")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--plot", default="results_trend_2026.png")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    use_mtf = not args.no_mtf

    if args.sweep:
        best = run_sweep(config, args.symbol, args.start, args.end, use_mtf)
        if best:
            plot_results(best, args.plot)
        return

    result = run_backtest(
        config,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        ema_fast=args.ema_fast,
        ema_slow=args.ema_slow,
        atr_stop_mult=args.atr_stop,
        atr_target_mult=args.atr_stop * 2.0,
        use_mtf=use_mtf,
        verbose=args.verbose,
    )

    print_report(result)
    plot_results(result, args.plot)


if __name__ == "__main__":
    main()

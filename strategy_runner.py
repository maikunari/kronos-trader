"""
strategy_runner.py
Pluggable backtest harness for agent-generated strategies.

Reuses the simulation pattern from trend_backtest.py (line-for-line where
possible) but accepts any object with `.evaluate(candles) -> dict` returning
the contract in strategy_template.md.

Public entry points:
  - run_simulation(strategy, candles, initial_capital, fee_rate, ...)
      called in-process
  - run_from_file(strategy_path, config_json)
      called by the agent as a subprocess; prints JSON metrics to stdout
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


REQUIRED_SIGNAL_KEYS = {"action", "stop_pct", "target_pct"}
VALID_ACTIONS = {"long", "short", "flat"}


@dataclass
class Trade:
    entry_time: Any
    exit_time: Any
    side: str
    entry_price: float
    exit_price: float
    size_usd: float
    net_pnl: float
    won: bool
    exit_reason: str


@dataclass
class RunMetrics:
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
    initial_capital: float
    final_equity: float
    long_trades: int
    short_trades: int
    skip_reasons: Dict[str, int] = field(default_factory=dict)


def run_simulation(
    strategy: Any,
    candles: pd.DataFrame,
    *,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.00035,
    slippage_rate: float = 0.0002,
    max_position_pct: float = 0.10,
    lookback: int = 75,
) -> RunMetrics:
    """
    Walk forward through `candles` one bar at a time. When flat, query the
    strategy for a signal; when in a trade, check the bar's high/low for
    stop/target hits.

    Position size is fixed at `max_position_pct * initial_capital` (no
    compounding) to keep variant comparisons apples-to-apples.
    """
    if lookback >= len(candles):
        raise RuntimeError(
            f"candles too short ({len(candles)}) for lookback {lookback}"
        )

    size_usd = initial_capital * max_position_pct
    equity = initial_capital
    equity_curve: List[float] = [initial_capital]
    trades: List[Trade] = []
    skip_reasons: Dict[str, int] = defaultdict(int)
    open_trade: Optional[dict] = None

    df = candles.reset_index(drop=True)

    for i in range(lookback, len(df)):
        candle = df.iloc[i]
        window = df.iloc[i - lookback:i + 1]

        # --- Manage open trade: check if this bar hit stop or target ---
        if open_trade is not None:
            side = open_trade["side"]
            stop = open_trade["stop"]
            target = open_trade["target"]
            ep = open_trade["entry_price"]
            et = open_trade["entry_time"]

            exit_price = None
            exit_reason = None

            if side == "long":
                if candle["low"] <= stop:
                    exit_price, exit_reason = stop, "stop"
                elif candle["high"] >= target:
                    exit_price, exit_reason = target, "target"
            else:  # short
                if candle["high"] >= stop:
                    exit_price, exit_reason = stop, "stop"
                elif candle["low"] <= target:
                    exit_price, exit_reason = target, "target"

            if exit_price is not None:
                if exit_reason == "stop":
                    exit_price *= (1 - slippage_rate) if side == "long" else (1 + slippage_rate)

                if side == "long":
                    gross = (exit_price - ep) / ep * size_usd
                else:
                    gross = (ep - exit_price) / ep * size_usd
                fees = size_usd * fee_rate * 2
                slip = size_usd * slippage_rate * 2
                net = gross - fees - slip

                trades.append(Trade(
                    entry_time=et,
                    exit_time=candle["timestamp"],
                    side=side,
                    entry_price=ep,
                    exit_price=exit_price,
                    size_usd=size_usd,
                    net_pnl=net,
                    won=net > 0,
                    exit_reason=exit_reason,
                ))
                equity += net
                equity_curve.append(equity)
                open_trade = None

        # --- Look for new signal when flat ---
        if open_trade is None:
            try:
                signal = strategy.evaluate(window)
            except Exception as e:
                # A strategy that crashes once is disqualified.
                raise RuntimeError(f"strategy.evaluate raised: {type(e).__name__}: {e}") from e

            if not isinstance(signal, dict):
                raise RuntimeError(f"signal must be dict, got {type(signal).__name__}")
            missing = REQUIRED_SIGNAL_KEYS - signal.keys()
            if missing:
                raise RuntimeError(f"signal missing keys: {sorted(missing)}")
            action = signal["action"]
            if action not in VALID_ACTIONS:
                raise RuntimeError(f"invalid action: {action!r}")

            if action == "flat":
                reason = signal.get("skip_reason", "unspecified")
                skip_reasons[str(reason)[:80]] += 1
                continue

            stop_pct = float(signal["stop_pct"])
            target_pct = float(signal["target_pct"])
            if not (0.0005 <= stop_pct <= 0.20):
                raise RuntimeError(f"stop_pct out of range: {stop_pct}")
            if not (0.0005 <= target_pct <= 0.50):
                raise RuntimeError(f"target_pct out of range: {target_pct}")

            if i + 1 >= len(df):
                break
            next_open = float(df.iloc[i + 1]["open"])
            if action == "long":
                ep = next_open * (1 + slippage_rate)
                stop = ep * (1 - stop_pct)
                target = ep * (1 + target_pct)
            else:
                ep = next_open * (1 - slippage_rate)
                stop = ep * (1 + stop_pct)
                target = ep * (1 - target_pct)

            open_trade = {
                "side": action,
                "entry_price": ep,
                "stop": stop,
                "target": target,
                "entry_time": df.iloc[i + 1]["timestamp"],
            }

    # Close any lingering trade at the last close
    if open_trade is not None:
        exit_price = float(df.iloc[-1]["close"])
        ep = open_trade["entry_price"]
        side = open_trade["side"]
        if side == "long":
            gross = (exit_price - ep) / ep * size_usd
        else:
            gross = (ep - exit_price) / ep * size_usd
        fees = size_usd * fee_rate * 2
        net = gross - fees
        trades.append(Trade(
            entry_time=open_trade["entry_time"],
            exit_time=df.iloc[-1]["timestamp"],
            side=side,
            entry_price=ep,
            exit_price=exit_price,
            size_usd=size_usd,
            net_pnl=net,
            won=net > 0,
            exit_reason="end_of_data",
        ))
        equity += net
        equity_curve.append(equity)

    return _compute_metrics(trades, equity_curve, initial_capital, equity, dict(skip_reasons))


def _compute_metrics(
    trades: List[Trade],
    equity_curve: List[float],
    initial_capital: float,
    final_equity: float,
    skip_reasons: Dict[str, int],
) -> RunMetrics:
    total = len(trades)
    if total == 0:
        return RunMetrics(
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0, avg_win_pct=0.0, avg_loss_pct=0.0,
            profit_factor=0.0, total_return_pct=0.0,
            max_drawdown_pct=0.0, sharpe_ratio=0.0,
            initial_capital=initial_capital, final_equity=final_equity,
            long_trades=0, short_trades=0, skip_reasons=skip_reasons,
        )

    winners = [t for t in trades if t.won]
    losers = [t for t in trades if not t.won]
    wins = [t.net_pnl / t.size_usd for t in winners] if winners else [0.0]
    losses = [t.net_pnl / t.size_usd for t in losers] if losers else [0.0]

    gross_w = sum(t.net_pnl for t in winners) if winners else 0.0
    gross_l = abs(sum(t.net_pnl for t in losers)) if losers else 0.0
    if gross_l > 0:
        pf = gross_w / gross_l
    else:
        pf = float("inf") if gross_w > 0 else 0.0

    eq = np.array(equity_curve, dtype=float)
    rmax = np.maximum.accumulate(eq)
    max_dd = float(((eq - rmax) / rmax).min())

    daily = defaultdict(float)
    for t in trades:
        ts = t.entry_time
        day = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
        daily[day] += t.net_pnl
    if len(daily) > 1:
        daily_rets = np.array([v / initial_capital for v in daily.values()])
        sharpe = float(np.mean(daily_rets) / (np.std(daily_rets) + 1e-10) * np.sqrt(252))
    else:
        sharpe = 0.0

    return RunMetrics(
        total_trades=total,
        winning_trades=len(winners),
        losing_trades=len(losers),
        win_rate=len(winners) / total,
        avg_win_pct=float(np.mean(wins)),
        avg_loss_pct=float(np.mean(losses)),
        profit_factor=pf,
        total_return_pct=(final_equity - initial_capital) / initial_capital,
        max_drawdown_pct=max_dd,
        sharpe_ratio=sharpe,
        initial_capital=initial_capital,
        final_equity=final_equity,
        long_trades=sum(1 for t in trades if t.side == "long"),
        short_trades=sum(1 for t in trades if t.side == "short"),
        skip_reasons=skip_reasons,
    )


def _load_strategy_from_file(path: str) -> Any:
    """Import a strategy module by file path and instantiate its Strategy class."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(f"_generated_{p.stem}", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # the AST validator runs BEFORE this
    if not hasattr(module, "Strategy"):
        raise AttributeError("module has no `Strategy` class")
    return module.Strategy()


def run_from_file(strategy_path: str, candles_parquet: str, config: dict) -> dict:
    """
    Subprocess entry point. Loads the strategy, runs the simulation against
    pre-fetched candles on disk, and returns a JSON-serializable dict.
    """
    candles = pd.read_parquet(candles_parquet)
    strategy = _load_strategy_from_file(strategy_path)
    metrics = run_simulation(
        strategy,
        candles,
        initial_capital=float(config.get("initial_capital", 10_000.0)),
        fee_rate=float(config.get("fee_rate", 0.00035)),
        slippage_rate=float(config.get("slippage_rate", 0.0002)),
        max_position_pct=float(config.get("max_position_pct", 0.10)),
        lookback=int(config.get("lookback", 75)),
    )
    d = metrics.__dict__.copy()
    return d


def _cli():
    """
    Tiny CLI so the agent can invoke this in a subprocess:

        python strategy_runner.py <strategy.py> <candles.parquet> <config.json>

    On success, prints one line of JSON metrics to stdout. Nonzero exit on
    any failure (validation error, strategy raised, timeout handled by
    caller).
    """
    if len(sys.argv) != 4:
        print("usage: strategy_runner.py <strategy.py> <candles.parquet> <config.json>",
              file=sys.stderr)
        sys.exit(2)
    strategy_path, candles_path, config_path = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(config_path) as f:
        config = json.load(f)
    metrics = run_from_file(strategy_path, candles_path, config)
    print(json.dumps(metrics))


if __name__ == "__main__":
    _cli()

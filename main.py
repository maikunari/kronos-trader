"""
main.py
Entry point for the Kronos + TimesFM trading bot.

Modes:
  --mode backtest   Run historical backtest (no API keys needed)
  --mode paper      Live candle feed, signals logged but no orders placed
  --mode live       Live trading on Hyperliquid (requires .env credentials)
"""

import argparse
import logging
import os
import sys
import time
from collections import deque

import yaml
from dotenv import load_dotenv

from hyperliquid_feed import LiveFeed, TIMEFRAME_MS
from kronos_model import KronosModel
from timesfm_model import TimesFMModel
from signal_engine import SignalEngine
from risk_manager import RiskManager
from executor import Executor
from mtf_filter import MTFFilter
from optimizer import WalkForwardOptimizer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_live(config: dict, mode: str, config_path: str = "config.yaml"):
    """Run paper or live trading loop."""
    t = config["trading"]
    m = config["models"]
    r = config["risk"]
    hl = config["hyperliquid"]

    symbol = t["symbol"]
    timeframe = t["timeframe"]
    lookback = m["lookback_candles"]

    logger.info(f"Starting {mode.upper()} trading: {symbol}/{timeframe}")

    # Init models
    kronos = KronosModel(model_size=m["kronos_model_size"], cache_dir=m["model_cache_dir"])
    timesfm_model = TimesFMModel(
        model_name=m["timesfm_model"],
        horizon=m["trend_horizon"],
        cache_dir=m["model_cache_dir"],
    )

    # MTF filter
    mtf_cfg = config.get("mtf", {})
    mtf = None
    if mtf_cfg.get("enabled", False):
        mtf = MTFFilter(
            symbol=symbol,
            ema_fast=mtf_cfg.get("ema_fast", 20),
            ema_slow=mtf_cfg.get("ema_slow", 50),
            require_both=mtf_cfg.get("require_both", True),
        )
        logger.info("MTF filter enabled (1H + 4H EMA trend)")

    engine = SignalEngine(
        kronos=kronos,
        timesfm=timesfm_model,
        min_move_threshold=t["min_move_threshold"],
        stop_pct=t["stop_pct"],
        target_pct=t["target_pct"],
        mtf_filter=mtf,
    )

    # Optimizer
    opt_cfg = config.get("optimizer", {})
    optimizer = None
    if opt_cfg.get("enabled", False):
        optimizer = WalkForwardOptimizer(config_path=config_path)
        optimizer.start_background()
        logger.info("Walk-forward optimizer started")
    executor = Executor(mode=mode, mainnet=hl["mainnet"])
    equity = executor.get_account_equity()
    risk = RiskManager(
        initial_equity=equity,
        max_position_pct=r["max_position_pct"],
        daily_loss_limit_pct=r["daily_loss_limit_pct"],
        max_concurrent_positions=r["max_concurrent_positions"],
    )

    logger.info(f"Account equity: ${equity:,.2f}")

    # Rolling candle buffer — we only act on completed candles
    candle_buffer: deque = deque(maxlen=lookback + 1)
    open_trade = None

    interval_ms = TIMEFRAME_MS.get(timeframe, 900_000)
    last_completed_ts = None

    def on_candle(candle: dict):
        nonlocal open_trade, last_completed_ts

        # Buffer every tick; act only when a new candle opens (prev one completed)
        candle_buffer.append(candle)

        current_ts = candle["timestamp"]
        if last_completed_ts is None:
            last_completed_ts = current_ts
            return

        # Detect candle close: new candle opened
        if current_ts != last_completed_ts:
            last_completed_ts = current_ts

            if len(candle_buffer) < lookback:
                return

            import pandas as pd
            df = pd.DataFrame(list(candle_buffer)[:-1])  # exclude current forming candle
            df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
            df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

            current_price = float(df["close"].iloc[-1])

            # Check exit conditions for open trade
            if open_trade is not None:
                side = open_trade["side"]
                stop = open_trade["stop"]
                target = open_trade["target"]
                size = open_trade["size"]

                hit_stop = (side == "long" and current_price <= stop) or \
                           (side == "short" and current_price >= stop)
                hit_target = (side == "long" and current_price >= target) or \
                             (side == "short" and current_price <= target)

                if hit_stop or hit_target:
                    exit_price = stop if hit_stop else target
                    result = executor.market_close(symbol, side, size, exit_price)
                    if result.success:
                        pnl = _calc_pnl(side, open_trade["entry_price"], result.filled_price, open_trade["size_usd"])
                        risk.on_trade_close(pnl, pnl > 0)
                        if optimizer:
                            optimizer.record_trade(won=pnl > 0, net_pnl=pnl)
                        logger.info(f"Closed trade: {'STOP' if hit_stop else 'TARGET'} | pnl=${pnl:.2f}")
                        open_trade = None

            # Look for new signal
            if open_trade is None:
                signal = engine.evaluate(df)
                approval = risk.approve_trade(signal.action, current_price)

                if approval.approved:
                    result = executor.market_open(symbol, signal.action, approval.position_size_usd, current_price)
                    if result.success:
                        ep = result.filled_price
                        if signal.action == "long":
                            stop_p = ep * (1 - t["stop_pct"])
                            target_p = ep * (1 + t["target_pct"])
                        else:
                            stop_p = ep * (1 + t["stop_pct"])
                            target_p = ep * (1 - t["target_pct"])

                        open_trade = {
                            "side": signal.action,
                            "entry_price": ep,
                            "stop": stop_p,
                            "target": target_p,
                            "size": result.size,
                            "size_usd": approval.position_size_usd,
                        }
                        risk.on_trade_open()

    feed = LiveFeed(symbol=symbol, timeframe=timeframe, on_candle=on_candle)
    feed.start()

    try:
        while True:
            time.sleep(60)
            stats = risk.get_daily_stats()
            logger.info(
                f"Status: equity=${risk.get_equity():,.2f} | "
                f"today: {stats.trades_taken} trades, {stats.win_rate:.0%} win rate, "
                f"pnl=${stats.realized_pnl:.2f} | "
                f"{'HALTED' if stats.halted else 'active'}"
            )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        feed.stop()
        if optimizer:
            optimizer.stop()


def _calc_pnl(side: str, entry: float, exit_p: float, size_usd: float) -> float:
    if side == "long":
        return (exit_p - entry) / entry * size_usd
    return (entry - exit_p) / entry * size_usd


def main():
    parser = argparse.ArgumentParser(description="Kronos + TimesFM Trading Bot")
    parser.add_argument("--mode", choices=["backtest", "paper", "live", "agent"], default="paper")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--symbol", help="Override symbol")
    parser.add_argument("--timeframe", help="Override timeframe")
    parser.add_argument("--start", dest="start_date")
    parser.add_argument("--end", dest="end_date")
    parser.add_argument("--plot", default="equity_curve.png")

    # Agent-mode flags (ignored in other modes)
    parser.add_argument("--in-sample-start",  dest="in_sample_start", default="2024-01-01")
    parser.add_argument("--in-sample-end",    dest="in_sample_end",   default="2024-10-01")
    parser.add_argument("--oos-start",        dest="oos_start",       default="2024-10-01")
    parser.add_argument("--oos-end",          dest="oos_end",         default="2025-04-01")
    parser.add_argument("--iterations",       type=int, default=2,    help="Agent: LLM refinement iterations")
    parser.add_argument("--variants",         type=int, default=4,    help="Agent: variants per iteration")
    parser.add_argument("--model",            default="claude-sonnet-4-6", help="Agent: Anthropic model ID")
    parser.add_argument("--dry-run",          action="store_true",    help="Agent: use fixtures, no API key needed")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    if args.mode == "backtest":
        from backtest import run_backtest, print_report, plot_equity_curve
        result = run_backtest(
            config,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        print_report(result)
        plot_equity_curve(result, output_path=args.plot)

    elif args.mode == "agent":
        from strategy_agent import run_agent_loop
        run_agent_loop(config, args)

    else:
        if args.mode == "live":
            confirm = input("WARNING: Live trading mode. Real funds at risk. Type 'yes' to continue: ")
            if confirm.strip().lower() != "yes":
                print("Aborted.")
                sys.exit(0)
        run_live(config, mode=args.mode, config_path=args.config)


if __name__ == "__main__":
    main()

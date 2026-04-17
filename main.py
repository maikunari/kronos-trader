"""
main.py
Entry point for the Phase 1 sniper bot.

Modes:
  --mode backtest   Run historical backtest (no API keys needed)
  --mode paper      Live feed, signals logged but no real orders
  --mode live       Live trading on Hyperliquid (requires .env credentials)

Backtest is the primary daily-driver until paper-trading graduation. The
live loop is intentionally small: build MarketContext each bar close,
evaluate signal, approve via RiskManager, place via Executor+ExecutionPolicy.
Per-bar position management (Chandelier trail + stop/target checks) is
handled inline.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from dotenv import load_dotenv

from atr_engine import ChandelierTrail
from backtest import BacktestResult, run_snipe_backtest
from execution_policy import BookSnapshot, ExecutionPolicy, OrderIntent
from executor import Executor
from hyperliquid_feed import (
    TIMEFRAME_MS,
    LiveFeed,
    fetch_historical_binance,
    fetch_historical_hl,
)
from mtf_filter import MTFFilter
from regime import RegimeDetector
from risk_manager import RiskManager
from snipe_signal_engine import MarketContext, SnipeSignalEngine

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Config + factory helpers
# ------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_engine(config: dict) -> SnipeSignalEngine:
    s = config.get("snipe", {})
    r = config.get("regime", {})
    detector = RegimeDetector(
        adx_period=r.get("adx_period", 14),
        adx_threshold=r.get("adx_threshold", 20.0),
        hurst_window=r.get("hurst_window", 200),
        hurst_threshold=r.get("hurst_threshold", 0.50),
        rv_window=r.get("rv_window", 96),
        rv_hist_window=r.get("rv_hist_window", 2880),
    )
    return SnipeSignalEngine(
        regime=detector,
        donchian_period=s.get("donchian_period", 20),
        keltner_period=s.get("keltner_period", 20),
        keltner_atr_mult=s.get("keltner_atr_mult", 2.0),
        atr_period=s.get("atr_period", 14),
        stop_atr_mult=s.get("stop_atr_mult", 1.5),
        target_atr_mult=s.get("target_atr_mult", 3.0),
        supertrend_period=s.get("supertrend_period", 10),
        supertrend_mult=s.get("supertrend_mult", 3.0),
        funding_veto_pct_hourly=s.get("funding_veto_pct_hourly", 0.0003),
        composite_threshold=s.get("composite_threshold", 0.0),
    )


def build_risk(config: dict, initial_equity: float) -> RiskManager:
    r = config.get("risk", {})
    return RiskManager(
        initial_equity=initial_equity,
        target_annual_vol=r.get("target_annual_vol", 0.20),
        max_position_pct=r.get("max_position_pct", 0.25),
        max_concurrent_positions=r.get("max_concurrent_positions", 1),
        daily_loss_limit_pct=r.get("daily_loss_limit_pct", 0.03),
        consecutive_loss_limit=r.get("consecutive_loss_limit", 3),
        consecutive_loss_cooloff_seconds=r.get("consecutive_loss_cooloff_seconds", 3600),
        weekly_drawdown_halve_pct=r.get("weekly_drawdown_halve_pct", 0.08),
        tripwire_file=r.get("tripwire_file"),
        kelly_fraction=r.get("kelly_fraction", 0.0),
    )


def build_execution_policy(config: dict) -> ExecutionPolicy:
    e = config.get("execution", {})
    return ExecutionPolicy(
        post_only_first=e.get("post_only_first", True),
        taker_timeout_ms=e.get("taker_timeout_ms", 3_000),
        retreat_bps=e.get("retreat_bps", 10.0),
        max_book_fraction=e.get("max_book_fraction", 0.30),
        spread_anomaly_mult=e.get("spread_anomaly_mult", 5.0),
        enable_taker_fallback=e.get("enable_taker_fallback", True),
    )


# ------------------------------------------------------------------
# Backtest mode
# ------------------------------------------------------------------

def cmd_backtest(args, config: dict) -> int:
    b = config.get("backtest", {})
    t = config.get("trading", {})
    symbol = args.symbol or t.get("symbol", "BTC")
    timeframe = args.timeframe or t.get("timeframe", "15m")
    start_date = args.start_date or b.get("start_date", "2024-01-01")
    end_date = args.end_date or b.get("end_date", "2024-12-31")

    start_ms = int(datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).timestamp() * 1000)

    logger.info("Fetching %s/%s from Hyperliquid (%s -> %s)", symbol, timeframe, start_date, end_date)
    df = fetch_historical_hl(symbol, timeframe, start_ms, end_ms)
    if df.empty:
        logger.warning("HL returned empty; falling back to Binance")
        df = fetch_historical_binance(symbol, timeframe, start_ms, end_ms)
    if df.empty:
        logger.error("No historical data available for %s/%s", symbol, timeframe)
        return 1

    df_1h = fetch_historical_hl(symbol, "1h", start_ms, end_ms)
    df_4h = fetch_historical_hl(symbol, "4h", start_ms, end_ms)

    engine = build_engine(config)
    result = run_snipe_backtest(
        df, engine=engine,
        candles_1h=df_1h if not df_1h.empty else None,
        candles_4h=df_4h if not df_4h.empty else None,
        initial_capital=b.get("initial_capital", 10_000.0),
        fee_rate=b.get("fee_rate", 0.00035),
        slippage_base_bps=b.get("slippage_base_bps", 2.0),
        slippage_atr_frac=b.get("slippage_atr_frac", 0.05),
        funding_rate_hourly=b.get("funding_rate_hourly", 0.0),
        use_chandelier_trail=b.get("use_chandelier_trail", True),
        chandelier_atr_mult=b.get("chandelier_atr_mult", 3.0),
        max_hold_bars=b.get("max_hold_bars", 0),
        target_annual_vol=config.get("risk", {}).get("target_annual_vol", 0.20),
    )
    print_backtest_report(result, symbol, timeframe)
    return 0


def print_backtest_report(result: BacktestResult, symbol: str, timeframe: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  BACKTEST: {symbol} / {timeframe}")
    print(f"  {result.start_date} -> {result.end_date}")
    print(f"{'=' * 60}")
    for k, v in result.summary().items():
        print(f"  {k:<18} {v}")
    print(f"  final_equity       ${result.final_equity:,.2f}")
    print(f"  initial_capital    ${result.initial_capital:,.2f}")
    if result.trades_count:
        print(
            f"  avg_win            {result.avg_win_pct * 100:+.3f}%  | "
            f"avg_loss {result.avg_loss_pct * 100:+.3f}%"
        )
        print(
            f"  cost_breakdown     fees=${result.fees_total:,.2f} "
            f"slip=${result.slippage_total:,.2f} "
            f"funding=${result.funding_total:,.2f}"
        )
    print()


# ------------------------------------------------------------------
# Paper / live loop
# ------------------------------------------------------------------

def cmd_live(args, config: dict, *, live: bool) -> int:
    t = config.get("trading", {})
    symbol = args.symbol or t.get("symbol", "BTC")
    timeframe = args.timeframe or t.get("timeframe", "15m")
    lookback = config.get("snipe", {}).get("lookback_candles", 300)

    engine = build_engine(config)
    policy = build_execution_policy(config)

    executor = Executor(
        mode="live" if live else "paper",
        mainnet=config.get("hyperliquid", {}).get("mainnet", True),
        execution_policy=policy,
    )
    equity = executor.get_account_equity()
    risk = build_risk(config, initial_equity=equity)
    logger.info("Starting %s loop — %s/%s equity=$%.2f", "LIVE" if live else "PAPER", symbol, timeframe, equity)

    candle_buffer: deque = deque(maxlen=lookback + 1)
    open_pos: Optional[dict] = None
    last_completed_ts: Optional[int] = None

    def on_candle(candle: dict) -> None:
        nonlocal open_pos, last_completed_ts
        candle_buffer.append(candle)
        ts = candle["timestamp"]

        if last_completed_ts is None:
            last_completed_ts = ts
            return
        if ts == last_completed_ts:
            return
        last_completed_ts = ts
        if len(candle_buffer) < lookback:
            return

        window = pd.DataFrame(list(candle_buffer)[:-1])
        window["timestamp"] = pd.to_datetime(window["timestamp"], unit="ms", utc=True)
        window = window.astype(
            {"open": float, "high": float, "low": float, "close": float, "volume": float}
        )
        cur_price = float(window["close"].iloc[-1])
        cur_ts = window["timestamp"].iloc[-1]

        # --- Manage open position ---
        if open_pos is not None:
            exit_info = _position_exit_check(open_pos, candle)
            if exit_info is None and open_pos.get("trail") is not None:
                # Update Chandelier trail with this bar
                from atr_engine import _atr
                atr_now = float(_atr(
                    window["high"].to_numpy(dtype=float),
                    window["low"].to_numpy(dtype=float),
                    window["close"].to_numpy(dtype=float),
                    period=14,
                )[-1])
                stop = open_pos["trail"].update(
                    high=float(candle["high"]), low=float(candle["low"]), atr=atr_now,
                )
                if stop is not None:
                    if open_pos["direction"] == "long":
                        open_pos["stop"] = max(open_pos["stop"], stop)
                    else:
                        open_pos["stop"] = min(open_pos["stop"], stop)
                exit_info = _position_exit_check(open_pos, candle)

            if exit_info is not None:
                reason, exit_price = exit_info
                close_result = executor.market_close(
                    symbol, open_pos["direction"], open_pos["size_contracts"], cur_price,
                )
                if close_result.success:
                    ep_fill = close_result.filled_price
                    pnl = _pnl(open_pos["direction"], open_pos["entry_price"], ep_fill, open_pos["size_usd"])
                    risk.on_trade_close(pnl=pnl, won=pnl > 0)
                    logger.info("Closed %s: %s @ %.4f pnl=$%.2f",
                                open_pos["direction"], reason, ep_fill, pnl)
                    open_pos = None

        # --- New signal ---
        if open_pos is None:
            ctx = MarketContext(candles_15m=window, current_price=cur_price, timestamp=cur_ts)
            sig = engine.evaluate(ctx)
            if sig.action not in ("long", "short"):
                logger.debug("No signal: %s", sig.skip_reason)
                return
            approval = risk.approve_trade(
                sig.action, current_price=cur_price, symbol=symbol,
                instrument_annual_vol=sig.regime.rv if sig.regime else None,
            )
            if not approval.approved:
                logger.info("Signal rejected by RM: %s", approval.reason)
                return
            logger.info("Placing %s %s size=$%.2f", sig.action.upper(), symbol, approval.position_size_usd)
            size_contracts = approval.position_size_contracts
            # For paper/live simplicity, use market_open on initial fill; execution policy
            # applies to limit entries with a book snapshot (future enhancement).
            fill = executor.market_open(symbol, sig.action, approval.position_size_usd, cur_price)
            if not fill.success:
                logger.error("Order rejected: %s", fill.error)
                return
            open_pos = {
                "direction": sig.action,
                "entry_price": fill.filled_price,
                "stop": sig.stop_price,
                "target": sig.target_price,
                "size_usd": approval.position_size_usd,
                "size_contracts": size_contracts,
                "trail": ChandelierTrail(direction=sig.action, atr_mult=3.0),
            }
            risk.on_trade_open()

    feed = LiveFeed(symbol=symbol, timeframe=timeframe, on_candle=on_candle)
    feed.start()
    try:
        while True:
            time.sleep(60)
            stats = risk.get_daily_stats()
            logger.info(
                "Status: eq=$%.2f today:%d trades wr=%.0f%% pnl=$%.2f %s",
                risk.get_equity(), stats.trades_taken, stats.win_rate * 100, stats.realized_pnl,
                "HALTED" if risk.is_halted() else "active",
            )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        feed.stop()
    return 0


def _position_exit_check(pos: dict, candle: dict) -> Optional[tuple[str, float]]:
    high, low = float(candle["high"]), float(candle["low"])
    if pos["direction"] == "long":
        if low <= pos["stop"]:
            return "stop", pos["stop"]
        if high >= pos["target"]:
            return "target", pos["target"]
    else:
        if high >= pos["stop"]:
            return "stop", pos["stop"]
        if low <= pos["target"]:
            return "target", pos["target"]
    return None


def _pnl(direction: str, entry: float, exit_p: float, size_usd: float) -> float:
    if direction == "long":
        return (exit_p - entry) / entry * size_usd
    return (entry - exit_p) / entry * size_usd


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Kronos-trader Phase 1 sniper")
    parser.add_argument("--mode", choices=["backtest", "paper", "live"], default="backtest")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--symbol", help="Override config symbol")
    parser.add_argument("--timeframe", help="Override config timeframe")
    parser.add_argument("--start", dest="start_date", help="Backtest start YYYY-MM-DD")
    parser.add_argument("--end", dest="end_date", help="Backtest end YYYY-MM-DD")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 1
    config = load_config(str(cfg_path))

    if args.mode == "backtest":
        return cmd_backtest(args, config)
    if args.mode == "live":
        confirm = input("WARNING: LIVE trading — real funds at risk. Type 'yes' to continue: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return 0
        return cmd_live(args, config, live=True)
    return cmd_live(args, config, live=False)


if __name__ == "__main__":
    sys.exit(main())

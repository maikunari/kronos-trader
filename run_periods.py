"""
run_periods.py
Run the Phase 1 sniper backtest across multiple named historical windows
and print a side-by-side comparison. Primary use: stress-testing against
known high-variance regimes (Terra, FTX, post-halving, etc.).

Usage:
    python run_periods.py --symbol BTC
    python run_periods.py --symbol ETH --periods "bull:2023-06-01:2024-04-30"
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml

from backtest import BacktestResult, run_snipe_backtest
from hyperliquid_feed import fetch_historical_binance, fetch_historical_hl
from main import build_engine

logger = logging.getLogger(__name__)


DEFAULT_PERIODS: list[tuple[str, str, str]] = [
    ("2022-terra",  "2022-04-01", "2022-06-30"),   # Terra collapse
    ("2022-ftx",    "2022-10-15", "2022-12-15"),   # FTX collapse
    ("2023-bull",   "2023-10-01", "2024-04-30"),   # BTC ETF approval rally
    ("2024",        "2024-01-01", "2024-12-31"),
    ("2025-YTD",    "2025-01-01", "2025-12-31"),
    ("2026-YTD",    "2026-01-01", "2026-04-14"),
]


def parse_period_spec(specs: list[str]) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for s in specs:
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError(f"bad period spec: {s!r} (expected name:YYYY-MM-DD:YYYY-MM-DD)")
        out.append((parts[0], parts[1], parts[2]))
    return out


def _fetch(symbol: str, timeframe: str, start: str, end: str):
    start_ms = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp() * 1000)
    df = fetch_historical_hl(symbol, timeframe, start_ms, end_ms)
    if df.empty:
        df = fetch_historical_binance(symbol, timeframe, start_ms, end_ms)
    return df


def run_periods(
    symbol: str,
    timeframe: str,
    config: dict,
    periods: Iterable[tuple[str, str, str]],
) -> dict[str, BacktestResult]:
    engine = build_engine(config)
    b = config.get("backtest", {})
    results: dict[str, BacktestResult] = {}
    for name, start, end in periods:
        logger.info("=== %s %s/%s %s -> %s", name, symbol, timeframe, start, end)
        df = _fetch(symbol, timeframe, start, end)
        if df.empty:
            logger.warning("No data for %s — skipping", name)
            continue
        df_1h = _fetch(symbol, "1h", start, end)
        df_4h = _fetch(symbol, "4h", start, end)
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
        )
        results[name] = result
    return results


def print_comparison(results: dict[str, BacktestResult], symbol: str, timeframe: str) -> None:
    if not results:
        print("No results to compare.")
        return

    metrics = [
        ("Return", lambda r: f"{r.total_return_pct * 100:+.2f}%"),
        ("Sharpe", lambda r: f"{r.sharpe_ratio:.2f}"),
        ("Max DD", lambda r: f"{r.max_drawdown_pct * 100:.1f}%"),
        ("Trades", lambda r: f"{r.trades_count}"),
        ("Win rate", lambda r: f"{r.win_rate * 100:.1f}%"),
        ("Profit factor", lambda r: f"{r.profit_factor:.2f}"),
        ("Avg win", lambda r: f"{r.avg_win_pct * 100:+.3f}%"),
        ("Avg loss", lambda r: f"{r.avg_loss_pct * 100:+.3f}%"),
        ("Cost drag", lambda r: f"${r.fees_total + r.slippage_total + r.funding_total:,.2f}"),
    ]

    col_w = 16
    header = "Metric".ljust(18) + "".join(f" {n:>{col_w}}" for n in results.keys())
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print(f"  PERIOD COMPARISON: {symbol}/{timeframe}")
    print("=" * len(header))
    print(header)
    print(sep)
    for label, fmt in metrics:
        row = f"{label:<18}" + "".join(f" {fmt(r):>{col_w}}" for r in results.values())
        print(row)
    print(sep)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Multi-period backtest comparison")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--periods", nargs="*", default=None,
        help="Override default periods; format 'name:YYYY-MM-DD:YYYY-MM-DD'",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 1
    config = yaml.safe_load(cfg_path.read_text())
    t = config.get("trading", {})
    symbol = args.symbol or t.get("symbol", "BTC")
    timeframe = args.timeframe or t.get("timeframe", "15m")
    periods = parse_period_spec(args.periods) if args.periods else DEFAULT_PERIODS

    results = run_periods(symbol, timeframe, config, periods)
    print_comparison(results, symbol, timeframe)
    return 0


if __name__ == "__main__":
    sys.exit(main())

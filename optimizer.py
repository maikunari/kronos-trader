"""
optimizer.py
Grid-search parameter tuner for the Phase 1 sniper, run via purged
walk-forward on historical candles.

Unlike the old Kronos/TimesFM tuner, this evaluates each parameter
combination across multiple (is | embargo | oos) folds and ranks by
*median OOS* metric — in-sample Sharpe is hopelessly biased. Results
are written to JSON for reproducibility + dashboard consumption.

Usage:
    python optimizer.py --symbol BTC --start 2024-01-01 --end 2025-01-01 \\
        --grid grids/donchian_basic.yaml --out optimizer_results.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import yaml

from backtest import WalkForwardFold, walk_forward
from hyperliquid_feed import fetch_historical_binance, fetch_historical_hl
from regime import RegimeDetector
from snipe_signal_engine import SnipeSignalEngine

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Grid expansion
# ------------------------------------------------------------------

def expand_grid(grid: dict[str, list]) -> list[dict]:
    """Cartesian-product expansion of a grid dict into a list of param dicts."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def build_engine_from_params(params: dict) -> SnipeSignalEngine:
    """Construct an engine with the flat-param namespace used by the optimizer grid."""
    detector = RegimeDetector(
        adx_period=params.get("adx_period", 14),
        adx_threshold=params.get("adx_threshold", 20.0),
        hurst_window=params.get("hurst_window", 200),
        hurst_threshold=params.get("hurst_threshold", 0.50),
        rv_window=params.get("rv_window", 96),
        rv_hist_window=params.get("rv_hist_window", 2_880),
    )
    return SnipeSignalEngine(
        regime=detector,
        donchian_period=params.get("donchian_period", 20),
        keltner_period=params.get("keltner_period", 20),
        keltner_atr_mult=params.get("keltner_atr_mult", 2.0),
        atr_period=params.get("atr_period", 14),
        stop_atr_mult=params.get("stop_atr_mult", 1.5),
        target_atr_mult=params.get("target_atr_mult", 3.0),
        supertrend_period=params.get("supertrend_period", 10),
        supertrend_mult=params.get("supertrend_mult", 3.0),
        funding_veto_pct_hourly=params.get("funding_veto_pct_hourly", 0.0003),
        composite_threshold=params.get("composite_threshold", 0.0),
    )


# ------------------------------------------------------------------
# Fold aggregation
# ------------------------------------------------------------------

def aggregate_folds(folds: list[WalkForwardFold]) -> dict:
    """Summarize per-fold OOS results into medians/means for ranking."""
    if not folds:
        return {
            "folds": 0, "median_oos_sharpe": 0.0, "mean_oos_return": 0.0,
            "median_oos_max_dd": 0.0, "total_trades": 0, "median_oos_win_rate": 0.0,
            "median_oos_profit_factor": 0.0,
        }
    sharpes = [f.oos_result.sharpe_ratio for f in folds]
    returns = [f.oos_result.total_return_pct for f in folds]
    max_dds = [f.oos_result.max_drawdown_pct for f in folds]
    win_rates = [f.oos_result.win_rate for f in folds]
    pfs = [f.oos_result.profit_factor for f in folds]
    trades = sum(f.oos_result.trades_count for f in folds)
    return {
        "folds": len(folds),
        "median_oos_sharpe": float(np.median(sharpes)),
        "mean_oos_sharpe": float(np.mean(sharpes)),
        "mean_oos_return": float(np.mean(returns)),
        "median_oos_max_dd": float(np.median(max_dds)),
        "median_oos_win_rate": float(np.median(win_rates)),
        "median_oos_profit_factor": float(np.median(pfs)),
        "total_trades": trades,
    }


# ------------------------------------------------------------------
# Grid search
# ------------------------------------------------------------------

def grid_search(
    candles_15m: pd.DataFrame,
    grid: dict[str, list],
    *,
    in_sample_days: int = 60,
    out_of_sample_days: int = 14,
    embargo_days: int = 1,
    step_days: Optional[int] = None,
    backtest_kwargs: Optional[dict] = None,
    progress: Optional[Callable[[int, int, dict], None]] = None,
) -> list[dict]:
    """
    Run walk-forward on each combination of `grid`. Returns all results
    sorted by median OOS Sharpe descending.
    """
    combos = expand_grid(grid)
    logger.info("Optimizer: %d param combinations x WF folds", len(combos))

    results: list[dict] = []
    for idx, params in enumerate(combos):
        def factory(p=params):   # capture by default arg
            return build_engine_from_params(p)

        folds = walk_forward(
            candles_15m,
            engine_factory=factory,
            in_sample_days=in_sample_days,
            out_of_sample_days=out_of_sample_days,
            embargo_days=embargo_days,
            step_days=step_days,
            backtest_kwargs=backtest_kwargs,
        )
        agg = aggregate_folds(folds)
        record = {"params": params, **agg}
        results.append(record)
        if progress:
            progress(idx + 1, len(combos), record)

    results.sort(key=lambda r: (r["median_oos_sharpe"], r["mean_oos_return"]), reverse=True)
    return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _load_grid(path: str) -> dict[str, list]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _fetch_candles(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    start_ms = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp() * 1000)
    df = fetch_historical_hl(symbol, timeframe, start_ms, end_ms)
    if df.empty:
        df = fetch_historical_binance(symbol, timeframe, start_ms, end_ms)
    if df.empty:
        raise SystemExit(f"No data for {symbol}/{timeframe} {start} -> {end}")
    return df


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Walk-forward grid-search optimizer")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--grid", required=True, help="YAML grid file")
    parser.add_argument("--out", default="optimizer_results.json")
    parser.add_argument("--in-sample-days", type=int, default=60)
    parser.add_argument("--oos-days", type=int, default=14)
    parser.add_argument("--embargo-days", type=int, default=1)
    args = parser.parse_args()

    grid = _load_grid(args.grid)
    df = _fetch_candles(args.symbol, args.timeframe, args.start, args.end)
    logger.info("Loaded %d candles for %s/%s", len(df), args.symbol, args.timeframe)

    def progress(i: int, total: int, rec: dict) -> None:
        logger.info(
            "[%d/%d] sharpe=%.2f ret=%.2f%% dd=%.2f%% wr=%.1f%% trades=%d params=%s",
            i, total, rec["median_oos_sharpe"], rec["mean_oos_return"] * 100,
            rec["median_oos_max_dd"] * 100, rec["median_oos_win_rate"] * 100,
            rec["total_trades"], rec["params"],
        )

    results = grid_search(
        df, grid,
        in_sample_days=args.in_sample_days,
        out_of_sample_days=args.oos_days,
        embargo_days=args.embargo_days,
        progress=progress,
    )
    Path(args.out).write_text(json.dumps({
        "symbol": args.symbol, "timeframe": args.timeframe,
        "start": args.start, "end": args.end,
        "grid": grid, "results": results,
    }, indent=2, default=str))
    best = results[0] if results else None
    if best:
        logger.info("Best: %s", best)
    logger.info("Wrote %d results to %s", len(results), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

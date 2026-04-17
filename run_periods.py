"""
run_periods.py
Run backtests across three distinct market periods and print a comparison table.

Periods:
  2024  — Jan 1 2024 → Dec 31 2024  (bull run / up-only)
  2025  — Jan 1 2025 → Dec 31 2025  (SOL drawdown / macro uncertainty)
  2026  — Jan 1 2026 → Apr 14 2026  (recent / tariff sell-off)

Usage:
    ./venv/bin/python run_periods.py
    ./venv/bin/python run_periods.py --symbol ETH
"""
import argparse
import copy
import sys
from pathlib import Path

import yaml

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from backtest import run_backtest, BacktestResult, plot_equity_curve

PERIODS = [
    ("2024 (bull)",     "2024-01-01", "2024-12-31"),
    ("2025 (drawdown)", "2025-01-01", "2025-12-31"),
    ("2026 YTD",        "2026-01-01", "2026-04-14"),
]


def label(r: BacktestResult) -> dict:
    return {
        "Return":        f"{r.total_return_pct*100:+.2f}%",
        "Win rate":      f"{r.win_rate*100:.1f}%",
        "Trades":        str(r.total_trades),
        "Profit factor": f"{r.profit_factor:.2f}",
        "Max drawdown":  f"{r.max_drawdown_pct*100:.1f}%",
        "Sharpe":        f"{r.sharpe_ratio:.2f}",
        "Avg win":       f"{r.avg_win_pct*100:.3f}%",
        "Avg loss":      f"{r.avg_loss_pct*100:.3f}%",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None, help="Override symbol")
    parser.add_argument("--timeframe", default=None, help="Override timeframe")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--plot", action="store_true", help="Save equity curve PNGs")
    args = parser.parse_args()

    with open(BASE / args.config) as f:
        base_cfg = yaml.safe_load(f)

    symbol = args.symbol or base_cfg["trading"]["symbol"]
    tf = args.timeframe or base_cfg["trading"]["timeframe"]

    results = {}
    for name, start, end in PERIODS:
        cfg = copy.deepcopy(base_cfg)
        cfg["backtest"]["start_date"] = start
        cfg["backtest"]["end_date"] = end
        print(f"\n{'='*60}")
        print(f"  Running: {name}  ({start} → {end})  {symbol}/{tf}")
        print(f"{'='*60}")
        r = run_backtest(cfg, symbol=symbol, timeframe=tf)
        results[name] = r
        if args.plot:
            safe_name = name.split()[0]
            plot_equity_curve(r, output_path=f"results_{symbol}_{tf}_{safe_name}.png")

    # --- Comparison table ---
    col_w = 16
    period_names = list(results.keys())
    header = f"\n{'':28}" + "".join(f" {n:>{col_w}}" for n in period_names)
    separator = "-" * (28 + (col_w + 1) * len(period_names))

    print(f"\n{'='*60}")
    print(f"  PERIOD COMPARISON: {symbol}/{tf}")
    print(f"{'='*60}")
    print(header)
    print(separator)

    sample_label = label(list(results.values())[0])
    for key in sample_label:
        row = f"  {key:<26}"
        for name in period_names:
            row += f" {label(results[name])[key]:>{col_w}}"
        print(row)

    print(separator)
    print()

    # MTF filter info
    mtf_cfg = base_cfg.get("mtf", {})
    if mtf_cfg.get("enabled"):
        d1 = "ON" if mtf_cfg.get("use_d1", False) else "OFF"
        rb = "H4+H1" if mtf_cfg.get("require_both", False) else "H4 only"
        print(f"  MTF: D1={d1}, {rb}, EMA{mtf_cfg.get('ema_fast',20)}/{mtf_cfg.get('ema_slow',50)}")
        print(f"  Neutral bias = BLOCK (momentum-only mode)")
    else:
        print("  MTF: disabled")
    print()


if __name__ == "__main__":
    main()

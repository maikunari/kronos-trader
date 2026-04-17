"""
compare_tf.py
Runs the same strategy across M15 / M5 / M1 and prints a side-by-side comparison.

Uses a 90-day window to keep M1 data volume manageable (~135K candles vs 525K for a full year).

Usage:
    python compare_tf.py
    python compare_tf.py --days 180   # longer window if you want more signal
    python compare_tf.py --symbol ETH
"""
import argparse
import copy
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yaml

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from backtest import run_backtest, BacktestResult

parser = argparse.ArgumentParser()
parser.add_argument("--days",   type=int,   default=90,    help="Lookback window in days (default: 90)")
parser.add_argument("--symbol", type=str,   default=None,  help="Symbol override (default: from config)")
args = parser.parse_args()

with open(BASE / "config.yaml") as f:
    base_cfg = yaml.safe_load(f)

end   = datetime.now(timezone.utc)
start = end - timedelta(days=args.days)
base_cfg["backtest"]["start_date"] = start.strftime("%Y-%m-%d")
base_cfg["backtest"]["end_date"]   = end.strftime("%Y-%m-%d")

symbol = args.symbol or base_cfg["trading"]["symbol"]
timeframes = ["15m", "5m", "1m"]

def label(r: BacktestResult) -> dict:
    return {
        "Return":        f"{r.total_return_pct * 100:+.2f}%",
        "Win rate":      f"{r.win_rate * 100:.1f}%",
        "Trades":        str(r.total_trades),
        "Profit factor": f"{r.profit_factor:.2f}",
        "Max drawdown":  f"{r.max_drawdown_pct * 100:.1f}%",
        "Sharpe":        f"{r.sharpe_ratio:.2f}",
        "Avg win":       f"{r.avg_win_pct * 100:.3f}%",
        "Avg loss":      f"{r.avg_loss_pct * 100:.3f}%",
    }

print(f"\nBacktest period: {base_cfg['backtest']['start_date']} → {base_cfg['backtest']['end_date']} ({symbol}, {args.days}d window)")
print(f"Same params across all TFs — stop {base_cfg['trading']['stop_pct']*100:.2f}% / target {base_cfg['trading']['target_pct']*100:.2f}% / RR {base_cfg['trading']['rr_ratio']}\n")

results = {}
for tf in timeframes:
    cfg = copy.deepcopy(base_cfg)
    print(f"Running {tf}...")
    results[tf] = run_backtest(cfg, symbol=symbol, timeframe=tf)

# Side-by-side print
col_w = 14
header = f"{'':28}" + "".join(f" {tf:>{col_w}}" for tf in timeframes)
print(f"\n{header}")
print("-" * (28 + col_w * len(timeframes) + len(timeframes)))

sample = label(results[timeframes[0]])
for k in sample:
    row = f"  {k:<26}"
    for tf in timeframes:
        row += f" {label(results[tf])[k]:>{col_w}}"
    print(row)

print()

# Quick verdict
best_tf = max(timeframes, key=lambda tf: results[tf].total_return_pct)
print(f"Winner by return: {best_tf} ({results[best_tf].total_return_pct*100:+.2f}%)")
best_pf = max(timeframes, key=lambda tf: results[tf].profit_factor)
print(f"Winner by profit factor: {best_pf} ({results[best_pf].profit_factor:.2f})")
print()

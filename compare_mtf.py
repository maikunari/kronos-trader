"""
compare_mtf.py
Runs two backtests back-to-back and prints a side-by-side comparison:
  A) require_both=True  (current config — 4H AND 1H must agree)
  B) require_both=False (4H only — faster to react to trend shifts)
"""
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yaml

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from backtest import run_backtest, BacktestResult

with open(BASE / "config.yaml") as f:
    base_cfg = yaml.safe_load(f)

end = datetime.now(timezone.utc)
start = end - timedelta(days=365)
base_cfg["backtest"]["start_date"] = start.strftime("%Y-%m-%d")
base_cfg["backtest"]["end_date"]   = end.strftime("%Y-%m-%d")

import copy

cfg_a = copy.deepcopy(base_cfg)
cfg_a["mtf"]["require_both"] = True

cfg_b = copy.deepcopy(base_cfg)
cfg_b["mtf"]["require_both"] = False

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

print(f"\nBacktest period: {base_cfg['backtest']['start_date']} → {base_cfg['backtest']['end_date']} (SOL/15m)\n")

print("Running A: require_both=True  (4H + 1H both must agree)...")
ra = run_backtest(cfg_a)
da = label(ra)

print("Running B: require_both=False (4H only)...")
rb = run_backtest(cfg_b)
db = label(rb)

# Side-by-side print
col_w = 16
print(f"\n{'':30} {'A: 4H+1H':>{col_w}} {'B: 4H only':>{col_w}}")
print("-" * (30 + col_w * 2 + 2))
for k in da:
    print(f"  {k:<28} {da[k]:>{col_w}} {db[k]:>{col_w}}")
print()

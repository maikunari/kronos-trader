"""
dashboard.py
Streamlit dashboard for the Phase 1 sniper.

Panels:
  - Backtest: load a result JSON or run one on-the-fly; equity curve +
    trade table + cost attribution.
  - Optimizer: leaderboard of most-recent grid search from
    optimizer_results.json.
  - Regime: last-N bars regime classification snapshot.

Start with:
    streamlit run dashboard.py --server.port 8502

Streamlit is an optional dependency — the helper functions below are
pure and tested independently.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backtest import BacktestResult, Trade

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Pure helpers (tested; no streamlit dependency)
# ------------------------------------------------------------------

def trades_to_frame(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=[
            "entry_ts", "exit_ts", "direction", "entry_price", "exit_price",
            "size_usd", "pnl_usd", "exit_reason", "fees_usd", "slippage_usd",
            "funding_usd",
        ])
    return pd.DataFrame([asdict(t) for t in trades])


def cost_attribution(result: BacktestResult) -> pd.DataFrame:
    gross = sum(t.pnl_usd + t.fees_usd + t.slippage_usd + t.funding_usd for t in result.trades)
    rows = [
        ("Gross PnL",   gross,                    100.0 if gross else 0.0),
        ("Fees",        -result.fees_total,       _pct(result.fees_total, gross)),
        ("Slippage",    -result.slippage_total,   _pct(result.slippage_total, gross)),
        ("Funding",     -result.funding_total,    _pct(result.funding_total, gross)),
        ("Net PnL",     sum(t.pnl_usd for t in result.trades), 0.0),
    ]
    return pd.DataFrame(rows, columns=["component", "usd", "pct_of_gross"])


def _pct(value: float, base: float) -> float:
    return float(value / base * 100) if base else 0.0


def exit_reason_breakdown(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["reason", "count", "avg_pnl_usd", "total_pnl_usd"])
    df = pd.DataFrame([asdict(t) for t in trades])
    agg = (
        df.groupby("exit_reason", as_index=False)
        .agg(count=("pnl_usd", "size"),
             avg_pnl_usd=("pnl_usd", "mean"),
             total_pnl_usd=("pnl_usd", "sum"))
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    return agg


def load_optimizer_results(path: str = "optimizer_results.json") -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse %s: %s", path, exc)
        return None


def optimizer_leaderboard(results_json: dict, top_n: int = 20) -> pd.DataFrame:
    if not results_json:
        return pd.DataFrame()
    rows = []
    for rec in results_json.get("results", [])[:top_n]:
        row = {
            "median_oos_sharpe": rec.get("median_oos_sharpe"),
            "mean_oos_return": rec.get("mean_oos_return"),
            "median_oos_max_dd": rec.get("median_oos_max_dd"),
            "median_oos_win_rate": rec.get("median_oos_win_rate"),
            "median_oos_profit_factor": rec.get("median_oos_profit_factor"),
            "folds": rec.get("folds"),
            "total_trades": rec.get("total_trades"),
            **rec.get("params", {}),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Streamlit UI (imported lazily — only when this file is run by Streamlit)
# ------------------------------------------------------------------

def _render_ui() -> None:
    import streamlit as st

    st.set_page_config(page_title="Kronos Sniper", layout="wide")
    st.title("Kronos-Trader — Phase 1 Sniper")

    tab_bt, tab_opt, tab_live = st.tabs(["Backtest", "Optimizer", "Live"])

    with tab_bt:
        st.header("Backtest Results")
        path = st.text_input("Backtest result JSON path", value="backtest_result.json")
        if Path(path).exists():
            data = json.loads(Path(path).read_text())
            st.json({k: data[k] for k in data if k != "equity_curve" and k != "trades"})
            if "equity_curve" in data:
                st.line_chart(pd.Series(data["equity_curve"]), height=300)
        else:
            st.info("No backtest_result.json yet — run `python main.py --mode backtest`.")

    with tab_opt:
        st.header("Optimizer Leaderboard")
        res = load_optimizer_results()
        if res is None:
            st.info("No optimizer_results.json yet — run `python optimizer.py`.")
        else:
            st.caption(
                f"{res.get('symbol')} / {res.get('timeframe')} — "
                f"{res.get('start')} → {res.get('end')}"
            )
            lb = optimizer_leaderboard(res, top_n=30)
            st.dataframe(lb, use_container_width=True)

    with tab_live:
        st.header("Live State")
        st.info(
            "Live-state panel stub. The paper/live loop will write a heartbeat JSON "
            "(equity, regime, fill quality) that this panel consumes."
        )


if __name__ == "__main__":
    _render_ui()

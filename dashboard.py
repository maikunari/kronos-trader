"""
dashboard.py
Kronos Trader — Live Dashboard
Run: venv/bin/streamlit run dashboard.py --server.port 8502
"""

import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent
LOG = BASE / "paper_trade.log"
OPT_LOG = BASE / "optimizer_history.jsonl"
REPORT_CACHE = BASE / "last_backtest.json"

st.set_page_config(
    page_title="Kronos Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.metric-label { font-size: 0.8rem; color: #888; }
.metric-value { font-size: 1.8rem; font-weight: 700; }
.positive { color: #00c896; }
.negative { color: #ff4444; }
.neutral  { color: #aaa; }
</style>
""", unsafe_allow_html=True)


# ── helpers ────────────────────────────────────────────────────────────────

def service_status() -> str:
    try:
        r = subprocess.run(
            ["systemctl", "is-active", "kronos-paper"],
            capture_output=True, text=True
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


def parse_paper_log(n: int = 200) -> list[dict]:
    """Parse the last N lines of paper_trade.log into structured events."""
    if not LOG.exists():
        return []
    lines = LOG.read_text().splitlines()[-n:]
    events = []
    for line in lines:
        m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(\w+)\s+(\S+): (.+)", line)
        if m:
            events.append({
                "time": m.group(1),
                "level": m.group(2),
                "module": m.group(3),
                "msg": m.group(4),
            })
    return events


def load_optimizer_history() -> list[dict]:
    if not OPT_LOG.exists():
        return []
    entries = []
    for line in OPT_LOG.read_text().splitlines():
        try:
            entries.append(json.loads(line))
        except Exception:
            pass
    return entries


@st.cache_data(ttl=3600, show_spinner="Running 30-day backtest...")
def run_backtest_cached():
    """Run backtest and return results as dict (cached 1h)."""
    sys.path.insert(0, str(BASE))
    import yaml
    from backtest import run_backtest, BacktestResult

    with open(BASE / "config.yaml") as f:
        config = yaml.safe_load(f)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)
    config["backtest"]["start_date"] = start.strftime("%Y-%m-%d")
    config["backtest"]["end_date"] = end.strftime("%Y-%m-%d")

    result: BacktestResult = run_backtest(config)
    return {
        "total_return_pct": result.total_return_pct,
        "win_rate": result.win_rate,
        "total_trades": result.total_trades,
        "profit_factor": result.profit_factor,
        "max_drawdown_pct": result.max_drawdown_pct,
        "sharpe_ratio": result.sharpe_ratio,
        "avg_win_pct": result.avg_win_pct,
        "avg_loss_pct": result.avg_loss_pct,
        "equity_curve": result.equity_curve,
        "initial_capital": result.initial_capital,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "trades": [
            {
                "entry_time": str(t.entry_time),
                "exit_time": str(t.exit_time),
                "side": t.side,
                "net_pnl": t.net_pnl,
                "won": t.won,
                "exit_reason": t.exit_reason,
            }
            for t in result.trades
        ],
    }


# ── layout ─────────────────────────────────────────────────────────────────

status = service_status()
status_color = "#00c896" if status == "active" else "#ff4444"
status_label = "● PAPER TRADING LIVE" if status == "active" else f"● {status.upper()}"

st.markdown(f"""
## 📈 Kronos Trader &nbsp; <span style="font-size:0.9rem; color:{status_color}">{status_label}</span>
""", unsafe_allow_html=True)

tab_bt, tab_paper, tab_opt = st.tabs(["30-Day Backtest", "Paper Trading Feed", "Optimizer"])

# ── TAB 1: BACKTEST ────────────────────────────────────────────────────────
with tab_bt:
    col_run, _ = st.columns([1, 5])
    with col_run:
        force_refresh = st.button("🔄 Re-run Backtest")
    if force_refresh:
        run_backtest_cached.clear()

    try:
        r = run_backtest_cached()
    except Exception as e:
        st.error(f"Backtest failed: {e}")
        st.stop()

    st.caption(f"Period: **{r['start_date']}** → **{r['end_date']}** (SOL/15m)")

    # Metric cards
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    ret_pct = r["total_return_pct"] * 100
    dd_pct  = r["max_drawdown_pct"] * 100
    wr_pct  = r["win_rate"] * 100

    def _card(col, label, value, color_class="neutral"):
        col.markdown(f"""
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_class}">{value}</div>
        """, unsafe_allow_html=True)

    _card(m1, "30-Day Return",  f"{ret_pct:+.2f}%",   "positive" if ret_pct > 0 else "negative")
    _card(m2, "Win Rate",       f"{wr_pct:.1f}%",      "positive" if wr_pct > 50 else "negative")
    _card(m3, "Profit Factor",  f"{r['profit_factor']:.2f}", "positive" if r['profit_factor'] > 1.5 else "neutral")
    _card(m4, "Sharpe",         f"{r['sharpe_ratio']:.2f}",  "positive" if r['sharpe_ratio'] > 1 else "neutral")
    _card(m5, "Max Drawdown",   f"{dd_pct:.1f}%",      "negative")
    _card(m6, "Trades",         str(r["total_trades"]), "neutral")

    st.divider()

    # Equity curve
    eq = r["equity_curve"]
    ic = r["initial_capital"]
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        y=eq, mode="lines",
        line=dict(color="#00c896", width=2),
        name="Equity",
        hovertemplate="Trade #%{x}<br>$%{y:,.2f}<extra></extra>",
    ))
    fig_eq.add_hline(y=ic, line_dash="dash", line_color="#555", annotation_text="Initial capital")
    fig_eq.update_layout(
        title="Equity Curve (30-day backtest)",
        yaxis_title="Portfolio Value ($)",
        xaxis_title="Trade #",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#ddd",
        height=350,
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # Daily P&L bar chart + rolling win rate
    trades_df = pd.DataFrame(r["trades"])
    if not trades_df.empty:
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
        trades_df["date"] = trades_df["entry_time"].dt.date

        col_pnl, col_wr = st.columns(2)

        with col_pnl:
            daily = trades_df.groupby("date")["net_pnl"].sum().reset_index()
            daily.columns = ["date", "pnl"]
            fig_pnl = px.bar(
                daily, x="date", y="pnl",
                color=daily["pnl"].apply(lambda x: "profit" if x >= 0 else "loss"),
                color_discrete_map={"profit": "#00c896", "loss": "#ff4444"},
                title="Daily P&L",
                labels={"pnl": "Net P&L ($)", "date": ""},
            )
            fig_pnl.update_layout(
                showlegend=False,
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font_color="#ddd", height=280, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

        with col_wr:
            trades_df["won_int"] = trades_df["won"].astype(int)
            trades_df["rolling_wr"] = trades_df["won_int"].rolling(20, min_periods=5).mean() * 100
            fig_wr = go.Figure()
            fig_wr.add_trace(go.Scatter(
                x=list(range(len(trades_df))), y=trades_df["rolling_wr"],
                mode="lines", line=dict(color="#7b8cff", width=2),
                name="Rolling WR (20)",
            ))
            fig_wr.add_hline(y=50, line_dash="dash", line_color="#555", annotation_text="50%")
            fig_wr.update_layout(
                title="Rolling Win Rate (20-trade window)",
                yaxis_title="Win Rate (%)", xaxis_title="Trade #",
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font_color="#ddd", height=280, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_wr, use_container_width=True)

        # Trade scatter
        trades_df["color"] = trades_df["won"].map({True: "#00c896", False: "#ff4444"})
        trades_df["pnl_label"] = trades_df["net_pnl"].apply(lambda x: f"${x:+.2f}")
        fig_sc = go.Figure()
        for won, grp in trades_df.groupby("won"):
            fig_sc.add_trace(go.Scatter(
                x=grp["entry_time"],
                y=grp["net_pnl"],
                mode="markers",
                marker=dict(color="#00c896" if won else "#ff4444", size=7, opacity=0.8),
                name="Win" if won else "Loss",
                hovertemplate="%{x}<br>%{y:+.2f}<extra></extra>",
            ))
        fig_sc.add_hline(y=0, line_color="#555")
        fig_sc.update_layout(
            title="Individual Trades",
            yaxis_title="Net P&L ($)", xaxis_title="",
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="#ddd", height=280, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

# ── TAB 2: PAPER TRADING ───────────────────────────────────────────────────
with tab_paper:
    st.markdown(f"**Service status:** `{status}`")
    auto_refresh = st.checkbox("Auto-refresh every 30s", value=True)
    if auto_refresh:
        st.markdown(
            '<meta http-equiv="refresh" content="30">',
            unsafe_allow_html=True,
        )

    events = parse_paper_log(300)
    if not events:
        st.info("No log data yet — paper trading may still be loading models.")
    else:
        df_log = pd.DataFrame(events)
        # Highlight signal rows
        signal_rows = df_log[df_log["msg"].str.contains("Signal:|signal:|LONG|SHORT|FLAT", case=False, na=False)]
        trade_rows  = df_log[df_log["msg"].str.contains("Trade|pnl=", case=False, na=False)]

        st.markdown(f"**Last signal events** (of {len(events)} log lines):")
        if not signal_rows.empty:
            st.dataframe(signal_rows[["time","module","msg"]].tail(30), use_container_width=True, hide_index=True)
        else:
            st.info("No signals yet — waiting for next 15m candle close.")

        if not trade_rows.empty:
            st.markdown("**Trade events:**")
            st.dataframe(trade_rows[["time","module","msg"]].tail(20), use_container_width=True, hide_index=True)

        with st.expander("Full log (last 100 lines)"):
            st.code("\n".join(e["time"] + " " + e["level"].ljust(8) + " " + e["module"] + ": " + e["msg"]
                              for e in events[-100:]))

# ── TAB 3: OPTIMIZER ───────────────────────────────────────────────────────
with tab_opt:
    entries = load_optimizer_history()
    if not entries:
        st.info("No optimizer retune history yet.")
    else:
        df_opt = pd.DataFrame(entries)
        df_opt["status"] = df_opt["applied"].map({True: "✅ applied", False: "⏭️ skipped"})

        st.markdown(f"**{len(df_opt)} total retunes**")
        st.dataframe(
            df_opt[["timestamp", "trigger", "in_sample_pf", "out_of_sample_pf", "status"]].sort_values("timestamp", ascending=False),
            use_container_width=True, hide_index=True,
        )

        fig_pf = go.Figure()
        fig_pf.add_trace(go.Bar(x=df_opt["timestamp"], y=df_opt["in_sample_pf"], name="In-sample PF", marker_color="#7b8cff"))
        fig_pf.add_trace(go.Bar(x=df_opt["timestamp"], y=df_opt["out_of_sample_pf"], name="OOS PF", marker_color="#00c896"))
        fig_pf.add_hline(y=1.0, line_dash="dash", line_color="#ff4444", annotation_text="PF=1 (breakeven)")
        fig_pf.update_layout(
            title="Optimizer Profit Factors", barmode="group",
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="#ddd", height=300, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_pf, use_container_width=True)

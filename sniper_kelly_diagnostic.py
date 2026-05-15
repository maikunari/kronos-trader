"""
sniper_kelly_diagnostic.py
Compute Kelly-criterion growth-rate (G) for the Phase 1 sniper across a
representative universe of tickers.

The point: answer "does the sniper actually grow capital in the long run
under Kelly-optimal sizing?" — the same question we're asking of the
Phase C setup detectors, but applied to the parallel many-small-edges
track.

For each ticker:
  1. Run a single-path backtest with the current config.yaml params
  2. Extract per-trade returns (trade.pnl_usd / trade.size_usd)
  3. Compute Kelly fraction f* and growth rate G via validation.kelly

Output: tasks/sniper_kelly_diagnostic.md side-by-side with the Phase C
report so we can compare. The decisive question is which track has
higher G after costs (fees + slippage + funding ARE included in
trade.pnl_usd).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from backtest import BacktestResult, run_snipe_backtest
from hyperliquid_feed import fetch_historical_binance, fetch_historical_hl
from main import build_engine
from validation.kelly import growth_rate


def _safe_fetch(symbol: str, timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Try HL, fall back to Binance, return empty df on any error."""
    try:
        df = fetch_historical_hl(symbol, timeframe, start_ms, end_ms)
    except Exception:
        df = pd.DataFrame()
    if df is None or df.empty:
        try:
            df = fetch_historical_binance(symbol, timeframe, start_ms, end_ms)
        except Exception:
            df = pd.DataFrame()
    return df if df is not None else pd.DataFrame()


def _backtest_one(symbol: str, timeframe: str, start: str, end: str, config: dict) -> BacktestResult | None:
    start_ms = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp() * 1000)
    df = _safe_fetch(symbol, timeframe, start_ms, end_ms)
    if df.empty:
        return None
    df_1h = _safe_fetch(symbol, "1h", start_ms, end_ms)
    df_4h = _safe_fetch(symbol, "4h", start_ms, end_ms)
    engine = build_engine(config)
    b = config.get("backtest", {})
    return run_snipe_backtest(
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


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    # Silence noisy per-trade risk_manager output — we want the summary only.
    logging.getLogger("risk_manager").setLevel(logging.ERROR)
    logging.getLogger("hyperliquid_feed").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Sniper Kelly diagnostic")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--start", default="2025-04-01")
    parser.add_argument("--end", default="2026-04-01")
    parser.add_argument("--tickers", nargs="*",
                        default=["BTC", "ETH", "SOL", "HYPE"],
                        help="Tickers to backtest (default: BTC ETH SOL HYPE)")
    parser.add_argument("--out", default="tasks/sniper_kelly_diagnostic.md")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())

    print(f"Sniper Kelly diagnostic: {len(args.tickers)} tickers, "
          f"{args.timeframe}, {args.start} → {args.end}", file=sys.stderr)

    rows: list[dict] = []
    all_returns: list[float] = []   # universe-level pool

    for ticker in args.tickers:
        print(f"  backtest {ticker}...", file=sys.stderr)
        try:
            result = _backtest_one(ticker, args.timeframe, args.start, args.end, config)
        except Exception as exc:
            print(f"    skipped {ticker}: {exc}", file=sys.stderr)
            rows.append({"ticker": ticker, "trades": 0, "skipped": True,
                         "skip_reason": str(exc)[:120]})
            continue
        if result is None or not result.trades:
            rows.append({"ticker": ticker, "trades": 0, "skipped": True,
                         "skip_reason": "no data or no trades"})
            continue
        returns = [t.pnl_usd / t.size_usd for t in result.trades if t.size_usd > 0]
        all_returns.extend(returns)
        k = growth_rate(returns)
        rows.append({
            "ticker": ticker,
            "trades": len(returns),
            "win_rate": k.win_rate,
            "payoff_ratio": k.payoff_ratio,
            "kelly_fraction": k.kelly_fraction,
            "growth_rate": k.growth_rate,
            "growth_rate_full_unit": k.growth_rate_full_unit,
            "mean_return": k.mean_return,
            "sharpe": result.sharpe_ratio,
            "total_return_pct": result.total_return_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "profit_factor": result.profit_factor,
        })

    # Universe-pooled Kelly
    pooled = growth_rate(all_returns)

    lines: list[str] = []
    lines.append("# Sniper Kelly diagnostic")
    lines.append("")
    lines.append(f"- Timeframe: {args.timeframe}")
    lines.append(f"- Window: {args.start} → {args.end}")
    lines.append(f"- Tickers: {', '.join(args.tickers)}")
    lines.append(f"- Total trades pooled: {pooled.n_trades}")
    lines.append("")

    lines.append("## Per-ticker")
    lines.append("")
    lines.append(
        "| ticker | trades | win% | payoff | f* | G | G(f=1) | "
        "mean_ret | Sharpe | total_ret | max_dd |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("skipped"):
            lines.append(f"| {r['ticker']} | 0 | — | — | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {r['ticker']} | {r['trades']} | "
            f"{r['win_rate']*100:.1f}% | {r['payoff_ratio']:.2f} | "
            f"{r['kelly_fraction']:.3f} | {r['growth_rate']:+.5f} | "
            f"{r['growth_rate_full_unit']:+.5f} | "
            f"{r['mean_return']*100:+.2f}% | "
            f"{r['sharpe']:.2f} | {r['total_return_pct']*100:+.1f}% | "
            f"{r['max_drawdown_pct']*100:.1f}% |"
        )
    lines.append("")

    lines.append("## Universe-pooled Kelly")
    lines.append("")
    lines.append(f"- n_trades:          **{pooled.n_trades}**")
    lines.append(f"- win_rate:          **{pooled.win_rate*100:.1f}%**")
    lines.append(f"- avg_win:           {pooled.avg_win*100:+.2f}%")
    lines.append(f"- avg_loss:          {pooled.avg_loss*100:+.2f}%")
    lines.append(f"- payoff_ratio:      {pooled.payoff_ratio:.2f}")
    lines.append(f"- Kelly fraction f*: **{pooled.kelly_fraction:.3f}**")
    lines.append(f"- growth rate G:     **{pooled.growth_rate:+.5f}** per trade")
    lines.append(f"- G(f=1.0):          {pooled.growth_rate_full_unit:+.5f}")
    lines.append(f"- mean return:       {pooled.mean_return*100:+.3f}%")
    lines.append("")
    lines.append(
        "**Verdict:** "
        + (
            "✓ Sniper has POSITIVE edge under Kelly sizing — capital "
            f"grows at {pooled.growth_rate:+.4f} per trade in log units."
            if pooled.has_positive_edge
            else "✗ Sniper has NO positive edge under Kelly — capital "
            "decays under any positive sizing fraction. Either retune "
            "or abandon."
        )
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    out_path.with_suffix(".json").write_text(json.dumps({
        "rows": rows,
        "pooled": {
            "n_trades": pooled.n_trades,
            "win_rate": pooled.win_rate,
            "payoff_ratio": pooled.payoff_ratio,
            "kelly_fraction": pooled.kelly_fraction,
            "growth_rate": pooled.growth_rate,
            "growth_rate_full_unit": pooled.growth_rate_full_unit,
            "mean_return": pooled.mean_return,
        },
        "config": {
            "timeframe": args.timeframe,
            "start": args.start,
            "end": args.end,
            "tickers": args.tickers,
        },
    }, indent=2))
    print(f"\nReport: {out_path}", file=sys.stderr)
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

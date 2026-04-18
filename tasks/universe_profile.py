"""
universe_profile.py
One-off diagnostic: pull HL per-asset liquidity stats for the Krillin
watchlist and produce a homogeneity report.

HL exposes (via info endpoint, type=metaAndAssetCtxs):
  - markPx          current mark price
  - dayNtlVlm       24h notional volume in USD
  - openInterest    current OI in base units
  - funding         current funding rate
  - premium         perp premium over oracle

We also estimate "30d avg volume" from historical 1d candle data since
HL's context only gives current-day.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import requests
import yaml

from hyperliquid_feed import fetch_historical_hl


def pull_asset_ctx() -> dict[str, dict]:
    """Fetch HL's per-asset context. Returns dict keyed by symbol."""
    resp = requests.post(
        "https://api.hyperliquid.xyz/info",
        json={"type": "metaAndAssetCtxs"}, timeout=15,
    )
    resp.raise_for_status()
    meta, ctxs = resp.json()
    universe = meta["universe"]
    out = {}
    for asset, ctx in zip(universe, ctxs):
        out[asset["name"]] = {
            "markPx": float(ctx.get("markPx", 0) or 0),
            "dayNtlVlm": float(ctx.get("dayNtlVlm", 0) or 0),
            "openInterest": float(ctx.get("openInterest", 0) or 0),
            "funding": float(ctx.get("funding", 0) or 0),
            "premium": float(ctx.get("premium", 0) or 0),
            "szDecimals": int(asset.get("szDecimals", 0)),
        }
    return out


def avg_daily_volume_30d(symbol: str) -> float:
    """Compute 30-day average daily notional volume from 1d candles."""
    import time
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - 30 * 24 * 3600 * 1000
    df = fetch_historical_hl(symbol, "1d", start_ms, end_ms)
    if df.empty:
        return 0.0
    # Notional volume = close * volume (in base units)
    return float((df["close"] * df["volume"]).mean())


def main() -> int:
    tickers = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "markets/krillin_watchlist.yaml").read_text()
    )["tickers"]
    print(f"Profiling {len(tickers)} tickers...\n", file=sys.stderr)

    ctx = pull_asset_ctx()

    rows = []
    for t in tickers:
        c = ctx.get(t)
        if c is None:
            print(f"  {t}: not in HL universe (skip)", file=sys.stderr)
            continue
        print(f"  {t}...", file=sys.stderr)
        avg_30d = avg_daily_volume_30d(t)
        mark = c["markPx"]
        oi_notional = c["openInterest"] * mark
        rows.append({
            "ticker": t,
            "price": mark,
            "vol_24h_usd": c["dayNtlVlm"],
            "vol_30d_avg_usd": avg_30d,
            "oi_usd": oi_notional,
            "funding_bp_hr": c["funding"] * 10_000,
            "premium_bp": c["premium"] * 10_000,
        })

    df = pd.DataFrame(rows).sort_values("vol_30d_avg_usd", ascending=False).reset_index(drop=True)

    # Print table
    print(f"\n{'='*110}")
    print(f"  UNIVERSE LIQUIDITY PROFILE  ({len(df)} tickers)")
    print(f"{'='*110}\n")
    print(f"{'ticker':<8} {'price':>12} {'24h vol $M':>12} "
          f"{'30d avg vol $M':>16} {'OI $M':>10} "
          f"{'funding bp/h':>14} {'premium bp':>12}")
    print("-" * 110)
    for _, r in df.iterrows():
        print(f"{r['ticker']:<8} {r['price']:>12.4f} "
              f"{r['vol_24h_usd']/1e6:>12.2f} "
              f"{r['vol_30d_avg_usd']/1e6:>16.2f} "
              f"{r['oi_usd']/1e6:>10.2f} "
              f"{r['funding_bp_hr']:>+14.3f} "
              f"{r['premium_bp']:>+12.2f}")

    # Distribution summary
    vols = df["vol_30d_avg_usd"] / 1e6
    ois = df["oi_usd"] / 1e6
    print(f"\n{'='*70}")
    print("  DISTRIBUTION SUMMARY (30d avg volume, $M)")
    print(f"{'='*70}")
    print(f"  min       {vols.min():>10.2f}")
    print(f"  p25       {vols.quantile(0.25):>10.2f}")
    print(f"  median    {vols.median():>10.2f}")
    print(f"  p75       {vols.quantile(0.75):>10.2f}")
    print(f"  max       {vols.max():>10.2f}")
    print(f"  spread    {vols.max() / max(vols.min(), 0.01):>10.1f}x  (max / min)")

    # Quick tiering
    print(f"\n{'='*70}")
    print("  TIERS BY 30d AVG DAILY VOLUME")
    print(f"{'='*70}")
    thick = df[df["vol_30d_avg_usd"] >= 50e6]
    mid = df[(df["vol_30d_avg_usd"] >= 10e6) & (df["vol_30d_avg_usd"] < 50e6)]
    thin = df[df["vol_30d_avg_usd"] < 10e6]
    print(f"  thick (>=$50M/day, {len(thick)}): {', '.join(thick['ticker'].tolist())}")
    print(f"  mid ($10-50M/day, {len(mid)}): {', '.join(mid['ticker'].tolist())}")
    print(f"  thin (<$10M/day, {len(thin)}): {', '.join(thin['ticker'].tolist())}")
    print()

    # Write CSV for future use
    out_csv = Path(__file__).resolve().parent / "universe_profile.csv"
    df.to_csv(out_csv, index=False)
    print(f"CSV written to {out_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

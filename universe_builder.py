"""
universe_builder.py
Construct a filtered trading universe from live HL + CoinGecko data.

Produces markets/core_universe.yaml with tickers passing the configured
filters (default: MC $100M-$10B, HL 30d avg daily vol >= $1M, OI >= $1M).

The resulting universe is the "production" list for validation and live
trading. The hand-curated Krillin list (markets/krillin_watchlist.yaml)
stays separately as a research reference.

Usage:
    ./venv/bin/python universe_builder.py
    ./venv/bin/python universe_builder.py --mc-min 500e6 --mc-max 5e9 --vol-min 3e6

CoinGecko free tier (no key) is rate-limited to ~10-30 calls/min. The
builder caches /coins/list and the final ticker->CoinGecko-ID map to
cut subsequent runs to a handful of calls.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yaml

from hyperliquid_feed import fetch_historical_hl

logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache/universe")
CG_BASE = "https://api.coingecko.com/api/v3"
HL_INFO = "https://api.hyperliquid.xyz/info"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TickerProfile:
    symbol: str                    # HL ticker
    price: float                   # HL mark
    vol_24h_usd: float             # HL 24h notional
    vol_30d_avg_usd: float         # 30d avg from daily candles
    oi_usd: float                  # OI × price
    funding_bp_hr: float
    market_cap_usd: Optional[float] = None
    coingecko_id: Optional[str] = None
    excluded_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Hyperliquid data
# ---------------------------------------------------------------------------

def fetch_hl_universe() -> dict[str, dict]:
    """Pull HL perp universe with per-asset context."""
    resp = requests.post(HL_INFO, json={"type": "metaAndAssetCtxs"}, timeout=15)
    resp.raise_for_status()
    meta, ctxs = resp.json()
    out = {}
    for asset, ctx in zip(meta["universe"], ctxs):
        sym = asset["name"]
        out[sym] = {
            "markPx": float(ctx.get("markPx", 0) or 0),
            "dayNtlVlm": float(ctx.get("dayNtlVlm", 0) or 0),
            "openInterest": float(ctx.get("openInterest", 0) or 0),
            "funding": float(ctx.get("funding", 0) or 0),
        }
    return out


def fetch_hl_30d_volume(symbol: str) -> float:
    """Compute 30-day avg notional volume from HL 1d candles."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - 30 * 24 * 3600 * 1000
    df = fetch_historical_hl(symbol, "1d", start_ms, end_ms)
    if df.empty:
        return 0.0
    return float((df["close"] * df["volume"]).mean())


# ---------------------------------------------------------------------------
# CoinGecko mapping
# ---------------------------------------------------------------------------

def cg_coins_list(cache_path: Path = CACHE_DIR / "cg_coins_list.json",
                   max_age_hours: int = 24) -> list[dict]:
    """Fetch and cache CoinGecko's full coins list."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < max_age_hours:
            return json.loads(cache_path.read_text())
    logger.info("Fetching CoinGecko /coins/list (will cache for %dh)", max_age_hours)
    resp = requests.get(f"{CG_BASE}/coins/list", timeout=30)
    resp.raise_for_status()
    data = resp.json()
    cache_path.write_text(json.dumps(data))
    return data


def cg_markets(ids: list[str], per_page: int = 250) -> list[dict]:
    """Fetch market data (inc. MC) for a batch of CoinGecko IDs."""
    if not ids:
        return []
    result = []
    for i in range(0, len(ids), per_page):
        batch = ids[i : i + per_page]
        params = {
            "vs_currency": "usd",
            "ids": ",".join(batch),
            "per_page": per_page,
            "page": 1,
            "order": "market_cap_desc",
        }
        resp = requests.get(f"{CG_BASE}/coins/markets", params=params, timeout=30)
        if resp.status_code == 429:
            logger.warning("CoinGecko rate limited — sleeping 60s")
            time.sleep(60)
            resp = requests.get(f"{CG_BASE}/coins/markets", params=params, timeout=30)
        resp.raise_for_status()
        result.extend(resp.json())
        time.sleep(2)   # be polite to free tier
    return result


def build_symbol_to_cg_map(
    hl_symbols: list[str],
    cache_path: Path = CACHE_DIR / "hl_to_cg_map.yaml",
    max_age_hours: int = 168,  # 1 week
) -> dict[str, str]:
    """
    Resolve HL ticker -> CoinGecko ID. Uses cache when available.

    For ambiguous symbols (multiple CG entries with the same symbol), picks
    the one with the highest market cap via /coins/markets.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < max_age_hours:
            cached = yaml.safe_load(cache_path.read_text()) or {}
            existing = cached.get("mapping", {})
            # Check if all our HL symbols are covered
            missing = [s for s in hl_symbols if s not in existing]
            if not missing:
                return existing
            logger.info("Cache missing %d symbols; refreshing", len(missing))

    coins_list = cg_coins_list()
    # Build lower-symbol -> list of CG ids
    by_symbol: dict[str, list[str]] = {}
    for c in coins_list:
        by_symbol.setdefault(c["symbol"].lower(), []).append(c["id"])

    mapping: dict[str, str] = {}
    ambiguous: list[tuple[str, list[str]]] = []
    unresolved: list[str] = []

    for sym in hl_symbols:
        candidates = by_symbol.get(sym.lower(), [])
        if len(candidates) == 1:
            mapping[sym] = candidates[0]
        elif len(candidates) > 1:
            ambiguous.append((sym, candidates))
        else:
            unresolved.append(sym)

    # Resolve ambiguous by highest market cap
    if ambiguous:
        logger.info("Resolving %d ambiguous symbols via market cap...", len(ambiguous))
        all_candidate_ids = [cid for _, cids in ambiguous for cid in cids]
        markets = cg_markets(all_candidate_ids)
        mc_by_id = {m["id"]: m.get("market_cap") or 0 for m in markets}
        for sym, cids in ambiguous:
            best = max(cids, key=lambda cid: mc_by_id.get(cid, 0))
            mapping[sym] = best

    # Persist
    cache_path.write_text(yaml.safe_dump({
        "built_at": datetime.now(timezone.utc).isoformat(),
        "mapping": mapping,
        "unresolved": unresolved,
    }, sort_keys=True))
    if unresolved:
        logger.info("Could not resolve %d HL symbols in CoinGecko: %s",
                    len(unresolved), unresolved)
    return mapping


def fetch_market_caps(cg_ids: list[str]) -> dict[str, float]:
    """CG id -> market cap in USD."""
    markets = cg_markets(cg_ids)
    return {m["id"]: float(m.get("market_cap") or 0) for m in markets}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_universe(
    *,
    mc_min: float = 100e6,
    mc_max: float = 10e9,
    vol_min: float = 1e6,
    oi_min: float = 1e6,
    allow_missing_mc: bool = False,
) -> tuple[list[TickerProfile], list[TickerProfile]]:
    """
    Apply the filter criteria and return (included, excluded) ticker profiles.
    """
    logger.info("Fetching HL universe...")
    hl = fetch_hl_universe()
    logger.info("HL universe size: %d", len(hl))

    logger.info("Building CoinGecko symbol map (may be slow on first run)...")
    cg_map = build_symbol_to_cg_map(list(hl.keys()))
    logger.info("Mapped %d of %d HL tickers to CoinGecko", len(cg_map), len(hl))

    # Pull market caps in one batched call
    cg_ids = list(set(cg_map.values()))
    logger.info("Fetching market caps for %d CoinGecko IDs...", len(cg_ids))
    mc_by_id = fetch_market_caps(cg_ids)

    included: list[TickerProfile] = []
    excluded: list[TickerProfile] = []

    for sym, ctx in hl.items():
        price = ctx["markPx"]
        vol_24h = ctx["dayNtlVlm"]
        oi_usd = ctx["openInterest"] * price
        funding_bp = ctx["funding"] * 10_000

        cg_id = cg_map.get(sym)
        mc = mc_by_id.get(cg_id) if cg_id else None

        # Fetch 30d avg volume — expensive, only for candidates that look close
        # on 24h. Skip if 24h < vol_min/4 (clearly too thin).
        if vol_24h < vol_min / 4:
            profile = TickerProfile(
                symbol=sym, price=price, vol_24h_usd=vol_24h,
                vol_30d_avg_usd=0.0, oi_usd=oi_usd, funding_bp_hr=funding_bp,
                market_cap_usd=mc, coingecko_id=cg_id,
                excluded_reason=f"vol_24h<{vol_min/4:.0e}",
            )
            excluded.append(profile)
            continue

        vol_30d = fetch_hl_30d_volume(sym)

        profile = TickerProfile(
            symbol=sym, price=price, vol_24h_usd=vol_24h,
            vol_30d_avg_usd=vol_30d, oi_usd=oi_usd, funding_bp_hr=funding_bp,
            market_cap_usd=mc, coingecko_id=cg_id,
        )

        reasons: list[str] = []
        if vol_30d < vol_min:
            reasons.append(f"vol_30d<{vol_min:.0e}")
        if oi_usd < oi_min:
            reasons.append(f"oi<{oi_min:.0e}")
        if mc is None:
            if not allow_missing_mc:
                reasons.append("no_cg_mc")
        else:
            if mc < mc_min:
                reasons.append(f"mc<{mc_min:.0e}")
            if mc > mc_max:
                reasons.append(f"mc>{mc_max:.0e}")

        if reasons:
            profile.excluded_reason = ",".join(reasons)
            excluded.append(profile)
        else:
            included.append(profile)

    included.sort(key=lambda t: t.vol_30d_avg_usd, reverse=True)
    return included, excluded


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_universe_yaml(
    included: list[TickerProfile],
    excluded: list[TickerProfile],
    *,
    out_path: Path,
    filters: dict,
) -> None:
    doc = {
        "metadata": {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "builder_version": "0.1",
            "filters": filters,
            "counts": {"included": len(included), "excluded": len(excluded)},
        },
        "tickers": [t.symbol for t in included],
        "profiles": [
            {
                "symbol": t.symbol,
                "price": round(t.price, 6),
                "market_cap_usd": round(t.market_cap_usd) if t.market_cap_usd else None,
                "vol_30d_avg_usd": round(t.vol_30d_avg_usd),
                "vol_24h_usd": round(t.vol_24h_usd),
                "oi_usd": round(t.oi_usd),
                "funding_bp_hr": round(t.funding_bp_hr, 3),
                "coingecko_id": t.coingecko_id,
            }
            for t in included
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(doc, sort_keys=False, width=120))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Build HL-tradeable universe from filters.")
    parser.add_argument("--mc-min", type=float, default=100e6, help="Min market cap USD (default 100M)")
    parser.add_argument("--mc-max", type=float, default=10e9, help="Max market cap USD (default 10B)")
    parser.add_argument("--vol-min", type=float, default=1e6, help="Min 30d avg daily vol USD (default 1M)")
    parser.add_argument("--oi-min", type=float, default=1e6, help="Min OI USD (default 1M)")
    parser.add_argument("--allow-missing-mc", action="store_true",
                        help="Include tickers CoinGecko doesn't have MC for (default: exclude)")
    parser.add_argument("--out", default="markets/core_universe.yaml")
    args = parser.parse_args()

    included, excluded = build_universe(
        mc_min=args.mc_min, mc_max=args.mc_max,
        vol_min=args.vol_min, oi_min=args.oi_min,
        allow_missing_mc=args.allow_missing_mc,
    )

    filters = {
        "market_cap_min_usd": args.mc_min,
        "market_cap_max_usd": args.mc_max,
        "vol_30d_min_usd": args.vol_min,
        "oi_min_usd": args.oi_min,
        "allow_missing_mc": args.allow_missing_mc,
    }
    out_path = Path(args.out)
    write_universe_yaml(included, excluded, out_path=out_path, filters=filters)

    # Console summary
    print(f"\n{'='*90}")
    print(f"  CORE UNIVERSE — {len(included)} tickers (from {len(included)+len(excluded)} HL perps)")
    print(f"{'='*90}")
    print(f"  Filters: MC ${args.mc_min/1e6:,.0f}M-${args.mc_max/1e9:.0f}B, "
          f"vol30d >= ${args.vol_min/1e6:.0f}M, OI >= ${args.oi_min/1e6:.0f}M")
    print(f"  Written to: {out_path}\n")
    print(f"  {'symbol':<8} {'price':>10} {'mc $M':>10} {'vol30d $M':>12} {'oi $M':>10}")
    print("  " + "-" * 55)
    for t in included:
        mc_str = f"{t.market_cap_usd/1e6:,.0f}" if t.market_cap_usd else "?"
        print(f"  {t.symbol:<8} {t.price:>10.4f} {mc_str:>10} "
              f"{t.vol_30d_avg_usd/1e6:>12.2f} {t.oi_usd/1e6:>10.2f}")
    print()

    # Exclusion reason breakdown
    reasons: dict[str, int] = {}
    for t in excluded:
        key = t.excluded_reason or "unknown"
        reasons[key] = reasons.get(key, 0) + 1
    print(f"  Exclusion reasons ({len(excluded)} total):")
    for r, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {n:>4}  {r}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

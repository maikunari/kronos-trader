"""
microstructure.py
Pure-math microstructure features for signal confirmation and ML inputs.

Each function is stateless and takes pre-fetched series/dataframes. Data
sourcing (Hyperliquid OI, Binance spot, Coinglass liquidations) lives in
other modules; this file is testable without any network access.

Features:
- oi_delta_pct: rate of change of open interest over last N bars.
- basis_pct / basis_expansion_pct: perp-spot premium and its change.
- cvd: cumulative volume delta from taker-buy / total-volume.
- cvd_slope: linear-regression slope of CVD over last N bars.
- liquidation_proximity: distance and size of the nearest liquidation cluster
  in a given direction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --- Open interest ------------------------------------------------------------

def oi_delta_pct(oi: pd.Series, window: int = 3) -> float:
    """
    Percentage change in open interest over the last `window` bars.

    Positive = OI expanding (new positions opening alongside the move).
    Negative = OI contracting (positions closing — often distribution / fade).
    Returns NaN if insufficient data.
    """
    oi = pd.Series(oi).astype(float)
    if len(oi) < window + 1:
        return float("nan")
    prev = oi.iloc[-window - 1]
    cur = oi.iloc[-1]
    if prev <= 0:
        return float("nan")
    return float((cur - prev) / prev)


# --- Basis (perp - spot) ------------------------------------------------------

def basis_pct(perp_close: float, spot_close: float) -> float:
    """
    Perp-spot basis as a fraction of spot. Positive = perp trades at a premium.

    Typical regimes:
        > +0.1%  -> perp premium, longs crowded, funding positive, fade-the-long
        ~ 0      -> flat, neutral
        < -0.1%  -> perp discount, shorts crowded, fade-the-short
    """
    if spot_close <= 0:
        return float("nan")
    return float((perp_close - spot_close) / spot_close)


def basis_expansion_pct(
    perp_closes: pd.Series, spot_closes: pd.Series, window: int = 3
) -> float:
    """
    Change in basis over the last `window` bars — captures premium
    building in the direction of a move.

    Positive value in a rising market = perp leading spot = confirmation.
    Returns NaN if series are too short or misaligned.
    """
    p = pd.Series(perp_closes).astype(float)
    s = pd.Series(spot_closes).astype(float)
    if len(p) < window + 1 or len(s) < window + 1:
        return float("nan")
    cur = basis_pct(float(p.iloc[-1]), float(s.iloc[-1]))
    prev = basis_pct(float(p.iloc[-window - 1]), float(s.iloc[-window - 1]))
    if np.isnan(cur) or np.isnan(prev):
        return float("nan")
    return cur - prev


# --- Cumulative Volume Delta --------------------------------------------------

def cvd(taker_buy_volume: pd.Series, total_volume: pd.Series) -> pd.Series:
    """
    Cumulative volume delta: running sum of (taker_buy - taker_sell).

    Inputs are per-bar taker-buy volume and per-bar total volume.
    Taker-sell volume is derived as total - taker_buy.

    Rising CVD = aggressive buying dominates; falling = selling.
    """
    buy = pd.Series(taker_buy_volume).astype(float)
    total = pd.Series(total_volume).astype(float)
    if len(buy) != len(total):
        raise ValueError("taker_buy_volume and total_volume must have same length")
    sell = total - buy
    delta = buy - sell
    return delta.cumsum()


def cvd_slope(cvd_series: pd.Series, window: int = 20) -> float:
    """
    Linear-regression slope of CVD over the last `window` bars.

    Sign indicates net aggression direction; magnitude is absolute volume
    per bar, so normalize by average bar volume for cross-asset comparison.
    """
    series = pd.Series(cvd_series).astype(float).dropna()
    if len(series) < window:
        return float("nan")
    y = series.iloc[-window:].to_numpy()
    x = np.arange(window)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


# --- Liquidation clusters -----------------------------------------------------

@dataclass
class LiquidationCluster:
    price: float
    volume: float    # USD notional queued to be liquidated at this level
    side: str        # "long" (longs get liquidated if price drops to here) or "short"


def liquidation_proximity(
    current_price: float,
    clusters: list[LiquidationCluster],
    direction: str,
    max_distance_pct: float = 0.05,
) -> tuple[Optional[float], Optional[float]]:
    """
    Find the nearest liquidation cluster in the direction the price is
    expected to move. Liquidation cascades pull price toward clusters, so
    for a long entry we want the nearest *upside* cluster.

    A long-side cluster sits below current price (longs liquidated on drop);
    a short-side cluster sits above current price (shorts liquidated on rally).
    For a long entry we target upside short-liquidation clusters; for a short
    entry we target downside long-liquidation clusters.

    Args:
        current_price: live price.
        clusters: list of cluster objects.
        direction: "long" or "short" — entry direction we are confirming.
        max_distance_pct: ignore clusters farther than this fraction away.

    Returns:
        (distance_pct, cluster_volume) or (None, None) if no cluster found.
        distance_pct is positive regardless of direction.
    """
    if direction == "long":
        target_side = "short"
        candidates = [c for c in clusters if c.price > current_price and c.side == target_side]
    elif direction == "short":
        target_side = "long"
        candidates = [c for c in clusters if c.price < current_price and c.side == target_side]
    else:
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

    if not candidates or current_price <= 0:
        return None, None

    # Nearest by absolute distance
    nearest = min(candidates, key=lambda c: abs(c.price - current_price))
    distance_pct = abs(nearest.price - current_price) / current_price
    if distance_pct > max_distance_pct:
        return None, None
    return float(distance_pct), float(nearest.volume)

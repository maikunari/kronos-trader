"""
support_resistance.py
Detect structural support and resistance zones from OHLCV candles.

Follows CBS's approach:
  * Pivots detected on the CLOSE series (line chart view) — not wicks.
    Wicks are noise; price that settles at a level is what matters.
  * Pivots clustered into zones by proximity (0.5% band for liquid
    tickers, wider for thin ones).
  * Touch-count scoring is CBS's sweet-spot-at-3 rule. 3 touches is
    the strongest; more touches actually *weaken* the level because
    order flow at that level has been exhausted.
  * Age decay: older zones contribute less strength.
  * Volume weighting: touches with above-median volume score higher.

Exports `SRZone` dataclasses classified by role (support vs resistance)
relative to a reference price. Flip detection lives in a Phase D
extension — this module handles fresh zones only.

Zone strength ∈ [0, 1] is the composite used downstream for:
  * Trigger confidence (setups in §4 of architecture.md)
  * TP laddering (higher-strength zones have higher fill probability)
  * Consolidation/weak-bounce detectors (only fire on strong zones)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd

from pivot import Pivot, PivotKind, find_pivots

logger = logging.getLogger(__name__)

Side = Literal["resistance", "support"]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TouchEvent:
    """A single pivot that touched a zone."""
    bar_index: int
    price: float
    volume: float
    pivot_kind: PivotKind
    timestamp: Optional[pd.Timestamp] = None


@dataclass(frozen=True)
class SRZone:
    price_mid: float
    price_low: float
    price_high: float
    side: Side
    strength: float                       # composite, 0..1
    touches: tuple[TouchEvent, ...]

    @property
    def touch_count(self) -> int:
        return len(self.touches)

    @property
    def last_touch_bar(self) -> int:
        return max(t.bar_index for t in self.touches) if self.touches else -1

    def contains(self, price: float) -> bool:
        return self.price_low <= price <= self.price_high

    def distance_pct(self, reference: float) -> float:
        """Signed distance from reference to zone mid, as fraction."""
        if reference <= 0:
            return float("inf")
        return (self.price_mid - reference) / reference


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------

def touch_strength(n_touches: int) -> float:
    """
    CBS's sweet-spot-at-3 rule.

    2 touches = 0.6 (tentative)
    3 touches = 1.0 (ideal)
    4+ touches = linear decay (order flow exhausted)
    """
    if n_touches < 2:
        return 0.0
    if n_touches == 2:
        return 0.6
    if n_touches == 3:
        return 1.0
    return max(0.3, 1.0 - (n_touches - 3) * 0.15)


def age_factor(
    last_touch_bar: int,
    current_bar: int,
    start_decay_age: int = 500,
    full_decay_age: int = 2000,
) -> float:
    """
    Linear decay with deadband.

    Zones untouched for < `start_decay_age` bars contribute full strength.
    Strength decays linearly to 0 between start_decay_age and full_decay_age.
    Older zones contribute nothing.
    """
    age = current_bar - last_touch_bar
    if age <= start_decay_age:
        return 1.0
    if age >= full_decay_age:
        return 0.0
    return 1.0 - (age - start_decay_age) / (full_decay_age - start_decay_age)


def volume_weight(touches: list[TouchEvent], median_volume: float) -> float:
    """
    Median touch volume relative to overall median. Clipped to [0.5, 1.5].

    A zone whose touches happened on high-volume bars is a more reliable
    structural level than one built during thin sessions.
    """
    if not touches or median_volume <= 0:
        return 1.0
    touch_volumes = [t.volume for t in touches if t.volume > 0]
    if not touch_volumes:
        return 1.0
    ratio = float(np.median(touch_volumes)) / float(median_volume)
    return float(np.clip(ratio, 0.5, 1.5))


# ---------------------------------------------------------------------------
# Zone detection
# ---------------------------------------------------------------------------

def detect_sr_zones(
    candles: pd.DataFrame,
    *,
    pivot_window: int = 5,
    merge_pct: float = 0.005,
    min_touches: int = 2,
    reference_price: Optional[float] = None,
    start_decay_age: int = 500,
    full_decay_age: int = 2000,
) -> list[SRZone]:
    """
    Detect support/resistance zones from `candles`.

    Args:
        candles:        DataFrame with columns [close, volume]; optionally
                        [timestamp]. Close-based pivot detection per CBS.
        pivot_window:   fractal window for swing detection (default 5 bars
                        each side).
        merge_pct:      zones within this fractional band merge. 0.005 =
                        0.5%, appropriate for liquid majors. Mid-caps may
                        need 0.01-0.02 for the narrative volatility.
        min_touches:    minimum pivots required to form a zone. 2 is
                        tentative (strength 0.6); 3 is CBS ideal.
        reference_price: price used to classify side. Defaults to the last
                        close. Zones above → resistance, below → support.
        start_decay_age / full_decay_age: age-decay parameters in bars.

    Returns:
        List of SRZone, sorted by strength descending.
    """
    if "close" not in candles.columns or "volume" not in candles.columns:
        raise ValueError("candles must have columns [close, volume]")
    if len(candles) < 2 * pivot_window + 2:
        return []

    closes = candles["close"].astype(float).reset_index(drop=True)
    volumes = candles["volume"].astype(float).reset_index(drop=True)
    timestamps = (
        candles["timestamp"].reset_index(drop=True)
        if "timestamp" in candles.columns
        else pd.Series([None] * len(candles))
    )

    ref = float(reference_price) if reference_price is not None else float(closes.iloc[-1])
    current_bar = len(closes) - 1
    median_volume = float(volumes.median()) if len(volumes) else 0.0

    pivots = find_pivots(closes, window=pivot_window)
    if not pivots:
        return []

    # Build TouchEvent per pivot
    events: list[TouchEvent] = []
    for p in pivots:
        events.append(TouchEvent(
            bar_index=p.index,
            price=p.value,
            volume=float(volumes.iloc[p.index]) if p.index < len(volumes) else 0.0,
            pivot_kind=p.kind,
            timestamp=pd.Timestamp(timestamps.iloc[p.index]) if timestamps.iloc[p.index] is not None else None,
        ))

    clusters = _cluster_by_price(events, merge_pct=merge_pct)

    zones: list[SRZone] = []
    for cluster in clusters:
        if len(cluster) < min_touches:
            continue
        prices = [t.price for t in cluster]
        price_mid = float(np.median(prices))
        # The band spans the actual touch range, but capped at merge_pct of mid
        merge_band = price_mid * merge_pct
        price_low = max(min(prices), price_mid - merge_band)
        price_high = min(max(prices), price_mid + merge_band)

        last_bar = max(t.bar_index for t in cluster)
        raw_strength = (
            touch_strength(len(cluster))
            * age_factor(last_bar, current_bar, start_decay_age, full_decay_age)
            * volume_weight(cluster, median_volume)
        )
        strength = float(np.clip(raw_strength, 0.0, 1.0))
        if strength <= 0:
            continue

        side: Side = "resistance" if price_mid > ref else "support"
        zones.append(SRZone(
            price_mid=price_mid,
            price_low=price_low,
            price_high=price_high,
            side=side,
            strength=strength,
            touches=tuple(sorted(cluster, key=lambda t: t.bar_index)),
        ))

    zones.sort(key=lambda z: z.strength, reverse=True)
    return zones


def _cluster_by_price(
    events: list[TouchEvent],
    merge_pct: float,
) -> list[list[TouchEvent]]:
    """
    Greedy price-proximity clustering.

    Sorts events by price, walks through them, merges into the current
    cluster if the next event is within merge_pct of the cluster's
    running midpoint. Otherwise starts a new cluster.
    """
    if not events:
        return []
    sorted_events = sorted(events, key=lambda t: t.price)
    clusters: list[list[TouchEvent]] = []
    current: list[TouchEvent] = [sorted_events[0]]
    current_mid = sorted_events[0].price

    for ev in sorted_events[1:]:
        if current_mid > 0 and abs(ev.price - current_mid) / current_mid <= merge_pct:
            current.append(ev)
            # Update running midpoint to keep cluster cohesive
            current_mid = float(np.mean([t.price for t in current]))
        else:
            clusters.append(current)
            current = [ev]
            current_mid = ev.price
    clusters.append(current)
    return clusters


# ---------------------------------------------------------------------------
# Breakout detection (close-based)
# ---------------------------------------------------------------------------

def is_close_above(zone: SRZone, close: float) -> bool:
    """Strict close above the zone's upper band."""
    return close > zone.price_high


def is_close_below(zone: SRZone, close: float) -> bool:
    """Strict close below the zone's lower band."""
    return close < zone.price_low


def confirmed_breakout(
    zone: SRZone,
    closes: pd.Series,
    required_closes: int = 2,
) -> Literal["up", "down", "none"]:
    """
    Has price closed through `zone` on `required_closes` consecutive bars?

    Follows CBS: breakouts are measured on closes, not wicks. A single
    close through is provisional; two is confirmation.
    """
    if len(closes) < required_closes:
        return "none"
    recent = closes.tail(required_closes).to_numpy(dtype=float)
    if all(c > zone.price_high for c in recent):
        return "up"
    if all(c < zone.price_low for c in recent):
        return "down"
    return "none"


# ---------------------------------------------------------------------------
# Convenience filters
# ---------------------------------------------------------------------------

def nearest_in_direction(
    zones: list[SRZone],
    reference: float,
    direction: Literal["up", "down"],
    min_strength: float = 0.0,
    max_distance_pct: Optional[float] = None,
) -> Optional[SRZone]:
    """Return the nearest zone above (up) or below (down) `reference`."""
    candidates = []
    for z in zones:
        if z.strength < min_strength:
            continue
        dist = z.distance_pct(reference)
        if direction == "up" and dist > 0:
            candidates.append((dist, z))
        elif direction == "down" and dist < 0:
            candidates.append((-dist, z))
    if not candidates:
        return None
    candidates.sort()
    dist, z = candidates[0]
    if max_distance_pct is not None and dist > max_distance_pct:
        return None
    return z


def zones_in_direction(
    zones: list[SRZone],
    reference: float,
    direction: Literal["up", "down"],
    min_strength: float = 0.0,
    max_distance_pct: Optional[float] = None,
) -> list[SRZone]:
    """
    All zones in a given direction from reference, ordered by proximity.

    Used by TP-ladder builder (§6.4 of architecture.md).
    """
    candidates = []
    for z in zones:
        if z.strength < min_strength:
            continue
        dist = z.distance_pct(reference)
        if direction == "up" and dist > 0:
            candidates.append((dist, z))
        elif direction == "down" and dist < 0:
            candidates.append((-dist, z))
    candidates.sort()
    if max_distance_pct is not None:
        candidates = [(d, z) for d, z in candidates if d <= max_distance_pct]
    return [z for _, z in candidates]

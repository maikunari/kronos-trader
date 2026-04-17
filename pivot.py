"""
pivot.py
Fractal swing-high / swing-low detection on a time series.

Used throughout the Shiller-style engine:
  * support_resistance.py clusters pivot levels into S/R zones
  * divergence detector compares pivot values on price vs indicators
  * position manager's leg-based trailing stop uses recent pivots

CBS's rule "draw levels at closes, not wicks" is reflected by having
S/R default to pivot detection on close prices. Other callers (e.g.,
V-reversal structure) can feed high/low series when wicks matter.

Core idea (fractal pivot):
  A swing high at index i requires series[i] to be strictly greater than
  all values in [i-window, i+window] except itself. A swing low is
  symmetric. The window parameter trades off noise-rejection (larger
  window = fewer, more significant pivots) vs responsiveness.

Design:
  * Only "confirmed" pivots are returned — we need `window` bars on each
    side. The most recent `window` bars can contain a pivot whose
    confirmation is still pending; the caller should refresh.
  * Strict inequality by default; plateaus do not produce pivots. This
    matches human visual judgment — a flat run isn't a swing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PivotKind = Literal["high", "low"]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pivot:
    """A confirmed swing high or swing low.

    index:     integer position in the source series (0-based)
    value:     series value at the pivot bar
    kind:      "high" or "low"
    timestamp: source-series timestamp if the series had a DatetimeIndex or
               timestamp column; otherwise None
    window:    the fractal window size that confirmed this pivot (for
               downstream strength/age reasoning)
    """
    index: int
    value: float
    kind: PivotKind
    timestamp: Optional[pd.Timestamp] = None
    window: int = 5

    def __lt__(self, other: "Pivot") -> bool:
        return self.index < other.index


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def find_pivots(
    series: pd.Series,
    window: int = 5,
    kinds: Iterable[PivotKind] = ("high", "low"),
) -> list[Pivot]:
    """
    Return all confirmed pivots in `series`.

    Args:
        series: numeric pd.Series. If the index is a DatetimeIndex, timestamps
                populate Pivot.timestamp; otherwise Pivot.timestamp is None.
        window: fractal window. A pivot requires `window` bars on each side
                to confirm. Larger window = fewer, stronger pivots.
        kinds:  iterable of "high" and/or "low" to restrict the search.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if series is None or len(series) < 2 * window + 1:
        return []

    values = series.to_numpy(dtype=float)
    timestamps = (
        series.index
        if isinstance(series.index, pd.DatetimeIndex)
        else [None] * len(series)
    )
    want_high = "high" in kinds
    want_low = "low" in kinds

    pivots: list[Pivot] = []
    for i in range(window, len(values) - window):
        v = values[i]
        if np.isnan(v):
            continue
        left = values[i - window : i]
        right = values[i + 1 : i + 1 + window]

        if want_high and np.all(v > left) and np.all(v > right):
            pivots.append(Pivot(
                index=i, value=float(v), kind="high",
                timestamp=pd.Timestamp(timestamps[i]) if timestamps[i] is not None else None,
                window=window,
            ))
        elif want_low and np.all(v < left) and np.all(v < right):
            pivots.append(Pivot(
                index=i, value=float(v), kind="low",
                timestamp=pd.Timestamp(timestamps[i]) if timestamps[i] is not None else None,
                window=window,
            ))

    return pivots


def swing_highs(series: pd.Series, window: int = 5) -> list[Pivot]:
    """Convenience: only return confirmed swing highs."""
    return find_pivots(series, window=window, kinds=("high",))


def swing_lows(series: pd.Series, window: int = 5) -> list[Pivot]:
    """Convenience: only return confirmed swing lows."""
    return find_pivots(series, window=window, kinds=("low",))


# ---------------------------------------------------------------------------
# Recent-pivot helpers (for divergence / leg-trail logic)
# ---------------------------------------------------------------------------

def most_recent(
    pivots: Sequence[Pivot],
    n: int,
    kind: Optional[PivotKind] = None,
) -> list[Pivot]:
    """
    Return the `n` most recent pivots, optionally filtered by kind, in
    chronological order (oldest first, newest last).
    """
    if n < 1:
        return []
    filtered = [p for p in pivots if kind is None or p.kind == kind]
    filtered.sort()
    return filtered[-n:]


def current_leg_bottom(
    closes: pd.Series,
    direction: Literal["long", "short"],
    window: int = 3,
) -> Optional[float]:
    """
    Return the 'bottom of the current leg' used by CBS's leg-based trailing
    stop rule.

    For a long position: the most recent swing-low after the most recent
    swing-high. In practice: the highest-low since the last higher-high.

    Conservative fallback: if no confirmed swing exists in the recent series,
    returns None (caller should fall back to ATR-based trail).
    """
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

    pivots = find_pivots(closes, window=window)
    if not pivots:
        return None
    pivots.sort()

    if direction == "long":
        # Find the most recent swing-high, then the most recent swing-low after it
        last_high_idx = -1
        for p in reversed(pivots):
            if p.kind == "high":
                last_high_idx = p.index
                break
        if last_high_idx == -1:
            return None
        lows_after = [p for p in pivots if p.kind == "low" and p.index > last_high_idx]
        if not lows_after:
            # No low confirmed after the high yet — use the swing high's own basis
            return None
        return lows_after[-1].value
    else:
        last_low_idx = -1
        for p in reversed(pivots):
            if p.kind == "low":
                last_low_idx = p.index
                break
        if last_low_idx == -1:
            return None
        highs_after = [p for p in pivots if p.kind == "high" and p.index > last_low_idx]
        if not highs_after:
            return None
        return highs_after[-1].value

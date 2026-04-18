"""
validation/capture.py
Single-trade capture simulator for matched trigger × pop pairs.

For each true-positive match from the matcher, simulate the trade's
lifecycle over subsequent bars and report what fraction of the pop's
magnitude the bot would actually have banked — accounting for the
trigger's stop, its TP ladder, and partial fills along the way.

Simulator conventions:
  * Entry is at trigger.entry_price at the trigger's bar (no latency modeled).
  * Each subsequent bar's [low, high] range is scanned.
  * Stop check runs first on each bar (conservative — if both stop and
    a TP lie within the same bar's range, assume stop filled first).
  * TP tiers are filled in price order (nearest to entry first for the
    trade direction). Each tier can fill at most its `fraction` of the
    position, or the remaining fraction if less.
  * **Binary outcomes only**: each trade resolves via full TP ladder fill
    or stop hit. No timeout close. If neither has fired by the end of the
    available candles, the trade is marked "unresolved" and contributes
    zero realized PnL — it's flagged so the report can count them
    separately, but not folded into capture stats.
  * No trailing stop migration in v1 — the simulator tracks the same
    initial stop throughout. Chandelier / leg-based trails are position-
    manager logic (Phase D); validation here measures the bare-bones
    trigger + static-stop + ladder combo.

capture_ratio = realized_return_pct / pop.magnitude.
  * 1.0 → we banked the whole pop
  * 0.5 → we banked half of it (left some meat on the bone)
  * 0.0 → break-even (cost-less stop-out)
  * negative → we lost money despite being on the right side
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from setups.base import Trigger
from validation.labeler import PopEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TierFill:
    """A single partial close record."""
    level: str       # "stop" | "target" | "timeout"
    price: float
    fraction: float
    bar_index: int


@dataclass(frozen=True)
class CaptureResult:
    trigger: Trigger
    pop: PopEvent
    entry_price: float
    weighted_exit_price: float
    realized_return_pct: float    # signed by direction: +5% on longs where price rose 5%
    capture_ratio: float           # realized / pop.magnitude
    exit_reason: str               # "stop" | "target" | "timeout" | "end_of_data"
    bars_held: int
    tier_fills: tuple[TierFill, ...]


def _bar_index_at(candles: pd.DataFrame, timestamp: pd.Timestamp) -> Optional[int]:
    """Find the bar index whose timestamp matches `timestamp`. None if not found."""
    ts_col = pd.to_datetime(candles["timestamp"], utc=True).reset_index(drop=True)
    match = ts_col[ts_col == timestamp]
    if len(match) > 0:
        return int(match.index[0])
    # Nearest-within-one-bar fallback (timestamp might be at bar close, not exact)
    diffs = (ts_col - timestamp).abs()
    idx = int(diffs.idxmin())
    if diffs.iloc[idx] <= pd.Timedelta(minutes=1):
        return idx
    return None


def simulate_capture(
    trigger: Trigger,
    pop: PopEvent,
    candles: pd.DataFrame,
    *,
    max_hold_bars: Optional[int] = None,
) -> Optional[CaptureResult]:
    """
    Simulate the trade from `trigger` forward over `candles`.

    Binary outcomes: the trade resolves via stop hit or full TP ladder
    fill. If neither happens by the end of the simulation window, the
    result has exit_reason='unresolved' and zero realized PnL — not
    counted in capture stats.

    Args:
        max_hold_bars: hard cap on bars to simulate. Default None = no cap
                       (run to end of candles). Pass a number only to
                       bound worst-case simulation time.

    Returns None if the trigger's bar can't be located in `candles`.
    """
    required = {"timestamp", "open", "high", "low", "close"}
    if not required.issubset(candles.columns):
        raise ValueError(f"candles missing columns: {required - set(candles.columns)}")
    entry_idx = _bar_index_at(candles, trigger.timestamp)
    if entry_idx is None:
        logger.warning("Could not locate trigger bar at %s in candles", trigger.timestamp)
        return None

    direction = trigger.direction
    entry = trigger.entry_price
    stop = trigger.stop_price
    if direction == "long":
        tiers = sorted(trigger.tp_ladder, key=lambda tp: tp.price)
    else:
        tiers = sorted(trigger.tp_ladder, key=lambda tp: -tp.price)
    tiers_remaining = list(tiers)

    remaining = 1.0
    realized = 0.0
    fills: list[TierFill] = []
    exit_reason = "unresolved"
    last_bar_seen = entry_idx

    highs = candles["high"].to_numpy(dtype=float)
    lows = candles["low"].to_numpy(dtype=float)

    end_idx = len(candles) if max_hold_bars is None else min(
        entry_idx + 1 + max_hold_bars, len(candles)
    )

    for i in range(entry_idx + 1, end_idx):
        last_bar_seen = i
        high = highs[i]
        low = lows[i]

        # Stop check first (conservative)
        if direction == "long" and low <= stop:
            frac = remaining
            pnl = (stop - entry) / entry * frac
            realized += pnl
            fills.append(TierFill("stop", stop, frac, i))
            remaining = 0.0
            exit_reason = "stop"
            break
        if direction == "short" and high >= stop:
            frac = remaining
            pnl = (entry - stop) / entry * frac
            realized += pnl
            fills.append(TierFill("stop", stop, frac, i))
            remaining = 0.0
            exit_reason = "stop"
            break

        # TP ladder in direction order; may fill multiple in one bar
        while tiers_remaining and remaining > 0:
            tp = tiers_remaining[0]
            if direction == "long" and high >= tp.price:
                frac = min(tp.fraction, remaining)
                pnl = (tp.price - entry) / entry * frac
                realized += pnl
                fills.append(TierFill("target", tp.price, frac, i))
                remaining -= frac
                tiers_remaining.pop(0)
            elif direction == "short" and low <= tp.price:
                frac = min(tp.fraction, remaining)
                pnl = (entry - tp.price) / entry * frac
                realized += pnl
                fills.append(TierFill("target", tp.price, frac, i))
                remaining -= frac
                tiers_remaining.pop(0)
            else:
                break

        if remaining <= 1e-9:
            exit_reason = "target"
            break

    # If neither stop nor full TP ladder resolved by end of data: unresolved.
    # Do NOT force-close at last bar — trade is still open.
    # Realized PnL is whatever partial TPs hit (if any); remaining contributes 0.
    total_fraction = sum(f.fraction for f in fills)
    if total_fraction > 0:
        weighted_exit = sum(f.price * f.fraction for f in fills) / total_fraction
    else:
        weighted_exit = entry

    capture_ratio = realized / pop.magnitude if pop.magnitude > 0 else 0.0

    return CaptureResult(
        trigger=trigger,
        pop=pop,
        entry_price=entry,
        weighted_exit_price=weighted_exit,
        realized_return_pct=realized,
        capture_ratio=capture_ratio,
        exit_reason=exit_reason,
        bars_held=last_bar_seen - entry_idx,
        tier_fills=tuple(fills),
    )


def median_capture_ratio(results: list[CaptureResult]) -> float:
    if not results:
        return 0.0
    ratios = sorted(r.capture_ratio for r in results)
    n = len(ratios)
    if n % 2 == 1:
        return ratios[n // 2]
    return (ratios[n // 2 - 1] + ratios[n // 2]) / 2


def mean_realized_return(results: list[CaptureResult]) -> float:
    if not results:
        return 0.0
    return sum(r.realized_return_pct for r in results) / len(results)


def exit_reason_breakdown(results: list[CaptureResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.exit_reason] = counts.get(r.exit_reason, 0) + 1
    return counts

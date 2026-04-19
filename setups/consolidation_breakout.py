"""
setups/consolidation_breakout.py
Consolidation-breakout setup (architecture §4.3, "diagonal breakout").

Fires when:

  1. The last N bars *before* the current bar form a tight range.
     Range = (max(high) - min(low)) / mean(close) over that window,
     must be ≤ max_range_pct (default 3%).
  2. The current bar is the **breakout bar**: its close is decisively
     above the consolidation high (long) or below the consolidation
     low (short). "Decisively" = beyond the boundary by at least
     `breakout_buffer_pct` (default 10 bps).
  3. **Close-based confirmation**: the last `required_confirm_closes`
     closes (default 2) are all on the breakout side. Matches the
     locked-in rule from §5.3 ("a wick through a level isn't a
     breakout — only a close is").
  4. **Fresh breakout guard**: the bar immediately before the breakout
     must still be inside the consolidation range. Prevents firing
     three bars into a trend.

Design notes:

  * HTF trend filter from §4.3 of the architecture doc is **skipped in
    v1**. The validation scanner doesn't populate `ctx.candles_htf`;
    wiring HTF alignment is a follow-on task. Divergence operates the
    same way today (no HTF gate). Accepting this now so we can measure
    the detector's raw signal.

  * Wick-based stop placement, per the handoff-locked rule. Long stop
    sits just below the min-low of the consolidation (wick, not close)
    with a small buffer; short stop sits just above the max-high.

  * `min_target_distance_pct` filter (default 10%) reuses the
    "look-left for meaningful TP" CBS rule, same as divergence.
    If no TP on the ladder is far enough from entry, reject.

  * Stateless: the scanner handles dedup across repeated bars that
    satisfy the same underlying setup.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from setups.base import (
    Direction,
    MarketContext,
    Trigger,
    build_tp_ladder,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _consolidation_range(
    candles: pd.DataFrame,
    window_bars: int,
    confirm_closes: int = 1,
) -> Optional[tuple[float, float, float]]:
    """Return (low, high, mean_close) over `window_bars` bars BEFORE the
    confirmation sequence, or None if we don't have enough history.

    The last `confirm_closes` bars (the breakout-confirmation sequence)
    are explicitly excluded — they're the breakout itself and would
    expand the range if included.
    """
    if len(candles) < window_bars + confirm_closes:
        return None
    end = len(candles) - confirm_closes
    window = candles.iloc[end - window_bars : end]
    low = float(window["low"].min())
    high = float(window["high"].max())
    mean_close = float(window["close"].mean())
    if mean_close <= 0:
        return None
    return low, high, mean_close


def _is_breakout_close(
    current_close: float,
    range_low: float,
    range_high: float,
    direction: Direction,
    breakout_buffer_pct: float,
) -> bool:
    """True if `current_close` is decisively through the range boundary
    in the trade direction."""
    if direction == "long":
        threshold = range_high * (1 + breakout_buffer_pct)
        return current_close > threshold
    threshold = range_low * (1 - breakout_buffer_pct)
    return current_close < threshold


def _confirmation_closes_ok(
    closes: pd.Series,
    range_low: float,
    range_high: float,
    direction: Direction,
    required_closes: int,
) -> bool:
    """All of the last `required_closes` closes must sit beyond the
    relevant consolidation boundary on the trade side.

    Required_closes=1 is effectively just the current close check; the
    usual default is 2, so a single wick-driven close alone won't fire."""
    if required_closes < 1 or len(closes) < required_closes:
        return False
    recent = closes.iloc[-required_closes:]
    if direction == "long":
        return bool((recent > range_high).all())
    return bool((recent < range_low).all())


def _prior_bar_inside_range(
    candles: pd.DataFrame,
    range_low: float,
    range_high: float,
    required_confirm_closes: int,
) -> bool:
    """Fresh-breakout guard: the bar BEFORE the breakout sequence must
    still have closed within the consolidation range.

    With N required confirmation closes, the breakout sequence occupies
    candles[-N:]; the bar that preceded it is candles[-(N+1)]. That bar
    must be inside the range for the breakout to count as fresh.
    Prevents firing on already-extended breakouts (e.g. several bars
    past the range break), which are chasing rather than setup-driven.
    """
    offset = required_confirm_closes + 1
    if len(candles) < offset:
        return False
    prior_close = float(candles["close"].iloc[-offset])
    return range_low <= prior_close <= range_high


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class ConsolidationBreakoutDetector:
    """Phase C's first detector — architecture §4.3 diagonal breakout."""

    name = "consolidation_breakout"

    def __init__(
        self,
        *,
        window_bars: int = 20,
        max_range_pct: float = 0.03,
        breakout_buffer_pct: float = 0.001,
        required_confirm_closes: int = 2,
        stop_buffer_pct: float = 0.003,
        base_confidence: float = 0.65,
        size_fraction: float = 0.5,
        atr_period: int = 14,
        min_target_distance_pct: float = 0.10,
    ) -> None:
        self.window_bars = window_bars
        self.max_range_pct = max_range_pct
        self.breakout_buffer_pct = breakout_buffer_pct
        self.required_confirm_closes = required_confirm_closes
        self.stop_buffer_pct = stop_buffer_pct
        self.base_confidence = base_confidence
        self.size_fraction = size_fraction
        self.atr_period = atr_period
        self.min_target_distance_pct = min_target_distance_pct

    def detect(self, ctx: MarketContext) -> Optional[Trigger]:
        candles = ctx.candles
        if len(candles) < self.window_bars + max(self.required_confirm_closes, 2):
            return None

        rng = _consolidation_range(
            candles, self.window_bars,
            confirm_closes=self.required_confirm_closes,
        )
        if rng is None:
            return None
        range_low, range_high, mean_close = rng

        # Tight-range check
        range_ratio = (range_high - range_low) / mean_close
        if range_ratio > self.max_range_pct:
            return None

        # Fresh-breakout guard
        if not _prior_bar_inside_range(
            candles, range_low, range_high, self.required_confirm_closes,
        ):
            return None

        current_close = float(candles["close"].iloc[-1])

        for direction in ("long", "short"):
            if not _is_breakout_close(
                current_close, range_low, range_high,
                direction, self.breakout_buffer_pct,  # type: ignore[arg-type]
            ):
                continue
            if not _confirmation_closes_ok(
                candles["close"].astype(float), range_low, range_high,
                direction, self.required_confirm_closes,  # type: ignore[arg-type]
            ):
                continue

            trig = self._build_trigger(ctx, direction, range_low, range_high)  # type: ignore[arg-type]
            if trig is not None:
                return trig

        return None

    # -- helpers --------------------------------------------------------------

    def _build_trigger(
        self,
        ctx: MarketContext,
        direction: Direction,
        range_low: float,
        range_high: float,
    ) -> Optional[Trigger]:
        entry = ctx.current_price

        # Wick-based stop
        if direction == "long":
            stop = range_low * (1 - self.stop_buffer_pct)
        else:
            stop = range_high * (1 + self.stop_buffer_pct)

        atr = self._atr(ctx.candles)
        tp_ladder = build_tp_ladder(
            entry=entry, direction=direction,
            sr_zones=ctx.sr_zones, atr=atr,
        )

        # "Look left" filter: reject if no meaningful runway
        if tp_ladder:
            max_distance = max(abs(tp.price - entry) / entry for tp in tp_ladder)
            if max_distance < self.min_target_distance_pct:
                return None
        else:
            return None

        return Trigger(
            ticker=ctx.ticker,
            timestamp=ctx.timestamp or pd.Timestamp.now(tz="UTC"),
            action="open_new",
            direction=direction,
            entry_price=entry,
            stop_price=stop,
            tp_ladder=tp_ladder,
            setup=self.name,
            confidence=self.base_confidence,
            size_fraction=self.size_fraction,
            components={
                "range_low": range_low,
                "range_high": range_high,
                "range_ratio": (range_high - range_low) / entry if entry > 0 else 0.0,
                "window_bars": self.window_bars,
                "required_confirm_closes": self.required_confirm_closes,
            },
        )

    def _atr(self, candles: pd.DataFrame) -> float:
        if len(candles) < self.atr_period + 1:
            return 0.0
        high = candles["high"].astype(float).to_numpy()
        low = candles["low"].astype(float).to_numpy()
        close = candles["close"].astype(float).to_numpy()
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        if len(tr) < self.atr_period:
            return float(np.mean(tr))
        atr = float(np.mean(tr[: self.atr_period]))
        for v in tr[self.atr_period :]:
            atr = (atr * (self.atr_period - 1) + v) / self.atr_period
        return atr

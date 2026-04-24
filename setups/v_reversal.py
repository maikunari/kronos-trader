"""
setups/v_reversal.py
V-reversal-at-extended-move setup (architecture §4.4).

Fires when:

  1. **Extended prior move.** Price has dropped ≥ `min_move_pct`
     (default 10% long / 15% short per arch §4.4) from a local
     extreme to a subsequent V-extreme over the last `lookback_bars`.
  2. **Bounce off the V-extreme.** After the V-bottom (for longs),
     price made a local high meaningfully above the V-low.
  3. **Higher-low confirmation.** Between the bounce peak and the
     current bar, price pulled back to a low that's STRICTLY above
     the V-low. The HL is the "confirmed V-structure" the architecture
     describes.
  4. **Fresh entry.** The HL is recent (within `max_bars_since_hl`)
     and the current close is above it, i.e. momentum has turned back
     up off the HL.

Deferred from v1 (noted in architecture §4.4 pseudocode):

  * Capitulation gate (wide-range bar + above-median volume on the
    V-extreme bar). Adding this as a precision filter is a follow-up;
    skipping now to measure the structural signal first.

  * HTF trend filter — same accepted-limitation as divergence and
    consolidation_breakout. Our validation scanner doesn't populate
    `ctx.candles_htf`.

Stop placement: WICK of the V-extreme bar minus buffer, per the
handoff-locked rule (commit b8d09e3). Takeouts of the V-low by wicks
are normal; a CLOSE through the V-low kills the setup.

Stateless — the scanner handles dedup.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
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
# V-structure helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VStructure:
    """Geometry of a detected V-reversal."""
    direction: Direction
    prior_extreme_idx: int      # pre-move high (long) or low (short)
    v_idx: int                  # V-bottom (long) or V-top (short)
    bounce_peak_idx: int        # local extreme after V
    hl_idx: int                 # higher-low (long) or lower-high (short)
    v_extreme_close: float
    v_extreme_wick: float       # low (long) or high (short)
    hl_extreme_close: float
    move_magnitude: float       # fraction, positive


def _extended_move_long(
    closes: pd.Series,
    lookback_bars: int,
    min_move_pct: float,
) -> Optional[tuple[int, int, float]]:
    """Return (prior_high_idx, v_low_idx, magnitude) if a decline of
    at least `min_move_pct` exists in the lookback window.

    The V-low is the lowest close; the prior high is the max close
    BEFORE that low (within the window).
    """
    n = len(closes)
    if n < lookback_bars:
        return None
    window = closes.iloc[-lookback_bars:]
    v_idx_local = int(np.argmin(window.to_numpy()))
    if v_idx_local == 0:
        return None
    pre = window.iloc[:v_idx_local]
    if pre.empty:
        return None
    prior_high_local = int(np.argmax(pre.to_numpy()))
    v_low = float(window.iloc[v_idx_local])
    prior_high = float(window.iloc[prior_high_local])
    if prior_high <= 0:
        return None
    magnitude = (prior_high - v_low) / prior_high
    if magnitude < min_move_pct:
        return None
    offset = n - lookback_bars
    return offset + prior_high_local, offset + v_idx_local, magnitude


def _extended_move_short(
    closes: pd.Series,
    lookback_bars: int,
    min_move_pct: float,
) -> Optional[tuple[int, int, float]]:
    """Inverse: symmetric structure for a rally followed by topping."""
    n = len(closes)
    if n < lookback_bars:
        return None
    window = closes.iloc[-lookback_bars:]
    v_idx_local = int(np.argmax(window.to_numpy()))
    if v_idx_local == 0:
        return None
    pre = window.iloc[:v_idx_local]
    if pre.empty:
        return None
    prior_low_local = int(np.argmin(pre.to_numpy()))
    v_high = float(window.iloc[v_idx_local])
    prior_low = float(window.iloc[prior_low_local])
    if prior_low <= 0:
        return None
    magnitude = (v_high - prior_low) / prior_low
    if magnitude < min_move_pct:
        return None
    offset = n - lookback_bars
    return offset + prior_low_local, offset + v_idx_local, magnitude


def _find_v_structure(
    candles: pd.DataFrame,
    direction: Direction,
    lookback_bars: int,
    min_move_pct: float,
    min_bounce_pct: float,
    max_bars_since_hl: int,
) -> Optional[VStructure]:
    """Search for a valid V (long) or Λ (short) structure ending near
    the current bar."""
    closes = candles["close"].astype(float)
    lows = candles["low"].astype(float)
    highs = candles["high"].astype(float)

    if direction == "long":
        ext = _extended_move_long(closes, lookback_bars, min_move_pct)
    else:
        ext = _extended_move_short(closes, lookback_bars, min_move_pct)
    if ext is None:
        return None
    prior_idx, v_idx, magnitude = ext

    n = len(candles)
    # Need at least a few bars after the V for bounce + HL + entry
    if v_idx >= n - 3:
        return None

    v_close = float(closes.iloc[v_idx])
    v_wick = float(lows.iloc[v_idx]) if direction == "long" else float(highs.iloc[v_idx])

    # Bounce peak: extreme close strictly after v_idx
    post_v = closes.iloc[v_idx + 1 :].to_numpy()
    if direction == "long":
        bp_rel = int(np.argmax(post_v))
    else:
        bp_rel = int(np.argmin(post_v))
    bp_idx = v_idx + 1 + bp_rel
    bp_close = float(closes.iloc[bp_idx])

    # Minimum bounce magnitude
    bounce_pct = (
        (bp_close - v_close) / abs(v_close) if direction == "long"
        else (v_close - bp_close) / abs(v_close)
    )
    if bounce_pct < min_bounce_pct:
        return None

    # HL (long) / LH (short): extreme close after the bounce peak
    if bp_idx >= n - 1:
        return None
    post_bp = closes.iloc[bp_idx + 1 :].to_numpy()
    if direction == "long":
        hl_rel = int(np.argmin(post_bp))
    else:
        hl_rel = int(np.argmax(post_bp))
    hl_idx = bp_idx + 1 + hl_rel
    hl_close = float(closes.iloc[hl_idx])

    # HL must be STRICTLY above V (long) or below V (short)
    if direction == "long":
        if not (hl_close > v_close):
            return None
    else:
        if not (hl_close < v_close):
            return None

    # Freshness: HL must be recent
    bars_since_hl = (n - 1) - hl_idx
    if bars_since_hl > max_bars_since_hl:
        return None

    # Current bar must continue in the trade direction from the HL
    current_close = float(closes.iloc[-1])
    if direction == "long":
        if current_close < hl_close:
            return None
    else:
        if current_close > hl_close:
            return None

    return VStructure(
        direction=direction,
        prior_extreme_idx=prior_idx,
        v_idx=v_idx,
        bounce_peak_idx=bp_idx,
        hl_idx=hl_idx,
        v_extreme_close=v_close,
        v_extreme_wick=v_wick,
        hl_extreme_close=hl_close,
        move_magnitude=magnitude,
    )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class VReversalDetector:
    """Phase C — architecture §4.4 V-reversal at extended move."""

    name = "v_reversal"

    def __init__(
        self,
        *,
        lookback_bars: int = 120,
        min_move_pct_long: float = 0.15,      # tuned 2026-04-20 sweep: 10% -> 15%
        min_move_pct_short: float = 0.20,     # tuned proportionally
        min_bounce_pct: float = 0.05,         # tuned: 3% -> 5%
        max_bars_since_hl: int = 2,           # tuned: 5 -> 2 (fresher only)
        stop_buffer_pct: float = 0.003,
        base_confidence: float = 0.70,
        size_fraction: float = 0.4,
        atr_period: int = 14,
        min_target_distance_pct: float = 0.10,
    ) -> None:
        self.lookback_bars = lookback_bars
        self.min_move_pct_long = min_move_pct_long
        self.min_move_pct_short = min_move_pct_short
        self.min_bounce_pct = min_bounce_pct
        self.max_bars_since_hl = max_bars_since_hl
        self.stop_buffer_pct = stop_buffer_pct
        self.base_confidence = base_confidence
        self.size_fraction = size_fraction
        self.atr_period = atr_period
        self.min_target_distance_pct = min_target_distance_pct

    def detect(self, ctx: MarketContext) -> Optional[Trigger]:
        if len(ctx.candles) < self.lookback_bars + 5:
            return None

        for direction in ("long", "short"):
            min_move = (
                self.min_move_pct_long if direction == "long"
                else self.min_move_pct_short
            )
            v = _find_v_structure(
                ctx.candles, direction,  # type: ignore[arg-type]
                self.lookback_bars, min_move, self.min_bounce_pct,
                self.max_bars_since_hl,
            )
            if v is None:
                continue
            trig = self._build_trigger(ctx, v)
            if trig is not None:
                return trig
        return None

    # -- helpers --------------------------------------------------------------

    def _build_trigger(
        self,
        ctx: MarketContext,
        v: VStructure,
    ) -> Optional[Trigger]:
        entry = ctx.current_price
        if v.direction == "long":
            stop = v.v_extreme_wick * (1 - self.stop_buffer_pct)
        else:
            stop = v.v_extreme_wick * (1 + self.stop_buffer_pct)

        atr = self._atr(ctx.candles)
        tp_ladder = build_tp_ladder(
            entry=entry, direction=v.direction,
            sr_zones=ctx.sr_zones, atr=atr,
        )
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
            direction=v.direction,
            entry_price=entry,
            stop_price=stop,
            tp_ladder=tp_ladder,
            setup=self.name,
            confidence=self.base_confidence,
            size_fraction=self.size_fraction,
            components={
                "move_magnitude": v.move_magnitude,
                "v_extreme_close": v.v_extreme_close,
                "v_extreme_wick": v.v_extreme_wick,
                "hl_close": v.hl_extreme_close,
                "prior_extreme_idx": v.prior_extreme_idx,
                "v_idx": v.v_idx,
                "bounce_peak_idx": v.bounce_peak_idx,
                "hl_idx": v.hl_idx,
                "bars_since_hl": (len(ctx.candles) - 1) - v.hl_idx,
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

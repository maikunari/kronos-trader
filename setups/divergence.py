"""
setups/divergence.py
Divergence reversal detector.

Detects the pattern CBS shows in his RSI Masterclass screenshots:
  1. Price makes a new pivot extreme (lower low for bullish, higher high for bearish)
  2. AO or RSI (or both — 'double divergence') makes a less-extreme pivot
     on the same bar (i.e., indicator diverges from price)
  3. The price pivot is at a price extreme (below lower BB, above upper BB,
     or equivalent distance-from-mean metric)
  4. Two consecutive AO bars in the reversal direction confirm entry
     ("two bars of a certain colour are very hard to reverse from")
  5. Triple divergence (3 successive divergent pivots) boosts confidence

Design notes:

  * We evaluate AO and RSI *at the price pivot bars*, not as independent
    pivot series. This matches how traders read divergences visually:
    "price made a new low but RSI at that low is higher than at the
    previous low." Aligning separate pivot sets would be fiddly and
    introduce mismatch edge cases.

  * Stop is placed just beyond the divergent pivot (recent low for
    longs, recent high for shorts). That's the invalidation per CBS
    — if price breaks the divergent extreme, the setup is dead.

  * TP ladder comes from S/R zones in the direction of the trade, with
    an ATR-based fallback if the zone list is empty.

  * Stateless: `detect(ctx)` returns a trigger or None for the current
    bar. The validation framework / scanner handles dedup (don't fire
    repeatedly on the same setup).

  * Max age: a divergence stops being actionable if price has already
    run too far from the pivot. `max_bars_since_pivot` (default 8 bars)
    limits how late we can still enter.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from indicators.awesome_oscillator import two_bar_same_color
from pivot import Pivot, find_pivots
from setups.base import (
    Direction,
    MarketContext,
    TPLevel,
    Trigger,
    build_tp_ladder,
)
from support_resistance import SRZone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extreme-price filter
# ---------------------------------------------------------------------------

def _is_at_extreme(
    candles: pd.DataFrame,
    bar_index: int,
    direction: Direction,
    bb_period: int = 20,
    bb_std: float = 2.0,
) -> bool:
    """True if the close at `bar_index` was outside the Bollinger band.

    Long setups require the bar at bar_index to have closed BELOW the
    lower band (oversold extreme). Short setups require a close ABOVE
    the upper band.
    """
    if bar_index < bb_period:
        return False
    closes = candles["close"].astype(float)
    window = closes.iloc[bar_index - bb_period + 1 : bar_index + 1]
    mean = float(window.mean())
    std = float(window.std(ddof=1))
    if std <= 0:
        return False
    px = float(closes.iloc[bar_index])
    if direction == "long":
        return px < mean - bb_std * std
    return px > mean + bb_std * std


# ---------------------------------------------------------------------------
# Divergence logic
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DivergenceFinding:
    """A detected divergence between two price pivots."""
    direction: Direction
    recent_pivot: Pivot
    prior_pivot: Pivot
    ao_diverged: bool
    rsi_diverged: bool
    is_triple: bool = False

    @property
    def both_diverged(self) -> bool:
        return self.ao_diverged and self.rsi_diverged


def _detect_divergence(
    price_pivots: list[Pivot],
    ao: pd.Series,
    rsi: pd.Series,
    direction: Direction,
) -> Optional[DivergenceFinding]:
    """
    Check the last two same-kind pivots for a divergence.

    Bullish: most recent LOW is lower than previous LOW, but the indicator
             at the recent low is higher than at the previous low.
    Bearish: symmetric on highs.

    Returns None if fewer than 2 same-kind pivots exist.
    """
    want_kind = "low" if direction == "long" else "high"
    same_kind = [p for p in price_pivots if p.kind == want_kind]
    if len(same_kind) < 2:
        return None

    prior = same_kind[-2]
    recent = same_kind[-1]

    # Price-divergence precondition
    if direction == "long":
        if not (recent.value < prior.value):
            return None
    else:
        if not (recent.value > prior.value):
            return None

    ao_prior = float(ao.iloc[prior.index]) if prior.index < len(ao) else float("nan")
    ao_recent = float(ao.iloc[recent.index]) if recent.index < len(ao) else float("nan")
    rsi_prior = float(rsi.iloc[prior.index]) if prior.index < len(rsi) else float("nan")
    rsi_recent = float(rsi.iloc[recent.index]) if recent.index < len(rsi) else float("nan")

    if direction == "long":
        ao_diverged = not np.isnan(ao_prior) and not np.isnan(ao_recent) and ao_recent > ao_prior
        rsi_diverged = not np.isnan(rsi_prior) and not np.isnan(rsi_recent) and rsi_recent > rsi_prior
    else:
        ao_diverged = not np.isnan(ao_prior) and not np.isnan(ao_recent) and ao_recent < ao_prior
        rsi_diverged = not np.isnan(rsi_prior) and not np.isnan(rsi_recent) and rsi_recent < rsi_prior

    if not (ao_diverged or rsi_diverged):
        return None

    # Triple divergence: check 3rd-most-recent pivot as well
    is_triple = False
    if len(same_kind) >= 3:
        oldest = same_kind[-3]
        ao_oldest = float(ao.iloc[oldest.index]) if oldest.index < len(ao) else float("nan")
        rsi_oldest = float(rsi.iloc[oldest.index]) if oldest.index < len(rsi) else float("nan")
        if direction == "long":
            price_chain = recent.value < prior.value < oldest.value
            ind_chain = (
                (not np.isnan(ao_oldest) and ao_prior > ao_oldest and ao_recent > ao_prior)
                or (not np.isnan(rsi_oldest) and rsi_prior > rsi_oldest and rsi_recent > rsi_prior)
            )
        else:
            price_chain = recent.value > prior.value > oldest.value
            ind_chain = (
                (not np.isnan(ao_oldest) and ao_prior < ao_oldest and ao_recent < ao_prior)
                or (not np.isnan(rsi_oldest) and rsi_prior < rsi_oldest and rsi_recent < rsi_prior)
            )
        is_triple = bool(price_chain and ind_chain)

    return DivergenceFinding(
        direction=direction, recent_pivot=recent, prior_pivot=prior,
        ao_diverged=ao_diverged, rsi_diverged=rsi_diverged, is_triple=is_triple,
    )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class DivergenceReversalDetector:
    """Phase B's first setup detector."""

    name = "divergence_reversal"

    def __init__(
        self,
        *,
        pivot_window: int = 5,
        bb_period: int = 20,
        bb_std: float = 2.0,
        max_bars_since_pivot: int = 8,
        require_both_indicators: bool = False,
        triple_confidence_boost: float = 1.5,
        base_confidence: float = 0.55,
        atr_period: int = 14,
        min_target_distance_pct: float = 0.10,
    ) -> None:
        self.pivot_window = pivot_window
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.max_bars_since_pivot = max_bars_since_pivot
        self.require_both_indicators = require_both_indicators
        self.triple_confidence_boost = triple_confidence_boost
        self.base_confidence = base_confidence
        self.atr_period = atr_period
        # Per CBS "look left for TP": only fire if there's at least one
        # structural target this far from entry. Rejects setups with no
        # meaningful runway.
        self.min_target_distance_pct = min_target_distance_pct

    def detect(self, ctx: MarketContext) -> Optional[Trigger]:
        if ctx.ao is None or ctx.rsi is None:
            return None
        closes = ctx.candles["close"].astype(float)
        if len(closes) < max(self.bb_period, self.pivot_window * 3) + 2:
            return None

        pivots = find_pivots(closes, window=self.pivot_window)
        if not pivots:
            return None

        # Evaluate both directions; return the first valid trigger we find.
        for direction in ("long", "short"):
            finding = _detect_divergence(pivots, ctx.ao, ctx.rsi, direction)  # type: ignore[arg-type]
            if finding is None:
                continue

            # Require both indicators to diverge if configured
            if self.require_both_indicators and not finding.both_diverged:
                continue

            # Filter: divergent pivot must be recent
            bars_since = ctx.current_bar_index - finding.recent_pivot.index
            if bars_since > self.max_bars_since_pivot:
                continue

            # Filter: must be at price extreme on the pivot bar
            if not _is_at_extreme(
                ctx.candles, finding.recent_pivot.index, direction,
                bb_period=self.bb_period, bb_std=self.bb_std,
            ):
                continue

            # Filter: two-bar AO confirmation in reversal direction
            needed_color = "green" if direction == "long" else "red"
            if not two_bar_same_color(ctx.ao, needed_color):
                continue

            trig = self._build_trigger(ctx, finding, direction)  # type: ignore[arg-type]
            if trig is not None:
                return trig
            # else: target-distance filter rejected; try the other direction

        return None

    # --- helpers -------------------------------------------------------------

    def _build_trigger(
        self,
        ctx: MarketContext,
        finding: DivergenceFinding,
        direction: Direction,
    ) -> Optional[Trigger]:
        entry = ctx.current_price

        # Stop placement uses the WICK of the divergent pivot bar, not its
        # close. Pivots are detected on the close series (per CBS "draw
        # levels at closes"), but the stop needs to sit below the actual
        # price the bar reached, otherwise normal retests wick us out.
        # The recent_pivot.index points to the pivot bar; we look up its
        # actual low/high and place the stop beyond it with a small buffer.
        pivot_idx = finding.recent_pivot.index
        buffer = 0.003   # 30 bps buffer below/above the wick
        if direction == "long":
            pivot_wick = float(ctx.candles["low"].iloc[pivot_idx])
            stop = pivot_wick * (1 - buffer)
        else:
            pivot_wick = float(ctx.candles["high"].iloc[pivot_idx])
            stop = pivot_wick * (1 + buffer)

        atr = self._atr(ctx.candles)
        tp_ladder = build_tp_ladder(
            entry=entry, direction=direction,
            sr_zones=ctx.sr_zones, atr=atr,
        )

        # "Look left" filter: reject setups with no meaningful target runway.
        # Require at least one TP to be `min_target_distance_pct` away from
        # entry in the trade direction. Prevents grabbing near-field noise
        # zones as TPs when there's no real structural target far above/below.
        if tp_ladder:
            max_distance = max(abs(tp.price - entry) / entry for tp in tp_ladder)
            if max_distance < self.min_target_distance_pct:
                return None
        else:
            # No ladder at all (no zones AND no ATR) — not tradeable
            return None

        # Confidence: base × (double-div bonus) × (triple bonus)
        confidence = self.base_confidence
        if finding.both_diverged:
            confidence = min(confidence * 1.3, 1.0)
        if finding.is_triple:
            confidence = min(confidence * self.triple_confidence_boost, 1.0)

        return Trigger(
            ticker=ctx.ticker,
            timestamp=ctx.timestamp or pd.Timestamp.now(tz="UTC"),
            action="open_new",
            direction=direction,
            entry_price=entry,
            stop_price=stop,
            tp_ladder=tp_ladder,
            setup=self.name,
            confidence=confidence,
            size_fraction=0.4,   # per architecture §6.2 divergence initial fraction
            components={
                "ao_diverged": finding.ao_diverged,
                "rsi_diverged": finding.rsi_diverged,
                "triple": finding.is_triple,
                "bars_since_pivot": ctx.current_bar_index - finding.recent_pivot.index,
                "divergent_price_close": finding.recent_pivot.value,
                "divergent_wick": pivot_wick,
                "prior_price": finding.prior_pivot.value,
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
        # Wilder's smoothing
        atr = float(np.mean(tr[: self.atr_period]))
        for v in tr[self.atr_period :]:
            atr = (atr * (self.atr_period - 1) + v) / self.atr_period
        return atr

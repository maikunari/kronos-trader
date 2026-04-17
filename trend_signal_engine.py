"""
trend_signal_engine.py
Simple EMA crossover trend-following signal engine with ATR-based stops.

STRATEGY:
  - Enter on EMA fast/slow crossover in the direction of the D1 trend
  - Stop: 1.5 * ATR(14) from entry  → exact 2:1 R:R with 3.0 * ATR target
  - Filter: D1 EMA(20/50) must agree with signal direction

PARAMETERS (all tunable):
  ema_fast:   default 9
  ema_slow:   default 21
  atr_period: default 14
  atr_stop_mult:   1.5  → stop distance = 1.5 * ATR
  atr_target_mult: 3.0  → target distance = 3.0 * ATR  (always 2:1)
  confirm_bars:    0    → how many bars signal must persist before entry
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    action: str           # "long" | "short" | "flat"
    entry_price: float
    stop_price: float
    target_price: float
    rr_ratio: float       # should always be ~2.0
    atr: float            # ATR value at signal time
    ema_fast: float
    ema_slow: float
    mtf_bias: str
    confidence: float
    skip_reason: str = ""


class TrendSignalEngine:
    """
    EMA crossover trend-following signal engine.

    Detects when EMA(fast) crosses EMA(slow) in the direction of the D1 trend,
    then enters with ATR-based stop/target for exact 2:1 R:R.
    """

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        atr_period: int = 14,
        atr_stop_mult: float = 1.5,
        atr_target_mult: float = 3.0,
        mtf_filter=None,
        require_mtf: bool = True,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.mtf_filter = mtf_filter
        self.require_mtf = require_mtf  # if False, trade without D1 filter

        self._rr = atr_target_mult / atr_stop_mult
        logger.info(
            f"TrendSignalEngine: EMA({ema_fast}/{ema_slow}), "
            f"ATR({atr_period}) stop={atr_stop_mult}x target={atr_target_mult}x "
            f"→ {self._rr:.1f}:1 R:R | MTF={'on' if mtf_filter else 'off'}"
        )

    def evaluate(
        self,
        candles: pd.DataFrame,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> TradeSignal:
        """
        Evaluate current bar for a crossover signal.

        candles: DataFrame with at least [open, high, low, close], oldest → newest.
        Needs at least ema_slow + atr_period + 2 bars.
        """
        min_bars = self.ema_slow + self.atr_period + 2
        if len(candles) < min_bars:
            return self._flat(0.0, 0.0, 0.0, "disabled", f"insufficient_bars({len(candles)}<{min_bars})")

        closes = candles["close"].values.astype(float)
        highs = candles["high"].values.astype(float)
        lows = candles["low"].values.astype(float)
        current_close = closes[-1]

        if timestamp is None and "timestamp" in candles.columns:
            timestamp = candles["timestamp"].iloc[-1]

        # --- EMAs ---
        ema_f_now = _ema(closes, self.ema_fast)
        ema_s_now = _ema(closes, self.ema_slow)
        ema_f_prev = _ema(closes[:-1], self.ema_fast)
        ema_s_prev = _ema(closes[:-1], self.ema_slow)

        # --- ATR ---
        atr = _atr(highs, lows, closes, self.atr_period)
        if atr <= 0:
            return self._flat(ema_f_now, ema_s_now, atr, "disabled", "atr_zero")

        # --- MTF bias ---
        mtf_bias = "disabled"
        if self.mtf_filter is not None and timestamp is not None:
            mtf_sig = self.mtf_filter.get_bias_at(timestamp)
            mtf_bias = mtf_sig.bias

        # --- Crossover detection ---
        crossed_up = (ema_f_prev <= ema_s_prev) and (ema_f_now > ema_s_now)
        crossed_down = (ema_f_prev >= ema_s_prev) and (ema_f_now < ema_s_now)

        if not crossed_up and not crossed_down:
            return self._flat(ema_f_now, ema_s_now, atr, mtf_bias, "no_crossover")

        raw_direction = "long" if crossed_up else "short"

        # --- MTF filter ---
        if self.require_mtf and mtf_bias not in ("disabled", "neutral"):
            if mtf_bias != raw_direction:
                return self._flat(
                    ema_f_now, ema_s_now, atr, mtf_bias,
                    f"mtf_conflict(signal={raw_direction},d1={mtf_bias})"
                )

        action = raw_direction
        stop_dist = self.atr_stop_mult * atr
        target_dist = self.atr_target_mult * atr

        if action == "long":
            stop_price = current_close - stop_dist
            target_price = current_close + target_dist
        else:
            stop_price = current_close + stop_dist
            target_price = current_close - target_dist

        rr = target_dist / stop_dist  # should be atr_target_mult / atr_stop_mult

        # Confidence: how far the EMAs have separated relative to ATR
        separation = abs(ema_f_now - ema_s_now) / atr
        confidence = min(separation / 2.0, 1.0)
        if mtf_bias == action:
            confidence = min(confidence * 1.2, 1.0)

        logger.debug(
            f"{action.upper()} crossover @ {current_close:.4f} | "
            f"EMA({self.ema_fast})={ema_f_now:.4f} EMA({self.ema_slow})={ema_s_now:.4f} | "
            f"ATR={atr:.4f} | stop={stop_price:.4f} target={target_price:.4f} | "
            f"R:R={rr:.2f} | MTF={mtf_bias}"
        )

        return TradeSignal(
            action=action,
            entry_price=current_close,
            stop_price=stop_price,
            target_price=target_price,
            rr_ratio=rr,
            atr=atr,
            ema_fast=ema_f_now,
            ema_slow=ema_s_now,
            mtf_bias=mtf_bias,
            confidence=confidence,
        )

    def _flat(self, ema_f, ema_s, atr, mtf_bias, reason) -> TradeSignal:
        return TradeSignal(
            action="flat",
            entry_price=0.0,
            stop_price=0.0,
            target_price=0.0,
            rr_ratio=self._rr,
            atr=atr,
            ema_fast=ema_f,
            ema_slow=ema_s,
            mtf_bias=mtf_bias,
            confidence=0.0,
            skip_reason=reason,
        )


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _ema(values: np.ndarray, period: int) -> float:
    """EMA of the full array, returns last value."""
    if len(values) < period:
        return float(values[-1])
    k = 2.0 / (period + 1)
    ema = float(values[0])
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
    """Wilder's ATR (RMA smoothing), returns current value."""
    n = len(close)
    if n < 2:
        return 0.0
    # True ranges
    tr = np.zeros(n - 1)
    for i in range(1, n):
        tr[i - 1] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    if len(tr) < period:
        return float(np.mean(tr))
    # Wilder's smoothing
    atr = float(np.mean(tr[:period]))
    for v in tr[period:]:
        atr = (atr * (period - 1) + v) / period
    return atr

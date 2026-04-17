"""
atr_engine.py
Pure math-based ATR Trend Following signal engine.
No ML models. Just EMA crossovers + ATR-based risk management.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ATRSignal:
    action: str               # "long" | "short" | "flat"
    entry_price: float
    stop_price: float
    target_price: float
    mtf_bias: str             # "long" | "short" | "neutral"
    confidence: float         # 0.0 → 1.0
    skip_reason: str = ""     # why signal was flat
    atr_value: float = 0.0


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average."""
    ema = np.zeros_like(values)
    alpha = 2.0 / (period + 1)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = values[i] * alpha + ema[i - 1] * (1 - alpha)
    return ema


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Average True Range."""
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    tr[0] = high[0] - low[0]
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, len(tr)):
        atr[i] = tr[i] * alpha + atr[i - 1] * (1 - alpha)
    return atr


class ATREngine:
    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        atr_period: int = 14,
        stop_multiplier: float = 1.5,
        target_multiplier: float = 3.0,
        mtf_filter=None,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr_period = atr_period
        self.stop_multiplier = stop_multiplier
        self.target_multiplier = target_multiplier
        self.mtf_filter = mtf_filter

    def evaluate(self, candles: pd.DataFrame, timestamp: Optional[pd.Timestamp] = None) -> ATRSignal:
        """
        Generate a trade signal based on EMA pullback + ATR risk.

        Args:
            candles: DataFrame with [open, high, low, close, volume], oldest → newest.
            timestamp: current bar timestamp (backtest mode).
        """
        if len(candles) < max(self.ema_fast, self.ema_slow, self.atr_period) + 5:
            return ATRSignal(
                action="flat",
                entry_price=0,
                stop_price=0,
                target_price=0,
                mtf_bias="neutral",
                confidence=0.0,
                skip_reason="insufficient_data"
            )

        closes = candles["close"].values
        highs = candles["high"].values
        lows = candles["low"].values
        current_close = float(closes[-1])
        current_high = float(highs[-1])
        current_low = float(lows[-1])

        # Calculate EMAs
        ema_f = _ema(closes, self.ema_fast)
        ema_s = _ema(closes, self.ema_slow)
        ema_f_current = ema_f[-1]
        ema_s_current = ema_s[-1]

        # Calculate ATR
        atr_vals = _atr(highs, lows, closes, self.atr_period)
        atr_current = atr_vals[-1]

        # Check MTF bias
        mtf_bias = "neutral"
        if self.mtf_filter is not None:
            ts = timestamp or (candles["timestamp"].iloc[-1] if "timestamp" in candles.columns else None)
            try:
                if ts is not None:
                    mtf_sig = self.mtf_filter.get_bias_at(ts)
                else:
                    mtf_sig = self.mtf_filter.get_bias_live()
                mtf_bias = mtf_sig.bias
            except Exception as e:
                logger.warning(f"MTF filter error: {e}")
                mtf_bias = "neutral"

        # Determine trend direction
        trend = "long" if ema_f_current > ema_s_current else "short"

        # Check for pullback + crossback entry
        prev_close = float(closes[-2]) if len(closes) > 1 else current_close

        def flat(reason: str) -> ATRSignal:
            return ATRSignal(
                action="flat",
                entry_price=0,
                stop_price=0,
                target_price=0,
                mtf_bias=mtf_bias,
                confidence=0.0,
                skip_reason=reason,
                atr_value=atr_current
            )

        # For LONG: price was below EMA20, now closes above
        if trend == "long":
            below_ema = prev_close < ema_f_current
            crosses_above = current_close > ema_f_current

            if not (below_ema and crosses_above):
                return flat("no_long_pullback_crossover")

            # MTF agreement
            if mtf_bias == "short":
                return flat("mtf_disagree_long")

            # Calculate 2:1 risk
            stop = current_close - (self.stop_multiplier * atr_current)
            target = current_close + (self.target_multiplier * atr_current)
            confidence = 0.7 if mtf_bias == "long" else 0.5

            logger.info(
                f"Signal: LONG @ {current_close:.4f} | "
                f"stop={stop:.4f} target={target:.4f} | "
                f"atr={atr_current:.4f} | MTF={mtf_bias} | confidence={confidence:.2f}"
            )

            return ATRSignal(
                action="long",
                entry_price=current_close,
                stop_price=stop,
                target_price=target,
                mtf_bias=mtf_bias,
                confidence=confidence,
                atr_value=atr_current
            )

        # For SHORT: price was above EMA20, now closes below
        else:
            above_ema = prev_close > ema_f_current
            crosses_below = current_close < ema_f_current

            if not (above_ema and crosses_below):
                return flat("no_short_pullback_crossover")

            # MTF agreement
            if mtf_bias == "long":
                return flat("mtf_disagree_short")

            # Calculate 2:1 risk
            stop = current_close + (self.stop_multiplier * atr_current)
            target = current_close - (self.target_multiplier * atr_current)
            confidence = 0.7 if mtf_bias == "short" else 0.5

            logger.info(
                f"Signal: SHORT @ {current_close:.4f} | "
                f"stop={stop:.4f} target={target:.4f} | "
                f"atr={atr_current:.4f} | MTF={mtf_bias} | confidence={confidence:.2f}"
            )

            return ATRSignal(
                action="short",
                entry_price=current_close,
                stop_price=stop,
                target_price=target,
                mtf_bias=mtf_bias,
                confidence=confidence,
                atr_value=atr_current
            )

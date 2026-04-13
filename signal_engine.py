"""
signal_engine.py
Combines Kronos + TimesFM + MTF filter into a final trade signal.

Rules:
- MTF filter must agree (4H AND 1H trend must match signal direction)
- Kronos and TimesFM must agree on direction
- |Kronos predicted move| >= min_move_threshold
- FLAT otherwise
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from kronos_model import KronosModel, KronosPrediction
from timesfm_model import TimesFMModel, TimesFMSignal

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    action: str               # "long" | "short" | "flat"
    entry_price: float
    stop_price: float
    target_price: float
    predicted_move_pct: float
    kronos_direction: str
    timesfm_direction: str
    mtf_bias: str             # "long" | "short" | "neutral" | "disabled"
    confidence: float         # 0.0 → 1.0, composite score
    skip_reason: str = ""     # why signal was flat, if applicable


class SignalEngine:
    def __init__(
        self,
        kronos: KronosModel,
        timesfm: TimesFMModel,
        min_move_threshold: float = 0.003,
        stop_pct: float = 0.002,
        target_pct: float = 0.004,
        mtf_filter=None,                    # MTFFilter instance or None
        mtf_timestamp: Optional[pd.Timestamp] = None,  # for backtest mode
    ):
        self.kronos = kronos
        self.timesfm = timesfm
        self.min_move_threshold = min_move_threshold
        self.stop_pct = stop_pct
        self.target_pct = target_pct
        self.mtf_filter = mtf_filter
        self.mtf_timestamp = mtf_timestamp  # updated each bar in backtest

    def evaluate(self, candles: pd.DataFrame, timestamp: Optional[pd.Timestamp] = None) -> TradeSignal:
        """
        Run models + MTF filter and return a trade signal.

        Args:
            candles: DataFrame with [open, high, low, close, volume], oldest → newest.
            timestamp: current bar timestamp (backtest mode). If None, uses live fetch.
        """
        current_close = float(candles["close"].iloc[-1])

        # --- MTF filter ---
        mtf_bias = "disabled"
        if self.mtf_filter is not None:
            ts = timestamp or (candles["timestamp"].iloc[-1] if "timestamp" in candles.columns else None)
            if ts is not None:
                mtf_sig = self.mtf_filter.get_bias_at(ts)
            else:
                mtf_sig = self.mtf_filter.get_bias_live()
            mtf_bias = mtf_sig.bias
            logger.debug(f"MTF: 4H={mtf_sig.tf_4h} 1H={mtf_sig.tf_1h} bias={mtf_bias}")

        # --- Kronos + TimesFM ---
        kronos_pred: Optional[KronosPrediction] = self.kronos.predict(candles)
        timesfm_sig: Optional[TimesFMSignal] = self.timesfm.predict(candles)

        k_dir = kronos_pred.direction if kronos_pred else "neutral"
        t_dir = timesfm_sig.direction if timesfm_sig else "neutral"
        k_move = abs(kronos_pred.predicted_move_pct) if kronos_pred else 0.0
        t_strength = timesfm_sig.trend_strength if timesfm_sig else 0.0

        logger.debug(
            f"Kronos: {k_dir} ({k_move:.4%}) | TimesFM: {t_dir} (str={t_strength:.2f}) | MTF: {mtf_bias}"
        )

        def flat(reason: str) -> TradeSignal:
            return TradeSignal(
                action="flat", entry_price=current_close,
                stop_price=0.0, target_price=0.0,
                predicted_move_pct=kronos_pred.predicted_move_pct if kronos_pred else 0.0,
                kronos_direction=k_dir, timesfm_direction=t_dir,
                mtf_bias=mtf_bias, confidence=0.0, skip_reason=reason,
            )

        # Model agreement
        if k_dir != t_dir or k_dir == "neutral":
            return flat(f"models_disagree({k_dir}≠{t_dir})")

        if k_move < self.min_move_threshold:
            return flat(f"move_too_small({k_move:.4%}<{self.min_move_threshold:.4%})")

        # MTF alignment
        if mtf_bias not in ("disabled", "neutral") and mtf_bias != k_dir:
            return flat(f"mtf_conflict(signal={k_dir},bias={mtf_bias})")

        action = k_dir
        confidence = min(k_move / self.min_move_threshold * 0.5 + t_strength * 0.5, 1.0)
        # Boost confidence when MTF agrees
        if mtf_bias == action:
            confidence = min(confidence * 1.2, 1.0)

        if action == "long":
            stop_price = current_close * (1 - self.stop_pct)
            target_price = current_close * (1 + self.target_pct)
        else:
            stop_price = current_close * (1 + self.stop_pct)
            target_price = current_close * (1 - self.target_pct)

        logger.info(
            f"Signal: {action.upper()} @ {current_close:.4f} | "
            f"stop={stop_price:.4f} target={target_price:.4f} | "
            f"MTF={mtf_bias} | confidence={confidence:.2f}"
        )

        return TradeSignal(
            action=action,
            entry_price=current_close,
            stop_price=stop_price,
            target_price=target_price,
            predicted_move_pct=kronos_pred.predicted_move_pct if kronos_pred else 0.0,
            kronos_direction=k_dir,
            timesfm_direction=t_dir,
            mtf_bias=mtf_bias,
            confidence=confidence,
        )

"""
kronos_model.py
Wrapper for the Kronos candlestick OHLCV prediction model.
https://github.com/shiyu-coder/Kronos

Predicts next candle's OHLCV given a lookback window.
Uses the mini model (4.1M params) by default for speed.
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Add local Kronos source to path
_KRONOS_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kronos_src")


@dataclass
class KronosPrediction:
    direction: str           # "long" | "short" | "neutral"
    predicted_close: float
    predicted_open: float
    predicted_high: float
    predicted_low: float
    predicted_volume: float
    current_close: float
    predicted_move_pct: float  # (predicted_close - current_close) / current_close


# HuggingFace model IDs
KRONOS_MODEL_IDS = {
    "mini":  ("NeoQuasar/Kronos-Tokenizer-2k", "NeoQuasar/Kronos-mini"),
    "small": ("NeoQuasar/Kronos-Tokenizer-base", "NeoQuasar/Kronos-small"),
    "base":  ("NeoQuasar/Kronos-Tokenizer-base", "NeoQuasar/Kronos-base"),
}


class KronosModel:
    """
    Lazy-loading wrapper for Kronos candlestick prediction model.

    Falls back to a statistical baseline model if Kronos source is not available,
    so backtesting can run even without the ML dependency.
    """

    def __init__(self, model_size: str = "mini", cache_dir: str = "./model_cache"):
        self.model_size = model_size
        self.cache_dir = cache_dir
        self._predictor = None
        self._loaded = False
        self._use_fallback = False
        os.makedirs(cache_dir, exist_ok=True)

    def _load(self):
        if self._loaded:
            return
        try:
            self._load_kronos()
        except Exception as e:
            logger.warning(f"Kronos not available ({e}) — using statistical fallback model.")
            self._use_fallback = True
        self._loaded = True

    def _load_kronos(self):
        if not os.path.isdir(_KRONOS_SRC):
            raise ImportError("kronos_src/ directory not found")

        # Add project root to sys.path so kronos_src is importable
        project_root = os.path.dirname(os.path.abspath(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        import torch
        from kronos_src.kronos import KronosTokenizer, Kronos, KronosPredictor  # type: ignore

        tok_id, model_id = KRONOS_MODEL_IDS.get(self.model_size, KRONOS_MODEL_IDS["mini"])
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading Kronos {self.model_size} from HuggingFace ({model_id}) on {device}...")
        tokenizer = KronosTokenizer.from_pretrained(tok_id, cache_dir=self.cache_dir)
        model = Kronos.from_pretrained(model_id, cache_dir=self.cache_dir)
        self._predictor = KronosPredictor(model, tokenizer, device=device)
        logger.info("Kronos loaded")

    def predict(self, candles: pd.DataFrame) -> Optional[KronosPrediction]:
        """
        Predict next candle given historical OHLCV candles.

        Args:
            candles: DataFrame with columns [open, high, low, close, volume],
                     sorted oldest → newest. Should have a 'timestamp' column.
                     Minimum 32 candles recommended.

        Returns:
            KronosPrediction or None if insufficient data.
        """
        self._load()

        if len(candles) < 32:
            logger.debug("Insufficient candles for Kronos prediction (need >=32)")
            return None

        current_close = float(candles["close"].iloc[-1])

        if self._use_fallback:
            return self._fallback_predict(candles, current_close)

        return self._kronos_predict(candles, current_close)

    def _kronos_predict(self, candles: pd.DataFrame, current_close: float) -> Optional[KronosPrediction]:
        """Run actual Kronos model inference."""
        try:
            df = candles[["open", "high", "low", "close", "volume"]].copy().astype(float)

            if "timestamp" in candles.columns:
                x_ts = pd.Series(pd.to_datetime(candles["timestamp"].values))
            else:
                x_ts = pd.Series(pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq="15min"))

            freq = x_ts.iloc[-1] - x_ts.iloc[-2]
            y_ts = pd.Series([x_ts.iloc[-1] + freq])

            pred_df = self._predictor.predict(
                df=df,
                x_timestamp=x_ts,
                y_timestamp=y_ts,
                pred_len=1,
                T=1.0,
                top_p=0.9,
                sample_count=1,
                verbose=False,
            )

            pred_open  = float(pred_df["open"].iloc[0])
            pred_high  = float(pred_df["high"].iloc[0])
            pred_low   = float(pred_df["low"].iloc[0])
            pred_close = float(pred_df["close"].iloc[0])
            pred_vol   = float(pred_df["volume"].iloc[0])

            return self._build_prediction(pred_open, pred_high, pred_low, pred_close, pred_vol, current_close)

        except Exception as e:
            logger.error(f"Kronos inference error: {e} — falling back to stat model")
            return self._fallback_predict(candles, current_close)

    def _fallback_predict(self, candles: pd.DataFrame, current_close: float) -> KronosPrediction:
        """Statistical fallback: exponentially weighted momentum + mean reversion."""
        closes = candles["close"].values.astype(float)
        returns = np.diff(closes) / closes[:-1]

        weights = np.exp(np.linspace(-2, 0, len(returns)))
        weights /= weights.sum()
        momentum = float(np.dot(weights, returns))

        recent_mean = float(closes[-10:].mean())
        reversion_signal = (recent_mean - current_close) / current_close * 0.1
        predicted_move = momentum * 0.6 + reversion_signal * 0.4
        predicted_move = float(np.clip(predicted_move, -0.02, 0.02))

        predicted_close = current_close * (1 + predicted_move)
        last = candles.iloc[-1]
        body = abs(float(last["high"]) - float(last["low"]))

        return self._build_prediction(
            pred_open=current_close,
            pred_high=predicted_close + body * 0.5,
            pred_low=predicted_close - body * 0.5,
            pred_close=predicted_close,
            pred_volume=float(candles["volume"].iloc[-5:].mean()),
            current_close=current_close,
        )

    def _build_prediction(self, pred_open, pred_high, pred_low, pred_close, pred_volume, current_close) -> KronosPrediction:
        move_pct = (pred_close - current_close) / current_close

        if move_pct > 0.0005:
            direction = "long"
        elif move_pct < -0.0005:
            direction = "short"
        else:
            direction = "neutral"

        return KronosPrediction(
            direction=direction,
            predicted_close=pred_close,
            predicted_open=pred_open,
            predicted_high=pred_high,
            predicted_low=pred_low,
            predicted_volume=pred_volume,
            current_close=current_close,
            predicted_move_pct=move_pct,
        )

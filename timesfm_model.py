"""
timesfm_model.py
Google TimesFM wrapper for macro trend direction signal.
https://github.com/google-research/timesfm

Used as a trend filter: only trade in the direction TimesFM agrees with.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TimesFMSignal:
    direction: str       # "long" | "short" | "neutral"
    trend_strength: float  # 0.0 → 1.0
    forecast: list       # predicted close values for next N candles


class TimesFMModel:
    """
    Lazy-loading wrapper for Google's TimesFM time-series foundation model.

    Used as a macro trend filter: given the last N candles of close prices,
    forecast the next `horizon` candles and derive trend direction.

    Falls back to a simple moving average trend if TimesFM is not installed.
    """

    def __init__(
        self,
        model_name: str = "timesfm-1.0-200m",
        horizon: int = 10,
        cache_dir: str = "./model_cache",
    ):
        self.model_name = model_name
        self.horizon = horizon
        self.cache_dir = cache_dir
        self._model = None
        self._loaded = False
        self._use_fallback = False
        os.makedirs(cache_dir, exist_ok=True)

    def _load(self):
        if self._loaded:
            return
        try:
            self._load_timesfm()
        except ImportError:
            logger.warning(
                "TimesFM not installed — using SMA trend fallback. "
                "Install via: pip install timesfm"
            )
            self._use_fallback = True
        self._loaded = True

    def _load_timesfm(self):
        import timesfm  # type: ignore

        logger.info(f"Loading TimesFM model: {self.model_name}...")
        self._tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="cpu",
                per_core_batch_size=32,
                horizon_len=self.horizon,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=f"google/{self.model_name}"
            ),
        )
        logger.info("TimesFM loaded")

    def predict(self, candles: pd.DataFrame) -> Optional[TimesFMSignal]:
        """
        Generate macro trend signal from close price history.

        Args:
            candles: DataFrame with 'close' column, sorted oldest → newest.
                     Minimum 32 candles recommended.

        Returns:
            TimesFMSignal or None if insufficient data.
        """
        self._load()

        if len(candles) < 32:
            logger.debug("Insufficient candles for TimesFM (need >=32)")
            return None

        closes = candles["close"].values.astype(np.float64)

        if self._use_fallback:
            return self._sma_trend(closes)

        return self._timesfm_predict(closes)

    def _timesfm_predict(self, closes: np.ndarray) -> TimesFMSignal:
        """Run actual TimesFM inference."""
        try:
            # TimesFM expects list of arrays
            forecast_input = [closes]
            freq_input = [0]  # 0 = high-frequency (sub-daily)

            point_forecast, _ = self._tfm.forecast(forecast_input, freq=freq_input)
            forecast = point_forecast[0].tolist()

            return self._signal_from_forecast(closes[-1], forecast)

        except Exception as e:
            logger.error(f"TimesFM inference error: {e}, using SMA fallback")
            return self._sma_trend(closes)

    def _sma_trend(self, closes: np.ndarray) -> TimesFMSignal:
        """
        Fallback: SMA crossover trend signal.
        Fast SMA (10) vs slow SMA (30).
        """
        if len(closes) < 30:
            return TimesFMSignal(direction="neutral", trend_strength=0.0, forecast=[])

        sma_fast = float(closes[-10:].mean())
        sma_slow = float(closes[-30:].mean())
        current = float(closes[-1])

        gap_pct = (sma_fast - sma_slow) / sma_slow

        # Simple linear projection for "forecast"
        slope = (closes[-1] - closes[-10]) / 10
        forecast = [float(closes[-1] + slope * i) for i in range(1, self.horizon + 1)]

        return self._signal_from_forecast(current, forecast, gap_pct)

    def _signal_from_forecast(
        self, current_close: float, forecast: list, extra_signal: float = 0.0
    ) -> TimesFMSignal:
        """Derive direction and strength from forecast values."""
        if not forecast:
            return TimesFMSignal(direction="neutral", trend_strength=0.0, forecast=[])

        end_forecast = forecast[-1]
        mid_forecast = forecast[len(forecast) // 2]

        total_move = (end_forecast - current_close) / current_close

        # Consistency check: is trend monotonically directional?
        diffs = np.diff(forecast)
        consistency = abs(float(np.mean(np.sign(diffs))))  # 0 = random, 1 = all same direction

        strength = min(abs(total_move) * 50 + consistency * 0.3, 1.0)

        threshold = 0.001  # 0.1% net move over horizon to call a direction
        if total_move > threshold:
            direction = "long"
        elif total_move < -threshold:
            direction = "short"
        else:
            direction = "neutral"

        return TimesFMSignal(direction=direction, trend_strength=strength, forecast=forecast)

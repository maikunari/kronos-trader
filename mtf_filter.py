"""
mtf_filter.py
Multi-timeframe trend filter.

Fetches 1H and 4H candles and computes EMA(fast) vs EMA(slow) to determine
the macro trend bias. Only trades in the direction of the higher-timeframe trend.

Logic:
  4H: EMA20 > EMA50 AND 1H: EMA20 > EMA50 → bias = "long"
  4H: EMA20 < EMA50 AND 1H: EMA20 < EMA50 → bias = "short"
  Otherwise → bias = "neutral" (skip trade)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from hyperliquid_feed import fetch_historical_hl, fetch_historical_binance

logger = logging.getLogger(__name__)


@dataclass
class MTFSignal:
    bias: str              # "long" | "short" | "neutral"
    tf_1h: str             # "long" | "short" | "neutral"
    tf_4h: str             # "long" | "short" | "neutral"
    ema_fast_1h: float
    ema_slow_1h: float
    ema_fast_4h: float
    ema_slow_4h: float


class MTFFilter:
    """
    Multi-timeframe EMA trend filter.

    In live mode: fetches fresh 1H/4H candles from Hyperliquid on each call.
    In backtest mode: slices pre-loaded DataFrames at the current timestamp.
    """

    def __init__(
        self,
        symbol: str,
        ema_fast: int = 20,
        ema_slow: int = 50,
        require_both: bool = True,     # True = 4H AND 1H must agree; False = 4H only
        data_source: str = "hyperliquid",
    ):
        self.symbol = symbol
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.require_both = require_both
        self.data_source = data_source

        # Pre-loaded for backtest mode
        self._df_1h: Optional[pd.DataFrame] = None
        self._df_4h: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Backtest mode: pre-load data once, then slice by timestamp
    # ------------------------------------------------------------------

    def load_backtest_data(self, start_date: str, end_date: str):
        """Pre-load 1H and 4H candles for backtest simulation."""
        logger.info(f"Loading MTF data: {self.symbol} 1h/4h ({start_date} → {end_date})")
        fetch_fn = fetch_historical_hl if self.data_source == "hyperliquid" else fetch_historical_binance

        from datetime import datetime, timezone
        start_ms = int(datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).timestamp() * 1000)

        for tf, attr in [("1h", "_df_1h"), ("4h", "_df_4h")]:
            df = fetch_historical_hl(self.symbol, tf, start_ms, end_ms)
            if df.empty:
                logger.warning(f"HL returned no {tf} data — falling back to Binance")
                df = fetch_historical_binance(self.symbol, tf, start_ms, end_ms)
            setattr(self, attr, df)

        if self._df_1h.empty or self._df_4h.empty:
            logger.warning("Could not load MTF data — filter will return neutral")
        else:
            logger.info(f"MTF data loaded: {len(self._df_1h)} 1h candles, {len(self._df_4h)} 4h candles")

    def get_bias_at(self, timestamp: pd.Timestamp) -> MTFSignal:
        """
        Backtest mode: get trend bias at a specific point in time.
        Uses only candles available BEFORE the given timestamp (no lookahead).
        """
        if self._df_1h is None or self._df_4h is None or self._df_1h.empty or self._df_4h.empty:
            return self._neutral()

        df_1h = self._df_1h[self._df_1h["timestamp"] < timestamp]
        df_4h = self._df_4h[self._df_4h["timestamp"] < timestamp]

        return self._compute_bias(df_1h, df_4h)

    # ------------------------------------------------------------------
    # Live mode: fetch fresh candles and compute current bias
    # ------------------------------------------------------------------

    def get_bias_live(self) -> MTFSignal:
        """
        Live mode: fetch latest 1H and 4H candles from Hyperliquid and
        compute current trend bias.
        """
        try:
            from datetime import datetime, timezone, timedelta
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            lookback_ms = 60 * 24 * 3600 * 1000  # 60 days back

            fetch_fn = fetch_historical_hl if self.data_source == "hyperliquid" else fetch_historical_binance
            df_1h = fetch_fn(self.symbol, "1h", now_ms - lookback_ms, now_ms)
            df_4h = fetch_fn(self.symbol, "4h", now_ms - lookback_ms, now_ms)

            return self._compute_bias(df_1h, df_4h)

        except Exception as e:
            logger.error(f"MTF live fetch failed: {e} — returning neutral")
            return self._neutral()

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute_bias(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> MTFSignal:
        """Compute EMA trend on 1H and 4H and return combined bias."""
        min_candles = self.ema_slow + 5

        if len(df_1h) < min_candles or len(df_4h) < min_candles:
            logger.debug("Insufficient candles for MTF filter — returning neutral")
            return self._neutral()

        trend_1h, ema_fast_1h, ema_slow_1h = self._ema_trend(df_1h["close"].values)
        trend_4h, ema_fast_4h, ema_slow_4h = self._ema_trend(df_4h["close"].values)

        if self.require_both:
            if trend_4h == trend_1h and trend_4h != "neutral":
                bias = trend_4h
            else:
                bias = "neutral"
        else:
            # 4H is primary; 1H confirms or weakens
            bias = trend_4h

        logger.debug(
            f"MTF: 4H={trend_4h} (EMA{self.ema_fast}={ema_fast_4h:.2f} vs EMA{self.ema_slow}={ema_slow_4h:.2f}) | "
            f"1H={trend_1h} (EMA{self.ema_fast}={ema_fast_1h:.2f} vs EMA{self.ema_slow}={ema_slow_1h:.2f}) | "
            f"bias={bias}"
        )

        return MTFSignal(
            bias=bias,
            tf_1h=trend_1h,
            tf_4h=trend_4h,
            ema_fast_1h=ema_fast_1h,
            ema_slow_1h=ema_slow_1h,
            ema_fast_4h=ema_fast_4h,
            ema_slow_4h=ema_slow_4h,
        )

    def _ema_trend(self, closes: np.ndarray):
        """Compute EMA crossover trend direction."""
        ema_f = _ema(closes, self.ema_fast)
        ema_s = _ema(closes, self.ema_slow)

        gap_pct = (ema_f - ema_s) / ema_s

        # Require a minimum gap to call it a trend (avoid false signals near crossover)
        min_gap = 0.001  # 0.1%
        if gap_pct > min_gap:
            trend = "long"
        elif gap_pct < -min_gap:
            trend = "short"
        else:
            trend = "neutral"

        return trend, float(ema_f), float(ema_s)

    def _neutral(self) -> MTFSignal:
        return MTFSignal(
            bias="neutral", tf_1h="neutral", tf_4h="neutral",
            ema_fast_1h=0.0, ema_slow_1h=0.0,
            ema_fast_4h=0.0, ema_slow_4h=0.0,
        )


def _ema(values: np.ndarray, period: int) -> float:
    """Compute the last EMA value for the given period."""
    if len(values) < period:
        return float(values[-1])
    k = 2.0 / (period + 1)
    ema = float(values[0])
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema

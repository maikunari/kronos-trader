"""
mtf_filter.py
Multi-timeframe EMA trend bias utility.

Exposes per-timeframe trend classification (1H, 4H) plus small helpers for
veto+confirm use by downstream strategies. The old require-both AND-gate
semantics are gone — the prior Kronos/TimesFM pipeline used this to combine
1H + 4H into a single must-agree bias, which starved signals. In the new
Phase 1 pipeline:

  * 1H agreement is enforced via SuperTrend in snipe_signal_engine (not here).
  * 4H is consumed as a veto: `.vetoes("long")` is True iff 4H explicitly
    opposes the entry direction; neutral or unavailable does not veto.
  * `.confirms(dir)` is available for strategies that still want a strong
    both-timeframes-agree check as a confidence booster.

Backtest: pre-load historical 1H/4H once, then call `get_bias_at(ts)` —
the slicer uses only candles strictly before `ts` (no lookahead).
Live: `get_bias_live()` fetches fresh candles from Hyperliquid.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from hyperliquid_feed import fetch_historical_binance, fetch_historical_hl

logger = logging.getLogger(__name__)


@dataclass
class MTFSignal:
    tf_1h: str               # "long" | "short" | "neutral"
    tf_4h: str               # same
    ema_fast_1h: float
    ema_slow_1h: float
    ema_fast_4h: float
    ema_slow_4h: float

    # --- Veto + confirm helpers ------------------------------------------------

    def vetoes(self, direction: str, which: str = "4h") -> bool:
        """True iff `which` timeframe explicitly opposes `direction`.

        Defaults to 4H as that's the intended veto in the Phase 1 sniper
        (1H is gated by SuperTrend elsewhere). `neutral` or `disabled`
        never vetoes.
        """
        if direction not in {"long", "short"}:
            return False
        opposite = "short" if direction == "long" else "long"
        field = "tf_4h" if which == "4h" else "tf_1h"
        return getattr(self, field) == opposite

    def confirms(self, direction: str) -> bool:
        """True iff both 1H and 4H agree with `direction`."""
        return self.tf_1h == direction and self.tf_4h == direction

    @property
    def bias(self) -> str:
        """Deprecated: legacy AND-gate bias field.

        Returns "long" iff both 1H and 4H agree long, "short" iff both agree
        short, else "neutral". Kept so the pre-Phase-1 engines (atr_engine,
        trend_signal_engine, derivatives_signal_engine) still compile while
        they're being replaced. New code should use `.vetoes()` / `.confirms()`.
        """
        if self.tf_1h == self.tf_4h and self.tf_1h in ("long", "short"):
            return self.tf_1h
        return "neutral"


class MTFFilter:
    def __init__(
        self,
        symbol: str,
        ema_fast: int = 20,
        ema_slow: int = 50,
        data_source: str = "hyperliquid",
        min_gap_pct: float = 0.001,
        require_both: Optional[bool] = None,   # deprecated, ignored
    ) -> None:
        self.symbol = symbol
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.data_source = data_source
        self.min_gap_pct = min_gap_pct
        if require_both is not None:
            logger.debug(
                "MTFFilter(require_both=...) is deprecated and ignored; "
                "use MTFSignal.vetoes()/confirms() instead."
            )

        # Pre-loaded for backtest mode
        self._df_1h: Optional[pd.DataFrame] = None
        self._df_4h: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Backtest mode: pre-load once, then slice by timestamp
    # ------------------------------------------------------------------

    def load_backtest_data(self, start_date: str, end_date: str) -> None:
        logger.info("Loading MTF data %s 1h/4h (%s -> %s)", self.symbol, start_date, end_date)
        fetch = fetch_historical_hl if self.data_source == "hyperliquid" else fetch_historical_binance

        start_ms = int(datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).timestamp() * 1000)

        for tf, attr in (("1h", "_df_1h"), ("4h", "_df_4h")):
            df = fetch(self.symbol, tf, start_ms, end_ms)
            if df.empty and fetch is not fetch_historical_binance:
                logger.warning("HL returned no %s data — falling back to Binance", tf)
                df = fetch_historical_binance(self.symbol, tf, start_ms, end_ms)
            setattr(self, attr, df)

        assert self._df_1h is not None and self._df_4h is not None
        if self._df_1h.empty or self._df_4h.empty:
            logger.warning("Could not load MTF data — filter will return neutral")
        else:
            logger.info("MTF data loaded: %d 1h candles, %d 4h candles",
                        len(self._df_1h), len(self._df_4h))

    def get_bias_at(self, timestamp: pd.Timestamp) -> MTFSignal:
        """No-lookahead: uses only candles strictly before `timestamp`."""
        if self._df_1h is None or self._df_4h is None or self._df_1h.empty or self._df_4h.empty:
            return self._neutral()
        df_1h = self._df_1h[self._df_1h["timestamp"] < timestamp]
        df_4h = self._df_4h[self._df_4h["timestamp"] < timestamp]
        return self._compute(df_1h, df_4h)

    # ------------------------------------------------------------------
    # Live mode
    # ------------------------------------------------------------------

    def get_bias_live(self) -> MTFSignal:
        try:
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            lookback_ms = 60 * 24 * 3600 * 1000  # 60 days
            fetch = fetch_historical_hl if self.data_source == "hyperliquid" else fetch_historical_binance
            return self._compute(
                fetch(self.symbol, "1h", now_ms - lookback_ms, now_ms),
                fetch(self.symbol, "4h", now_ms - lookback_ms, now_ms),
            )
        except Exception as exc:
            logger.error("MTF live fetch failed: %s — returning neutral", exc)
            return self._neutral()

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def _compute(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> MTFSignal:
        min_candles = self.ema_slow + 5
        if len(df_1h) < min_candles or len(df_4h) < min_candles:
            logger.debug("Insufficient candles for MTF filter — returning neutral")
            return self._neutral()

        trend_1h, ema_f_1h, ema_s_1h = self._ema_trend(df_1h["close"].to_numpy(dtype=float))
        trend_4h, ema_f_4h, ema_s_4h = self._ema_trend(df_4h["close"].to_numpy(dtype=float))
        return MTFSignal(
            tf_1h=trend_1h, tf_4h=trend_4h,
            ema_fast_1h=ema_f_1h, ema_slow_1h=ema_s_1h,
            ema_fast_4h=ema_f_4h, ema_slow_4h=ema_s_4h,
        )

    def _ema_trend(self, closes: np.ndarray) -> tuple[str, float, float]:
        ema_f = _ema_last(closes, self.ema_fast)
        ema_s = _ema_last(closes, self.ema_slow)
        gap = (ema_f - ema_s) / ema_s if ema_s else 0.0
        if gap > self.min_gap_pct:
            trend = "long"
        elif gap < -self.min_gap_pct:
            trend = "short"
        else:
            trend = "neutral"
        return trend, float(ema_f), float(ema_s)

    def _neutral(self) -> MTFSignal:
        return MTFSignal(
            tf_1h="neutral", tf_4h="neutral",
            ema_fast_1h=0.0, ema_slow_1h=0.0,
            ema_fast_4h=0.0, ema_slow_4h=0.0,
        )


def _ema_last(values: np.ndarray, period: int) -> float:
    """Compute the last EMA value for the given period."""
    if len(values) < period:
        return float(values[-1]) if len(values) else 0.0
    k = 2.0 / (period + 1)
    ema = float(values[0])
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema

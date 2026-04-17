"""
regime.py
Market regime classification: ADX, Hurst exponent, realized-volatility bucket.

Purpose: gate trend-following entries. Trend strategies lose in chop; the
regime detector is the single biggest lever against whipsaw. Skip-don't-trade
when the market is mean-reverting.

Primary API:
    detector = RegimeDetector(adx_period=14, hurst_window=200, rv_window=120)
    state = detector.classify(df)
    if state.is_trending: ...

Where state exposes:
    state.adx          - ADX(adx_period) current value
    state.hurst        - Hurst exponent over hurst_window recent bars (0.5 = random walk)
    state.rv           - rolling annualized realized volatility
    state.rv_quintile  - 0..4 bucket of rv vs its own trailing distribution
    state.is_trending  - adx >= adx_threshold AND hurst >= hurst_threshold
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from ta.trend import ADXIndicator

logger = logging.getLogger(__name__)


# --- Hurst (rescaled range) ----------------------------------------------------

def hurst_rs(values: np.ndarray, min_chunk: int = 8) -> float:
    """
    Hurst exponent via classical rescaled-range (R/S) analysis.

    Interpretation:
        H ≈ 0.5  -> random walk (no memory)
        H > 0.5  -> persistent / trending
        H < 0.5  -> mean-reverting / anti-persistent

    Returns 0.5 (random-walk default) when data is too short or degenerate —
    callers that need to distinguish "missing" from "computed 0.5" should
    check the series length themselves first.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2 * min_chunk or np.any(values <= 0):
        return 0.5

    log_ret = np.diff(np.log(values))
    if len(log_ret) < 2 * min_chunk:
        return 0.5

    max_chunk = len(log_ret) // 2
    if max_chunk < min_chunk:
        return 0.5

    # Geometrically spaced chunk sizes
    sizes = np.unique(
        np.logspace(np.log10(min_chunk), np.log10(max_chunk), num=8).astype(int)
    )
    sizes = sizes[sizes >= min_chunk]

    rs_points: list[tuple[int, float]] = []
    for size in sizes:
        rs_for_size = []
        for i in range(0, len(log_ret) - size + 1, size):
            chunk = log_ret[i : i + size]
            mean = chunk.mean()
            cum_dev = np.cumsum(chunk - mean)
            rng = cum_dev.max() - cum_dev.min()
            std = chunk.std(ddof=0)
            if std > 0 and rng > 0:
                rs_for_size.append(rng / std)
        if rs_for_size:
            rs_points.append((int(size), float(np.mean(rs_for_size))))

    if len(rs_points) < 2:
        return 0.5

    xs = np.log([p[0] for p in rs_points])
    ys = np.log([p[1] for p in rs_points])
    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope)


# --- Realized volatility ------------------------------------------------------

def realized_vol(closes: np.ndarray, window: int, bars_per_year: int = 35_040) -> float:
    """
    Annualized realized volatility over the last `window` bars.

    Defaults assume 15-minute bars: 35,040 bars per 365-day year.
    Crypto trades 24/7; for 1h bars use bars_per_year=8_760, for 1d use 365.
    """
    closes = np.asarray(closes, dtype=float)
    if len(closes) < window + 1:
        return float("nan")
    log_ret = np.diff(np.log(closes[-(window + 1) :]))
    sigma_bar = log_ret.std(ddof=1)
    return float(sigma_bar * np.sqrt(bars_per_year))


def rv_quintile(closes: np.ndarray, window: int, hist_window: int, bars_per_year: int = 35_040) -> int:
    """
    Bucket current RV into 0..4 against a rolling distribution.

    Compares rv(window) on the most recent `window` bars against the empirical
    distribution of rv(window) values computed on rolling windows over the last
    `hist_window` bars. Useful as a tabular feature for the ML layer.
    """
    closes = np.asarray(closes, dtype=float)
    if len(closes) < hist_window + window + 1:
        return 2  # middle bucket when insufficient history

    log_ret = np.diff(np.log(closes))
    # Rolling RV series over the history window
    s = pd.Series(log_ret[-(hist_window + window) :])
    rv_series = s.rolling(window).std(ddof=1) * np.sqrt(bars_per_year)
    rv_series = rv_series.dropna()
    if len(rv_series) < 5:
        return 2
    cur = float(rv_series.iloc[-1])
    quantiles = rv_series.quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
    return int(np.searchsorted(quantiles, cur))


# --- Regime state + detector --------------------------------------------------

@dataclass
class RegimeState:
    adx: float
    hurst: float
    rv: float
    rv_quintile: int
    is_trending: bool
    skip_reason: str = ""

    def __str__(self) -> str:
        kind = "TREND" if self.is_trending else "RANGE"
        return (
            f"{kind} adx={self.adx:.1f} hurst={self.hurst:.2f} "
            f"rv={self.rv:.3f} q={self.rv_quintile}"
            + (f" skip={self.skip_reason}" if self.skip_reason else "")
        )


class RegimeDetector:
    def __init__(
        self,
        adx_period: int = 14,
        adx_threshold: float = 20.0,
        hurst_window: int = 200,
        hurst_threshold: float = 0.50,
        rv_window: int = 96,          # 96 × 15m = 24 hours
        rv_hist_window: int = 2_880,  # 30 days of 15m bars
        bars_per_year: int = 35_040,
    ) -> None:
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.hurst_window = hurst_window
        self.hurst_threshold = hurst_threshold
        self.rv_window = rv_window
        self.rv_hist_window = rv_hist_window
        self.bars_per_year = bars_per_year

    def classify(self, candles: pd.DataFrame) -> RegimeState:
        """
        Classify the regime at the most recent bar.

        `candles` must have columns [high, low, close] and be ordered oldest → newest.
        Returns RegimeState; on insufficient data returns a non-trending state with
        skip_reason set.
        """
        min_bars = max(self.adx_period * 2, self.hurst_window, self.rv_window) + 2
        if len(candles) < min_bars:
            return RegimeState(
                adx=float("nan"), hurst=0.5, rv=float("nan"),
                rv_quintile=2, is_trending=False,
                skip_reason="insufficient_data",
            )

        high = candles["high"].astype(float)
        low = candles["low"].astype(float)
        close = candles["close"].astype(float)

        try:
            adx_val = float(
                ADXIndicator(high=high, low=low, close=close, window=self.adx_period)
                .adx()
                .iloc[-1]
            )
        except Exception as exc:  # ta can throw on degenerate input
            logger.warning("ADX compute failed: %s", exc)
            adx_val = float("nan")

        h = hurst_rs(close.to_numpy()[-self.hurst_window :])
        rv = realized_vol(close.to_numpy(), self.rv_window, self.bars_per_year)
        q = rv_quintile(close.to_numpy(), self.rv_window, self.rv_hist_window, self.bars_per_year)

        trending = (
            not np.isnan(adx_val)
            and adx_val >= self.adx_threshold
            and h >= self.hurst_threshold
        )
        skip = ""
        if not trending:
            if np.isnan(adx_val):
                skip = "adx_nan"
            elif adx_val < self.adx_threshold:
                skip = f"adx_too_low({adx_val:.1f}<{self.adx_threshold:.0f})"
            elif h < self.hurst_threshold:
                skip = f"hurst_too_low({h:.2f}<{self.hurst_threshold:.2f})"

        return RegimeState(
            adx=adx_val, hurst=h, rv=rv, rv_quintile=q,
            is_trending=trending, skip_reason=skip,
        )

    def is_trending(self, candles: pd.DataFrame) -> bool:
        """Shortcut: True iff the current regime is trend-tradeable."""
        return self.classify(candles).is_trending

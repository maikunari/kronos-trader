"""
indicators/rsi.py
Wilder's Relative Strength Index via the ta library.

CBS uses RSI alongside AO as his two-indicator stack. Default thresholds
are 70 (overbought) and 30 (oversold). If CBS's exact thresholds turn
out to be 80/20 (he was noncommittal in the transcripts we have), we
tune during validation — the helpers accept custom thresholds.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

from ta.momentum import RSIIndicator

RSIZone = Literal["overbought", "oversold", "neutral"]

DEFAULT_OVERBOUGHT = 70.0
DEFAULT_OVERSOLD = 30.0


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI over `period` bars."""
    return RSIIndicator(close=close, window=period).rsi()


def rsi_zone(
    value: float,
    overbought: float = DEFAULT_OVERBOUGHT,
    oversold: float = DEFAULT_OVERSOLD,
) -> RSIZone:
    """Classify an RSI value."""
    if value >= overbought:
        return "overbought"
    if value <= oversold:
        return "oversold"
    return "neutral"


def is_overbought(value: float, threshold: float = DEFAULT_OVERBOUGHT) -> bool:
    return value >= threshold


def is_oversold(value: float, threshold: float = DEFAULT_OVERSOLD) -> bool:
    return value <= threshold


def midline_cross(series: pd.Series) -> Literal["up", "down", "none"]:
    """Did RSI cross its 50 midline on the most recent bar?"""
    if len(series) < 2:
        return "none"
    prev = series.iloc[-2]
    cur = series.iloc[-1]
    if pd.isna(prev) or pd.isna(cur):
        return "none"
    if prev <= 50 and cur > 50:
        return "up"
    if prev >= 50 and cur < 50:
        return "down"
    return "none"

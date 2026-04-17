"""
indicators/awesome_oscillator.py
Bill Williams' Awesome Oscillator (AO).

Definition:
    midpoint(i) = (high(i) + low(i)) / 2
    AO(i)       = SMA_short(midpoint) - SMA_long(midpoint)
    default windows: short=5, long=34

A positive AO = short-term momentum above longer-term mean → bullish bias.
Negative = bearish bias.

Bar coloring (used by CBS's "two bars of a certain colour" rule):
    green if AO(i) > AO(i-1)   (momentum building)
    red   if AO(i) < AO(i-1)   (momentum fading)
    flat  if equal (rare)

This module implements the raw indicator + bar-color labeling + a
utility to detect "two consecutive same-color bars" which CBS uses
as a confirmation trigger.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

from ta.momentum import AwesomeOscillatorIndicator

BarColor = Literal["green", "red", "flat"]


def awesome_oscillator(
    high: pd.Series,
    low: pd.Series,
    short: int = 5,
    long: int = 34,
) -> pd.Series:
    """
    Raw AO series.

    Uses the ta library implementation for consistency with other
    ta-sourced indicators in this repo (ATR, ADX, etc.).
    """
    if short >= long:
        raise ValueError(f"short ({short}) must be less than long ({long})")
    if len(high) != len(low):
        raise ValueError("high and low must have matching length")
    return AwesomeOscillatorIndicator(high=high, low=low, window1=short, window2=long).awesome_oscillator()


def ao_bar_colors(ao: pd.Series) -> pd.Series:
    """
    Map AO values to {'green', 'red', 'flat'} bars.

    First bar is 'flat' (no previous to compare).
    """
    diff = ao.diff()

    def colorize(d: float) -> BarColor:
        if pd.isna(d) or d == 0:
            return "flat"
        return "green" if d > 0 else "red"

    return diff.map(colorize)


def two_bar_same_color(ao: pd.Series, color: BarColor) -> bool:
    """
    CBS's two-bar confirmation: last two AO bars are both `color`.

    Returns False on insufficient data.
    """
    if color not in ("green", "red"):
        raise ValueError(f"color must be 'green' or 'red', got {color!r}")
    if len(ao) < 3:  # need at least 3 bars so last 2 diffs exist
        return False
    colors = ao_bar_colors(ao)
    return bool(colors.iloc[-1] == color and colors.iloc[-2] == color)


def zero_line_state(ao: pd.Series) -> Literal["above", "below", "at_zero"]:
    """Whether the latest AO value sits above, below, or at the zero line."""
    if len(ao) == 0 or pd.isna(ao.iloc[-1]):
        return "at_zero"
    v = float(ao.iloc[-1])
    if v > 0:
        return "above"
    if v < 0:
        return "below"
    return "at_zero"


def zero_line_cross(ao: pd.Series) -> Literal["up", "down", "none"]:
    """
    Did AO cross the zero line on the most recent bar?

    Returns:
        'up'   — crossed from <=0 to >0
        'down' — crossed from >=0 to <0
        'none' — no cross or insufficient data
    """
    if len(ao) < 2:
        return "none"
    prev = ao.iloc[-2]
    cur = ao.iloc[-1]
    if pd.isna(prev) or pd.isna(cur):
        return "none"
    if prev <= 0 and cur > 0:
        return "up"
    if prev >= 0 and cur < 0:
        return "down"
    return "none"

"""Tests for the Chandelier trailing exit helpers in atr_engine.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atr_engine import ChandelierTrail, chandelier_exit_level


def _df(closes: np.ndarray) -> pd.DataFrame:
    highs = closes * 1.002
    lows = closes * 0.998
    ts = pd.date_range("2025-01-01", periods=len(closes), freq="15min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": closes, "high": highs, "low": lows,
        "close": closes, "volume": np.full_like(closes, 1000.0),
    })


# --- chandelier_exit_level (stateless) ---------------------------------------

def test_exit_level_long_is_below_recent_high():
    closes = np.linspace(100, 200, 120)
    level = chandelier_exit_level(_df(closes), direction="long", period=22, atr_mult=3.0)
    assert 0 < level < 200 * 1.002  # below the recent high


def test_exit_level_short_is_above_recent_low():
    closes = np.linspace(200, 100, 120)
    level = chandelier_exit_level(_df(closes), direction="short", period=22, atr_mult=3.0)
    assert level > 100 * 0.998


def test_exit_level_insufficient_data_returns_zero():
    closes = np.linspace(100, 110, 10)
    assert chandelier_exit_level(_df(closes), direction="long", period=22) == 0.0


def test_exit_level_rejects_bad_direction():
    closes = np.linspace(100, 200, 120)
    with pytest.raises(ValueError):
        chandelier_exit_level(_df(closes), direction="sideways")


# --- ChandelierTrail (stateful, ratcheting) ----------------------------------

def test_long_trail_ratchets_up_never_down():
    trail = ChandelierTrail(direction="long", atr_mult=3.0)
    # Bars with increasing highs -> trail should step up
    s1 = trail.update(high=100, low=98, atr=1.0)
    s2 = trail.update(high=105, low=99, atr=1.0)
    s3 = trail.update(high=103, low=100, atr=1.0)   # high went down but trail holds
    s4 = trail.update(high=110, low=105, atr=1.0)   # new high pushes trail further up
    assert s1 == 100 - 3        # 97
    assert s2 == 105 - 3        # 102, moved up
    assert s3 == s2             # ratcheted: doesn't move down when high retreats
    assert s4 == 110 - 3        # 107, moved up again


def test_short_trail_ratchets_down_never_up():
    trail = ChandelierTrail(direction="short", atr_mult=3.0)
    s1 = trail.update(high=102, low=100, atr=1.0)
    s2 = trail.update(high=101, low=95, atr=1.0)
    s3 = trail.update(high=103, low=97, atr=1.0)
    s4 = trail.update(high=98, low=90, atr=1.0)
    assert s1 == 100 + 3        # 103
    assert s2 == 95 + 3         # 98, moved down
    assert s3 == s2             # ratchet
    assert s4 == 90 + 3         # 93


def test_trail_ignores_nonpositive_atr():
    trail = ChandelierTrail(direction="long", atr_mult=3.0)
    assert trail.update(high=100, low=99, atr=0.0) is None
    assert trail.stop is None


def test_trail_rejects_bad_direction():
    with pytest.raises(ValueError):
        ChandelierTrail(direction="flat")

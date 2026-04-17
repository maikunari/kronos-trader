"""Shared pytest fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def uptrend_ohlcv(rng) -> pd.DataFrame:
    """200 candles with a clean uptrend: +0.1% drift/bar + small noise."""
    n = 200
    close = 100.0 * np.exp(np.cumsum(0.001 + rng.normal(0, 0.002, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = rng.uniform(100, 1000, n)
    ts = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": vol}
    )


@pytest.fixture
def downtrend_ohlcv(rng) -> pd.DataFrame:
    """200 candles with a clean downtrend."""
    n = 200
    close = 100.0 * np.exp(np.cumsum(-0.001 + rng.normal(0, 0.002, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = rng.uniform(100, 1000, n)
    ts = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": vol}
    )


@pytest.fixture
def flat_ohlcv(rng) -> pd.DataFrame:
    """200 candles with no drift — pure noise around 100."""
    n = 200
    close = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
    high = close + np.abs(rng.normal(0, 0.1, n))
    low = close - np.abs(rng.normal(0, 0.1, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = rng.uniform(100, 1000, n)
    ts = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": vol}
    )

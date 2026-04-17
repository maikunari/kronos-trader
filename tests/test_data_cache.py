"""Tests for data_cache.py. Fetcher is mocked — no network access."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_cache import (
    OHLCV_COLS,
    cache_path,
    clear_cache,
    get_candles,
    load_cached,
    prefetch_universe,
    save_cache,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _candles_df(start_iso: str, n_bars: int, freq: str = "1h") -> pd.DataFrame:
    """Generate a deterministic OHLCV dataframe starting at start_iso."""
    ts = pd.date_range(start_iso, periods=n_bars, freq=freq, tz="UTC")
    rng = np.random.default_rng(7)
    closes = 100 + rng.normal(0, 0.5, n_bars).cumsum()
    return pd.DataFrame({
        "timestamp": ts,
        "open": closes,
        "high": closes * 1.001,
        "low": closes * 0.999,
        "close": closes,
        "volume": rng.uniform(800, 1200, n_bars),
    })


def _mock_fetcher(df: pd.DataFrame):
    """Wrap a canned df as a fetcher. Returns only rows within the ms range."""
    captured_calls = []

    def fetcher(symbol: str, timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        captured_calls.append((symbol, timeframe, start_ms, end_ms))
        # Resolution-independent conversion: Timestamp.timestamp() -> seconds
        ts_series = df["timestamp"].apply(lambda t: int(t.timestamp() * 1000))
        mask = (ts_series >= start_ms) & (ts_series <= end_ms)
        return df.loc[mask].reset_index(drop=True)

    fetcher.calls = captured_calls  # type: ignore[attr-defined]
    return fetcher


@pytest.fixture
def tmp_cache(tmp_path: Path) -> Path:
    return tmp_path / "cache_candles"


# ---------------------------------------------------------------------------
# Path / I/O
# ---------------------------------------------------------------------------

def test_cache_path_upper_and_formats(tmp_cache):
    assert cache_path("btc", "1h", tmp_cache) == tmp_cache / "BTC_1h.csv"


def test_save_and_load_roundtrip(tmp_cache):
    df = _candles_df("2025-01-01", 100, freq="1h")
    save_cache(df, "TEST", "1h", tmp_cache)
    loaded = load_cached("TEST", "1h", tmp_cache)
    assert len(loaded) == 100
    # Timestamps are UTC-aware after load
    assert str(loaded["timestamp"].dt.tz) == "UTC"
    # Columns preserved in order
    assert list(loaded.columns) == OHLCV_COLS


def test_load_cached_missing_returns_empty(tmp_cache):
    df = load_cached("NOPE", "1h", tmp_cache)
    assert df.empty
    assert list(df.columns) == OHLCV_COLS


def test_save_cache_noop_for_empty_df(tmp_cache):
    save_cache(pd.DataFrame(columns=OHLCV_COLS), "TEST", "1h", tmp_cache)
    assert not cache_path("TEST", "1h", tmp_cache).exists()


# ---------------------------------------------------------------------------
# get_candles — cold cache
# ---------------------------------------------------------------------------

def test_cold_cache_fetches_full_range(tmp_cache):
    all_data = _candles_df("2025-01-01", 240, freq="1h")   # 10 days
    fetch = _mock_fetcher(all_data)

    got = get_candles(
        "BTC", "1h",
        start="2025-01-01", end="2025-01-10",
        fetcher=fetch, cache_dir=tmp_cache,
    )
    assert len(got) > 0
    # One fetch call made
    assert len(fetch.calls) == 1
    # Cache persisted
    assert cache_path("BTC", "1h", tmp_cache).exists()


def test_warm_cache_within_range_no_fetch(tmp_cache):
    all_data = _candles_df("2025-01-01", 240, freq="1h")
    fetch = _mock_fetcher(all_data)

    # Warm the cache
    get_candles("BTC", "1h", "2025-01-01", "2025-01-10",
                fetcher=fetch, cache_dir=tmp_cache)
    fetch.calls.clear()

    # Request a range strictly within the cached range — no fetch needed
    got = get_candles("BTC", "1h", "2025-01-03", "2025-01-07",
                      fetcher=fetch, cache_dir=tmp_cache)
    assert len(got) > 0
    assert len(fetch.calls) == 0   # cache hit, no network


def test_warm_cache_extending_forward_triggers_incremental_fetch(tmp_cache):
    all_data = _candles_df("2025-01-01", 480, freq="1h")   # 20 days available
    fetch = _mock_fetcher(all_data)

    # Cache first 10 days
    get_candles("BTC", "1h", "2025-01-01", "2025-01-10",
                fetcher=fetch, cache_dir=tmp_cache)
    fetch.calls.clear()

    # Extend forward to day 15
    got = get_candles("BTC", "1h", "2025-01-01", "2025-01-15",
                      fetcher=fetch, cache_dir=tmp_cache)
    assert len(got) > 0
    # Exactly one incremental fetch for the forward gap
    assert len(fetch.calls) == 1


def test_warm_cache_extending_backward_triggers_incremental_fetch(tmp_cache):
    all_data = _candles_df("2025-01-01", 480, freq="1h")
    fetch = _mock_fetcher(all_data)

    # Cache days 10-15
    get_candles("BTC", "1h", "2025-01-10", "2025-01-15",
                fetcher=fetch, cache_dir=tmp_cache)
    fetch.calls.clear()

    # Request days 5-15 — needs a backward incremental
    got = get_candles("BTC", "1h", "2025-01-05", "2025-01-15",
                      fetcher=fetch, cache_dir=tmp_cache)
    assert len(got) > 0
    assert len(fetch.calls) == 1


def test_force_refresh_bypasses_cache(tmp_cache):
    all_data = _candles_df("2025-01-01", 240, freq="1h")
    fetch = _mock_fetcher(all_data)

    get_candles("BTC", "1h", "2025-01-01", "2025-01-10",
                fetcher=fetch, cache_dir=tmp_cache)
    fetch.calls.clear()

    get_candles("BTC", "1h", "2025-01-03", "2025-01-07",
                fetcher=fetch, cache_dir=tmp_cache,
                force_refresh=True)
    # With force_refresh, we re-fetch even if cache covers the range
    assert len(fetch.calls) == 1


def test_invalid_date_order_raises(tmp_cache):
    fetch = _mock_fetcher(_candles_df("2025-01-01", 10))
    with pytest.raises(ValueError):
        get_candles("BTC", "1h", "2025-01-10", "2025-01-01",
                    fetcher=fetch, cache_dir=tmp_cache)


def test_dedup_across_merges(tmp_cache):
    all_data = _candles_df("2025-01-01", 240, freq="1h")
    fetch = _mock_fetcher(all_data)

    # Fetch overlapping ranges twice
    get_candles("BTC", "1h", "2025-01-01", "2025-01-05",
                fetcher=fetch, cache_dir=tmp_cache)
    get_candles("BTC", "1h", "2025-01-03", "2025-01-08",
                fetcher=fetch, cache_dir=tmp_cache)

    cached = load_cached("BTC", "1h", tmp_cache)
    # No duplicate timestamps
    assert not cached["timestamp"].duplicated().any()


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------

def test_clear_cache_specific_symbol(tmp_cache):
    fetch = _mock_fetcher(_candles_df("2025-01-01", 100))
    get_candles("BTC", "1h", "2025-01-01", "2025-01-03",
                fetcher=fetch, cache_dir=tmp_cache)
    get_candles("ETH", "1h", "2025-01-01", "2025-01-03",
                fetcher=fetch, cache_dir=tmp_cache)
    removed = clear_cache(symbol="BTC", cache_dir=tmp_cache)
    assert removed == 1
    assert not cache_path("BTC", "1h", tmp_cache).exists()
    assert cache_path("ETH", "1h", tmp_cache).exists()


def test_clear_cache_everything(tmp_cache):
    fetch = _mock_fetcher(_candles_df("2025-01-01", 100))
    get_candles("BTC", "1h", "2025-01-01", "2025-01-03",
                fetcher=fetch, cache_dir=tmp_cache)
    get_candles("ETH", "4h", "2025-01-01", "2025-01-03",
                fetcher=fetch, cache_dir=tmp_cache)
    removed = clear_cache(cache_dir=tmp_cache)
    assert removed == 2


def test_clear_cache_missing_dir_returns_zero(tmp_path: Path):
    assert clear_cache(cache_dir=tmp_path / "nope") == 0


# ---------------------------------------------------------------------------
# prefetch_universe
# ---------------------------------------------------------------------------

def test_prefetch_universe_populates_all_cells(tmp_cache):
    all_data = _candles_df("2024-01-01", 2000, freq="1h")
    fetch = _mock_fetcher(all_data)
    tickers = ["BTC", "ETH", "SOL"]
    result = prefetch_universe(
        tickers, timeframes=["1h"], lookback_days=10,
        fetcher=fetch, cache_dir=tmp_cache,
    )
    for t in tickers:
        assert "1h" in result[t]


def test_prefetch_universe_swallows_per_ticker_failures(tmp_cache):
    def flaky(symbol, timeframe, start_ms, end_ms):
        if symbol == "BAD":
            raise RuntimeError("no data for you")
        # Generate data covering exactly the requested range
        start = pd.Timestamp(start_ms, unit="ms", tz="UTC")
        end = pd.Timestamp(end_ms, unit="ms", tz="UTC")
        ts = pd.date_range(start, end, freq="1h", tz="UTC")
        closes = 100.0 + np.arange(len(ts))
        return pd.DataFrame({
            "timestamp": ts, "open": closes,
            "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": [1000.0] * len(ts),
        })
    result = prefetch_universe(
        ["GOOD", "BAD"], timeframes=["1h"], lookback_days=10,
        fetcher=flaky, cache_dir=tmp_cache,
    )
    assert len(result["GOOD"]["1h"]) > 0
    assert result["BAD"]["1h"].empty

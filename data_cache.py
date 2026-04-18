"""
data_cache.py
Local disk cache for OHLCV candles.

Avoids re-fetching historical bars on every run. Does incremental fetches
to extend the cache range, dedupes across runs, and returns a clean
in-memory DataFrame for the requested window.

Cache layout:
    cache/candles/{SYMBOL}_{TIMEFRAME}.csv
    cache/candles/meta.json       # not used yet; reserved for schema versioning

CSV is used instead of parquet to keep dependencies thin (pyarrow isn't
in requirements). 21 tickers × 3 timeframes × 1 year ≈ 3M rows total
across all files; CSV is plenty fast for the sizes we care about. If
scale becomes an issue, swap to parquet via a single function change.

Public API:
    get_candles(symbol, timeframe, start, end, *, force_refresh=False) -> DataFrame
    prefetch_universe(tickers, timeframes, lookback_days) -> dict
    clear_cache(symbol=None, timeframe=None)

A fetcher callable can be injected for testing, so the core cache logic
is testable without network access.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd

from hyperliquid_feed import fetch_historical_binance, fetch_historical_hl

logger = logging.getLogger(__name__)


CACHE_DIR = Path("cache/candles")
TIMESTAMP_COL = "timestamp"
OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]

Fetcher = Callable[[str, str, int, int], pd.DataFrame]


# ---------------------------------------------------------------------------
# Path + I/O
# ---------------------------------------------------------------------------

def cache_path(symbol: str, timeframe: str, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"{symbol.upper()}_{timeframe}.csv"


def load_cached(symbol: str, timeframe: str, cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Load a cached candle CSV. Empty DataFrame if no cache file."""
    path = cache_path(symbol, timeframe, cache_dir)
    if not path.exists():
        return pd.DataFrame(columns=OHLCV_COLS)
    df = pd.read_csv(path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df.sort_values(TIMESTAMP_COL).reset_index(drop=True)


def save_cache(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    cache_dir: Path = CACHE_DIR,
) -> None:
    """Atomic-ish cache write (write to tmp, rename)."""
    if df.empty:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(symbol, timeframe, cache_dir)
    tmp = path.with_suffix(".csv.tmp")
    df[OHLCV_COLS].to_csv(tmp, index=False)
    tmp.replace(path)


def clear_cache(
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    cache_dir: Path = CACHE_DIR,
) -> int:
    """
    Delete cache entries. Returns number of files removed.

    - clear_cache() removes everything
    - clear_cache(symbol="BTC") removes all timeframes for BTC
    - clear_cache(symbol="BTC", timeframe="15m") removes the one file
    """
    if not cache_dir.exists():
        return 0
    pattern = ""
    if symbol:
        pattern += symbol.upper() + "_"
    else:
        pattern += "*_"
    pattern += (timeframe or "*") + ".csv"
    removed = 0
    for p in cache_dir.glob(pattern):
        p.unlink()
        removed += 1
    return removed


# ---------------------------------------------------------------------------
# Default fetcher (HL primary, Binance fallback)
# ---------------------------------------------------------------------------

def default_fetcher(symbol: str, timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Try Hyperliquid first, fall back to Binance if HL returns empty.

    Binance fallback is best-effort — some HL-listed tickers (mid-caps,
    HL-exclusive listings) aren't on Binance at all. In that case we
    return HL's empty result rather than raising, so callers can degrade
    to "no data" instead of crashing.
    """
    df = fetch_historical_hl(symbol, timeframe, start_ms, end_ms)
    if not df.empty:
        return df
    logger.info("HL empty for %s %s — trying Binance", symbol, timeframe)
    try:
        return fetch_historical_binance(symbol, timeframe, start_ms, end_ms)
    except ValueError as exc:
        logger.info("Binance fallback unavailable for %s: %s", symbol, exc)
        return pd.DataFrame(columns=OHLCV_COLS)


# ---------------------------------------------------------------------------
# Core get_candles
# ---------------------------------------------------------------------------

def get_candles(
    symbol: str,
    timeframe: str,
    start: Union[str, pd.Timestamp, datetime],
    end: Union[str, pd.Timestamp, datetime],
    *,
    force_refresh: bool = False,
    fetcher: Optional[Fetcher] = None,
    cache_dir: Path = CACHE_DIR,
) -> pd.DataFrame:
    """
    Return candles for [start, end]. Uses cache; incrementally fetches
    only what's missing. Dedupes on timestamp; sorts; persists updated
    cache back to disk.

    Args:
        fetcher: injectable for tests. Defaults to HL primary / Binance
                 fallback. Signature: (symbol, timeframe, start_ms, end_ms) -> DataFrame.
    """
    start_ts = _to_utc_ts(start)
    end_ts = _to_utc_ts(end)
    if start_ts >= end_ts:
        raise ValueError(f"start ({start_ts}) must be before end ({end_ts})")

    fetch = fetcher or default_fetcher
    cached = pd.DataFrame(columns=OHLCV_COLS) if force_refresh else load_cached(symbol, timeframe, cache_dir)

    if cached.empty:
        new_data = fetch(symbol, timeframe, _to_ms(start_ts), _to_ms(end_ts))
        if not new_data.empty:
            save_cache(new_data, symbol, timeframe, cache_dir)
        return _slice(new_data, start_ts, end_ts)

    cache_start = cached[TIMESTAMP_COL].min()
    cache_end = cached[TIMESTAMP_COL].max()

    # Fetch any gap before the cache
    pieces: list[pd.DataFrame] = [cached]
    if start_ts < cache_start:
        gap = fetch(symbol, timeframe, _to_ms(start_ts), _to_ms(cache_start))
        if not gap.empty:
            pieces.append(gap)

    # Fetch any gap after the cache
    if end_ts > cache_end:
        gap = fetch(symbol, timeframe, _to_ms(cache_end), _to_ms(end_ts))
        if not gap.empty:
            pieces.append(gap)

    merged = (
        pd.concat(pieces, ignore_index=True)
        .drop_duplicates(subset=[TIMESTAMP_COL])
        .sort_values(TIMESTAMP_COL)
        .reset_index(drop=True)
    )
    # Normalize timestamp dtype after concat
    merged[TIMESTAMP_COL] = pd.to_datetime(merged[TIMESTAMP_COL], utc=True)

    if len(merged) > len(cached):
        save_cache(merged, symbol, timeframe, cache_dir)

    return _slice(merged, start_ts, end_ts)


# ---------------------------------------------------------------------------
# Universe-scale prefetch
# ---------------------------------------------------------------------------

def prefetch_universe(
    tickers: list[str],
    timeframes: list[str] = ("1h", "4h", "15m"),
    lookback_days: int = 180,
    *,
    fetcher: Optional[Fetcher] = None,
    cache_dir: Path = CACHE_DIR,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Populate the cache for every (ticker, timeframe) pair and return the
    result as a nested dict.

    Used by the S/R CLI and by validation runs to pay the fetch cost once.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for sym in tickers:
        result[sym] = {}
        for tf in timeframes:
            try:
                df = get_candles(sym, tf, start, end, fetcher=fetcher, cache_dir=cache_dir)
                result[sym][tf] = df
                logger.info("prefetch %s %s: %d bars", sym, tf, len(df))
            except Exception as exc:
                logger.warning("prefetch failed for %s %s: %s", sym, tf, exc)
                result[sym][tf] = pd.DataFrame(columns=OHLCV_COLS)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_utc_ts(v: Union[str, pd.Timestamp, datetime]) -> pd.Timestamp:
    ts = pd.Timestamp(v)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _to_ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def _slice(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    ts = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    mask = (ts >= start_ts) & (ts <= end_ts)
    return df.loc[mask].reset_index(drop=True)

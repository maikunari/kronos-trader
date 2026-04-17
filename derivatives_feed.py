"""
derivatives_feed.py
Fetches historical funding rates from Hyperliquid for backtesting.

Hyperliquid pays funding every 1 hour.
The fundingRate field is a per-hour rate (e.g. 0.0005 = 0.05%/hr).
"""

import logging
import time
from datetime import datetime, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)

HL_REST_URL = "https://api.hyperliquid.xyz/info"


def fetch_funding_history(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Fetch historical hourly funding rates from Hyperliquid.

    Returns DataFrame with [timestamp, funding_rate] sorted oldest → newest.
    funding_rate: per-hour rate as float (e.g. 0.0005 = 0.05%/hr)
    """
    all_records = []
    cursor = start_ms

    logger.info(
        f"Fetching {symbol} funding history: {_ts_str(start_ms)} → {_ts_str(end_ms)}"
    )

    while cursor < end_ms:
        payload = {
            "type": "fundingHistory",
            "coin": symbol,
            "startTime": cursor,
        }
        try:
            resp = requests.post(HL_REST_URL, json=payload, timeout=15)
            resp.raise_for_status()
            records = resp.json()
        except Exception as e:
            logger.error(f"Hyperliquid funding fetch error: {e}")
            break

        if not records:
            break

        # Filter to our window
        records = [r for r in records if r["time"] <= end_ms]
        if not records:
            break

        all_records.extend(records)
        last_t = records[-1]["time"]

        # Stop if we've reached the end or no progress
        if last_t >= end_ms or last_t <= cursor:
            break

        cursor = last_t + 1
        time.sleep(0.1)

    if not all_records:
        logger.warning(f"No funding history returned for {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = (
        df[["timestamp", "funding_rate"]]
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )
    logger.info(f"Fetched {len(df)} funding records for {symbol}")
    return df


def fetch_funding_for_backtest(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Convenience wrapper using ISO date strings."""
    start_ms = int(
        datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).timestamp() * 1000
    )
    end_ms = int(
        datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).timestamp() * 1000
    )
    return fetch_funding_history(symbol, start_ms, end_ms)


def _ts_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

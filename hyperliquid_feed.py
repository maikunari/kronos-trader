"""
hyperliquid_feed.py
Live WebSocket OHLCV feed, liquidation event stream, and historical data
fetcher for Hyperliquid. Falls back to Binance for historical data if needed.
"""

import bisect
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Deque, List, Optional

import pandas as pd
import requests
import websocket

from microstructure import LiquidationCluster

logger = logging.getLogger(__name__)

# Hyperliquid endpoints
HL_REST_URL = "https://api.hyperliquid.xyz/info"
HL_WS_URL = "wss://api.hyperliquid.xyz/ws"
BINANCE_REST_URL = "https://api.binance.com/api/v3/klines"

# Timeframe to milliseconds
TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

# Binance symbol mapping
BINANCE_SYMBOL_MAP = {
    "SOL": "SOLUSDT",
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "AVAX": "AVAXUSDT",
    "LINK": "LINKUSDT",
    "XRP": "XRPUSDT",
}


def fetch_historical_hl(symbol: str, timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch historical OHLCV candles from Hyperliquid REST API."""
    interval_ms = TIMEFRAME_MS[timeframe]
    all_candles = []
    cursor = start_ms

    logger.info(f"Fetching {symbol} {timeframe} from Hyperliquid ({_ts_str(start_ms)} → {_ts_str(end_ms)})")

    while cursor < end_ms:
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": timeframe,
                "startTime": cursor,
                "endTime": min(cursor + interval_ms * 500, end_ms),
            },
        }
        try:
            resp = requests.post(HL_REST_URL, json=payload, timeout=15)
            resp.raise_for_status()
            candles = resp.json()
        except Exception as e:
            logger.error(f"Hyperliquid fetch error: {e}")
            break

        if not candles:
            break

        all_candles.extend(candles)
        last_t = candles[-1]["t"]
        cursor = last_t + interval_ms

        if len(candles) < 500:
            break

        time.sleep(0.1)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles)
    df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": float}
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Fetched {len(df)} candles from Hyperliquid")
    return df


def fetch_historical_binance(symbol: str, timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fallback: fetch historical OHLCV from Binance."""
    binance_sym = BINANCE_SYMBOL_MAP.get(symbol.upper())
    if not binance_sym:
        raise ValueError(f"No Binance mapping for symbol: {symbol}")

    all_candles = []
    cursor = start_ms
    logger.info(f"Fetching {binance_sym} {timeframe} from Binance ({_ts_str(start_ms)} → {_ts_str(end_ms)})")

    while cursor < end_ms:
        params = {
            "symbol": binance_sym,
            "interval": timeframe,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        try:
            resp = requests.get(BINANCE_REST_URL, params=params, timeout=15)
            resp.raise_for_status()
            candles = resp.json()
        except Exception as e:
            logger.error(f"Binance fetch error: {e}")
            break

        if not candles:
            break

        all_candles.extend(candles)
        cursor = candles[-1][0] + TIMEFRAME_MS[timeframe]

        if len(candles) < 1000:
            break

        time.sleep(0.05)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": float}
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Fetched {len(df)} candles from Binance")
    return df


def fetch_historical(symbol: str, timeframe: str, start_date: str, end_date: str, source: str = "hyperliquid") -> pd.DataFrame:
    """Fetch historical OHLCV, trying Hyperliquid first then Binance."""
    start_ms = int(datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).timestamp() * 1000)

    if source == "hyperliquid":
        df = fetch_historical_hl(symbol, timeframe, start_ms, end_ms)
        if df.empty:
            logger.warning("Hyperliquid returned no data, falling back to Binance")
            df = fetch_historical_binance(symbol, timeframe, start_ms, end_ms)
    else:
        df = fetch_historical_binance(symbol, timeframe, start_ms, end_ms)

    if df.empty:
        raise RuntimeError(f"Could not fetch historical data for {symbol}/{timeframe}")

    return df


class LiveFeed:
    """WebSocket-based live candle feed from Hyperliquid."""

    def __init__(self, symbol: str, timeframe: str, on_candle: Callable[[dict], None]):
        self.symbol = symbol
        self.timeframe = timeframe
        self.on_candle = on_candle
        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"LiveFeed started: {self.symbol} {self.timeframe}")

    def stop(self):
        self._running = False
        if self._ws:
            self._ws.close()
        logger.info("LiveFeed stopped")

    def _run(self):
        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    HL_WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            if self._running:
                logger.info("Reconnecting in 5s...")
                time.sleep(5)

    def _on_open(self, ws):
        sub = {
            "method": "subscribe",
            "subscription": {
                "type": "candle",
                "coin": self.symbol,
                "interval": self.timeframe,
            },
        }
        ws.send(json.dumps(sub))
        logger.info(f"Subscribed to {self.symbol}/{self.timeframe} candles")

    def _on_message(self, ws, raw):
        try:
            msg = json.loads(raw)
            if msg.get("channel") == "candle":
                data = msg["data"]
                candle = {
                    "timestamp": data["t"],
                    "open": float(data["o"]),
                    "high": float(data["h"]),
                    "low": float(data["l"]),
                    "close": float(data["c"]),
                    "volume": float(data["v"]),
                    "is_closed": data.get("T", 0) < data["t"] + TIMEFRAME_MS.get(self.timeframe, 60000),
                }
                self.on_candle(candle)
        except Exception as e:
            logger.error(f"Message parse error: {e}")

    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws, code, msg):
        logger.warning(f"WebSocket closed: {code} {msg}")


def _ts_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


# ------------------------------------------------------------------
# Liquidation event stream + rolling cluster buffer
# ------------------------------------------------------------------

@dataclass
class LiquidationEvent:
    """A single liquidation on Hyperliquid."""
    timestamp_ms: int
    symbol: str
    price: float
    notional_usd: float
    side: str   # "long" = a long was liquidated (forced sell); "short" = short liquidated


def parse_hl_trade_liquidations(msg: dict, symbol: str) -> list[LiquidationEvent]:
    """
    Extract liquidation events from a Hyperliquid 'trades' websocket message.

    HL publishes trades via the 'trades' channel; liquidations appear as
    individual trades flagged with liquidation metadata. Schema has varied
    across HL API versions — this parser accepts either:
      - trade['liquidation'] = True  (bool flag)
      - trade['tag'] contains 'liq'  (string tag)
      - trade['users'] present with liquidator/liquidated pair

    Returns [] if the message is not a trades frame or contains no
    liquidations — this is the common case and must not log noisily.

    Side convention: HL 'A' = ask = taker sold (long was liquidated);
    'B' = bid = taker bought (short was liquidated).
    """
    if not isinstance(msg, dict) or msg.get("channel") != "trades":
        return []
    trades = msg.get("data") or []
    if not isinstance(trades, list):
        return []

    events: list[LiquidationEvent] = []
    for t in trades:
        if not isinstance(t, dict):
            continue
        if t.get("coin") and t["coin"] != symbol:
            continue
        if not _is_liquidation(t):
            continue
        try:
            px = float(t["px"])
            sz = float(t["sz"])
            ts = int(t.get("time") or t.get("t") or 0)
        except (KeyError, TypeError, ValueError):
            continue
        if px <= 0 or sz <= 0:
            continue
        raw_side = t.get("side", "")
        side = "long" if raw_side == "A" else "short" if raw_side == "B" else ""
        if not side:
            continue
        events.append(LiquidationEvent(
            timestamp_ms=ts, symbol=t.get("coin", symbol),
            price=px, notional_usd=px * sz, side=side,
        ))
    return events


def _is_liquidation(trade: dict) -> bool:
    if trade.get("liquidation") is True:
        return True
    tag = trade.get("tag")
    if isinstance(tag, str) and "liq" in tag.lower():
        return True
    # Some HL versions tuck the liquidated-user info in a 'users' array
    users = trade.get("users")
    if isinstance(users, list) and len(users) >= 2:
        # best-effort: a 'liquidation' hint in the users block
        for u in users:
            if isinstance(u, dict) and u.get("liquidation"):
                return True
    return False


class LiquidationClusterBuffer:
    """
    Rolling buffer of liquidation events → price-bucketed LiquidationCluster list.

    Buckets events by round price levels so a small number of clusters emerge
    from a large stream of trades. The bucket width is set as a fraction of
    current price (default 0.1%).

    Clusters are classified relative to `current_price` at read time:
      - bucket below current -> 'long' (longs would be liquidated on a drop)
      - bucket above current -> 'short'

    This complements coinglass_client (single-exchange native data vs
    cross-exchange aggregate).
    """

    def __init__(self, window_seconds: int = 3600, bucket_pct: float = 0.001) -> None:
        self.window_seconds = window_seconds
        self.bucket_pct = bucket_pct
        self._events: Deque[LiquidationEvent] = deque()

    def add(self, event: LiquidationEvent) -> None:
        self._events.append(event)
        self._expire(now_ms=event.timestamp_ms)

    def _expire(self, now_ms: int) -> None:
        cutoff = now_ms - self.window_seconds * 1000
        while self._events and self._events[0].timestamp_ms < cutoff:
            self._events.popleft()

    def get_clusters(self, current_price: float, now_ms: Optional[int] = None) -> list[LiquidationCluster]:
        if current_price <= 0:
            return []
        if now_ms is not None:
            self._expire(now_ms)
        if not self._events:
            return []
        bucket_size = current_price * self.bucket_pct
        if bucket_size <= 0:
            return []

        # Aggregate notional by rounded bucket price
        totals: dict[float, float] = {}
        for ev in self._events:
            bucket = round(ev.price / bucket_size) * bucket_size
            totals[bucket] = totals.get(bucket, 0.0) + ev.notional_usd

        clusters: list[LiquidationCluster] = []
        for price, notional in totals.items():
            if price <= 0 or notional <= 0:
                continue
            side = "long" if price < current_price else "short"
            clusters.append(LiquidationCluster(price=price, volume=notional, side=side))
        return clusters


class LiquidationFeed:
    """WebSocket subscription to Hyperliquid trades, surfacing liquidations only."""

    def __init__(
        self,
        symbol: str,
        on_liquidation: Callable[[LiquidationEvent], None],
    ) -> None:
        self.symbol = symbol
        self.on_liquidation = on_liquidation
        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("LiquidationFeed started: %s", self.symbol)

    def stop(self) -> None:
        self._running = False
        if self._ws:
            self._ws.close()
        logger.info("LiquidationFeed stopped")

    def _run(self) -> None:
        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    HL_WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=lambda _w, e: logger.error("LiquidationFeed WS error: %s", e),
                    on_close=lambda _w, c, m: logger.warning("LiquidationFeed WS closed: %s %s", c, m),
                )
                self._ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                logger.error("LiquidationFeed WS exception: %s", e)
            if self._running:
                time.sleep(5)

    def _on_open(self, ws) -> None:
        sub = {"method": "subscribe", "subscription": {"type": "trades", "coin": self.symbol}}
        ws.send(json.dumps(sub))
        logger.info("LiquidationFeed subscribed to %s trades", self.symbol)

    def _on_message(self, _ws, raw) -> None:
        try:
            msg = json.loads(raw)
        except ValueError:
            return
        for ev in parse_hl_trade_liquidations(msg, self.symbol):
            try:
                self.on_liquidation(ev)
            except Exception as exc:
                logger.error("on_liquidation callback raised: %s", exc)

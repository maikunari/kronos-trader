"""Unit tests for the liquidation parser and rolling cluster buffer."""
from __future__ import annotations

import pytest

from hyperliquid_feed import (
    LiquidationClusterBuffer,
    LiquidationEvent,
    parse_hl_trade_liquidations,
)


# --- parse_hl_trade_liquidations ---------------------------------------------

def _trades_msg(trades: list[dict]) -> dict:
    return {"channel": "trades", "data": trades}


def test_parses_flag_liquidation():
    msg = _trades_msg([
        {"coin": "BTC", "side": "A", "px": "50000", "sz": "0.5",
         "time": 1_700_000_000_000, "liquidation": True},
    ])
    events = parse_hl_trade_liquidations(msg, "BTC")
    assert len(events) == 1
    ev = events[0]
    assert ev.symbol == "BTC"
    assert ev.price == 50_000
    assert ev.notional_usd == pytest.approx(25_000)
    assert ev.side == "long"  # side=A = taker sold = long liquidated


def test_parses_bid_side_as_short_liquidation():
    msg = _trades_msg([
        {"coin": "ETH", "side": "B", "px": "2500", "sz": "1.0",
         "time": 1_700_000_000_000, "liquidation": True},
    ])
    events = parse_hl_trade_liquidations(msg, "ETH")
    assert len(events) == 1
    assert events[0].side == "short"


def test_parses_tag_liquidation():
    """Older HL versions tagged with a 'tag' string containing 'liq'."""
    msg = _trades_msg([
        {"coin": "BTC", "side": "A", "px": "50000", "sz": "0.5",
         "time": 1_700_000_000_000, "tag": "liquidation"},
    ])
    events = parse_hl_trade_liquidations(msg, "BTC")
    assert len(events) == 1


def test_ignores_non_liquidation_trades():
    msg = _trades_msg([
        {"coin": "BTC", "side": "A", "px": "50000", "sz": "0.5", "time": 1},
    ])
    assert parse_hl_trade_liquidations(msg, "BTC") == []


def test_ignores_wrong_symbol():
    msg = _trades_msg([
        {"coin": "ETH", "side": "A", "px": "2500", "sz": "1",
         "time": 1, "liquidation": True},
    ])
    assert parse_hl_trade_liquidations(msg, "BTC") == []


def test_non_trades_channel_returns_empty():
    assert parse_hl_trade_liquidations({"channel": "candle", "data": {}}, "BTC") == []


def test_invalid_message_structure_returns_empty():
    assert parse_hl_trade_liquidations("not-a-dict", "BTC") == []
    assert parse_hl_trade_liquidations({"channel": "trades", "data": "oops"}, "BTC") == []


def test_rejects_bad_numeric_fields():
    msg = _trades_msg([
        {"coin": "BTC", "side": "A", "px": "oops", "sz": "1",
         "time": 1, "liquidation": True},
        {"coin": "BTC", "side": "A", "px": "-1", "sz": "1",
         "time": 1, "liquidation": True},
    ])
    assert parse_hl_trade_liquidations(msg, "BTC") == []


def test_rejects_unknown_side():
    msg = _trades_msg([
        {"coin": "BTC", "side": "X", "px": "50000", "sz": "1",
         "time": 1, "liquidation": True},
    ])
    assert parse_hl_trade_liquidations(msg, "BTC") == []


# --- LiquidationClusterBuffer -------------------------------------------------

def _ev(px: float, usd: float, side_hl: str, t: int = 1_700_000_000_000):
    return LiquidationEvent(
        timestamp_ms=t, symbol="BTC", price=px,
        notional_usd=usd, side=("long" if side_hl == "A" else "short"),
    )


def test_buffer_aggregates_events_by_price_bucket():
    buf = LiquidationClusterBuffer(window_seconds=3600, bucket_pct=0.001)
    # current_price 50_000 -> bucket width = 50. Two events at 50,040 round to same bucket.
    buf.add(_ev(50_040, 100_000, "A"))
    buf.add(_ev(50_010, 50_000, "A"))
    buf.add(_ev(51_000, 200_000, "B"))
    clusters = buf.get_clusters(current_price=50_000)
    by_price = {c.price: c for c in clusters}
    # 50_040 rounds to 50_050 (nearest 50); 50_010 rounds to 50_000; 51_000 -> 51_000
    assert sum(c.volume for c in clusters) == 350_000
    # The bucket at or below 50,000 should be classified long; above short.
    above = [c for c in clusters if c.price > 50_000]
    below = [c for c in clusters if c.price < 50_000]
    assert all(c.side == "short" for c in above)
    assert all(c.side == "long" for c in below)


def test_buffer_expires_old_events():
    buf = LiquidationClusterBuffer(window_seconds=60, bucket_pct=0.001)
    # Event at t=0 -> expires (older than 60s at t=120s)
    buf.add(_ev(50_000, 100_000, "A", t=1_000_000))
    # Event at t=90s -> within window at t=120s
    buf.add(_ev(50_100, 50_000, "A", t=1_000_000 + 90 * 1000))
    # Event at t=120s -> within window
    buf.add(_ev(50_200, 30_000, "A", t=1_000_000 + 120 * 1000))
    clusters = buf.get_clusters(current_price=50_000, now_ms=1_000_000 + 120 * 1000)
    total = sum(c.volume for c in clusters)
    assert total == 80_000   # 50k + 30k, the 100k at t=0 expired


def test_buffer_empty_returns_no_clusters():
    buf = LiquidationClusterBuffer()
    assert buf.get_clusters(current_price=50_000) == []


def test_buffer_ignores_zero_price():
    buf = LiquidationClusterBuffer()
    buf.add(_ev(50_000, 100_000, "A"))
    assert buf.get_clusters(current_price=0) == []

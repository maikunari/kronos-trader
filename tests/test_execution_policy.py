"""Unit tests for execution_policy.py — pure logic, no exchange calls."""
from __future__ import annotations

import pytest

from execution_policy import (
    BookSnapshot,
    ExecutionPolicy,
    OrderDecision,
    OrderIntent,
    SpreadHistory,
)


def _tight_book(mid: float = 100.0, spread: float = 0.02, depth_size: float = 5.0) -> BookSnapshot:
    bid = mid - spread / 2
    ask = mid + spread / 2
    return BookSnapshot(
        bid=bid, ask=ask,
        bid_depth=[(bid, depth_size)] * 5,
        ask_depth=[(ask, depth_size)] * 5,
    )


# --- BookSnapshot properties --------------------------------------------------

def test_book_properties_are_consistent():
    b = _tight_book(mid=100.0, spread=0.10)
    assert b.mid == pytest.approx(100.0)
    assert b.spread == pytest.approx(0.10)
    assert b.spread_pct == pytest.approx(0.001)
    assert b.top_n_depth("bid", n=5) == pytest.approx(25.0)


# --- SpreadHistory ------------------------------------------------------------

def test_spread_history_median_requires_min_samples():
    h = SpreadHistory()
    for _ in range(20):
        h.add(0.0010)
    assert h.median is None   # not enough samples yet
    for _ in range(20):
        h.add(0.0010)
    assert h.median == pytest.approx(0.0010)


# --- post_only_first happy path ----------------------------------------------

def test_post_only_selected_on_first_attempt():
    p = ExecutionPolicy(post_only_first=True, taker_timeout_ms=3_000)
    intent = OrderIntent(side="buy", total_size=1.0, intent_price=100.0)
    decision = p.decide_entry(intent, _tight_book(), elapsed_ms=0)
    assert decision.order_type == "post_only"
    assert decision.price is not None
    # Post-only buy price should not cross ask
    assert decision.price < _tight_book().ask


def test_taker_fallback_after_timeout():
    p = ExecutionPolicy(taker_timeout_ms=3_000, enable_taker_fallback=True)
    intent = OrderIntent(side="buy", total_size=1.0, intent_price=100.0)
    decision = p.decide_entry(intent, _tight_book(), elapsed_ms=3_500)
    assert decision.order_type == "market"
    assert decision.reason.startswith("taker_fallback")


def test_taker_fallback_disabled_keeps_post_only():
    p = ExecutionPolicy(taker_timeout_ms=3_000, enable_taker_fallback=False)
    intent = OrderIntent(side="buy", total_size=1.0, intent_price=100.0)
    decision = p.decide_entry(intent, _tight_book(), elapsed_ms=30_000)
    assert decision.order_type == "post_only"


# --- Retreat on adverse move --------------------------------------------------

def test_retreat_cancels_when_mid_moved_adversely():
    p = ExecutionPolicy(retreat_bps=10.0)
    # intent formed at 100.0; book mid now 100.2 -> 20bps adverse for a buy
    book = _tight_book(mid=100.2)
    intent = OrderIntent(side="buy", total_size=1.0, intent_price=100.0)
    decision = p.decide_entry(intent, book, elapsed_ms=500)
    assert decision.order_type == "skip"
    assert decision.reason.startswith("retreat")


def test_favorable_move_does_not_trigger_retreat():
    p = ExecutionPolicy(retreat_bps=10.0)
    # Mid dropped 0.2 — favorable for buy intent
    book = _tight_book(mid=99.8)
    intent = OrderIntent(side="buy", total_size=1.0, intent_price=100.0)
    decision = p.decide_entry(intent, book, elapsed_ms=500)
    assert decision.order_type == "post_only"


# --- Spread anomaly skip ------------------------------------------------------

def test_spread_anomaly_skip_when_spread_blows_out():
    p = ExecutionPolicy(spread_anomaly_mult=5.0)
    # Seed history with tight spreads
    for _ in range(50):
        p.spread_history.add(0.0002)  # 2bps
    anomaly_book = BookSnapshot(bid=99.9, ask=100.2,  # 30bps spread
                                 bid_depth=[(99.9, 5)] * 5, ask_depth=[(100.2, 5)] * 5)
    intent = OrderIntent(side="buy", total_size=1.0, intent_price=100.0)
    decision = p.decide_entry(intent, anomaly_book)
    assert decision.order_type == "skip"
    assert decision.reason.startswith("spread_anomaly")


# --- Iceberg chunking ---------------------------------------------------------

def test_iceberg_single_chunk_when_small_enough():
    p = ExecutionPolicy(max_book_fraction=0.30)
    # top-5 ask depth = 25; 30% = 7.5
    book = _tight_book(depth_size=5.0)
    chunks = p.iceberg_chunks(total_size=5.0, book=book, side="buy")
    assert chunks == [5.0]


def test_iceberg_splits_into_multiple_chunks():
    p = ExecutionPolicy(max_book_fraction=0.30)
    book = _tight_book(depth_size=5.0)   # top-5 = 25; max chunk = 7.5
    chunks = p.iceberg_chunks(total_size=20.0, book=book, side="buy")
    assert len(chunks) >= 3
    assert sum(chunks) == pytest.approx(20.0)
    assert max(chunks) <= 7.5 + 1e-9


def test_iceberg_zero_size_returns_empty():
    p = ExecutionPolicy()
    chunks = p.iceberg_chunks(0.0, _tight_book(), side="buy")
    assert chunks == []


# --- decide_entry size-cap ----------------------------------------------------

def test_decide_entry_size_is_capped_by_book_depth():
    p = ExecutionPolicy(max_book_fraction=0.30)
    book = _tight_book(depth_size=5.0)   # max child = 7.5
    intent = OrderIntent(side="buy", total_size=20.0, intent_price=100.0)
    decision = p.decide_entry(intent, book, elapsed_ms=0)
    assert decision.size <= 7.5 + 1e-9

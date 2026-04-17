"""Tests for Executor.place_entry — paper mode only (no exchange calls)."""
from __future__ import annotations

import pytest

from execution_policy import BookSnapshot, ExecutionPolicy, OrderIntent
from executor import Executor


def _book(mid: float = 100.0, spread: float = 0.02, depth: float = 10.0) -> BookSnapshot:
    bid = mid - spread / 2
    ask = mid + spread / 2
    return BookSnapshot(
        bid=bid, ask=ask,
        bid_depth=[(bid, depth)] * 5,
        ask_depth=[(ask, depth)] * 5,
    )


def _executor(policy: ExecutionPolicy) -> Executor:
    return Executor(mode="paper", execution_policy=policy)


def test_missing_policy_raises():
    ex = Executor(mode="paper")
    with pytest.raises(RuntimeError):
        ex.place_entry(
            "BTC",
            OrderIntent(side="buy", total_size=1.0, intent_price=100.0),
            _book(),
        )


def test_post_only_paper_fills_at_policy_price():
    """A post-only decision should be simulated as filling at our passive price."""
    ex = _executor(ExecutionPolicy(post_only_first=True))
    intent = OrderIntent(side="buy", total_size=1.0, intent_price=100.0)
    book = _book(mid=100.0, spread=0.02)
    result = ex.place_entry("BTC", intent, book, elapsed_ms=0)
    assert result.success
    assert result.filled_price == pytest.approx(100.0 - 0.01)   # mid - half-spread
    assert result.side == "long"


def test_taker_fallback_paper_fills_at_far_side():
    """Market fallback should simulate crossing the ask for a buy."""
    ex = _executor(ExecutionPolicy(post_only_first=True, taker_timeout_ms=1000))
    intent = OrderIntent(side="buy", total_size=1.0, intent_price=100.0)
    book = _book(mid=100.0, spread=0.02)
    result = ex.place_entry("BTC", intent, book, elapsed_ms=2000)
    assert result.success
    assert result.filled_price == pytest.approx(book.ask)


def test_skip_returns_unsuccessful_with_reason():
    ex = _executor(ExecutionPolicy(retreat_bps=10.0))
    # Mid moved 20bps against a buy -> retreat skip
    intent = OrderIntent(side="buy", total_size=1.0, intent_price=100.0)
    book = _book(mid=100.2, spread=0.02)
    result = ex.place_entry("BTC", intent, book, elapsed_ms=500)
    assert not result.success
    assert "retreat" in (result.error or "")


def test_short_side_paper_fill_at_bid_on_taker():
    ex = _executor(ExecutionPolicy(post_only_first=True, taker_timeout_ms=1000))
    intent = OrderIntent(side="sell", total_size=1.0, intent_price=100.0)
    book = _book(mid=100.0, spread=0.02)
    result = ex.place_entry("BTC", intent, book, elapsed_ms=2000)
    assert result.success
    assert result.filled_price == pytest.approx(book.bid)
    assert result.side == "short"

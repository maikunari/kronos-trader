"""
execution_policy.py
Order-placement policy: post-only-first, timeout-to-taker, iceberg splitting,
anomaly skip. Pulled out of the executor so it's testable without an exchange.

Phase 1 motivation: short-TF sniping lives and dies on execution quality.
A taker fill on every entry costs ~3.5 bps per side and eats most of the
edge. Capturing HL maker rebates (or even flat maker fees) on entry makes
the strategy significantly more profitable in practice than in a naive
market-order backtest.

Policy state machine, per order intent:
  t=0        : submit post-only limit at mid ± half-spread
  t<timeout  : if mid drifted > retreat_bps, cancel+retreat
  t>=timeout : cross with a taker market order (if enabled)
  pre-flight : skip entirely if spread > spread_anomaly_mult × 30d median
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class BookSnapshot:
    bid: float
    ask: float
    bid_depth: list[tuple[float, float]] = field(default_factory=list)   # top-5 [(price, size), ...]
    ask_depth: list[tuple[float, float]] = field(default_factory=list)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        return self.spread / self.mid if self.mid > 0 else float("inf")

    def top_n_depth(self, side: str, n: int = 5) -> float:
        """Total size on the first n levels of `side` ('bid' or 'ask')."""
        depth = self.bid_depth if side == "bid" else self.ask_depth
        return float(sum(sz for _, sz in depth[:n]))


@dataclass
class OrderIntent:
    side: str             # "buy" | "sell"
    total_size: float     # base units
    intent_price: float   # price when intent was formed; used for retreat check


@dataclass
class OrderDecision:
    order_type: str       # "post_only" | "market" | "skip"
    price: Optional[float]
    size: float
    reason: str


# ------------------------------------------------------------------
# Spread history (for anomaly detection)
# ------------------------------------------------------------------

class SpreadHistory:
    """Rolling store of spread_pct samples for anomaly detection."""

    def __init__(self, max_samples: int = 10_000) -> None:
        self.max_samples = max_samples
        self._samples: list[float] = []

    def add(self, spread_pct: float) -> None:
        if spread_pct > 0:
            self._samples.append(spread_pct)
        if len(self._samples) > self.max_samples:
            self._samples = self._samples[-self.max_samples :]

    @property
    def median(self) -> Optional[float]:
        if len(self._samples) < 30:
            return None
        return statistics.median(self._samples)


# ------------------------------------------------------------------
# Policy
# ------------------------------------------------------------------

class ExecutionPolicy:
    def __init__(
        self,
        *,
        post_only_first: bool = True,
        taker_timeout_ms: int = 3_000,
        retreat_bps: float = 10.0,
        max_book_fraction: float = 0.30,
        spread_anomaly_mult: float = 5.0,
        enable_taker_fallback: bool = True,
    ) -> None:
        self.post_only_first = post_only_first
        self.taker_timeout_ms = taker_timeout_ms
        self.retreat_bps = retreat_bps
        self.max_book_fraction = max_book_fraction
        self.spread_anomaly_mult = spread_anomaly_mult
        self.enable_taker_fallback = enable_taker_fallback
        self.spread_history = SpreadHistory()

    # --- Anomaly skip ---------------------------------------------------------

    def should_skip_on_spread(self, book: BookSnapshot) -> Optional[str]:
        """Return a skip reason if the current spread is anomalously wide, else None."""
        if book.spread <= 0 or book.mid <= 0:
            return "invalid_book"
        median = self.spread_history.median
        if median is None:
            return None
        if book.spread_pct > self.spread_anomaly_mult * median:
            return f"spread_anomaly({book.spread_pct:.4%}>{self.spread_anomaly_mult}x{median:.4%})"
        return None

    # --- Decision -------------------------------------------------------------

    def decide_entry(
        self,
        intent: OrderIntent,
        book: BookSnapshot,
        elapsed_ms: int = 0,
    ) -> OrderDecision:
        """
        Decide the next action for an open order intent.

        `elapsed_ms` is the age of the intent. 0 means first attempt.
        """
        skip = self.should_skip_on_spread(book)
        if skip is not None:
            return OrderDecision(order_type="skip", price=None, size=0.0, reason=skip)

        # Retreat: if mid has moved adversely beyond threshold, cancel and wait.
        adverse_move = self._adverse_move_bps(intent, book)
        if adverse_move > self.retreat_bps:
            return OrderDecision(
                order_type="skip",
                price=None,
                size=0.0,
                reason=f"retreat(adverse={adverse_move:.1f}bps>{self.retreat_bps:.1f})",
            )

        # Taker fallback after timeout (only if enabled and we've tried maker)
        if (
            self.enable_taker_fallback
            and elapsed_ms >= self.taker_timeout_ms
            and self.post_only_first
        ):
            size = self._cap_by_book(intent.total_size, book, intent.side)
            return OrderDecision(
                order_type="market",
                price=None,
                size=size,
                reason=f"taker_fallback(elapsed={elapsed_ms}ms)",
            )

        # Post-only maker first
        if self.post_only_first:
            price = self._post_only_price(book, intent.side)
            size = self._cap_by_book(intent.total_size, book, intent.side)
            return OrderDecision(
                order_type="post_only",
                price=price,
                size=size,
                reason="post_only_first",
            )

        # If post-only is disabled, cross at market immediately
        size = self._cap_by_book(intent.total_size, book, intent.side)
        return OrderDecision(order_type="market", price=None, size=size, reason="market_always")

    # --- Iceberg --------------------------------------------------------------

    def iceberg_chunks(
        self,
        total_size: float,
        book: BookSnapshot,
        side: str,
    ) -> list[float]:
        """
        Split `total_size` into child orders no larger than max_book_fraction
        of the top-5 depth on the taking side (ask depth for buys, bid for sells).

        Returns [total_size] when the single order already fits.
        """
        book_side = "ask" if side == "buy" else "bid"
        depth = book.top_n_depth(book_side, n=5)
        if depth <= 0 or total_size <= 0:
            return [total_size] if total_size > 0 else []
        max_chunk = self.max_book_fraction * depth
        if max_chunk <= 0 or total_size <= max_chunk:
            return [total_size]

        chunks: list[float] = []
        remaining = total_size
        while remaining > max_chunk:
            chunks.append(max_chunk)
            remaining -= max_chunk
        if remaining > 0:
            chunks.append(remaining)
        return chunks

    # --- Internals ------------------------------------------------------------

    def _post_only_price(self, book: BookSnapshot, side: str) -> float:
        """Place post-only at mid ± half-spread, but don't cross."""
        mid = book.mid
        half = book.spread / 2.0
        # Sit one tick inside the book from best quote to earn maker without
        # risk of crossing on next tick: equivalent to mid - half for buy, mid + half for sell.
        if side == "buy":
            return round(mid - half, 10)
        return round(mid + half, 10)

    def _adverse_move_bps(self, intent: OrderIntent, book: BookSnapshot) -> float:
        """
        How much has mid moved *against* the intent, in bps, since intent formed?
        Positive = move adverse; 0 or negative = move favorable.
        """
        if intent.intent_price <= 0:
            return 0.0
        delta = (book.mid - intent.intent_price) / intent.intent_price
        bps = delta * 10_000
        return bps if intent.side == "buy" else -bps

    def _cap_by_book(self, size: float, book: BookSnapshot, side: str) -> float:
        """Clip a single child size to max_book_fraction of the taking side's depth."""
        book_side = "ask" if side == "buy" else "bid"
        depth = book.top_n_depth(book_side, n=5)
        if depth <= 0:
            return size
        cap = self.max_book_fraction * depth
        return min(size, cap)

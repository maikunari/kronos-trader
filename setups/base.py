"""
setups/base.py
Common types shared by every setup detector.

Protocol:
    class SetupDetector(Protocol):
        name: str
        def detect(self, ctx: MarketContext) -> Optional[Trigger]: ...

Every concrete detector (divergence, two-bar, diagonal, etc.) consumes a
MarketContext and optionally emits a Trigger. The validation framework
and (later) the live scanner are both agnostic of which detector produced
a given trigger — they only look at the `setup` label on the result.

MarketContext is the input bundle. Build one per (ticker, timeframe, as-of-bar)
via MarketContext.build(); it computes AO/RSI and optionally pulls S/R
zones on demand.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol, runtime_checkable

import pandas as pd

from indicators.awesome_oscillator import awesome_oscillator
from indicators.rsi import rsi as compute_rsi
from support_resistance import SRZone, detect_sr_zones

logger = logging.getLogger(__name__)


Direction = Literal["long", "short"]
Action = Literal["open_new", "add_on"]


# ---------------------------------------------------------------------------
# TP ladder
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TPLevel:
    """A single tier in a trade's take-profit ladder.

    fraction: share of target total size to close at this level. Entire
              ladder's fractions should sum to ≤ 1.0; residual fraction
              becomes a runner.
    source:   where this level came from ('sr_zone', 'atr_fallback',
              'fib_extension') — for observability.
    """
    price: float
    fraction: float
    source: str


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Trigger:
    """A setup's entry signal.

    Consumed by RiskManager (for approval/sizing) and then by position
    manager (to open a new position or add a leg to an existing one).
    """
    ticker: str
    timestamp: pd.Timestamp
    action: Action
    direction: Direction
    entry_price: float
    stop_price: float
    tp_ladder: tuple[TPLevel, ...]
    setup: str
    confidence: float
    size_fraction: float = 1.0                # of the intended total position
    components: dict = field(default_factory=dict)   # setup-specific metadata

    @property
    def rr_to_first_tp(self) -> float:
        """Risk:reward to the nearest TP level, or 0 if ladder is empty."""
        if not self.tp_ladder:
            return 0.0
        risk = abs(self.entry_price - self.stop_price)
        if risk <= 0:
            return 0.0
        reward = abs(self.tp_ladder[0].price - self.entry_price)
        return reward / risk


# ---------------------------------------------------------------------------
# MarketContext
# ---------------------------------------------------------------------------

@dataclass
class MarketContext:
    """Everything a setup needs to reason about a single bar on one ticker.

    Callers typically use `MarketContext.build(...)` rather than instantiating
    directly — the factory computes indicators and S/R once so multiple
    detectors can share them.
    """
    ticker: str
    timeframe: str
    candles: pd.DataFrame                     # primary TF, OHLCV + timestamp
    candles_htf: Optional[pd.DataFrame] = None  # higher TF for trend bias
    ao: Optional[pd.Series] = None            # computed if None at build-time
    rsi: Optional[pd.Series] = None
    sr_zones: Optional[list[SRZone]] = None
    timestamp: Optional[pd.Timestamp] = None  # "as-of" — defaults to last bar's ts

    @classmethod
    def build(
        cls,
        *,
        ticker: str,
        timeframe: str,
        candles: pd.DataFrame,
        candles_htf: Optional[pd.DataFrame] = None,
        rsi_period: int = 14,
        ao_short: int = 5,
        ao_long: int = 34,
        compute_sr: bool = True,
        sr_pivot_window: int = 5,
        sr_merge_pct: float = 0.01,
        sr_min_touches: int = 2,
    ) -> "MarketContext":
        """Factory — computes AO, RSI, and optionally S/R zones from candles."""
        if not {"open", "high", "low", "close", "volume", "timestamp"}.issubset(candles.columns):
            raise ValueError("candles must have OHLCV + timestamp columns")

        candles = candles.reset_index(drop=True)
        ao = awesome_oscillator(candles["high"], candles["low"], ao_short, ao_long)
        rsi = compute_rsi(candles["close"], period=rsi_period)

        sr_zones: Optional[list[SRZone]] = None
        if compute_sr and len(candles) >= 2 * sr_pivot_window + 2:
            current_price = float(candles["close"].iloc[-1])
            sr_zones = detect_sr_zones(
                candles,
                pivot_window=sr_pivot_window,
                merge_pct=sr_merge_pct,
                min_touches=sr_min_touches,
                reference_price=current_price,
            )

        ts = candles["timestamp"].iloc[-1]
        return cls(
            ticker=ticker, timeframe=timeframe, candles=candles,
            candles_htf=candles_htf, ao=ao, rsi=rsi, sr_zones=sr_zones,
            timestamp=pd.Timestamp(ts) if ts is not None else None,
        )

    @property
    def current_price(self) -> float:
        return float(self.candles["close"].iloc[-1])

    @property
    def current_bar_index(self) -> int:
        return len(self.candles) - 1


# ---------------------------------------------------------------------------
# Detector protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class SetupDetector(Protocol):
    """Every setup exposes this interface. Concrete detectors declare a
    `name` attribute (str) and implement `detect(ctx) -> Optional[Trigger]`.

    Detectors are stateless; any state (e.g., 'fired on this setup already')
    is managed by the scanner / validation layer.
    """

    name: str

    def detect(self, ctx: MarketContext) -> Optional[Trigger]:  # pragma: no cover
        ...

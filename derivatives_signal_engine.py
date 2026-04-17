"""
derivatives_signal_engine.py
Signal engine based on Funding Rate extremes + MTF D1 trend gate.

STRATEGY: Counter-leverage squeeze plays
  - Extreme positive funding (everyone long) + D1 downtrend → SHORT the longs
  - Extreme negative funding (everyone short) + D1 uptrend  → LONG the shorts

LOGIC: Over-leveraged positions get forcibly liquidated. We ride the flush.
Funding rate threshold = 0.0005/hr (0.05%/hr) by default, matching HL convention.

The MTF D1 filter gates trades — only take the squeeze if the daily trend agrees.
Without MTF, trades on extreme funding alone (riskier, more trades).
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Default threshold: 0.05%/hr on Hyperliquid is elevated; 0.1%/hr is extreme
DEFAULT_THRESHOLD = 0.0005  # 0.05%/hr


@dataclass
class TradeSignal:
    action: str               # "long" | "short" | "flat"
    entry_price: float
    stop_price: float
    target_price: float
    funding_rate: float       # the funding rate that triggered (or blocked) the signal
    mtf_bias: str             # "long" | "short" | "neutral" | "disabled"
    confidence: float         # 0.0 → 1.0
    skip_reason: str = ""


class DerivativesSignalEngine:
    """
    Funding-rate based signal engine. Drop-in replacement for SignalEngine
    without any ML model dependency.

    Parameters:
        funding_df: Pre-loaded DataFrame [timestamp, funding_rate] from derivatives_feed.py
        funding_threshold: per-hour rate threshold. Default 0.0005 (0.05%/hr).
        stop_pct: stop loss as fraction of price. Default 0.005 (0.5%).
        target_pct: take profit. Default 0.010 (1.0%) → 2:1 R:R.
        mtf_filter: MTFFilter instance or None.
    """

    def __init__(
        self,
        funding_df: pd.DataFrame,
        funding_threshold: float = DEFAULT_THRESHOLD,
        stop_pct: float = 0.005,
        target_pct: float = 0.010,
        mtf_filter=None,
    ):
        self.funding_df = funding_df.sort_values("timestamp").reset_index(drop=True)
        self.funding_threshold = funding_threshold
        self.stop_pct = stop_pct
        self.target_pct = target_pct
        self.mtf_filter = mtf_filter

        logger.info(
            f"DerivativesSignalEngine: threshold=±{funding_threshold:.4%}/hr, "
            f"stop={stop_pct:.2%}, target={target_pct:.2%}, "
            f"mtf={'on' if mtf_filter else 'off'}"
        )

    def get_funding_at(self, timestamp: pd.Timestamp) -> Optional[float]:
        """Most recent funding rate at or before timestamp (no lookahead)."""
        mask = self.funding_df["timestamp"] <= timestamp
        if not mask.any():
            return None
        return float(self.funding_df.loc[mask, "funding_rate"].iloc[-1])

    def evaluate(
        self,
        candles: pd.DataFrame,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> TradeSignal:
        """
        Evaluate current bar. candles must include 'close' column.
        timestamp is required for backtest mode.
        """
        current_close = float(candles["close"].iloc[-1])

        if timestamp is None and "timestamp" in candles.columns:
            timestamp = candles["timestamp"].iloc[-1]

        def flat(reason: str, fr: float = 0.0) -> TradeSignal:
            return TradeSignal(
                action="flat",
                entry_price=current_close,
                stop_price=0.0,
                target_price=0.0,
                funding_rate=fr,
                mtf_bias="disabled",
                confidence=0.0,
                skip_reason=reason,
            )

        if timestamp is None:
            return flat("no_timestamp")

        # --- Funding rate ---
        funding_rate = self.get_funding_at(timestamp)
        if funding_rate is None:
            return flat("no_funding_data")

        # --- MTF bias ---
        mtf_bias = "disabled"
        if self.mtf_filter is not None:
            mtf_sig = self.mtf_filter.get_bias_at(timestamp)
            mtf_bias = mtf_sig.bias

        thresh = self.funding_threshold

        # --- Signal decision ---
        if funding_rate >= thresh:
            # Longs are over-leveraged — market will flush them
            if mtf_bias in ("short", "disabled"):
                action = "short"
                confidence = min(funding_rate / thresh, 3.0) / 3.0
                if mtf_bias == "short":
                    confidence = min(confidence * 1.3, 1.0)  # bonus for alignment
            else:
                # Extreme funding but trend says up — conflict, skip
                return flat(f"extreme_long_funding_but_mtf={mtf_bias}", funding_rate)

        elif funding_rate <= -thresh:
            # Shorts are over-leveraged — market will squeeze them
            if mtf_bias in ("long", "disabled"):
                action = "long"
                confidence = min(abs(funding_rate) / thresh, 3.0) / 3.0
                if mtf_bias == "long":
                    confidence = min(confidence * 1.3, 1.0)
            else:
                return flat(f"extreme_short_funding_but_mtf={mtf_bias}", funding_rate)

        else:
            return flat(f"funding_normal({funding_rate:.6f})", funding_rate)

        # --- Stops and targets ---
        if action == "long":
            stop_price = current_close * (1 - self.stop_pct)
            target_price = current_close * (1 + self.target_pct)
        else:
            stop_price = current_close * (1 + self.stop_pct)
            target_price = current_close * (1 - self.target_pct)

        logger.debug(
            f"{action.upper()} @ {current_close:.4f} | "
            f"funding={funding_rate:.6f} ({funding_rate*100:.4f}%/hr) | "
            f"mtf={mtf_bias} | conf={confidence:.2f}"
        )

        return TradeSignal(
            action=action,
            entry_price=current_close,
            stop_price=stop_price,
            target_price=target_price,
            funding_rate=funding_rate,
            mtf_bias=mtf_bias,
            confidence=confidence,
        )

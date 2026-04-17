"""
snipe_signal_engine.py
Classical short-TF trend sniper built from regime-gated channel breakouts
confirmed by microstructure.

Pipeline (at 15m bar close):
  1. Regime gate   -- RegimeDetector.is_trending()   (ADX + Hurst)
  2. Breakout      -- Donchian(N) or Keltner(N,k*ATR) break direction
  3. Trend gate    -- SuperTrend on 1H must agree (optional, skipped if no 1h data)
  4. 4H veto       -- MTFFilter used as veto-only: reject entries opposing the 4H trend
                      (the old AND-gate semantics are gone)
  5. Microstructure confirmation  -- funding veto + composite score over
                      OI delta, basis expansion, CVD slope, liquidation proximity
  6. Entry/stop/target -- entry at current_price, stop = k*ATR, target = R*k*ATR

This is the Phase 1 replacement for trend_signal_engine.py. No ML in the
hot path; a LightGBM vetoer can layer on top later without changing this.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange, DonchianChannel, KeltnerChannel

from microstructure import (
    LiquidationCluster,
    basis_expansion_pct,
    cvd,
    cvd_slope,
    liquidation_proximity,
    oi_delta_pct,
)
from regime import RegimeDetector, RegimeState

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Inputs + outputs
# ------------------------------------------------------------------

@dataclass
class MarketContext:
    """Bundle of all the data the engine may consume on a single bar.

    Everything except candles_15m is optional; when absent, the corresponding
    confirmation/veto is skipped (not counted against the composite score).
    """
    candles_15m: pd.DataFrame                   # oldest -> newest, cols: ts/o/h/l/c/v
    candles_1h: Optional[pd.DataFrame] = None
    candles_4h: Optional[pd.DataFrame] = None
    current_price: Optional[float] = None
    funding_rate_hourly: float = 0.0
    oi_series: Optional[pd.Series] = None
    perp_closes: Optional[pd.Series] = None
    spot_closes: Optional[pd.Series] = None
    taker_buy_volume: Optional[pd.Series] = None
    total_volume: Optional[pd.Series] = None
    liq_clusters: list[LiquidationCluster] = field(default_factory=list)
    timestamp: Optional[pd.Timestamp] = None


@dataclass
class SnipeSignal:
    action: str                       # "long" | "short" | "flat"
    entry_price: float
    stop_price: float
    target_price: float
    atr: float
    rr_ratio: float
    regime: Optional[RegimeState]
    composite_score: float            # net microstructure score in [-1, 1]
    breakout_channel: str             # "donchian" | "keltner" | ""
    mtf_bias: str                     # "long"|"short"|"neutral"|"disabled"
    supertrend_1h: str                # "long"|"short"|"disabled"
    skip_reason: str = ""
    components: dict = field(default_factory=dict)  # per-feature scores for debug


# ------------------------------------------------------------------
# SuperTrend (rolled in-repo; `ta` does not ship one)
# ------------------------------------------------------------------

def supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series,
    period: int = 10, multiplier: float = 3.0,
) -> pd.Series:
    """
    SuperTrend direction series: +1 when uptrend, -1 when downtrend.

    Uses Wilder's ATR over `period` and a k×ATR band around hl2. The band
    'locks' only in the non-adverse direction — prevents bands jittering
    when in a trend.
    """
    hl2 = (high + low) / 2.0
    atr = AverageTrueRange(high=high, low=low, close=close, window=period).average_true_range()

    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    upper = upper_basic.copy()
    lower = lower_basic.copy()
    direction = pd.Series(index=close.index, dtype=float)

    for i in range(len(close)):
        if i == 0:
            direction.iloc[i] = 1.0
            continue
        # Band locking
        if upper_basic.iloc[i] < upper.iloc[i - 1] or close.iloc[i - 1] > upper.iloc[i - 1]:
            upper.iloc[i] = upper_basic.iloc[i]
        else:
            upper.iloc[i] = upper.iloc[i - 1]

        if lower_basic.iloc[i] > lower.iloc[i - 1] or close.iloc[i - 1] < lower.iloc[i - 1]:
            lower.iloc[i] = lower_basic.iloc[i]
        else:
            lower.iloc[i] = lower.iloc[i - 1]

        prev = direction.iloc[i - 1]
        if prev == 1.0 and close.iloc[i] < lower.iloc[i]:
            direction.iloc[i] = -1.0
        elif prev == -1.0 and close.iloc[i] > upper.iloc[i]:
            direction.iloc[i] = 1.0
        else:
            direction.iloc[i] = prev
    return direction


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------

class SnipeSignalEngine:
    def __init__(
        self,
        *,
        regime: Optional[RegimeDetector] = None,
        # Breakout channels
        donchian_period: int = 20,
        keltner_period: int = 20,
        keltner_atr_mult: float = 2.0,
        # Stops / targets
        atr_period: int = 14,
        stop_atr_mult: float = 1.5,
        target_atr_mult: float = 3.0,
        # 1H trend gate
        supertrend_period: int = 10,
        supertrend_mult: float = 3.0,
        # 4H veto (MTFFilter instance operating in veto-only mode)
        mtf_veto=None,
        # Microstructure
        funding_veto_pct_hourly: float = 0.0003,   # 0.03%/hr
        oi_confirm_pct: float = 0.005,
        basis_confirm_pct: float = 0.0005,
        cvd_confirm_slope: float = 0.0,
        liq_distance_max_pct: float = 0.03,
        composite_threshold: float = 0.0,
    ) -> None:
        self.regime = regime or RegimeDetector()
        self.donchian_period = donchian_period
        self.keltner_period = keltner_period
        self.keltner_atr_mult = keltner_atr_mult
        self.atr_period = atr_period
        self.stop_atr_mult = stop_atr_mult
        self.target_atr_mult = target_atr_mult
        self.supertrend_period = supertrend_period
        self.supertrend_mult = supertrend_mult
        self.mtf_veto = mtf_veto
        self.funding_veto_pct_hourly = funding_veto_pct_hourly
        self.oi_confirm_pct = oi_confirm_pct
        self.basis_confirm_pct = basis_confirm_pct
        self.cvd_confirm_slope = cvd_confirm_slope
        self.liq_distance_max_pct = liq_distance_max_pct
        self.composite_threshold = composite_threshold

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def evaluate(self, ctx: MarketContext) -> SnipeSignal:
        df = ctx.candles_15m
        min_bars = max(self.donchian_period, self.keltner_period, self.atr_period) + 5
        if df is None or len(df) < min_bars:
            return self._flat(ctx, "insufficient_15m_bars", regime=None)

        # --- Regime gate ---
        regime_state = self.regime.classify(df)
        if not regime_state.is_trending:
            return self._flat(ctx, f"regime:{regime_state.skip_reason or 'range'}", regime=regime_state)

        # --- Breakout detection ---
        direction, channel = self._detect_breakout(df)
        if direction == "flat":
            return self._flat(ctx, "no_breakout", regime=regime_state)

        # --- 1H SuperTrend trend gate ---
        st_1h = self._supertrend_1h(ctx)
        if st_1h not in ("disabled", direction):
            return self._flat(
                ctx, f"supertrend_1h_disagree({st_1h}!={direction})",
                regime=regime_state, st_1h=st_1h,
            )

        # --- 4H veto ---
        mtf_bias = self._mtf_veto_bias(ctx)
        if mtf_bias in ("long", "short") and mtf_bias != direction:
            return self._flat(
                ctx, f"mtf_4h_veto({mtf_bias}!={direction})",
                regime=regime_state, mtf_bias=mtf_bias, st_1h=st_1h,
            )

        # --- Funding veto (paying carry into the move) ---
        if direction == "long" and ctx.funding_rate_hourly > self.funding_veto_pct_hourly:
            return self._flat(
                ctx, f"funding_too_positive({ctx.funding_rate_hourly:.5f})",
                regime=regime_state, mtf_bias=mtf_bias, st_1h=st_1h,
            )
        if direction == "short" and ctx.funding_rate_hourly < -self.funding_veto_pct_hourly:
            return self._flat(
                ctx, f"funding_too_negative({ctx.funding_rate_hourly:.5f})",
                regime=regime_state, mtf_bias=mtf_bias, st_1h=st_1h,
            )

        # --- Microstructure composite score ---
        composite, components = self._composite_score(ctx, direction)
        if composite < self.composite_threshold:
            return self._flat(
                ctx,
                f"composite_below_threshold({composite:.2f}<{self.composite_threshold:.2f})",
                regime=regime_state, mtf_bias=mtf_bias, st_1h=st_1h,
                composite_score=composite, components=components,
            )

        # --- Risk structure (ATR-based) ---
        atr_val = self._atr(df)
        if atr_val <= 0:
            return self._flat(ctx, "atr_zero", regime=regime_state)

        entry = float(ctx.current_price or df["close"].iloc[-1])
        stop_dist = self.stop_atr_mult * atr_val
        target_dist = self.target_atr_mult * atr_val
        if direction == "long":
            stop = entry - stop_dist
            target = entry + target_dist
        else:
            stop = entry + stop_dist
            target = entry - target_dist
        rr = target_dist / stop_dist

        return SnipeSignal(
            action=direction,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            atr=atr_val,
            rr_ratio=rr,
            regime=regime_state,
            composite_score=composite,
            breakout_channel=channel,
            mtf_bias=mtf_bias,
            supertrend_1h=st_1h,
            components=components,
        )

    # ------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------

    def _detect_breakout(self, df: pd.DataFrame) -> tuple[str, str]:
        """Return (direction, channel_name). direction in {'long','short','flat'}."""
        high, low, close = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
        cur_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])

        # Donchian: break of the prior-N-bar high/low (exclude current bar)
        prior_high = float(high.iloc[-self.donchian_period - 1:-1].max())
        prior_low = float(low.iloc[-self.donchian_period - 1:-1].min())
        if cur_close > prior_high and prev_close <= prior_high:
            return "long", "donchian"
        if cur_close < prior_low and prev_close >= prior_low:
            return "short", "donchian"

        # Keltner: break of EMA ± k*ATR band
        kc = KeltnerChannel(
            high=high, low=low, close=close,
            window=self.keltner_period,
            window_atr=self.atr_period,
            multiplier=self.keltner_atr_mult,
            original_version=False,
        )
        kc_high = float(kc.keltner_channel_hband().iloc[-1])
        kc_low = float(kc.keltner_channel_lband().iloc[-1])
        kc_high_prev = float(kc.keltner_channel_hband().iloc[-2])
        kc_low_prev = float(kc.keltner_channel_lband().iloc[-2])

        if cur_close > kc_high and prev_close <= kc_high_prev:
            return "long", "keltner"
        if cur_close < kc_low and prev_close >= kc_low_prev:
            return "short", "keltner"

        return "flat", ""

    def _atr(self, df: pd.DataFrame) -> float:
        high, low, close = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
        atr = AverageTrueRange(high=high, low=low, close=close, window=self.atr_period).average_true_range()
        val = float(atr.iloc[-1])
        return val if np.isfinite(val) else 0.0

    def _supertrend_1h(self, ctx: MarketContext) -> str:
        df = ctx.candles_1h
        if df is None or len(df) < self.supertrend_period * 3:
            return "disabled"
        direction_series = supertrend(
            df["high"].astype(float), df["low"].astype(float), df["close"].astype(float),
            period=self.supertrend_period, multiplier=self.supertrend_mult,
        )
        last = float(direction_series.iloc[-1])
        return "long" if last > 0 else "short"

    def _mtf_veto_bias(self, ctx: MarketContext) -> str:
        """Run the optional 4H MTF filter in veto-only mode.

        The filter still classifies 1H/4H trend; we only reject trades that
        directly oppose it. Neutral or unavailable = no veto.
        """
        if self.mtf_veto is None:
            return "disabled"
        try:
            ts = ctx.timestamp or (ctx.candles_15m["timestamp"].iloc[-1] if "timestamp" in ctx.candles_15m.columns else None)
            if ts is None:
                sig = self.mtf_veto.get_bias_live()
            else:
                sig = self.mtf_veto.get_bias_at(ts)
            # The veto uses the 4H leg specifically — richer than combined bias.
            return sig.tf_4h
        except Exception as exc:
            logger.warning("MTF veto lookup failed: %s", exc)
            return "disabled"

    # ------------------------------------------------------------
    # Microstructure composite score
    # ------------------------------------------------------------

    def _composite_score(self, ctx: MarketContext, direction: str) -> tuple[float, dict]:
        """
        Composite microstructure score in [-1, 1].

        Each present component contributes a signed {-1, 0, +1} aligned with
        the entry direction; the score is the average of present components.
        Missing components are omitted (not counted against the signal).
        Funding is handled separately as a hard veto, not a score contributor.
        """
        sign = 1 if direction == "long" else -1
        contribs: list[float] = []
        components: dict = {}

        # OI delta: expanding OI in the direction confirms; contraction = fade
        if ctx.oi_series is not None and len(ctx.oi_series) >= 4:
            oi = oi_delta_pct(ctx.oi_series, window=3)
            components["oi_delta_pct"] = oi
            if np.isfinite(oi):
                if oi > self.oi_confirm_pct:
                    contribs.append(+1 * sign)  # OI rising; for long this is +1, short this is -1 (bad)
                elif oi < -self.oi_confirm_pct:
                    contribs.append(-1 * sign)
                else:
                    contribs.append(0.0)

        # Basis expansion (perp-spot): premium building in the direction
        if ctx.perp_closes is not None and ctx.spot_closes is not None:
            be = basis_expansion_pct(ctx.perp_closes, ctx.spot_closes, window=3)
            components["basis_expansion_pct"] = be
            if np.isfinite(be):
                if direction == "long" and be > self.basis_confirm_pct:
                    contribs.append(+1.0)
                elif direction == "short" and be < -self.basis_confirm_pct:
                    contribs.append(+1.0)
                elif direction == "long" and be < -self.basis_confirm_pct:
                    contribs.append(-1.0)
                elif direction == "short" and be > self.basis_confirm_pct:
                    contribs.append(-1.0)
                else:
                    contribs.append(0.0)

        # CVD slope: net aggression direction
        if ctx.taker_buy_volume is not None and ctx.total_volume is not None:
            cvd_series = cvd(ctx.taker_buy_volume, ctx.total_volume)
            slope = cvd_slope(cvd_series, window=20)
            components["cvd_slope"] = slope
            if np.isfinite(slope):
                if direction == "long" and slope > self.cvd_confirm_slope:
                    contribs.append(+1.0)
                elif direction == "short" and slope < -self.cvd_confirm_slope:
                    contribs.append(+1.0)
                else:
                    contribs.append(-1.0 if abs(slope) > self.cvd_confirm_slope else 0.0)

        # Liquidation proximity: near-cluster in direction = magnet -> confirm
        if ctx.liq_clusters and ctx.current_price:
            dist, vol = liquidation_proximity(
                current_price=float(ctx.current_price),
                clusters=ctx.liq_clusters,
                direction=direction,
                max_distance_pct=self.liq_distance_max_pct,
            )
            components["liq_distance_pct"] = dist
            components["liq_volume"] = vol
            if dist is not None:
                # Closer and bigger cluster -> stronger confirm
                contribs.append(1.0)
            else:
                contribs.append(0.0)

        if not contribs:
            return 0.0, components
        score = float(np.mean(contribs))
        return score, components

    # ------------------------------------------------------------
    # Flat helper
    # ------------------------------------------------------------

    def _flat(
        self,
        ctx: MarketContext,
        reason: str,
        *,
        regime: Optional[RegimeState] = None,
        mtf_bias: str = "disabled",
        st_1h: str = "disabled",
        composite_score: float = 0.0,
        components: Optional[dict] = None,
    ) -> SnipeSignal:
        entry = float(ctx.current_price or 0.0)
        return SnipeSignal(
            action="flat",
            entry_price=entry,
            stop_price=0.0,
            target_price=0.0,
            atr=0.0,
            rr_ratio=self.target_atr_mult / self.stop_atr_mult,
            regime=regime,
            composite_score=composite_score,
            breakout_channel="",
            mtf_bias=mtf_bias,
            supertrend_1h=st_1h,
            skip_reason=reason,
            components=components or {},
        )

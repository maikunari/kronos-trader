"""
validation/kelly.py
Kelly-criterion growth-rate diagnostics for setup detectors and strategies.

The key insight from Kelly (1956): the long-run optimization target is
not expected return, not Sharpe, not win rate — it's the expected
logarithm of capital. For a series of independent bets with returns
r_1, r_2, ..., r_N (each a signed fraction of capital risked, e.g.
+0.05 = won 5%, -0.02 = lost 2%), the growth rate under fractional
Kelly sizing `f` is:

    G(f) = (1/N) * sum_i log(1 + f * r_i)

The optimal Kelly fraction f* maximizes G. For a strategy with f* > 0,
G(f*) is the per-trade exponential growth rate that strategy can
sustain — the actual metric that determines whether a strategy makes
money in the long run.

This module computes f* and G(f*) from a list of per-trade returns,
plus a `simulate_all_triggers` helper to produce that list for any
detector (covering both true-positive AND false-positive triggers,
because a real bot eats the losses on its false positives).

What G tells us:
  G > 0   →  capital grows in the long run (alive)
  G == 0  →  break-even (treadmill)
  G < 0   →  capital decays in the long run (dead)

A detector with G < 0 should not be traded regardless of its recall
or precision. A detector with G > 0 and low precision can still be
worth running — Kelly cares about expected log-return per trade, not
how often you're right.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd
from scipy.optimize import brentq

from setups.base import Trigger
from validation.capture import CaptureResult, simulate_capture
from validation.labeler import PopEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kelly maths
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KellyResult:
    """Diagnostics for a series of trade returns under Kelly sizing."""
    n_trades: int
    win_rate: float                # fraction of returns > 0
    avg_win: float                 # mean of positive returns
    avg_loss: float                # mean of negative returns (signed, i.e. negative)
    payoff_ratio: float            # avg_win / |avg_loss|, 0 if no losses
    kelly_fraction: float          # f* in [0, 1]; 0 if no positive edge
    growth_rate: float             # G(f*) per trade, in natural log units
    growth_rate_full_unit: float   # G(1.0) — growth at full-capital sizing
    mean_return: float             # simple arithmetic mean (for reference)

    @property
    def has_positive_edge(self) -> bool:
        """True iff G(f*) > 0 — capital grows under Kelly sizing."""
        return self.growth_rate > 0


def growth_rate(
    returns: Iterable[float],
    *,
    fraction_cap: float = 1.0,
) -> KellyResult:
    """Compute the Kelly-optimal fraction and growth rate from a series of
    per-trade returns.

    Args:
        returns: signed per-trade returns as fractions of capital risked.
                 E.g. [+0.05, -0.02, +0.10, -0.05]. Use raw fractions, not
                 percentages.
        fraction_cap: maximum allowed Kelly fraction (default 1.0). Some
                      practitioners cap at 0.5 or 0.25 to reduce drawdowns.

    Returns:
        KellyResult with f*, G(f*), and supporting stats. If the returns
        have no positive expectation, f* = 0 and G = 0.

    Algorithm:
        G(f) = (1/N) * sum_i log(1 + f * r_i)
        G'(f) = (1/N) * sum_i r_i / (1 + f * r_i)

        At f* > 0 either G'(f*) = 0 (interior optimum) or f* hits the cap.
        We find the root of G'(f) = 0 on (0, f_max) where f_max is the
        smaller of `fraction_cap` and the bet-survival upper bound
        1/|min_r| - epsilon. If G'(0) <= 0, no positive Kelly exists.
    """
    rs = [float(r) for r in returns]
    n = len(rs)
    if n == 0:
        return KellyResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    win_rate = len(wins) / n
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    payoff = (avg_win / abs(avg_loss)) if losses and avg_loss != 0 else 0.0
    mean_r = sum(rs) / n

    # Quick-out: if no positive returns, no Kelly bet
    if not wins:
        return KellyResult(n, win_rate, avg_win, avg_loss, payoff, 0.0, 0.0,
                           _g_at_full_unit(rs), mean_r)

    # G'(0) = mean(r). If negative-or-zero, no Kelly bet is positive-EV.
    if mean_r <= 0:
        return KellyResult(n, win_rate, avg_win, avg_loss, payoff, 0.0, 0.0,
                           _g_at_full_unit(rs), mean_r)

    # Bet-survival upper bound: 1 + f*r must stay > 0 for all r → f < 1/|min_r|
    worst = min(rs)   # most negative
    if worst < 0:
        survival_cap = (1.0 - 1e-6) / abs(worst)
    else:
        survival_cap = float("inf")
    f_max = min(fraction_cap, survival_cap)

    # G'(f_max) might already be positive (capped optimum). Check both
    # endpoints of (0, f_max).
    g_prime_0 = mean_r   # G'(0) = mean(r)
    g_prime_fmax = _g_prime(rs, f_max)

    if g_prime_fmax >= 0:
        f_star = f_max   # corner solution
    else:
        try:
            f_star = brentq(lambda f: _g_prime(rs, f), 1e-9, f_max, xtol=1e-6)
        except ValueError:
            # Fallback grid search if brentq fails (shouldn't happen given
            # the sign check above, but defensive).
            f_star = _grid_argmax_g(rs, f_max)

    g_star = _g(rs, f_star)
    g_full = _g_at_full_unit(rs)
    return KellyResult(
        n_trades=n,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=payoff,
        kelly_fraction=f_star,
        growth_rate=g_star,
        growth_rate_full_unit=g_full,
        mean_return=mean_r,
    )


def _g(returns: list[float], f: float) -> float:
    n = len(returns)
    if n == 0:
        return 0.0
    s = 0.0
    for r in returns:
        v = 1.0 + f * r
        if v <= 0:
            return float("-inf")
        s += math.log(v)
    return s / n


def _g_prime(returns: list[float], f: float) -> float:
    n = len(returns)
    if n == 0:
        return 0.0
    s = 0.0
    for r in returns:
        denom = 1.0 + f * r
        if denom <= 0:
            return float("-inf")
        s += r / denom
    return s / n


def _g_at_full_unit(returns: list[float]) -> float:
    """G at f = 1 (bet whole capital each trade). Returns -inf if any
    return is <= -1, i.e. ruin-risking on a single trade."""
    return _g(returns, 1.0)


def _grid_argmax_g(returns: list[float], f_max: float) -> float:
    best_f, best_g = 0.0, _g(returns, 0.0)
    step = f_max / 100
    f = step
    while f <= f_max:
        g = _g(returns, f)
        if g > best_g:
            best_g, best_f = g, f
        f += step
    return best_f


# ---------------------------------------------------------------------------
# Trigger -> returns adapter
# ---------------------------------------------------------------------------

_DUMMY_POP_CACHE: dict[str, PopEvent] = {}


def _dummy_pop(ticker: str, direction: str) -> PopEvent:
    """A throwaway PopEvent the capture simulator needs as an argument;
    its magnitude is only used for the capture_ratio computation which
    we don't read here."""
    key = f"{ticker}_{direction}"
    if key in _DUMMY_POP_CACHE:
        return _DUMMY_POP_CACHE[key]
    p = PopEvent(
        ticker=ticker,
        timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        direction=direction,
        magnitude=1.0,
        peak_timestamp=pd.Timestamp("2024-01-02", tz="UTC"),
        start_bar_index=0,
        peak_bar_index=0,
        threshold=0.0,
        start_price=100.0,
        peak_price=200.0 if direction == "long" else 50.0,
    )
    _DUMMY_POP_CACHE[key] = p
    return p


def simulate_trigger_returns(
    triggers: list[Trigger],
    candles_by_ticker: dict[str, pd.DataFrame],
    *,
    max_hold_bars: Optional[int] = None,
    include_unresolved_as_zero: bool = False,
) -> list[float]:
    """Run each trigger through simulate_capture and collect realized
    returns. Covers ALL triggers — both true positives and false
    positives — because in a real bot, every fire eats stop-loss risk.

    Args:
        triggers: trigger objects to simulate
        candles_by_ticker: candle DataFrames keyed by ticker
        max_hold_bars: passed through to simulate_capture
        include_unresolved_as_zero: when True, trades that ran out of
            candles without hitting stop or full TP ladder contribute
            r = 0 to the returns list. When False (default), they are
            dropped — appropriate for the diagnostic mode where we
            want to measure resolved-trade economics only.

    Returns:
        list of realized_return_pct values (signed fractions of capital).
    """
    out: list[float] = []
    for trig in triggers:
        candles = candles_by_ticker.get(trig.ticker)
        if candles is None or candles.empty:
            continue
        result = simulate_capture(
            trig, _dummy_pop(trig.ticker, trig.direction), candles,
            max_hold_bars=max_hold_bars,
        )
        if result is None:
            continue
        if result.exit_reason == "unresolved":
            if include_unresolved_as_zero:
                out.append(0.0)
            continue
        out.append(result.realized_return_pct)
    return out


def returns_from_captures(captures: list[CaptureResult]) -> list[float]:
    """Pull realized returns from already-simulated capture results.

    Returns only RESOLVED trades (stop / target / end_of_data — anything
    except `unresolved`). Use this when stats are already computed by
    `validation.report._compute_stats` and you want to layer Kelly on top
    without re-simulating.
    """
    return [
        c.realized_return_pct
        for c in captures
        if c.exit_reason != "unresolved"
    ]

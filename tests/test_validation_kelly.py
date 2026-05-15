"""Tests for validation/kelly.py — the Kelly-criterion growth-rate
diagnostics. Validates the math against known closed-form cases plus
synthetic edge cases."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from setups.base import TPLevel, Trigger
from validation.capture import CaptureResult
from validation.kelly import (
    KellyResult,
    growth_rate,
    returns_from_captures,
    simulate_trigger_returns,
)


# ---------------------------------------------------------------------------
# growth_rate — closed-form binary cases
# ---------------------------------------------------------------------------

def test_zero_trades_returns_zero():
    r = growth_rate([])
    assert r.n_trades == 0
    assert r.kelly_fraction == 0.0
    assert r.growth_rate == 0.0


def test_all_losses_returns_zero_kelly():
    """If every trade lost money, Kelly is 0 (don't bet)."""
    r = growth_rate([-0.05, -0.10, -0.02, -0.08])
    assert r.kelly_fraction == 0.0
    assert r.growth_rate == 0.0
    assert r.has_positive_edge is False


def test_negative_expectation_returns_zero_kelly():
    """If mean return <= 0 even with some wins, Kelly is 0."""
    # 1 win of +5%, 9 losses of -1% each → mean = -0.4%
    r = growth_rate([0.05] + [-0.01] * 9)
    assert r.kelly_fraction == 0.0
    assert r.growth_rate == 0.0


def test_known_binary_kelly_50_50_2x_payoff():
    """Classical 50/50 with 2:1 payoff: win 2 units per 1 risked, lose
    full bet. In our 'r = signed return on capital risked' framing this
    is r = +2.0 on win, r = -1.0 on loss → f* = (p*b - q)/b = 0.25 with
    b = 2.

    Use 100 deterministic alternating trades, no noise.
    """
    returns = [2.0, -1.0] * 50
    r = growth_rate(returns)
    assert r.win_rate == pytest.approx(0.5, rel=0.01)
    assert r.kelly_fraction == pytest.approx(0.25, abs=0.01)
    assert r.growth_rate > 0
    # Closed-form: G(0.25) = 0.5*log(1.5) + 0.5*log(0.75)
    expected_g = 0.5 * math.log(1.5) + 0.5 * math.log(0.75)
    assert r.growth_rate == pytest.approx(expected_g, abs=1e-4)


def test_all_wins_caps_at_fraction_cap():
    """If every trade wins, optimal f = 1 (or the cap)."""
    r = growth_rate([0.05, 0.10, 0.03, 0.07], fraction_cap=1.0)
    assert r.kelly_fraction == pytest.approx(1.0, abs=0.01)
    assert r.growth_rate > 0


def test_fraction_cap_is_respected():
    """Even strong edges should not exceed fraction_cap."""
    r = growth_rate([1.0, -0.5] * 50, fraction_cap=0.10)
    assert r.kelly_fraction <= 0.10 + 1e-9


def test_survival_cap_prevents_ruin():
    """If any return is -1.0 (full ruin on one trade), Kelly fraction
    must be strictly less than 1 to keep capital > 0."""
    r = growth_rate([0.5, 0.5, 0.5, -1.0])
    assert r.kelly_fraction < 1.0


def test_strong_edge_yields_high_growth():
    """Very profitable, low-variance strategy → high G."""
    r = growth_rate([0.10] * 100)   # 100 wins of 10% each
    assert r.kelly_fraction == pytest.approx(1.0, abs=0.01)
    # G(1) = log(1.10) ~= 0.0953
    assert r.growth_rate == pytest.approx(math.log(1.10), abs=1e-4)


def test_break_even_yields_zero_growth():
    """Equal-magnitude wins and losses → mean = 0 → no positive Kelly."""
    r = growth_rate([0.05, -0.05] * 50)
    assert r.kelly_fraction == 0.0
    assert r.growth_rate == 0.0


# ---------------------------------------------------------------------------
# growth_rate — derived stats
# ---------------------------------------------------------------------------

def test_win_rate_and_payoff_ratio_computed_correctly():
    returns = [0.10, 0.20, -0.05, -0.10, 0.05]
    r = growth_rate(returns)
    assert r.n_trades == 5
    assert r.win_rate == pytest.approx(0.6, abs=0.001)
    assert r.avg_win == pytest.approx((0.10 + 0.20 + 0.05) / 3, abs=1e-6)
    assert r.avg_loss == pytest.approx((-0.05 - 0.10) / 2, abs=1e-6)
    assert r.payoff_ratio == pytest.approx(
        ((0.10 + 0.20 + 0.05) / 3) / (0.075), abs=1e-3,
    )


def test_growth_rate_full_unit_reflects_full_kelly_sizing():
    """G(f=1.0) is reported as growth_rate_full_unit — useful for comparing
    naive 'bet it all' sizing to optimal Kelly."""
    r = growth_rate([0.10, -0.05] * 50)
    expected = 0.5 * math.log(1.10) + 0.5 * math.log(0.95)
    assert r.growth_rate_full_unit == pytest.approx(expected, abs=1e-4)


def test_has_positive_edge_threshold():
    """`has_positive_edge` should track G > 0 strictly."""
    r_pos = growth_rate([1.0, -0.5] * 50)
    r_neg = growth_rate([-0.05, -0.10, -0.02])
    assert r_pos.has_positive_edge
    assert not r_neg.has_positive_edge


# ---------------------------------------------------------------------------
# simulate_trigger_returns — uses the existing capture simulator
# ---------------------------------------------------------------------------

def _ohlc_df(closes: list[float], start: str = "2025-01-01") -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    highs = closes * 1.001
    lows = closes * 0.999
    ts = pd.date_range(start, periods=len(closes), freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": closes, "high": highs, "low": lows,
        "close": closes, "volume": [1000.0] * len(closes),
    })


def _trig(ticker="TEST", direction="long", entry=100.0, stop=98.0,
          tp=110.0, ts="2025-01-01"):
    return Trigger(
        ticker=ticker,
        timestamp=pd.Timestamp(ts, tz="UTC"),
        action="open_new",
        direction=direction,
        entry_price=entry,
        stop_price=stop,
        tp_ladder=(TPLevel(price=tp, fraction=1.0, source="test"),),
        setup="test",
        confidence=0.6,
    )


def test_simulate_trigger_returns_collects_resolved_outcomes():
    # Candle path goes up to 115 then back down — hits TP at 110 on bar 1
    closes = [100.0, 115.0, 112.0]
    df = _ohlc_df(closes)
    # need to raise the high on bar 1 so it touches TP at 110
    df.loc[1, "high"] = 115.0
    trig = _trig(ticker="TEST", entry=100.0, tp=110.0, stop=98.0)
    rets = simulate_trigger_returns([trig], {"TEST": df})
    assert len(rets) == 1
    # Hit TP @ 110 from entry 100 → +10% on full size
    assert rets[0] == pytest.approx(0.10, rel=0.01)


def test_simulate_trigger_returns_skips_missing_candles():
    trig = _trig(ticker="NOSUCH")
    rets = simulate_trigger_returns([trig], {"OTHER": _ohlc_df([100, 101])})
    assert rets == []


def test_simulate_trigger_returns_drops_unresolved_by_default():
    # Path stays at 100 — neither stop nor TP fires
    df = _ohlc_df([100.0, 100.5, 99.5, 100.2, 99.8])
    trig = _trig(ticker="TEST", entry=100.0, stop=90.0, tp=120.0)
    rets = simulate_trigger_returns([trig], {"TEST": df})
    assert rets == []


def test_simulate_trigger_returns_includes_unresolved_as_zero_when_asked():
    df = _ohlc_df([100.0, 100.5, 99.5, 100.2, 99.8])
    trig = _trig(ticker="TEST", entry=100.0, stop=90.0, tp=120.0)
    rets = simulate_trigger_returns(
        [trig], {"TEST": df}, include_unresolved_as_zero=True,
    )
    assert rets == [0.0]


# ---------------------------------------------------------------------------
# returns_from_captures — pulls from existing capture results
# ---------------------------------------------------------------------------

def _capture(realized: float, exit_reason: str = "target") -> CaptureResult:
    """Synthetic CaptureResult for testing."""
    dummy_trig = _trig()
    from validation.labeler import PopEvent
    dummy_pop = PopEvent(
        ticker="TEST",
        timestamp=pd.Timestamp("2025-01-01", tz="UTC"),
        direction="long",
        magnitude=0.20,
        peak_timestamp=pd.Timestamp("2025-01-02", tz="UTC"),
        start_bar_index=0, peak_bar_index=24,
        threshold=0.20, start_price=100.0, peak_price=120.0,
    )
    return CaptureResult(
        trigger=dummy_trig, pop=dummy_pop,
        entry_price=100.0, weighted_exit_price=100 * (1 + realized),
        realized_return_pct=realized,
        capture_ratio=realized / 0.20,
        exit_reason=exit_reason, bars_held=5, tier_fills=(),
    )


def test_returns_from_captures_drops_unresolved():
    caps = [
        _capture(0.10, "target"),
        _capture(-0.02, "stop"),
        _capture(0.0, "unresolved"),    # should be dropped
    ]
    rets = returns_from_captures(caps)
    assert rets == [0.10, -0.02]


def test_returns_from_captures_handles_empty():
    assert returns_from_captures([]) == []

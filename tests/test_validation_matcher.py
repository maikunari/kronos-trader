"""Tests for validation/matcher.py."""
from __future__ import annotations

import pandas as pd
import pytest

from setups.base import TPLevel, Trigger
from validation.labeler import PopEvent
from validation.matcher import (
    Match,
    match_counts,
    match_triggers_to_pops,
    median_lead_time,
    precision,
    recall,
)


# --- Constructors ------------------------------------------------------------

def _pop(ticker="BTC", direction="long", start="2025-01-01 10:00", peak="2025-01-02 10:00",
         magnitude=0.25):
    return PopEvent(
        ticker=ticker,
        timestamp=pd.Timestamp(start, tz="UTC"),
        direction=direction,
        magnitude=magnitude,
        peak_timestamp=pd.Timestamp(peak, tz="UTC"),
        start_bar_index=0,
        peak_bar_index=24,
        threshold=0.20,
        start_price=100.0,
        peak_price=125.0 if direction == "long" else 75.0,
    )


def _trigger(ticker="BTC", direction="long", timestamp="2025-01-01 08:00",
             confidence=0.7, setup="divergence_reversal"):
    return Trigger(
        ticker=ticker,
        timestamp=pd.Timestamp(timestamp, tz="UTC"),
        action="open_new",
        direction=direction,
        entry_price=100.0,
        stop_price=98.0,
        tp_ladder=(TPLevel(price=110.0, fraction=1.0, source="test"),),
        setup=setup,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Basic matching
# ---------------------------------------------------------------------------

def test_matches_trigger_firing_within_lead_window():
    pops = [_pop(start="2025-01-01 10:00")]
    trigs = [_trigger(timestamp="2025-01-01 08:00")]   # 2h lead
    matches = match_triggers_to_pops(trigs, pops)
    outcomes = [m.outcome for m in matches]
    assert outcomes == ["true_positive"]
    tp = matches[0]
    assert tp.pop is pops[0]
    assert tp.trigger is trigs[0]
    assert tp.lead_time == pd.Timedelta(hours=2)


def test_matches_trigger_firing_slightly_after_pop_start():
    pops = [_pop(start="2025-01-01 10:00", peak="2025-01-02 10:00")]
    trigs = [_trigger(timestamp="2025-01-01 12:00")]   # 2h lag, within max_lag=4h
    matches = match_triggers_to_pops(trigs, pops, max_lag=pd.Timedelta(hours=4))
    assert [m.outcome for m in matches] == ["true_positive"]
    # Lead time negative = trigger after pop start
    assert matches[0].lead_time == pd.Timedelta(hours=-2)


def test_false_negative_when_no_trigger_in_window():
    pops = [_pop(start="2025-01-01 10:00")]
    trigs = [_trigger(timestamp="2024-12-30 08:00")]   # way too early
    matches = match_triggers_to_pops(trigs, pops)
    outcomes = {m.outcome for m in matches}
    assert "false_negative" in outcomes
    assert "false_positive" in outcomes


def test_false_positive_for_trigger_without_pop():
    pops = []
    trigs = [_trigger(timestamp="2025-01-01 10:00")]
    matches = match_triggers_to_pops(trigs, pops)
    assert [m.outcome for m in matches] == ["false_positive"]


def test_mismatched_direction_does_not_match():
    pops = [_pop(direction="long", start="2025-01-01 10:00")]
    trigs = [_trigger(direction="short", timestamp="2025-01-01 08:00")]
    matches = match_triggers_to_pops(trigs, pops)
    outcomes = {m.outcome for m in matches}
    assert outcomes == {"false_negative", "false_positive"}


def test_mismatched_ticker_does_not_match():
    pops = [_pop(ticker="BTC", start="2025-01-01 10:00")]
    trigs = [_trigger(ticker="ETH", timestamp="2025-01-01 08:00")]
    matches = match_triggers_to_pops(trigs, pops)
    outcomes = {m.outcome for m in matches}
    assert "false_negative" in outcomes
    assert "false_positive" in outcomes


def test_trigger_after_peak_does_not_match():
    pops = [_pop(start="2025-01-01 10:00", peak="2025-01-01 20:00")]
    trigs = [_trigger(timestamp="2025-01-01 22:00")]   # after peak
    matches = match_triggers_to_pops(trigs, pops, max_lag=pd.Timedelta(hours=48))
    outcomes = {m.outcome for m in matches}
    assert "false_negative" in outcomes
    assert "false_positive" in outcomes


# ---------------------------------------------------------------------------
# Best-candidate selection
# ---------------------------------------------------------------------------

def test_picks_highest_confidence_trigger():
    pops = [_pop(start="2025-01-01 10:00")]
    low = _trigger(timestamp="2025-01-01 08:00", confidence=0.4)
    high = _trigger(timestamp="2025-01-01 09:00", confidence=0.9)
    matches = match_triggers_to_pops([low, high], pops)
    tp = next(m for m in matches if m.outcome == "true_positive")
    assert tp.trigger is high


def test_tiebreaks_on_earliest_timestamp():
    pops = [_pop(start="2025-01-01 10:00")]
    earlier = _trigger(timestamp="2025-01-01 07:00", confidence=0.7)
    later = _trigger(timestamp="2025-01-01 09:00", confidence=0.7)
    matches = match_triggers_to_pops([earlier, later], pops)
    tp = next(m for m in matches if m.outcome == "true_positive")
    assert tp.trigger is earlier


def test_greedy_assignment_one_trigger_per_pop():
    # Two pops, one shared window. Only one trigger available.
    pops = [
        _pop(start="2025-01-01 10:00", peak="2025-01-02 10:00"),
        _pop(start="2025-01-01 14:00", peak="2025-01-02 14:00"),
    ]
    trigs = [_trigger(timestamp="2025-01-01 08:00")]
    matches = match_triggers_to_pops(trigs, pops, max_lead=pd.Timedelta(hours=24))
    tp_count = sum(1 for m in matches if m.outcome == "true_positive")
    fn_count = sum(1 for m in matches if m.outcome == "false_negative")
    # Only one pop matched; the other is missed
    assert tp_count == 1
    assert fn_count == 1


def test_cross_ticker_isolation():
    pops = [_pop(ticker="BTC", start="2025-01-01 10:00"),
            _pop(ticker="ETH", start="2025-01-01 10:00")]
    trigs = [_trigger(ticker="BTC", timestamp="2025-01-01 08:00"),
             _trigger(ticker="ETH", timestamp="2025-01-01 08:00")]
    matches = match_triggers_to_pops(trigs, pops)
    tps = [m for m in matches if m.outcome == "true_positive"]
    tickers_matched = {m.pop.ticker for m in tps}
    assert tickers_matched == {"BTC", "ETH"}


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def test_match_counts_totals():
    pops = [_pop(start="2025-01-01 10:00"),
            _pop(start="2025-02-01 10:00")]
    trigs = [_trigger(timestamp="2025-01-01 08:00"),      # TP
             _trigger(timestamp="2025-03-01 08:00")]       # FP (no pop)
    matches = match_triggers_to_pops(trigs, pops)
    counts = match_counts(matches)
    assert counts["true_positive"] == 1
    assert counts["false_negative"] == 1
    assert counts["false_positive"] == 1


def test_recall_formula():
    pops = [_pop(start="2025-01-01 10:00"),
            _pop(start="2025-02-01 10:00")]
    trigs = [_trigger(timestamp="2025-01-01 08:00")]
    matches = match_triggers_to_pops(trigs, pops)
    # 1 TP / (1 TP + 1 FN) = 0.5
    assert recall(matches) == pytest.approx(0.5)


def test_precision_formula():
    pops = [_pop(start="2025-01-01 10:00")]
    trigs = [_trigger(timestamp="2025-01-01 08:00"),
             _trigger(timestamp="2025-03-01 08:00"),
             _trigger(timestamp="2025-04-01 08:00")]
    matches = match_triggers_to_pops(trigs, pops)
    # 1 TP / (1 TP + 2 FP) = 0.333
    assert precision(matches) == pytest.approx(1 / 3, abs=1e-6)


def test_recall_zero_when_no_pops():
    # Pure false positives; recall is 0 (formula has 0 denominator → 0.0)
    trigs = [_trigger(timestamp="2025-01-01 08:00")]
    matches = match_triggers_to_pops(trigs, [])
    assert recall(matches) == 0.0


def test_median_lead_time_reports_positive_when_triggers_lead():
    pops = [
        _pop(start="2025-01-01 10:00", peak="2025-01-02 10:00"),
        _pop(start="2025-01-03 10:00", peak="2025-01-04 10:00"),
        _pop(start="2025-01-05 10:00", peak="2025-01-06 10:00"),
    ]
    trigs = [
        _trigger(timestamp="2025-01-01 08:00"),   # 2h lead
        _trigger(timestamp="2025-01-03 06:00"),   # 4h lead
        _trigger(timestamp="2025-01-05 04:00"),   # 6h lead
    ]
    matches = match_triggers_to_pops(trigs, pops)
    lt = median_lead_time(matches)
    assert lt == pd.Timedelta(hours=4)


def test_median_lead_time_none_when_no_true_positives():
    assert median_lead_time([]) is None


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------

def test_all_empty_returns_empty_matches():
    assert match_triggers_to_pops([], []) == []


def test_input_order_does_not_affect_outcome():
    pops = [
        _pop(ticker="BTC", start="2025-02-01 10:00"),
        _pop(ticker="ETH", start="2025-01-01 10:00"),
    ]
    trigs = [
        _trigger(ticker="ETH", timestamp="2025-01-01 08:00"),
        _trigger(ticker="BTC", timestamp="2025-02-01 08:00"),
    ]
    matches_a = match_triggers_to_pops(trigs, pops)
    matches_b = match_triggers_to_pops(list(reversed(trigs)), list(reversed(pops)))
    assert match_counts(matches_a) == match_counts(matches_b)

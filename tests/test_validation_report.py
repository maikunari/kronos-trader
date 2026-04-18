"""Tests for validation/report.py — uses a mocked fetcher, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from setups.base import TPLevel, Trigger
from validation.report import (
    SetupDirectionStats,
    ValidationReport,
    _compute_stats,
    run_validation,
)


# ---------------------------------------------------------------------------
# SetupDirectionStats.passes_graduation
# ---------------------------------------------------------------------------

def _stats(recall=0.5, precision=0.3, capture=0.4, **kw):
    defaults = dict(
        setup="x", direction="long", pops=10, triggers=10,
        true_positives=5, false_negatives=5, false_positives=5,
        recall=recall, precision=precision, median_capture_ratio=capture,
        mean_realized_return=0.05, median_lead_hours=4.0, exit_reasons={},
    )
    defaults.update(kw)
    return SetupDirectionStats(**defaults)


def test_passes_graduation_all_above_thresholds():
    assert _stats(recall=0.5, precision=0.3, capture=0.4).passes_graduation


def test_fails_graduation_on_low_recall():
    assert not _stats(recall=0.35).passes_graduation


def test_fails_graduation_on_low_precision():
    assert not _stats(precision=0.20).passes_graduation


def test_fails_graduation_on_low_capture():
    assert not _stats(capture=0.20).passes_graduation


def test_passes_graduation_at_exact_thresholds():
    # Inclusive boundaries
    assert _stats(recall=0.40, precision=0.25, capture=0.30).passes_graduation


# ---------------------------------------------------------------------------
# run_validation — tiny end-to-end on synthetic data
# ---------------------------------------------------------------------------

def _synthetic_candles(
    start: str = "2025-01-01",
    n_bars: int = 1000,
    freq: str = "1h",
) -> pd.DataFrame:
    """Simple trending-with-noise series, with one clear 30% pop."""
    rng = np.random.default_rng(42)
    closes = 100 + rng.normal(0, 1, n_bars).cumsum() * 0.1
    # Inject a big 30% rally around bar 500
    closes[500:550] = np.linspace(closes[499], closes[499] * 1.30, 50)
    closes[550:] *= closes[549] / closes[550]   # hold new level
    highs = closes * 1.002
    lows = closes * 0.998
    ts = pd.date_range(start, periods=n_bars, freq=freq, tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": closes, "high": highs, "low": lows,
        "close": closes, "volume": [1000.0] * n_bars,
    })


class _NoOpDetector:
    """Never fires — gives a pure false-negative baseline."""
    name = "noop"

    def detect(self, ctx):
        return None


def _mock_fetcher_for(df: pd.DataFrame):
    """Return a fetcher that ignores ticker and returns the same df sliced by ms."""
    def fetcher(symbol, timeframe, start_ms, end_ms):
        ts_ms = df["timestamp"].apply(lambda t: int(t.timestamp() * 1000))
        mask = (ts_ms >= start_ms) & (ts_ms <= end_ms)
        return df.loc[mask].reset_index(drop=True)
    return fetcher


def test_run_validation_with_noop_detector_records_pops_and_zero_recall(tmp_path):
    df = _synthetic_candles(n_bars=800)
    fetcher = _mock_fetcher_for(df)

    report = run_validation(
        detectors=[_NoOpDetector()],
        tickers=["TEST"],
        timeframe="1h",
        start=df["timestamp"].iloc[0],
        end=df["timestamp"].iloc[-1],
        threshold_pct=0.20,
        window_hours=72,
        fetcher=fetcher,
        min_warm_bars=100,
    )
    # One pop (the 30% rally) should be labeled
    assert report.total_pops >= 1
    # NoOp detector emits zero triggers, so zero TPs and 100% FN rate
    assert report.total_triggers == 0
    for s in report.stats:
        assert s.true_positives == 0
        assert s.recall == 0.0


def test_run_validation_markdown_contains_expected_sections(tmp_path):
    df = _synthetic_candles(n_bars=800)
    fetcher = _mock_fetcher_for(df)
    report = run_validation(
        detectors=[_NoOpDetector()],
        tickers=["TEST"],
        timeframe="1h",
        start=df["timestamp"].iloc[0],
        end=df["timestamp"].iloc[-1],
        fetcher=fetcher,
        min_warm_bars=100,
    )
    md = report.format_markdown()
    assert "Phase B signal validation" in md
    assert "Results per setup × direction" in md
    assert "Pops per ticker" in md
    assert "Graduation gate" in md


# ---------------------------------------------------------------------------
# Direct trigger injection (bypasses detector scan)
# ---------------------------------------------------------------------------

def test_compute_stats_directly_injected_triggers_match_pops():
    """Feed _compute_stats a known trigger list + candle fetcher; verify TP and capture."""
    df = _synthetic_candles(n_bars=800)
    fetcher = _mock_fetcher_for(df)

    # The labeler places pop.start at the first bar where forward 72h shows 20%+.
    # For this synthetic series that's around bar 461. Place trigger just before.
    trigger_bar = 460
    trigger = Trigger(
        ticker="TEST",
        timestamp=df["timestamp"].iloc[trigger_bar],
        action="open_new", direction="long",
        entry_price=float(df["close"].iloc[trigger_bar]),
        stop_price=float(df["close"].iloc[trigger_bar]) * 0.97,
        tp_ladder=(TPLevel(price=float(df["close"].iloc[trigger_bar]) * 1.15, fraction=1.0, source="test"),),
        setup="divergence_reversal", confidence=0.8,
    )

    class _InjectedDetector:
        name = "divergence_reversal"
        def detect(self, ctx): return None

    # We're not going through the full scan — just run _compute_stats with explicit
    # trigger dict and labeled pops
    from validation.labeler import label_pops
    pops = label_pops(df, "TEST", threshold_pct=0.20, timeframe="1h")
    assert len(pops) >= 1

    stats = _compute_stats(
        detectors=[_InjectedDetector()],
        triggers_by_setup={"divergence_reversal": [trigger]},
        all_pops=pops,
        tickers=["TEST"],
        timeframe="1h",
        max_lead=pd.Timedelta(hours=24),
        max_lag=pd.Timedelta(hours=4),
        max_hold_bars=120,
        fetcher=fetcher,
        start=df["timestamp"].iloc[0],
        end=df["timestamp"].iloc[-1],
    )
    long_stats = next(s for s in stats if s.direction == "long")
    assert long_stats.triggers >= 1
    assert long_stats.true_positives >= 1
    # Capture should be positive (trigger + TP at +15% during a 30% rally)
    assert long_stats.median_capture_ratio > 0

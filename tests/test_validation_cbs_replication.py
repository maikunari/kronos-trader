"""Tests for validation/cbs_replication.py — uses mocked fetcher, no network."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from setups.base import TPLevel, Trigger
from validation.cbs_replication import (
    CBSTrade,
    ReplicationReport,
    TradeReplication,
    _parse_entry_date,
    load_cbs_trades,
    replicate_trade,
    run_replication,
)


# ---------------------------------------------------------------------------
# _parse_entry_date
# ---------------------------------------------------------------------------

def test_parse_entry_date_from_date_object():
    ts = _parse_entry_date(_dt.date(2026, 4, 5))
    assert ts == pd.Timestamp("2026-04-05", tz="UTC")


def test_parse_entry_date_from_string():
    ts = _parse_entry_date("2026-04-05")
    assert ts == pd.Timestamp("2026-04-05", tz="UTC")


def test_parse_entry_date_localizes_naive_datetime():
    ts = _parse_entry_date(_dt.datetime(2026, 4, 5, 12))
    assert ts.tzinfo is not None
    assert ts == pd.Timestamp("2026-04-05 12:00", tz="UTC")


# ---------------------------------------------------------------------------
# load_cbs_trades
# ---------------------------------------------------------------------------

def test_load_cbs_trades_parses_seed_yaml(tmp_path):
    p = tmp_path / "trades.yaml"
    p.write_text(
        "trades:\n"
        "  - id: hype_demo\n"
        "    ticker: HYPE\n"
        "    exchange: Hyperliquid\n"
        "    timeframe: 1h\n"
        "    direction: long\n"
        "    entry_date: 2026-04-05\n"
        "    entry_price_approx: 36.0\n"
        "    stop_initial: 35.69\n"
        "    targets: [40.25, 45.53]\n"
        "    status: open\n"
        "    setup_type: consolidation_break_then_trend\n"
    )
    trades = load_cbs_trades(p)
    assert len(trades) == 1
    t = trades[0]
    assert t.ticker == "HYPE"
    assert t.direction == "long"
    assert t.entry_date == pd.Timestamp("2026-04-05", tz="UTC")
    assert t.targets == (40.25, 45.53)


def test_load_cbs_trades_handles_real_seed_file():
    """The actual seed file should load cleanly."""
    seed = Path(__file__).resolve().parents[1] / "validation" / "cbs_trades.yaml"
    trades = load_cbs_trades(seed)
    assert len(trades) >= 2
    ids = {t.id for t in trades}
    assert "hype_long_2026_04_05" in ids
    assert "spx_long_2026_04_05" in ids


# ---------------------------------------------------------------------------
# replicate_trade — mocked fetcher scenarios
# ---------------------------------------------------------------------------

def _synthetic_candles(n_bars=800, start="2026-03-01", freq="1h"):
    """Simple trending series, enough bars to warm indicators."""
    rng = np.random.default_rng(1)
    closes = 100 + rng.normal(0, 0.5, n_bars).cumsum() * 0.1
    ts = pd.date_range(start, periods=n_bars, freq=freq, tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": closes, "high": closes * 1.002, "low": closes * 0.998,
        "close": closes, "volume": [1000.0] * n_bars,
    })


def _mock_fetcher(df):
    def fetcher(symbol, timeframe, start_ms, end_ms):
        ts_ms = df["timestamp"].apply(lambda t: int(t.timestamp() * 1000))
        mask = (ts_ms >= start_ms) & (ts_ms <= end_ms)
        return df.loc[mask].reset_index(drop=True)
    return fetcher


class _FixedTriggerDetector:
    """Fires one trigger when ctx.timestamp equals the configured moment."""

    def __init__(self, fire_at: pd.Timestamp, direction: str = "long", name: str = "mock"):
        self.name = name
        self._fire_at = pd.Timestamp(fire_at)
        if self._fire_at.tzinfo is None:
            self._fire_at = self._fire_at.tz_localize("UTC")
        self._direction = direction

    def detect(self, ctx):
        if ctx.timestamp != self._fire_at:
            return None
        return Trigger(
            ticker=ctx.ticker,
            timestamp=ctx.timestamp,
            action="open_new",
            direction=self._direction,
            entry_price=ctx.current_price,
            stop_price=ctx.current_price * 0.98,
            tp_ladder=(TPLevel(price=ctx.current_price * 1.10, fraction=1.0, source="mock"),),
            setup=self.name,
            confidence=0.7,
        )


def _trade_for(entry_date, direction="long", ticker="HYPE"):
    ts = pd.Timestamp(entry_date)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return CBSTrade(
        id="mock",
        ticker=ticker,
        exchange="Hyperliquid",
        timeframe="1h",
        direction=direction,
        entry_date=ts,
        entry_price_approx=100.0,
        stop_initial=98.0,
        targets=(110.0,),
        status="open",
        setup_type="test",
    )


def test_replicate_trade_matches_when_detector_fires_in_window():
    df = _synthetic_candles(n_bars=800, start="2026-03-01")
    # Trade date near end of series; fire detector at a bar whose timestamp
    # sits ~6 hours before entry_date.
    entry_date = pd.Timestamp("2026-03-25", tz="UTC")
    fire_at = entry_date - pd.Timedelta(hours=6)

    detector = _FixedTriggerDetector(fire_at=fire_at, direction="long")
    trade = _trade_for(entry_date, direction="long")

    result = replicate_trade(
        trade, [detector],
        match_window=pd.Timedelta(hours=24),
        lookback_days=20, post_days=2,
        min_warm_bars=100, ctx_lookback_bars=300,
        fetcher=_mock_fetcher(df),
    )

    assert result.matched is True
    assert result.best_trigger is not None
    assert result.best_trigger.setup == "mock"
    assert result.lead_time == pd.Timedelta(hours=6)
    assert result.gap_reason is None
    assert result.fetch_error is None


def test_replicate_trade_marks_wrong_direction_gap():
    df = _synthetic_candles(n_bars=800, start="2026-03-01")
    entry_date = pd.Timestamp("2026-03-25", tz="UTC")
    fire_at = entry_date - pd.Timedelta(hours=2)

    # Detector fires SHORT, but the CBS trade is LONG → wrong-direction gap
    detector = _FixedTriggerDetector(fire_at=fire_at, direction="short")
    trade = _trade_for(entry_date, direction="long")

    result = replicate_trade(
        trade, [detector],
        match_window=pd.Timedelta(hours=24),
        lookback_days=20, post_days=2,
        min_warm_bars=100, ctx_lookback_bars=300,
        fetcher=_mock_fetcher(df),
    )

    assert result.matched is False
    assert result.gap_reason == "wrong_direction_only"
    assert len(result.opposite_direction_triggers) == 1


def test_replicate_trade_marks_no_trigger_when_window_empty():
    df = _synthetic_candles(n_bars=800, start="2026-03-01")
    entry_date = pd.Timestamp("2026-03-25", tz="UTC")
    # Fire FAR outside the ±24h window — 10 days before entry
    fire_at = entry_date - pd.Timedelta(days=10)
    detector = _FixedTriggerDetector(fire_at=fire_at, direction="long")
    trade = _trade_for(entry_date, direction="long")

    result = replicate_trade(
        trade, [detector],
        match_window=pd.Timedelta(hours=24),
        lookback_days=20, post_days=2,
        min_warm_bars=100, ctx_lookback_bars=300,
        fetcher=_mock_fetcher(df),
    )

    assert result.matched is False
    assert result.gap_reason == "no_trigger_in_window"
    assert result.same_direction_triggers == ()
    assert result.opposite_direction_triggers == ()


def test_replicate_trade_records_fetch_error(monkeypatch):
    def failing_fetcher(*a, **kw):
        raise RuntimeError("unfetchable exchange")

    trade = _trade_for("2026-03-25", direction="long", ticker="SPX")
    result = replicate_trade(
        trade, [],
        lookback_days=20, post_days=2,
        min_warm_bars=100, ctx_lookback_bars=300,
        fetcher=failing_fetcher,
    )
    assert result.matched is False
    assert result.fetch_error and "unfetchable" in result.fetch_error


# ---------------------------------------------------------------------------
# Report aggregation
# ---------------------------------------------------------------------------

def test_run_replication_computes_match_rate_and_skips_fetch_errors():
    df = _synthetic_candles(n_bars=800, start="2026-03-01")
    entry_date = pd.Timestamp("2026-03-25", tz="UTC")
    fire_at = entry_date - pd.Timedelta(hours=3)

    # Trade A: ticker HYPE, fetched OK, detector fires → match
    detector = _FixedTriggerDetector(fire_at=fire_at, direction="long")
    trade_ok = _trade_for(entry_date, direction="long", ticker="HYPE")

    # Trade B: ticker SPX, fetcher errors → skipped from assessed
    trade_err = _trade_for(entry_date, direction="long", ticker="SPX")

    def mixed_fetcher(symbol, timeframe, start_ms, end_ms):
        if symbol == "SPX":
            raise RuntimeError("no feed")
        ts_ms = df["timestamp"].apply(lambda t: int(t.timestamp() * 1000))
        mask = (ts_ms >= start_ms) & (ts_ms <= end_ms)
        return df.loc[mask].reset_index(drop=True)

    report = run_replication(
        [trade_ok, trade_err], [detector],
        match_window=pd.Timedelta(hours=24),
        lookback_days=20, post_days=2,
        fetcher=mixed_fetcher,
    )
    assert len(report.results) == 2
    assert len(report.assessed) == 1
    assert report.match_rate == 1.0   # 1 of 1 assessed matched


def test_report_markdown_renders_key_sections():
    trade = _trade_for("2026-03-25", direction="long", ticker="HYPE")
    r = TradeReplication(
        trade=trade, matched=False, best_trigger=None,
        same_direction_triggers=(), opposite_direction_triggers=(),
        gap_reason="no_trigger_in_window", fetch_error=None,
        lead_time=None, bars_scanned=500,
    )
    report = ReplicationReport(
        results=[r],
        config={"detectors": ["mock"], "match_window_hours": 24,
                "lookback_days": 45, "post_days": 2, "total_trades": 1},
    )
    md = report.format_markdown()
    assert "CBS replication-fidelity check" in md
    assert "Per-trade results" in md
    assert "Per-trade diagnosis" in md
    assert "no_trigger_in_window" in md
    assert "HYPE" in md

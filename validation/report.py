"""
validation/report.py
Orchestrator and formatted-report for Phase B signal validation.

Ties labeler + matcher + capture simulator together and produces the
per-setup-per-direction recall/precision/capture stats that gate
Phase C — the graduation criteria from §9.5 of the architecture doc:

    recall    >= 40%
    precision >= 25%
    capture   >= 30%

Runs a detector across every bar of historical data for every ticker
in the universe, matches emitted triggers against labeled pops, and
simulates what the trades would have actually captured.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from data_cache import default_fetcher, get_candles
from setups.base import MarketContext, SetupDetector, Trigger
from validation.capture import (
    CaptureResult,
    exit_reason_breakdown,
    mean_realized_return,
    median_capture_ratio,
    simulate_capture,
)
from validation.labeler import PopEvent, label_pops, pop_stats
from validation.matcher import (
    Match,
    match_counts,
    match_triggers_to_pops,
    median_lead_time,
    precision,
    recall,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SetupDirectionStats:
    """Recall/precision/capture breakdown for one (setup, direction) cell."""
    setup: str
    direction: str
    pops: int
    triggers: int
    true_positives: int
    false_negatives: int
    false_positives: int
    recall: float
    precision: float
    median_capture_ratio: float
    mean_realized_return: float
    median_lead_hours: Optional[float]
    exit_reasons: dict[str, int]

    @property
    def passes_graduation(self) -> bool:
        """Per architecture §9.5."""
        return (
            self.recall >= 0.40
            and self.precision >= 0.25
            and self.median_capture_ratio >= 0.30
        )


@dataclass
class ValidationReport:
    stats: list[SetupDirectionStats]
    per_ticker_pops: dict[str, int]
    total_pops: int
    total_triggers: int
    config: dict = field(default_factory=dict)

    def format_markdown(self) -> str:
        return _format_markdown(self)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_validation(
    *,
    detectors: list[SetupDetector],
    tickers: list[str],
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    threshold_pct: float = 0.20,
    window_hours: float = 72.0,
    max_hold_bars: int = 72,
    max_lead: pd.Timedelta = pd.Timedelta(hours=24),
    max_lag: pd.Timedelta = pd.Timedelta(hours=4),
    min_warm_bars: int = 300,
    ctx_lookback_bars: int = 500,
    fetcher=None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> ValidationReport:
    """Run the validation pipeline across (detectors × tickers)."""
    fetcher = fetcher or default_fetcher
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")

    all_pops: list[PopEvent] = []
    per_ticker_pops: dict[str, int] = {}
    # triggers grouped by (setup, ticker) for matcher-per-setup later
    triggers_by_setup: dict[str, list[Trigger]] = {d.name: [] for d in detectors}

    for ticker in tickers:
        if progress_cb:
            progress_cb(f"fetch {ticker}")
        try:
            candles = get_candles(ticker, timeframe, start_ts, end_ts, fetcher=fetcher)
        except Exception as exc:
            logger.warning("skip %s: fetch failed: %s", ticker, exc)
            per_ticker_pops[ticker] = 0
            continue
        if len(candles) < min_warm_bars + 10:
            logger.info("skip %s: insufficient bars (%d)", ticker, len(candles))
            per_ticker_pops[ticker] = 0
            continue

        pops = label_pops(
            candles, ticker,
            threshold_pct=threshold_pct,
            timeframe=timeframe,
            window_hours=window_hours,
        )
        all_pops.extend(pops)
        per_ticker_pops[ticker] = len(pops)

        for detector in detectors:
            if progress_cb:
                progress_cb(f"scan {ticker}/{detector.name}")
            triggers_by_setup[detector.name].extend(
                _scan_ticker(detector, candles, ticker, timeframe,
                             min_warm_bars, ctx_lookback_bars)
            )

    stats = _compute_stats(
        detectors=detectors,
        triggers_by_setup=triggers_by_setup,
        all_pops=all_pops,
        tickers=tickers,
        timeframe=timeframe,
        max_lead=max_lead,
        max_lag=max_lag,
        max_hold_bars=max_hold_bars,
        fetcher=fetcher,
        start=start_ts,
        end=end_ts,
    )

    return ValidationReport(
        stats=stats,
        per_ticker_pops=per_ticker_pops,
        total_pops=len(all_pops),
        total_triggers=sum(len(v) for v in triggers_by_setup.values()),
        config={
            "timeframe": timeframe,
            "threshold_pct": threshold_pct,
            "window_hours": window_hours,
            "max_hold_bars": max_hold_bars,
            "max_lead_hours": max_lead.total_seconds() / 3600,
            "max_lag_hours": max_lag.total_seconds() / 3600,
            "tickers": list(tickers),
            "detectors": [d.name for d in detectors],
            "start": str(start_ts),
            "end": str(end_ts),
        },
    )


# ---------------------------------------------------------------------------
# Scan one ticker with one detector
# ---------------------------------------------------------------------------

def _scan_ticker(
    detector: SetupDetector,
    candles: pd.DataFrame,
    ticker: str,
    timeframe: str,
    min_warm_bars: int,
    lookback_bars: int,
) -> list[Trigger]:
    """Run `detector.detect(ctx)` at each bar from min_warm_bars onward."""
    triggers: list[Trigger] = []
    prev_trigger_time: Optional[pd.Timestamp] = None
    dedup_window = pd.Timedelta(hours=8)

    for t in range(min_warm_bars, len(candles)):
        window_start = max(0, t - lookback_bars)
        window = candles.iloc[window_start : t + 1].reset_index(drop=True)
        try:
            ctx = MarketContext.build(
                ticker=ticker, timeframe=timeframe, candles=window,
                compute_sr=True,
            )
        except ValueError:
            continue
        trig = detector.detect(ctx)
        if trig is None:
            continue
        # Dedup: don't fire again within dedup_window hours
        if prev_trigger_time is not None and trig.timestamp - prev_trigger_time < dedup_window:
            continue
        triggers.append(trig)
        prev_trigger_time = trig.timestamp
    return triggers


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

def _compute_stats(
    *,
    detectors: list[SetupDetector],
    triggers_by_setup: dict[str, list[Trigger]],
    all_pops: list[PopEvent],
    tickers: list[str],
    timeframe: str,
    max_lead: pd.Timedelta,
    max_lag: pd.Timedelta,
    max_hold_bars: int,
    fetcher,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[SetupDirectionStats]:
    """Match triggers to pops per (setup, direction) and simulate captures."""
    stats: list[SetupDirectionStats] = []

    # Cache candles per ticker so capture simulator doesn't refetch
    ticker_candles: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            ticker_candles[t] = get_candles(t, timeframe, start, end, fetcher=fetcher)
        except Exception:
            ticker_candles[t] = pd.DataFrame()

    for detector in detectors:
        trigs = triggers_by_setup[detector.name]
        for direction in ("long", "short"):
            trigs_dir = [t for t in trigs if t.direction == direction]
            pops_dir = [p for p in all_pops if p.direction == direction]

            matches = match_triggers_to_pops(
                trigs_dir, pops_dir, max_lead=max_lead, max_lag=max_lag,
            )
            counts = match_counts(matches)

            # Simulate capture on true positives
            captures: list[CaptureResult] = []
            for m in matches:
                if m.outcome != "true_positive":
                    continue
                candles = ticker_candles.get(m.ticker)
                if candles is None or candles.empty:
                    continue
                cap = simulate_capture(m.trigger, m.pop, candles, max_hold_bars=max_hold_bars)
                if cap:
                    captures.append(cap)

            lead = median_lead_time(matches)
            lead_hours = lead.total_seconds() / 3600 if lead else None

            stats.append(SetupDirectionStats(
                setup=detector.name,
                direction=direction,
                pops=len(pops_dir),
                triggers=len(trigs_dir),
                true_positives=counts["true_positive"],
                false_negatives=counts["false_negative"],
                false_positives=counts["false_positive"],
                recall=recall(matches),
                precision=precision(matches),
                median_capture_ratio=median_capture_ratio(captures),
                mean_realized_return=mean_realized_return(captures),
                median_lead_hours=lead_hours,
                exit_reasons=exit_reason_breakdown(captures),
            ))

    return stats


# ---------------------------------------------------------------------------
# Markdown formatter
# ---------------------------------------------------------------------------

def _format_markdown(report: ValidationReport) -> str:
    lines: list[str] = []
    cfg = report.config
    lines.append(f"# Phase B signal validation")
    lines.append("")
    lines.append(f"- Timeframe: **{cfg.get('timeframe')}**")
    lines.append(f"- Window: {cfg.get('start')} → {cfg.get('end')}")
    lines.append(f"- Pop threshold: {cfg.get('threshold_pct', 0) * 100:.0f}% in {cfg.get('window_hours')}h")
    lines.append(f"- Detectors: {', '.join(cfg.get('detectors', []))}")
    lines.append(f"- Tickers: {len(cfg.get('tickers', []))}")
    lines.append("")
    lines.append(f"**Totals: {report.total_pops} pops labeled, {report.total_triggers} triggers emitted**")
    lines.append("")

    # Summary table per (setup, direction)
    lines.append("## Results per setup × direction")
    lines.append("")
    header = (
        "| setup | dir | pops | trigs | TP | FN | FP | recall | prec | cap% | "
        "mean_ret% | lead_h | pass |"
    )
    sep = "|" + "|".join(["---"] * 13) + "|"
    lines.append(header)
    lines.append(sep)
    for s in report.stats:
        lines.append(
            f"| {s.setup} | {s.direction} | {s.pops} | {s.triggers} | "
            f"{s.true_positives} | {s.false_negatives} | {s.false_positives} | "
            f"{s.recall*100:.1f}% | {s.precision*100:.1f}% | "
            f"{s.median_capture_ratio*100:.1f}% | "
            f"{s.mean_realized_return*100:+.2f}% | "
            f"{s.median_lead_hours if s.median_lead_hours is not None else '—'} | "
            f"{'✓' if s.passes_graduation else '✗'} |"
        )
    lines.append("")

    # Per-ticker pop distribution
    lines.append("## Pops per ticker")
    lines.append("")
    lines.append("| ticker | pops |")
    lines.append("|---|---|")
    for t, n in sorted(report.per_ticker_pops.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {t} | {n} |")
    lines.append("")

    # Exit-reason breakdown per setup × direction
    lines.append("## Exit-reason breakdown (true positives only)")
    lines.append("")
    for s in report.stats:
        if not s.exit_reasons:
            continue
        parts = ", ".join(f"{k}={v}" for k, v in s.exit_reasons.items())
        lines.append(f"- **{s.setup} / {s.direction}**: {parts}")
    lines.append("")

    # Graduation summary
    lines.append("## Graduation gate (§9.5)")
    lines.append("")
    lines.append("Thresholds: recall ≥ 40%, precision ≥ 25%, median capture ≥ 30%.")
    lines.append("")
    passes = [s for s in report.stats if s.passes_graduation]
    fails = [s for s in report.stats if not s.passes_graduation]
    if passes:
        lines.append(f"**Passing ({len(passes)}):**")
        for s in passes:
            lines.append(f"- {s.setup} / {s.direction}")
    if fails:
        lines.append(f"")
        lines.append(f"**Failing ({len(fails)}):**")
        for s in fails:
            reasons = []
            if s.recall < 0.40:
                reasons.append(f"recall {s.recall*100:.1f}%")
            if s.precision < 0.25:
                reasons.append(f"precision {s.precision*100:.1f}%")
            if s.median_capture_ratio < 0.30:
                reasons.append(f"capture {s.median_capture_ratio*100:.1f}%")
            lines.append(f"- {s.setup} / {s.direction}: " + ", ".join(reasons))

    return "\n".join(lines)

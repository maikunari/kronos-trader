"""
validation/cbs_replication.py
Replication-fidelity check — do our detectors fire on CBS's actual trades?

For each trade in cbs_trades.yaml:

  1. Fetch candles for [entry_date - warmup, entry_date + post_window]
  2. Run each registered detector bar-by-bar, same loop as validation/report.
  3. Collect triggers firing within ±match_window of entry_date.
  4. Pick the best matching-direction trigger; report the gap otherwise.

This is a different question from the recall/precision validation. Recall
measures "do we catch labeled pops?" This measures "would we have opened
the same trade CBS did, at roughly the same time?" The answer is the
ground truth for which Phase C setups matter most.

Output: per-trade diagnosis + summary match rate. Written as markdown
(human) + json (machine) alongside each other, mirroring the Phase B
report layout.
"""
from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from data_cache import default_fetcher, get_candles
from setups.base import Direction, MarketContext, SetupDetector, Trigger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CBSTrade:
    """One documented CBS trade from the yaml seed corpus."""
    id: str
    ticker: str
    exchange: str
    timeframe: str
    direction: Direction
    entry_date: pd.Timestamp      # midnight UTC on the entry date
    entry_price_approx: Optional[float]
    stop_initial: Optional[float]
    targets: tuple[float, ...]
    status: str
    setup_type: str
    source: str = ""
    notes: str = ""


def _parse_entry_date(raw) -> pd.Timestamp:
    """YAML gives us date or string; normalise to a UTC midnight Timestamp."""
    if isinstance(raw, _dt.datetime):
        ts = pd.Timestamp(raw)
    elif isinstance(raw, _dt.date):
        ts = pd.Timestamp(raw.isoformat())
    else:
        ts = pd.Timestamp(str(raw))
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def load_cbs_trades(path: str | Path) -> list[CBSTrade]:
    """Parse cbs_trades.yaml into CBSTrade records."""
    data = yaml.safe_load(Path(path).read_text())
    out: list[CBSTrade] = []
    for raw in data.get("trades", []):
        targets = tuple(float(t) for t in (raw.get("targets") or []))
        out.append(CBSTrade(
            id=raw["id"],
            ticker=raw["ticker"],
            exchange=raw.get("exchange", ""),
            timeframe=raw.get("timeframe", "1h"),
            direction=raw["direction"],
            entry_date=_parse_entry_date(raw["entry_date"]),
            entry_price_approx=_optional_float(raw.get("entry_price_approx")),
            stop_initial=_optional_float(raw.get("stop_initial")),
            targets=targets,
            status=raw.get("status", ""),
            setup_type=raw.get("setup_type", ""),
            source=raw.get("source", ""),
            notes=(raw.get("notes") or "").strip(),
        ))
    return out


def _optional_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    return float(v)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TradeReplication:
    """Outcome of running our detectors against one CBS trade."""
    trade: CBSTrade
    matched: bool
    best_trigger: Optional[Trigger]
    same_direction_triggers: tuple[Trigger, ...]
    opposite_direction_triggers: tuple[Trigger, ...]
    gap_reason: Optional[str]
    fetch_error: Optional[str]
    lead_time: Optional[pd.Timedelta]   # entry_date - trigger.timestamp (+ve = trigger leads)
    bars_scanned: int


@dataclass
class ReplicationReport:
    results: list[TradeReplication]
    config: dict = field(default_factory=dict)

    @property
    def assessed(self) -> list[TradeReplication]:
        return [r for r in self.results if r.fetch_error is None]

    @property
    def match_rate(self) -> float:
        if not self.assessed:
            return 0.0
        return sum(1 for r in self.assessed if r.matched) / len(self.assessed)

    def format_markdown(self) -> str:
        return _format_markdown(self)


# ---------------------------------------------------------------------------
# Scan helpers
# ---------------------------------------------------------------------------

def _scan_bars(
    detector: SetupDetector,
    candles: pd.DataFrame,
    ticker: str,
    timeframe: str,
    *,
    min_warm_bars: int,
    lookback_bars: int,
) -> list[Trigger]:
    """Run `detector.detect(ctx)` at each bar from min_warm_bars onward.

    No dedup: we want every firing within the window, because a single
    trade may correspond to several near-simultaneous triggers and we
    want to see all of them for diagnosis.
    """
    out: list[Trigger] = []
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
        out.append(trig)
    return out


# ---------------------------------------------------------------------------
# Per-trade replication
# ---------------------------------------------------------------------------

def replicate_trade(
    trade: CBSTrade,
    detectors: list[SetupDetector],
    *,
    match_window: pd.Timedelta = pd.Timedelta(hours=24),
    lookback_days: int = 45,
    post_days: int = 2,
    min_warm_bars: int = 300,
    ctx_lookback_bars: int = 500,
    fetcher=None,
) -> TradeReplication:
    """Replay detectors against one CBS trade and report match status."""
    fetcher = fetcher or default_fetcher
    start = trade.entry_date - pd.Timedelta(days=lookback_days)
    end = trade.entry_date + pd.Timedelta(days=post_days)

    try:
        candles = get_candles(trade.ticker, trade.timeframe, start, end, fetcher=fetcher)
    except Exception as exc:
        logger.warning("replicate_trade: fetch failed for %s: %s", trade.ticker, exc)
        return TradeReplication(
            trade=trade, matched=False, best_trigger=None,
            same_direction_triggers=(), opposite_direction_triggers=(),
            gap_reason=None, fetch_error=str(exc), lead_time=None,
            bars_scanned=0,
        )

    if candles.empty or len(candles) < min_warm_bars + 10:
        return TradeReplication(
            trade=trade, matched=False, best_trigger=None,
            same_direction_triggers=(), opposite_direction_triggers=(),
            gap_reason="insufficient_data",
            fetch_error=None if not candles.empty else "no_candles",
            lead_time=None, bars_scanned=len(candles),
        )

    # Collect triggers across all detectors
    all_triggers: list[Trigger] = []
    for detector in detectors:
        all_triggers.extend(_scan_bars(
            detector, candles, trade.ticker, trade.timeframe,
            min_warm_bars=min_warm_bars, lookback_bars=ctx_lookback_bars,
        ))

    # Filter to trade's match window
    lower = trade.entry_date - match_window
    upper = trade.entry_date + match_window
    in_window = [t for t in all_triggers if lower <= t.timestamp <= upper]
    same_dir = tuple(t for t in in_window if t.direction == trade.direction)
    opp_dir = tuple(t for t in in_window if t.direction != trade.direction)

    if same_dir:
        best = min(
            same_dir,
            key=lambda t: (abs((t.timestamp - trade.entry_date).total_seconds()),
                           -t.confidence),
        )
        lead = trade.entry_date - best.timestamp
        return TradeReplication(
            trade=trade, matched=True, best_trigger=best,
            same_direction_triggers=same_dir,
            opposite_direction_triggers=opp_dir,
            gap_reason=None, fetch_error=None, lead_time=lead,
            bars_scanned=len(candles),
        )

    gap = "wrong_direction_only" if opp_dir else "no_trigger_in_window"
    return TradeReplication(
        trade=trade, matched=False, best_trigger=None,
        same_direction_triggers=(),
        opposite_direction_triggers=opp_dir,
        gap_reason=gap, fetch_error=None, lead_time=None,
        bars_scanned=len(candles),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_replication(
    trades: list[CBSTrade],
    detectors: list[SetupDetector],
    *,
    match_window: pd.Timedelta = pd.Timedelta(hours=24),
    lookback_days: int = 45,
    post_days: int = 2,
    fetcher=None,
    progress_cb=None,
) -> ReplicationReport:
    """Run every (trade × detector) replay and build a report."""
    results: list[TradeReplication] = []
    for trade in trades:
        if progress_cb:
            progress_cb(f"replicate {trade.id}")
        result = replicate_trade(
            trade, detectors,
            match_window=match_window,
            lookback_days=lookback_days,
            post_days=post_days,
            fetcher=fetcher,
        )
        results.append(result)

    return ReplicationReport(
        results=results,
        config={
            "detectors": [d.name for d in detectors],
            "match_window_hours": match_window.total_seconds() / 3600,
            "lookback_days": lookback_days,
            "post_days": post_days,
            "total_trades": len(trades),
        },
    )


# ---------------------------------------------------------------------------
# Markdown formatter
# ---------------------------------------------------------------------------

def _format_markdown(report: ReplicationReport) -> str:
    cfg = report.config
    lines: list[str] = []
    lines.append("# CBS replication-fidelity check")
    lines.append("")
    lines.append(f"- Detectors: {', '.join(cfg.get('detectors', []))}")
    lines.append(f"- Match window: ±{cfg.get('match_window_hours')}h around entry_date")
    lines.append(f"- Trades in corpus: **{cfg.get('total_trades')}**")
    lines.append("")
    lines.append(
        f"**Match rate: {report.match_rate*100:.0f}% "
        f"({sum(1 for r in report.assessed if r.matched)}/{len(report.assessed)} assessed)**"
    )
    if len(report.results) != len(report.assessed):
        skipped = len(report.results) - len(report.assessed)
        lines.append(f"*{skipped} trade(s) skipped due to fetch errors.*")
    lines.append("")

    # Per-trade table
    lines.append("## Per-trade results")
    lines.append("")
    lines.append("| id | ticker | dir | entry_date | matched | detector | lead | gap |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in report.results:
        t = r.trade
        if r.fetch_error:
            lines.append(
                f"| {t.id} | {t.ticker} | {t.direction} | "
                f"{t.entry_date.date()} | — | — | — | fetch_error: {r.fetch_error} |"
            )
            continue
        matched = "✓" if r.matched else "✗"
        detector = r.best_trigger.setup if r.best_trigger else "—"
        if r.lead_time is not None:
            lead = f"{r.lead_time.total_seconds()/3600:+.1f}h"
        else:
            lead = "—"
        gap = r.gap_reason or ""
        lines.append(
            f"| {t.id} | {t.ticker} | {t.direction} | "
            f"{t.entry_date.date()} | {matched} | {detector} | {lead} | {gap} |"
        )
    lines.append("")

    # Detailed diagnosis per trade
    lines.append("## Per-trade diagnosis")
    lines.append("")
    for r in report.results:
        t = r.trade
        lines.append(f"### {t.id} — {t.ticker} {t.direction} on {t.entry_date.date()}")
        lines.append("")
        lines.append(f"- CBS setup type: `{t.setup_type}`")
        if r.fetch_error:
            lines.append(f"- **Fetch error:** {r.fetch_error}")
            lines.append(f"- Exchange in yaml: {t.exchange} (may not be supported)")
            lines.append("")
            continue
        lines.append(f"- Bars scanned: {r.bars_scanned}")
        lines.append(
            f"- Same-direction triggers in window: {len(r.same_direction_triggers)}"
        )
        lines.append(
            f"- Opposite-direction triggers in window: {len(r.opposite_direction_triggers)}"
        )
        if r.matched and r.best_trigger is not None:
            bt = r.best_trigger
            lines.append(
                f"- **Best match:** `{bt.setup}` fired at {bt.timestamp} "
                f"(lead={r.lead_time.total_seconds()/3600:+.1f}h, "
                f"conf={bt.confidence:.2f}, entry={bt.entry_price:.4f}, "
                f"stop={bt.stop_price:.4f})"
            )
        else:
            lines.append(f"- **Gap:** {r.gap_reason}")
            if r.opposite_direction_triggers:
                t0 = r.opposite_direction_triggers[0]
                lines.append(
                    f"  - opposite-dir example: `{t0.setup}` {t0.direction} "
                    f"at {t0.timestamp} (conf={t0.confidence:.2f})"
                )
        lines.append("")

    return "\n".join(lines)

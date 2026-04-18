"""
validation/matcher.py
Match setup-trigger stream against labeled pops to compute recall/precision.

A trigger "catches" a pop if:
  * ticker and direction agree
  * trigger fires within [pop.start - max_lead, pop.start + max_lag]
    (lead = fired before the pop started; lag = fired slightly after start
    but before peak)
  * trigger fires before the pop's peak (otherwise we're too late)

Greedy assignment: each trigger can only match one pop. Each pop picks
the best matching trigger (highest confidence; tiebreak by earliest).

Outputs:
  * True positives  — pops that found a matching trigger
  * False negatives — pops with no trigger in window (misses)
  * False positives — triggers not attached to any pop (fake-outs)

Use counts + per-case details to compute recall = TP / (TP + FN)
and precision = TP / (TP + FP) in the report generator.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import pandas as pd

from setups.base import Trigger
from validation.labeler import PopEvent

logger = logging.getLogger(__name__)

Outcome = Literal["true_positive", "false_negative", "false_positive"]


@dataclass(frozen=True)
class Match:
    """Single pairing result from the matcher."""
    outcome: Outcome
    pop: Optional[PopEvent]
    trigger: Optional[Trigger]
    lead_time: Optional[pd.Timedelta] = None   # pop.timestamp - trigger.timestamp (positive = trigger leads)

    @property
    def ticker(self) -> str:
        if self.pop:
            return self.pop.ticker
        if self.trigger:
            return self.trigger.ticker
        return "?"

    @property
    def direction(self) -> str:
        if self.pop:
            return self.pop.direction
        if self.trigger:
            return self.trigger.direction
        return "?"


def match_triggers_to_pops(
    triggers: Iterable[Trigger],
    pops: Iterable[PopEvent],
    *,
    max_lead: pd.Timedelta = pd.Timedelta(hours=24),
    max_lag: pd.Timedelta = pd.Timedelta(hours=4),
) -> list[Match]:
    """
    Pair triggers against pops and return a flat list of Match outcomes.

    Each input can be a list, generator, or any iterable. Triggers and
    pops do not need to be sorted — the matcher sorts internally.
    """
    trigs = sorted(triggers, key=lambda t: (t.ticker, t.direction, t.timestamp))
    pop_list = sorted(pops, key=lambda p: (p.ticker, p.direction, p.timestamp))

    used_trigger_ids: set[int] = set()
    matches: list[Match] = []

    for pop in pop_list:
        candidates = _candidates_for_pop(trigs, used_trigger_ids, pop, max_lead, max_lag)
        if not candidates:
            matches.append(Match(outcome="false_negative", pop=pop, trigger=None))
            continue

        chosen = _best_candidate(candidates)
        used_trigger_ids.add(id(chosen))
        matches.append(Match(
            outcome="true_positive",
            pop=pop,
            trigger=chosen,
            lead_time=pop.timestamp - chosen.timestamp,
        ))

    for trig in trigs:
        if id(trig) in used_trigger_ids:
            continue
        matches.append(Match(outcome="false_positive", pop=None, trigger=trig))

    return matches


def _candidates_for_pop(
    triggers: list[Trigger],
    used: set[int],
    pop: PopEvent,
    max_lead: pd.Timedelta,
    max_lag: pd.Timedelta,
) -> list[Trigger]:
    lower = pop.timestamp - max_lead
    upper = pop.timestamp + max_lag
    out = []
    for t in triggers:
        if id(t) in used:
            continue
        if t.ticker != pop.ticker:
            continue
        if t.direction != pop.direction:
            continue
        if t.timestamp < lower or t.timestamp > upper:
            continue
        # Must fire before pop peak — otherwise we're chasing
        if t.timestamp >= pop.peak_timestamp:
            continue
        out.append(t)
    return out


def _best_candidate(candidates: list[Trigger]) -> Trigger:
    """Highest confidence; tiebreak by earliest timestamp."""
    return max(candidates, key=lambda t: (t.confidence, -t.timestamp.value))


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def match_counts(matches: list[Match]) -> dict[str, int]:
    out = {"true_positive": 0, "false_negative": 0, "false_positive": 0}
    for m in matches:
        out[m.outcome] += 1
    return out


def recall(matches: list[Match]) -> float:
    counts = match_counts(matches)
    denom = counts["true_positive"] + counts["false_negative"]
    return counts["true_positive"] / denom if denom > 0 else 0.0


def precision(matches: list[Match]) -> float:
    counts = match_counts(matches)
    denom = counts["true_positive"] + counts["false_positive"]
    return counts["true_positive"] / denom if denom > 0 else 0.0


def median_lead_time(matches: list[Match]) -> Optional[pd.Timedelta]:
    """Median lead time across true-positive matches (positive = trigger leads pop)."""
    leads = [m.lead_time for m in matches
             if m.outcome == "true_positive" and m.lead_time is not None]
    if not leads:
        return None
    # Convert to seconds for median, back to Timedelta
    seconds = sorted(l.total_seconds() for l in leads)
    mid = seconds[len(seconds) // 2]
    return pd.Timedelta(seconds=mid)

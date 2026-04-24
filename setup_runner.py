"""
setup_runner.py
Evaluate one generated setup detector against the Phase C validation
harness. Intended for subprocess use by setup_agent.py (process
isolation + hard timeout).

Public entry points:
  - run_setup(detector, tickers, timeframe, start, end) -> dict
      called in-process, returns the SetupDirectionStats-shaped summary
  - run_from_file(detector_path, config_json) -> dict
      subprocess entry; prints JSON metrics to stdout, nonzero exit on
      any failure (validation + timeout handled by the caller)

Mirrors strategy_runner.py's shape so both agents look the same from
the outside.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from setups.base import SetupDetector
from validation.report import run_validation

logger = logging.getLogger(__name__)


def _load_detector_from_file(path: str, **init_kwargs: Any) -> SetupDetector:
    """Import a generated detector module by file path and instantiate
    its `Setup` class with any overridden kwargs."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(f"_generated_{p.stem}", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # AST validator must run BEFORE this
    if not hasattr(module, "Setup"):
        raise AttributeError("module has no `Setup` class")
    return module.Setup(**init_kwargs)


def run_setup(
    detector: SetupDetector,
    tickers: list[str],
    timeframe: str,
    start: str,
    end: str,
    *,
    threshold_pct: float = 0.20,
    window_hours: float = 72.0,
) -> dict:
    """Run a single detector through validation.report.run_validation
    and return a JSON-serialisable stats dict."""
    report = run_validation(
        detectors=[detector],
        tickers=tickers,
        timeframe=timeframe,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
        threshold_pct=threshold_pct,
        window_hours=window_hours,
    )
    long_stats = _stats_for(report, detector.name, "long")
    short_stats = _stats_for(report, detector.name, "short")

    def _serialize(s):
        if s is None:
            return None
        d = asdict(s)
        # exit_reasons is a Counter/dict — make sure it's JSON-serialisable
        d["exit_reasons"] = dict(d.get("exit_reasons") or {})
        return d

    return {
        "setup_name": detector.name,
        "total_pops": report.total_pops,
        "total_triggers": report.total_triggers,
        "per_ticker_pops": dict(report.per_ticker_pops),
        "long": _serialize(long_stats),
        "short": _serialize(short_stats),
        "config": report.config,
    }


def _stats_for(report, setup_name: str, direction: str):
    for s in report.stats:
        if s.setup == setup_name and s.direction == direction:
            return s
    return None


def run_from_file(detector_path: str, config: dict) -> dict:
    """Subprocess entry point. Loads the detector module and runs
    validation with it; returns the stats dict."""
    detector = _load_detector_from_file(detector_path)
    return run_setup(
        detector,
        tickers=config["tickers"],
        timeframe=config.get("timeframe", "1h"),
        start=config["start"],
        end=config["end"],
        threshold_pct=float(config.get("threshold_pct", 0.20)),
        window_hours=float(config.get("window_hours", 72.0)),
    )


def _cli():
    """
    Subprocess CLI:

        python setup_runner.py <detector.py> <config.json>

    On success, prints a single JSON line of stats to stdout. Nonzero
    exit on any failure.
    """
    if len(sys.argv) != 3:
        print(
            "usage: setup_runner.py <detector.py> <config.json>",
            file=sys.stderr,
        )
        sys.exit(2)
    detector_path, config_path = sys.argv[1], sys.argv[2]
    with open(config_path) as f:
        config = json.load(f)
    stats = run_from_file(detector_path, config)
    print(json.dumps(stats, default=str))


if __name__ == "__main__":
    _cli()

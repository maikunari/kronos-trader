"""Tests for run_periods helpers — parsers and formatters, no network."""
from __future__ import annotations

import pytest

from run_periods import DEFAULT_PERIODS, parse_period_spec, print_comparison


def test_parse_period_spec_valid():
    specs = ["2024:2024-01-01:2024-12-31", "q1:2025-01-01:2025-03-31"]
    out = parse_period_spec(specs)
    assert out == [
        ("2024", "2024-01-01", "2024-12-31"),
        ("q1", "2025-01-01", "2025-03-31"),
    ]


def test_parse_period_spec_rejects_malformed():
    with pytest.raises(ValueError):
        parse_period_spec(["broken:2024-01-01"])


def test_default_periods_have_sane_shape():
    for name, start, end in DEFAULT_PERIODS:
        assert name
        assert start < end


def test_print_comparison_empty(capsys):
    print_comparison({}, symbol="BTC", timeframe="15m")
    assert "No results" in capsys.readouterr().out

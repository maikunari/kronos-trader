"""Tests for the grid-search optimizer — synthetic data only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from optimizer import aggregate_folds, build_engine_from_params, expand_grid, grid_search


def test_expand_grid_cartesian_product():
    grid = {"a": [1, 2], "b": [3, 4]}
    out = expand_grid(grid)
    assert len(out) == 4
    assert {"a": 1, "b": 3} in out
    assert {"a": 2, "b": 4} in out


def test_expand_grid_handles_scalars():
    out = expand_grid({"a": 1, "b": [2, 3]})
    assert out == [{"a": 1, "b": 2}, {"a": 1, "b": 3}]


def test_expand_grid_empty():
    assert expand_grid({}) == [{}]


def test_build_engine_from_params_uses_defaults_for_missing():
    engine = build_engine_from_params({"donchian_period": 15})
    assert engine.donchian_period == 15
    assert engine.keltner_period == 20   # default


def test_aggregate_folds_on_empty_returns_zeros():
    agg = aggregate_folds([])
    assert agg["folds"] == 0
    assert agg["total_trades"] == 0


def _synth(n: int, drift: float = 0.0008, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = np.zeros(n)
    ret[0] = rng.normal(drift, 0.002)
    for i in range(1, n):
        ret[i] = drift + 0.3 * (ret[i - 1] - drift) + rng.normal(0, 0.002)
    close = 100.0 * np.exp(np.cumsum(ret))
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC"),
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": np.full(n, 1000.0),
    })


def test_grid_search_runs_end_to_end_on_small_data():
    df = _synth(n=96 * 60, drift=0.0008, seed=2)   # ~60 days
    grid = {"donchian_period": [15, 25], "stop_atr_mult": [1.5]}
    results = grid_search(
        df, grid,
        in_sample_days=20, out_of_sample_days=7, embargo_days=1, step_days=7,
    )
    assert len(results) == 2
    # Sorted by median_oos_sharpe desc
    assert results[0]["median_oos_sharpe"] >= results[1]["median_oos_sharpe"]
    for r in results:
        assert "params" in r
        assert "folds" in r
        assert "median_oos_sharpe" in r

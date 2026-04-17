"""Unit tests for microstructure.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microstructure import (
    LiquidationCluster,
    basis_expansion_pct,
    basis_pct,
    cvd,
    cvd_slope,
    liquidation_proximity,
    oi_delta_pct,
)


# --- OI delta -----------------------------------------------------------------

def test_oi_delta_positive_when_oi_grows():
    s = pd.Series([100, 101, 103, 108, 115], dtype=float)
    assert oi_delta_pct(s, window=3) == pytest.approx((115 - 101) / 101)


def test_oi_delta_negative_when_oi_shrinks():
    s = pd.Series([100, 98, 95, 90], dtype=float)
    assert oi_delta_pct(s, window=3) == pytest.approx((90 - 100) / 100)


def test_oi_delta_nan_on_short_series():
    assert np.isnan(oi_delta_pct(pd.Series([100.0, 101.0]), window=3))


def test_oi_delta_nan_on_nonpositive_base():
    assert np.isnan(oi_delta_pct(pd.Series([0, 1, 2, 3], dtype=float), window=3))


# --- Basis --------------------------------------------------------------------

def test_basis_pct_premium_is_positive():
    assert basis_pct(perp_close=101.0, spot_close=100.0) == pytest.approx(0.01)


def test_basis_pct_discount_is_negative():
    assert basis_pct(perp_close=99.0, spot_close=100.0) == pytest.approx(-0.01)


def test_basis_pct_nan_on_nonpositive_spot():
    assert np.isnan(basis_pct(100.0, 0.0))


def test_basis_expansion_grows_into_premium():
    """Perp rises faster than spot -> expansion is positive."""
    perp = pd.Series([100, 100.5, 101.2, 102.0, 103.0], dtype=float)
    spot = pd.Series([100, 100.2, 100.4, 100.6, 100.8], dtype=float)
    exp = basis_expansion_pct(perp, spot, window=3)
    assert exp > 0


def test_basis_expansion_nan_on_short_series():
    perp = pd.Series([100.0, 101.0])
    spot = pd.Series([100.0, 100.5])
    assert np.isnan(basis_expansion_pct(perp, spot, window=3))


# --- CVD ----------------------------------------------------------------------

def test_cvd_accumulates_imbalance_correctly():
    taker_buy = pd.Series([60, 40, 70], dtype=float)
    total = pd.Series([100, 100, 100], dtype=float)
    # Deltas per bar: +20, -20, +40
    result = cvd(taker_buy, total).tolist()
    assert result == [20, 0, 40]


def test_cvd_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        cvd(pd.Series([1, 2, 3]), pd.Series([1, 2]))


def test_cvd_slope_positive_under_accumulation():
    """A linearly rising CVD should give a clearly positive slope."""
    series = pd.Series(np.arange(50, dtype=float))
    assert cvd_slope(series, window=20) == pytest.approx(1.0, abs=1e-9)


def test_cvd_slope_negative_under_distribution():
    series = pd.Series(-np.arange(50, dtype=float))
    assert cvd_slope(series, window=20) == pytest.approx(-1.0, abs=1e-9)


def test_cvd_slope_nan_on_short_series():
    assert np.isnan(cvd_slope(pd.Series([1.0, 2.0, 3.0]), window=20))


# --- Liquidation proximity ----------------------------------------------------

def test_liquidation_proximity_long_finds_upside_short_cluster():
    clusters = [
        LiquidationCluster(price=99.0, volume=50_000, side="long"),   # downside long liqs
        LiquidationCluster(price=102.0, volume=200_000, side="short"),  # nearest upside short liqs
        LiquidationCluster(price=108.0, volume=300_000, side="short"),
    ]
    dist, vol = liquidation_proximity(current_price=100.0, clusters=clusters, direction="long")
    assert dist == pytest.approx(0.02)
    assert vol == 200_000


def test_liquidation_proximity_short_finds_downside_long_cluster():
    clusters = [
        LiquidationCluster(price=99.0, volume=50_000, side="long"),
        LiquidationCluster(price=95.0, volume=300_000, side="long"),
        LiquidationCluster(price=102.0, volume=200_000, side="short"),
    ]
    dist, vol = liquidation_proximity(current_price=100.0, clusters=clusters, direction="short")
    assert dist == pytest.approx(0.01)
    assert vol == 50_000


def test_liquidation_proximity_returns_none_when_no_cluster_in_direction():
    clusters = [LiquidationCluster(price=95.0, volume=50_000, side="long")]
    dist, vol = liquidation_proximity(current_price=100.0, clusters=clusters, direction="long")
    assert dist is None and vol is None


def test_liquidation_proximity_ignores_clusters_beyond_max_distance():
    clusters = [LiquidationCluster(price=110.0, volume=200_000, side="short")]
    dist, vol = liquidation_proximity(
        current_price=100.0, clusters=clusters, direction="long",
        max_distance_pct=0.05,
    )
    assert dist is None and vol is None


def test_liquidation_proximity_rejects_bad_direction():
    with pytest.raises(ValueError):
        liquidation_proximity(100.0, [], direction="sideways")

"""Unit tests for regime.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime import (
    RegimeDetector,
    RegimeState,
    hurst_rs,
    realized_vol,
    rv_quintile,
)


# --- Hurst --------------------------------------------------------------------

def _ar1_prices(n: int, phi: float, sigma: float, seed: int) -> np.ndarray:
    """Prices built from AR(1) log-returns: r_t = phi * r_{t-1} + eps_t."""
    rng = np.random.default_rng(seed)
    ret = np.zeros(n)
    ret[0] = rng.normal(0, sigma)
    for i in range(1, n):
        ret[i] = phi * ret[i - 1] + rng.normal(0, sigma)
    return 100 * np.exp(np.cumsum(ret))


def test_hurst_random_walk_is_near_half():
    """AR(1) with phi=0 (IID returns) ≡ random walk → H ≈ 0.5."""
    h = hurst_rs(_ar1_prices(5_000, phi=0.0, sigma=0.01, seed=0))
    assert 0.40 <= h <= 0.60, f"random walk Hurst {h:.3f} not near 0.5"


def test_hurst_positive_ar1_is_persistent():
    """Positively autocorrelated returns → H > 0.5 (persistent / momentum)."""
    h = hurst_rs(_ar1_prices(5_000, phi=0.6, sigma=0.01, seed=1))
    assert h > 0.60, f"positive AR(1) Hurst {h:.3f} not > 0.6"


def test_hurst_negative_ar1_is_mean_reverting():
    """Negatively autocorrelated returns → H < 0.5 (mean reverting).

    Note: R/S is known to biased upward for anti-persistent series (slow
    convergence toward the true H<0.5), so we assert H below 0.5 rather
    than some sharper bound.
    """
    h = hurst_rs(_ar1_prices(5_000, phi=-0.8, sigma=0.01, seed=2))
    assert h < 0.50, f"negative AR(1) Hurst {h:.3f} not < 0.5"


def test_hurst_short_series_returns_half():
    assert hurst_rs(np.array([100.0, 101.0, 100.5])) == 0.5


def test_hurst_nonpositive_prices_returns_half():
    assert hurst_rs(np.array([100.0, 50.0, -1.0, 10.0])) == 0.5


# --- Realized vol -------------------------------------------------------------

def test_realized_vol_matches_hand_calc():
    rng = np.random.default_rng(3)
    sigma_per_bar = 0.005
    log_ret = rng.normal(0, sigma_per_bar, 10_000)
    prices = 100 * np.exp(np.cumsum(log_ret))
    rv = realized_vol(prices, window=10_000 - 1, bars_per_year=35_040)
    # Expected annualized: sigma_bar * sqrt(bars_per_year)
    expected = sigma_per_bar * np.sqrt(35_040)
    assert abs(rv - expected) / expected < 0.05


def test_realized_vol_insufficient_data_is_nan():
    rv = realized_vol(np.array([100.0, 101.0]), window=10)
    assert np.isnan(rv)


def test_rv_quintile_in_valid_range():
    rng = np.random.default_rng(4)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, 5_000)))
    q = rv_quintile(prices, window=96, hist_window=2_000)
    assert 0 <= q <= 4


# --- RegimeDetector -----------------------------------------------------------

def _df(closes: np.ndarray) -> pd.DataFrame:
    highs = closes * 1.001
    lows = closes * 0.999
    ts = pd.date_range("2025-01-01", periods=len(closes), freq="15min", tz="UTC")
    return pd.DataFrame(
        {"timestamp": ts, "open": closes, "high": highs, "low": lows,
         "close": closes, "volume": np.full_like(closes, 1000.0)}
    )


def test_classify_insufficient_data():
    detector = RegimeDetector(hurst_window=200, rv_window=96, rv_hist_window=2880)
    df = _df(np.linspace(100, 110, 20))
    state = detector.classify(df)
    assert not state.is_trending
    assert state.skip_reason == "insufficient_data"


def test_classify_clean_uptrend_is_trending():
    """A trend with positive return autocorrelation (realistic momentum regime)
    should pass both the ADX and Hurst gates."""
    detector = RegimeDetector(
        adx_period=14, adx_threshold=20, hurst_window=500, hurst_threshold=0.50,
        rv_window=96, rv_hist_window=500,
    )
    # AR(1) positive-drift returns -> uptrend with momentum. SNR ~= 1 so the
    # ADX(14) at the tail still reads trending (ADX is a local measure).
    rng = np.random.default_rng(5)
    n = 1_500
    drift = 0.002
    phi = 0.5
    sigma = 0.002
    ret = np.zeros(n)
    ret[0] = rng.normal(drift, sigma)
    for i in range(1, n):
        ret[i] = drift + phi * (ret[i - 1] - drift) + rng.normal(0, sigma)
    closes = 100 * np.exp(np.cumsum(ret))
    state = detector.classify(_df(closes))
    assert state.is_trending, f"expected trending, got {state}"
    assert state.adx >= 20
    assert state.hurst >= 0.50


def test_classify_flat_noise_is_not_trending():
    detector = RegimeDetector(
        adx_period=14, adx_threshold=20, hurst_window=200, hurst_threshold=0.50,
        rv_window=96, rv_hist_window=500,
    )
    rng = np.random.default_rng(6)
    closes = 100 + rng.normal(0, 0.05, 800).cumsum() * 0.1  # near-flat
    state = detector.classify(_df(closes))
    # Flat noise should fail at least one of ADX/Hurst
    assert not state.is_trending
    assert state.skip_reason.startswith(("adx_too_low", "hurst_too_low"))


def test_is_trending_shortcut_matches_classify():
    detector = RegimeDetector(rv_hist_window=500)
    df = _df(np.linspace(100, 200, 800) + np.random.default_rng(7).normal(0, 0.2, 800))
    assert detector.is_trending(df) == detector.classify(df).is_trending

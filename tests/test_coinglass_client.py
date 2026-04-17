"""Unit tests for coinglass_client.py — all mocked, no network access."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from coinglass_client import CoinglassClient


def _mock_session(payload: dict | None = None, raise_exc: Exception | None = None):
    session = MagicMock(spec=requests.Session)
    if raise_exc is not None:
        session.get.side_effect = raise_exc
        return session
    resp = MagicMock()
    resp.json.return_value = payload or {}
    resp.raise_for_status.return_value = None
    session.get.return_value = resp
    return session


# --- enabled / key handling ---------------------------------------------------

def test_no_api_key_returns_empty_and_does_not_call(monkeypatch):
    monkeypatch.delenv("COINGLASS_API_KEY", raising=False)
    session = _mock_session()
    client = CoinglassClient(api_key=None, session=session)
    assert not client.enabled
    assert client.fetch_liquidation_clusters("BTC", current_price=50_000) == []
    session.get.assert_not_called()


def test_explicit_api_key_overrides_env(monkeypatch):
    monkeypatch.setenv("COINGLASS_API_KEY", "from-env")
    client = CoinglassClient(api_key="explicit-key")
    assert client.api_key == "explicit-key"


# --- Symbol normalization -----------------------------------------------------

def test_normalize_symbol_appends_usdt():
    assert CoinglassClient._normalize_symbol("btc") == "BTCUSDT"
    assert CoinglassClient._normalize_symbol("BTC") == "BTCUSDT"


def test_normalize_symbol_preserves_already_suffixed():
    assert CoinglassClient._normalize_symbol("BTCUSDT") == "BTCUSDT"
    assert CoinglassClient._normalize_symbol("ETHUSD") == "ETHUSD"


# --- Heatmap parsing ----------------------------------------------------------

def test_fetch_parses_heatmap_and_classifies_sides():
    payload = {
        "code": "0",
        "msg": "ok",
        "data": {
            "y": [49_000, 50_000, 51_000],
            "liq": [
                [0, 0, 100_000],   # $100k @ 49,000 (below current -> long)
                [1, 0, 50_000],    # another $50k @ 49,000
                [2, 2, 200_000],   # $200k @ 51,000 (above current -> short)
            ],
        },
    }
    client = CoinglassClient(api_key="k", session=_mock_session(payload))
    clusters = client.fetch_liquidation_clusters("BTC", current_price=50_000)

    by_price = {c.price: c for c in clusters}
    assert by_price[49_000].volume == 150_000
    assert by_price[49_000].side == "long"
    assert by_price[51_000].volume == 200_000
    assert by_price[51_000].side == "short"


def test_fetch_returns_empty_on_api_error_code():
    payload = {"code": "50001", "msg": "rate limited", "data": {}}
    client = CoinglassClient(api_key="k", session=_mock_session(payload))
    assert client.fetch_liquidation_clusters("BTC", current_price=50_000) == []


def test_fetch_returns_empty_on_request_exception():
    client = CoinglassClient(
        api_key="k", session=_mock_session(raise_exc=requests.ConnectionError("boom"))
    )
    assert client.fetch_liquidation_clusters("BTC", current_price=50_000) == []


def test_fetch_returns_empty_on_malformed_json():
    payload = {"code": "0", "data": {"y": [], "liq": []}}
    client = CoinglassClient(api_key="k", session=_mock_session(payload))
    assert client.fetch_liquidation_clusters("BTC", current_price=50_000) == []


def test_fetch_ignores_invalid_grid_entries():
    payload = {
        "code": "0",
        "data": {
            "y": [100.0, 110.0],
            "liq": [
                [0, 0, 1_000],         # valid
                [0, 99, 5_000],        # out-of-range index
                [0, "bad", 2_000],     # non-numeric
                [0, 1, -3_000],        # non-positive volume
                [0, 1, 500],           # valid
            ],
        },
    }
    client = CoinglassClient(api_key="k", session=_mock_session(payload))
    clusters = client.fetch_liquidation_clusters("BTC", current_price=105.0)
    by_price = {c.price: c.volume for c in clusters}
    assert by_price == {100.0: 1_000, 110.0: 500}

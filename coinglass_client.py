"""
coinglass_client.py
Thin HTTP client for Coinglass liquidation heatmap data.

Coinglass aggregates liquidation levels across major exchanges. We use it
specifically to populate the `microstructure.liquidation_proximity` feature
— price magnetism toward clusters is a measurable short-TF edge.

API key goes in the COINGLASS_API_KEY env var. If unset or a request fails,
fetch_liquidation_clusters returns an empty list and logs a warning — the
caller should treat absence of data as "no liquidation signal", not an error.

Coinglass v4 docs: https://docs.coinglass.com/
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import requests

from microstructure import LiquidationCluster

logger = logging.getLogger(__name__)

COINGLASS_BASE_URL = "https://open-api-v4.coinglass.com"
DEFAULT_TIMEOUT = 10.0


class CoinglassClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = COINGLASS_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("COINGLASS_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    # ------------------------------------------------------------------
    # Liquidation heatmap
    # ------------------------------------------------------------------

    def fetch_liquidation_clusters(
        self,
        symbol: str,
        current_price: float,
        exchange: str = "Binance",
        range_: str = "24h",
    ) -> list[LiquidationCluster]:
        """
        Fetch aggregated liquidation clusters for `symbol`.

        Each cluster is a (price, volume, side) triple classified as "long"
        (price < current_price, meaning longs would liquidate on a drop) or
        "short" (price > current_price). Clusters are what
        `microstructure.liquidation_proximity` consumes.

        Returns [] if API key is missing or the request fails — callers
        should treat that as "no signal", not as an error.
        """
        if not self.enabled:
            logger.debug("Coinglass API key not set — returning no liquidation clusters")
            return []

        url = f"{self.base_url}/api/futures/liquidation/heatmap/model1"
        params = {"symbol": self._normalize_symbol(symbol), "exchange": exchange, "range": range_}
        headers = {"accept": "application/json", "CG-API-KEY": self.api_key}

        try:
            resp = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Coinglass heatmap request failed: %s", exc)
            return []

        return self._parse_heatmap(payload, current_price)

    # ------------------------------------------------------------------
    # Internal parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """HL uses bare 'BTC'; Coinglass expects 'BTCUSDT' for most perps."""
        s = symbol.upper()
        if s.endswith("USDT") or s.endswith("USD"):
            return s
        return f"{s}USDT"

    @staticmethod
    def _parse_heatmap(payload: dict, current_price: float) -> list[LiquidationCluster]:
        """
        Coinglass v4 model1 payload shape (documented):
          {
            "code": "0",
            "data": {
              "y": [price_level_0, price_level_1, ...],       # monotonic price axis
              "liq": [[time_idx, price_idx, volume_usd], ...]  # sparse grid
            }
          }

        We sum volume at each price level across all time buckets, then
        classify each level as "long" (below current price) or "short"
        (above) per the LiquidationCluster convention.
        """
        if str(payload.get("code")) != "0":
            logger.warning("Coinglass response error: %s", payload.get("msg"))
            return []

        data = payload.get("data") or {}
        price_axis = data.get("y") or []
        grid = data.get("liq") or []
        if not price_axis or not grid:
            return []

        # Aggregate volume per price level
        totals: dict[int, float] = {}
        for entry in grid:
            if len(entry) < 3:
                continue
            _, price_idx, vol = entry[0], entry[1], entry[2]
            try:
                idx = int(price_idx)
                vol_f = float(vol)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(price_axis) and vol_f > 0:
                totals[idx] = totals.get(idx, 0.0) + vol_f

        clusters: list[LiquidationCluster] = []
        for idx, vol in totals.items():
            try:
                price = float(price_axis[idx])
            except (TypeError, ValueError):
                continue
            if price <= 0:
                continue
            side = "long" if price < current_price else "short"
            clusters.append(LiquidationCluster(price=price, volume=vol, side=side))

        return clusters

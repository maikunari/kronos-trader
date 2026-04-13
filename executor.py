"""
executor.py
Hyperliquid order execution layer.
Supports live and paper trading modes.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str]
    filled_price: float
    size: float
    side: str
    mode: str   # "live" | "paper"
    error: Optional[str] = None


class Executor:
    """
    Places orders on Hyperliquid perps.

    In paper mode: logs the intended trade without touching the exchange.
    In live mode: uses hyperliquid-python-sdk to place real orders.
    """

    def __init__(self, mode: str = "paper", mainnet: bool = True):
        if mode not in ("live", "paper"):
            raise ValueError(f"Invalid mode: {mode}. Use 'live' or 'paper'.")
        self.mode = mode
        self.mainnet = mainnet
        self._exchange = None
        self._info = None

    def _load_client(self):
        """Lazy-load Hyperliquid SDK client."""
        if self._exchange is not None:
            return

        from hyperliquid.exchange import Exchange  # type: ignore
        from hyperliquid.info import Info  # type: ignore
        from eth_account import Account  # type: ignore

        private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY")
        wallet_address = os.getenv("HYPERLIQUID_WALLET_ADDRESS")

        if not private_key or not wallet_address:
            raise EnvironmentError(
                "HYPERLIQUID_PRIVATE_KEY and HYPERLIQUID_WALLET_ADDRESS must be set in .env"
            )

        account = Account.from_key(private_key)
        base_url = "https://api.hyperliquid.xyz" if self.mainnet else "https://api.hyperliquid-testnet.xyz"

        self._info = Info(base_url, skip_ws=True)
        self._exchange = Exchange(account, base_url, account_address=wallet_address)
        logger.info(f"Hyperliquid client loaded ({'mainnet' if self.mainnet else 'testnet'})")

    def get_account_equity(self) -> float:
        """Fetch current account equity from Hyperliquid."""
        if self.mode == "paper":
            return 10000.0  # placeholder for paper mode

        self._load_client()
        wallet_address = os.getenv("HYPERLIQUID_WALLET_ADDRESS", "")
        state = self._info.user_state(wallet_address)
        return float(state["marginSummary"]["accountValue"])

    def market_open(self, symbol: str, side: str, size_usd: float, current_price: float) -> OrderResult:
        """
        Open a market position.

        Args:
            symbol: e.g. "SOL"
            side: "long" or "short"
            size_usd: USD notional value of position
            current_price: current market price (for size calculation)
        """
        is_buy = side == "long"
        size_contracts = round(size_usd / current_price, 4)

        if self.mode == "paper":
            logger.info(
                f"[PAPER] OPEN {side.upper()} {symbol} | "
                f"{size_contracts:.4f} contracts @ ~{current_price:.4f} | ${size_usd:.2f}"
            )
            return OrderResult(
                success=True,
                order_id=f"paper-{id(self)}",
                filled_price=current_price,
                size=size_contracts,
                side=side,
                mode="paper",
            )

        self._load_client()
        try:
            result = self._exchange.market_open(symbol, is_buy, size_contracts)
            logger.info(f"Order placed: {result}")
            filled_price = float(result.get("response", {}).get("data", {}).get("statuses", [{}])[0].get("filled", {}).get("avgPx", current_price))
            return OrderResult(
                success=True,
                order_id=str(result),
                filled_price=filled_price,
                size=size_contracts,
                side=side,
                mode="live",
            )
        except Exception as e:
            logger.error(f"Order failed: {e}")
            return OrderResult(
                success=False,
                order_id=None,
                filled_price=0.0,
                size=0.0,
                side=side,
                mode="live",
                error=str(e),
            )

    def market_close(self, symbol: str, side: str, size_contracts: float, current_price: float) -> OrderResult:
        """Close an existing position at market."""
        is_buy = side == "short"  # closing a short = buy; closing a long = sell

        if self.mode == "paper":
            logger.info(
                f"[PAPER] CLOSE {symbol} {side} | "
                f"{size_contracts:.4f} contracts @ ~{current_price:.4f}"
            )
            return OrderResult(
                success=True,
                order_id=f"paper-close-{id(self)}",
                filled_price=current_price,
                size=size_contracts,
                side=side,
                mode="paper",
            )

        self._load_client()
        try:
            result = self._exchange.market_close(symbol)
            filled_price = float(result.get("response", {}).get("data", {}).get("statuses", [{}])[0].get("filled", {}).get("avgPx", current_price))
            return OrderResult(
                success=True,
                order_id=str(result),
                filled_price=filled_price,
                size=size_contracts,
                side=side,
                mode="live",
            )
        except Exception as e:
            logger.error(f"Close order failed: {e}")
            return OrderResult(
                success=False,
                order_id=None,
                filled_price=current_price,
                size=size_contracts,
                side=side,
                mode="live",
                error=str(e),
            )

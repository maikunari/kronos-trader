"""
risk_manager.py
Position sizing, daily loss kill switch, and trade approval.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeApproval:
    approved: bool
    reason: str
    position_size_usd: float = 0.0
    position_size_contracts: float = 0.0


@dataclass
class DailyStats:
    date: date = field(default_factory=date.today)
    starting_equity: float = 0.0
    realized_pnl: float = 0.0
    trades_taken: int = 0
    trades_won: int = 0
    halted: bool = False

    @property
    def loss_pct(self) -> float:
        if self.starting_equity == 0:
            return 0.0
        return self.realized_pnl / self.starting_equity

    @property
    def win_rate(self) -> float:
        if self.trades_taken == 0:
            return 0.0
        return self.trades_won / self.trades_taken


class RiskManager:
    def __init__(
        self,
        initial_equity: float,
        max_position_pct: float = 0.10,
        daily_loss_limit_pct: float = 0.03,
        max_concurrent_positions: int = 1,
    ):
        self.equity = initial_equity
        self.max_position_pct = max_position_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_concurrent_positions = max_concurrent_positions

        self._daily = DailyStats(
            date=date.today(),
            starting_equity=initial_equity,
        )
        self._open_positions = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def approve_trade(self, signal_action: str, current_price: float) -> TradeApproval:
        """Check if a new trade is allowed and calculate position size."""
        self._check_day_rollover()

        if self._daily.halted:
            return TradeApproval(
                approved=False,
                reason=f"Trading halted for today — daily loss limit hit "
                       f"({self._daily.loss_pct:.2%})",
            )

        if self._open_positions >= self.max_concurrent_positions:
            return TradeApproval(
                approved=False,
                reason=f"Max concurrent positions reached ({self.max_concurrent_positions})",
            )

        if signal_action == "flat":
            return TradeApproval(approved=False, reason="No signal")

        size_usd = self.equity * self.max_position_pct
        size_contracts = size_usd / current_price

        logger.info(
            f"Trade approved: {signal_action.upper()} | "
            f"size=${size_usd:.2f} ({size_contracts:.4f} contracts @ {current_price:.4f})"
        )
        return TradeApproval(
            approved=True,
            reason="OK",
            position_size_usd=size_usd,
            position_size_contracts=size_contracts,
        )

    def on_trade_open(self):
        self._open_positions += 1
        self._daily.trades_taken += 1
        logger.debug(f"Trade opened. Open positions: {self._open_positions}")

    def on_trade_close(self, pnl: float, won: bool):
        self._open_positions = max(0, self._open_positions - 1)
        self.equity += pnl
        self._daily.realized_pnl += pnl
        if won:
            self._daily.trades_won += 1

        logger.info(
            f"Trade closed: pnl=${pnl:.2f} | equity=${self.equity:.2f} | "
            f"daily pnl={self._daily.realized_pnl:.2f} ({self._daily.loss_pct:.2%})"
        )

        # Check daily loss limit
        if self._daily.loss_pct <= -self.daily_loss_limit_pct:
            self._daily.halted = True
            logger.warning(
                f"DAILY LOSS LIMIT HIT: {self._daily.loss_pct:.2%} "
                f"(limit: -{self.daily_loss_limit_pct:.2%}) — trading halted for today"
            )

    def get_daily_stats(self) -> DailyStats:
        return self._daily

    def get_equity(self) -> float:
        return self.equity

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_day_rollover(self):
        today = date.today()
        if self._daily.date != today:
            logger.info(
                f"New trading day. Previous: trades={self._daily.trades_taken}, "
                f"win_rate={self._daily.win_rate:.1%}, pnl=${self._daily.realized_pnl:.2f}"
            )
            self._daily = DailyStats(date=today, starting_equity=self.equity)

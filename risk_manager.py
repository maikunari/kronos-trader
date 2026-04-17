"""
risk_manager.py
Position sizing, kill switches, and trade approval for the Phase 1 sniper.

Swapped out the flat 10%-per-trade rule for volatility targeting; added
multiple kill switches that the paper/live loop can reason about. Keeps
the `approve_trade / on_trade_open / on_trade_close` surface so backtests
and legacy callers don't need major rewrites.

Sizing:
  When the caller supplies `instrument_annual_vol` (e.g. RegimeState.rv),
  position size = equity * target_annual_vol / instrument_annual_vol,
  clipped to max_position_pct * equity. Otherwise falls back to a flat
  max_position_pct * equity so legacy callers still work.

Kill switches (`TradeApproval.approved=False` until resolved):
  - Daily loss limit hit
  - N consecutive losses (cool-off period)
  - Rolling 7-day drawdown > threshold (not a halt — halves sizing)
  - Manual tripwire file on disk (watchdog signal from outside the process)
  - max_concurrent_positions already open

Sizing breakdown is returned alongside the decision for observability /
dashboard use.
"""
from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Deque, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TradeApproval:
    approved: bool
    reason: str
    position_size_usd: float = 0.0
    position_size_contracts: float = 0.0
    sizing_breakdown: dict = field(default_factory=dict)


@dataclass
class DailyStats:
    date: date = field(default_factory=lambda: datetime.now(timezone.utc).date())
    starting_equity: float = 0.0
    realized_pnl: float = 0.0
    trades_taken: int = 0
    trades_won: int = 0
    halted: bool = False

    @property
    def loss_pct(self) -> float:
        return self.realized_pnl / self.starting_equity if self.starting_equity > 0 else 0.0

    @property
    def win_rate(self) -> float:
        return self.trades_won / self.trades_taken if self.trades_taken > 0 else 0.0


# ---------------------------------------------------------------------------
# Risk manager
# ---------------------------------------------------------------------------

class RiskManager:
    def __init__(
        self,
        initial_equity: float,
        *,
        # Vol-targeting
        target_annual_vol: float = 0.20,
        # Hard per-trade cap
        max_position_pct: float = 0.25,
        # Concurrency
        max_concurrent_positions: int = 1,
        # Kill switches
        daily_loss_limit_pct: float = 0.03,
        consecutive_loss_limit: int = 3,
        consecutive_loss_cooloff_seconds: int = 3600,
        weekly_drawdown_halve_pct: float = 0.08,
        tripwire_file: Optional[str] = None,
        # Kelly (optional; default off)
        kelly_fraction: float = 0.0,
    ) -> None:
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.target_annual_vol = target_annual_vol
        self.max_position_pct = max_position_pct
        self.max_concurrent_positions = max_concurrent_positions
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.consecutive_loss_limit = consecutive_loss_limit
        self.consecutive_loss_cooloff_seconds = consecutive_loss_cooloff_seconds
        self.weekly_drawdown_halve_pct = weekly_drawdown_halve_pct
        self.tripwire_file = tripwire_file
        self.kelly_fraction = max(0.0, min(kelly_fraction, 1.0))

        today = datetime.now(timezone.utc).date()
        self._daily = DailyStats(date=today, starting_equity=initial_equity)
        self._open_positions = 0
        self._consecutive_losses = 0
        self._cooloff_until: Optional[datetime] = None
        self._equity_history: Deque[tuple[datetime, float]] = deque()
        self._manually_halted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def approve_trade(
        self,
        signal_action: str,
        current_price: float,
        *,
        symbol: str = "",
        instrument_annual_vol: Optional[float] = None,
        win_rate: Optional[float] = None,
        payoff_ratio: Optional[float] = None,
        now: Optional[datetime] = None,
    ) -> TradeApproval:
        now = now or datetime.now(timezone.utc)
        self._check_day_rollover(now)

        # --- Hard gates ----------------------------------------------------
        if self._manually_halted:
            return TradeApproval(False, "manual_halt")

        if self.tripwire_file and os.path.exists(self.tripwire_file):
            return TradeApproval(False, f"tripwire:{self.tripwire_file}")

        if self._daily.halted:
            return TradeApproval(
                False,
                f"daily_loss_limit({self._daily.loss_pct:.2%} <= -{self.daily_loss_limit_pct:.2%})",
            )

        if self._cooloff_until is not None and now < self._cooloff_until:
            remaining = int((self._cooloff_until - now).total_seconds())
            return TradeApproval(False, f"consecutive_loss_cooloff({remaining}s_remaining)")

        if self._open_positions >= self.max_concurrent_positions:
            return TradeApproval(False, f"max_concurrent_positions({self.max_concurrent_positions})")

        if signal_action not in ("long", "short"):
            return TradeApproval(False, "no_signal")

        if current_price <= 0:
            return TradeApproval(False, "invalid_price")

        # --- Sizing --------------------------------------------------------
        size_usd, breakdown = self._size(
            instrument_annual_vol=instrument_annual_vol,
            win_rate=win_rate,
            payoff_ratio=payoff_ratio,
            now=now,
        )
        size_contracts = size_usd / current_price

        logger.info(
            "Trade approved: %s %s | size=$%.2f (%.4f ctx @ %.4f) | %s",
            signal_action.upper(), symbol or "?",
            size_usd, size_contracts, current_price, breakdown,
        )
        return TradeApproval(
            approved=True,
            reason="OK",
            position_size_usd=size_usd,
            position_size_contracts=size_contracts,
            sizing_breakdown=breakdown,
        )

    def on_trade_open(self, symbol: str = "") -> None:
        self._open_positions += 1
        self._daily.trades_taken += 1
        logger.debug("Trade opened (%s). Open positions: %d", symbol or "?", self._open_positions)

    def on_trade_close(self, pnl: float, won: bool, now: Optional[datetime] = None) -> None:
        now = now or datetime.now(timezone.utc)
        self._open_positions = max(0, self._open_positions - 1)
        self.equity += pnl
        self._daily.realized_pnl += pnl
        if won:
            self._daily.trades_won += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.consecutive_loss_limit:
                self._cooloff_until = now + timedelta(seconds=self.consecutive_loss_cooloff_seconds)
                logger.warning(
                    "Cooling off after %d consecutive losses until %s",
                    self._consecutive_losses, self._cooloff_until.isoformat(),
                )

        # Track equity for drawdown ratchet
        self._equity_history.append((now, self.equity))
        self._prune_equity_history(now)

        if self._daily.loss_pct <= -self.daily_loss_limit_pct:
            self._daily.halted = True
            logger.warning(
                "Daily loss limit hit: %.2f%% — halted until UTC rollover",
                self._daily.loss_pct * 100,
            )

    def halt(self) -> None:
        self._manually_halted = True
        logger.warning("Manual halt set — new entries blocked")

    def resume(self) -> None:
        self._manually_halted = False
        logger.info("Manual halt cleared")

    def is_halted(self) -> bool:
        if self._manually_halted:
            return True
        if self._daily.halted:
            return True
        if self.tripwire_file and os.path.exists(self.tripwire_file):
            return True
        return False

    def get_equity(self) -> float:
        return self.equity

    def get_daily_stats(self) -> DailyStats:
        return self._daily

    # ------------------------------------------------------------------
    # Sizing internals
    # ------------------------------------------------------------------

    def _size(
        self,
        *,
        instrument_annual_vol: Optional[float],
        win_rate: Optional[float],
        payoff_ratio: Optional[float],
        now: datetime,
    ) -> tuple[float, dict]:
        equity = self.equity
        hard_cap = equity * self.max_position_pct
        breakdown: dict = {"equity": equity, "hard_cap_usd": hard_cap}

        # Vol targeting (primary)
        if instrument_annual_vol and instrument_annual_vol > 0:
            target_size = equity * self.target_annual_vol / instrument_annual_vol
            breakdown["vol_target_size_usd"] = target_size
        else:
            target_size = hard_cap
            breakdown["vol_target_size_usd"] = None  # fallback to hard cap

        # Optional fractional Kelly overlay
        if self.kelly_fraction > 0 and win_rate is not None and payoff_ratio and payoff_ratio > 0:
            # Kelly f* = (bp - q) / b where p=win, q=1-p, b=payoff_ratio
            p, q, b = win_rate, 1 - win_rate, payoff_ratio
            kelly = (b * p - q) / b
            kelly = max(0.0, kelly) * self.kelly_fraction
            kelly_size = equity * kelly
            breakdown["kelly_f"] = kelly
            breakdown["kelly_size_usd"] = kelly_size
            target_size = min(target_size, kelly_size) if kelly_size > 0 else 0.0

        # 7d drawdown halving
        dd_factor = self._drawdown_halve_factor(now)
        if dd_factor < 1.0:
            breakdown["drawdown_halve_factor"] = dd_factor
            target_size *= dd_factor

        # Clip to hard cap
        final = min(target_size, hard_cap)
        final = max(0.0, final)
        breakdown["final_usd"] = final
        return final, breakdown

    def _drawdown_halve_factor(self, now: datetime) -> float:
        """Return 0.5 if rolling-7d drawdown > threshold, else 1.0."""
        if not self._equity_history:
            return 1.0
        cutoff = now - timedelta(days=7)
        recent = [e for t, e in self._equity_history if t >= cutoff]
        if not recent:
            return 1.0
        peak = max(recent)
        if peak <= 0:
            return 1.0
        dd = (peak - self.equity) / peak
        if dd > self.weekly_drawdown_halve_pct:
            return 0.5
        return 1.0

    def _prune_equity_history(self, now: datetime) -> None:
        cutoff = now - timedelta(days=14)   # keep 2 weeks for analysis
        while self._equity_history and self._equity_history[0][0] < cutoff:
            self._equity_history.popleft()

    # ------------------------------------------------------------------
    # Day rollover
    # ------------------------------------------------------------------

    def _check_day_rollover(self, now: datetime) -> None:
        today = now.date()
        if self._daily.date != today:
            logger.info(
                "New trading day. Previous: trades=%d win_rate=%.1f%% pnl=$%.2f",
                self._daily.trades_taken, self._daily.win_rate * 100, self._daily.realized_pnl,
            )
            self._daily = DailyStats(date=today, starting_equity=self.equity)

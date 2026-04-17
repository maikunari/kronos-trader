"""
backtest.py
Canonical Phase 1 backtester for snipe_signal_engine.

Models the pieces that matter for honest edge estimation:
  * Bar-close signals, next-bar-open fills (no lookahead)
  * Taker fees per side (HL-style 3.5 bps default)
  * Slippage as a function of ATR + a configurable base
  * Funding-cost accrual at configurable interval
  * ATR-based initial stop / target, plus optional Chandelier trailing exit
  * Per-trade records with exit reason, fees, slippage, funding broken out

Consumers:
  - Single backtest: `run_snipe_backtest(...)`
  - Purged walk-forward CV: `walk_forward(...)` — returns per-fold results.
    Purged and embargoed per López de Prado so no label leakage across splits.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from atr_engine import ChandelierTrail, _atr
from risk_manager import RiskManager
from snipe_signal_engine import MarketContext, SnipeSignalEngine

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class Trade:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    direction: str               # "long" | "short"
    entry_price: float
    exit_price: float
    size_usd: float
    pnl_usd: float
    exit_reason: str             # "target" | "stop" | "trail" | "timeout" | "end_of_data"
    fees_usd: float
    slippage_usd: float
    funding_usd: float


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: list[float]
    initial_capital: float
    final_equity: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    trades_count: int
    fees_total: float
    slippage_total: float
    funding_total: float
    start_date: str
    end_date: str

    def summary(self) -> dict:
        return {
            "return_pct": f"{self.total_return_pct * 100:+.2f}%",
            "sharpe": f"{self.sharpe_ratio:.2f}",
            "max_dd": f"{self.max_drawdown_pct * 100:.2f}%",
            "trades": self.trades_count,
            "win_rate": f"{self.win_rate * 100:.1f}%",
            "profit_factor": f"{self.profit_factor:.2f}",
            "cost_drag": f"${self.fees_total + self.slippage_total + self.funding_total:,.2f}",
        }


# ------------------------------------------------------------------
# Core backtest
# ------------------------------------------------------------------

def run_snipe_backtest(
    candles_15m: pd.DataFrame,
    *,
    engine: SnipeSignalEngine,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.00035,                 # 3.5 bps per side (HL taker)
    slippage_base_bps: float = 2.0,            # base slippage in bps
    slippage_atr_frac: float = 0.05,           # additional slippage as frac of ATR
    funding_rate_hourly: float = 0.0,          # flat funding used if no series given
    funding_series: Optional[pd.DataFrame] = None,  # cols: timestamp, rate_hourly
    candles_1h: Optional[pd.DataFrame] = None,
    candles_4h: Optional[pd.DataFrame] = None,
    use_chandelier_trail: bool = True,
    chandelier_atr_mult: float = 3.0,
    max_hold_bars: int = 0,                    # 0 = no timeout
    risk: Optional[RiskManager] = None,
    target_annual_vol: float = 0.20,
    bars_per_year: int = 35_040,
) -> BacktestResult:
    """
    Run a single-path backtest over `candles_15m` using `engine`.

    Signals come at bar close; entries execute at the next bar's open to
    avoid lookahead. Stops/targets and trailing exits are checked against
    the path of each subsequent bar.
    """
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(candles_15m.columns):
        raise ValueError(f"candles_15m missing columns: {required - set(candles_15m.columns)}")

    df = candles_15m.reset_index(drop=True).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    risk = risk or RiskManager(
        initial_equity=initial_capital,
        target_annual_vol=target_annual_vol,
        max_position_pct=0.5,
        daily_loss_limit_pct=0.05,
    )

    funding_lookup = _build_funding_lookup(funding_series, funding_rate_hourly)

    trades: list[Trade] = []
    equity_curve: list[float] = [initial_capital]
    open_pos: Optional[dict] = None
    lookback = 300  # enough for regime/Hurst/rv

    for i in range(lookback, len(df) - 1):
        bar = df.iloc[i]
        next_bar = df.iloc[i + 1]
        window = df.iloc[i - lookback : i + 1]
        now = bar["timestamp"].to_pydatetime()

        # --- Manage open position ---
        if open_pos is not None:
            exit_info = _check_exit(open_pos, bar)
            # Accrue funding for the hours this bar spans
            _accrue_funding(open_pos, bar, funding_lookup)

            # Chandelier trailing update (uses the current bar's range)
            if use_chandelier_trail and open_pos["trail"] is not None:
                atr_now = _atr_current(window)
                new_stop = open_pos["trail"].update(
                    high=float(bar["high"]), low=float(bar["low"]), atr=atr_now,
                )
                if new_stop is not None:
                    if open_pos["direction"] == "long":
                        open_pos["stop"] = max(open_pos["stop"], new_stop)
                    else:
                        open_pos["stop"] = min(open_pos["stop"], new_stop)
                    # Re-check exit against the updated stop on this bar
                    exit_info = _check_exit(open_pos, bar) or exit_info

            # Timeout
            if exit_info is None and max_hold_bars > 0:
                if i - open_pos["entry_idx"] >= max_hold_bars:
                    exit_info = ("timeout", float(bar["close"]))

            if exit_info is not None:
                reason, exit_price = exit_info
                trade = _close(open_pos, bar["timestamp"], exit_price, reason,
                               fee_rate, slippage_base_bps, slippage_atr_frac)
                trades.append(trade)
                risk.on_trade_close(pnl=trade.pnl_usd, won=trade.pnl_usd > 0, now=now)
                open_pos = None

        # --- Look for new signal ---
        if open_pos is None:
            ctx = MarketContext(
                candles_15m=window,
                candles_1h=_slice_before(candles_1h, bar["timestamp"]),
                candles_4h=_slice_before(candles_4h, bar["timestamp"]),
                current_price=float(bar["close"]),
                timestamp=bar["timestamp"],
                funding_rate_hourly=funding_lookup(bar["timestamp"]),
            )
            sig = engine.evaluate(ctx)
            if sig.action in ("long", "short"):
                approval = risk.approve_trade(
                    sig.action,
                    current_price=float(next_bar["open"]),
                    symbol="",
                    instrument_annual_vol=sig.regime.rv if sig.regime else None,
                    now=now,
                )
                if approval.approved and approval.position_size_usd > 0:
                    open_pos = _open(
                        direction=sig.action,
                        entry_idx=i + 1,
                        entry_ts=next_bar["timestamp"],
                        raw_entry_price=float(next_bar["open"]),
                        size_usd=approval.position_size_usd,
                        atr=sig.atr,
                        initial_stop=sig.stop_price,
                        initial_target=sig.target_price,
                        use_trail=use_chandelier_trail,
                        chandelier_atr_mult=chandelier_atr_mult,
                        slippage_base_bps=slippage_base_bps,
                        slippage_atr_frac=slippage_atr_frac,
                        fee_rate=fee_rate,
                    )
                    risk.on_trade_open()

        equity_curve.append(risk.get_equity())

    # Close any position still open at end of data at last close
    if open_pos is not None:
        last = df.iloc[-1]
        trade = _close(open_pos, last["timestamp"], float(last["close"]),
                       "end_of_data", fee_rate, slippage_base_bps, slippage_atr_frac)
        trades.append(trade)
        risk.on_trade_close(pnl=trade.pnl_usd, won=trade.pnl_usd > 0)
        equity_curve.append(risk.get_equity())

    return _summarize(
        trades=trades, equity_curve=equity_curve, initial_capital=initial_capital,
        start_date=str(df["timestamp"].iloc[0]), end_date=str(df["timestamp"].iloc[-1]),
        bars_per_year=bars_per_year,
    )


# ------------------------------------------------------------------
# Trade open/close helpers
# ------------------------------------------------------------------

def _open(
    *, direction: str, entry_idx: int, entry_ts, raw_entry_price: float,
    size_usd: float, atr: float, initial_stop: float, initial_target: float,
    use_trail: bool, chandelier_atr_mult: float,
    slippage_base_bps: float, slippage_atr_frac: float, fee_rate: float,
) -> dict:
    slip = _slippage_per_unit(raw_entry_price, atr, slippage_base_bps, slippage_atr_frac)
    entry_price = raw_entry_price + slip if direction == "long" else raw_entry_price - slip
    entry_fee = size_usd * fee_rate
    entry_slippage_usd = size_usd * (abs(entry_price - raw_entry_price) / raw_entry_price)
    trail = ChandelierTrail(direction=direction, atr_mult=chandelier_atr_mult) if use_trail else None
    return {
        "direction": direction,
        "entry_idx": entry_idx,
        "entry_ts": entry_ts,
        "entry_price": entry_price,
        "size_usd": size_usd,
        "stop": initial_stop,
        "target": initial_target,
        "atr_at_entry": atr,
        "trail": trail,
        "fee_accum": entry_fee,
        "slip_accum": entry_slippage_usd,
        "funding_accum": 0.0,
        "last_funding_ts": entry_ts,
    }


def _close(
    pos: dict, exit_ts, raw_exit_price: float, reason: str,
    fee_rate: float, slippage_base_bps: float, slippage_atr_frac: float,
) -> Trade:
    atr = pos["atr_at_entry"]
    slip = _slippage_per_unit(raw_exit_price, atr, slippage_base_bps, slippage_atr_frac)
    exit_price = raw_exit_price - slip if pos["direction"] == "long" else raw_exit_price + slip
    exit_fee = pos["size_usd"] * fee_rate
    exit_slippage = pos["size_usd"] * (abs(exit_price - raw_exit_price) / raw_exit_price)
    gross = (exit_price - pos["entry_price"]) / pos["entry_price"] * pos["size_usd"]
    if pos["direction"] == "short":
        gross = -gross
    total_fee = pos["fee_accum"] + exit_fee
    total_slip = pos["slip_accum"] + exit_slippage
    pnl = gross - total_fee - total_slip - pos["funding_accum"]
    return Trade(
        entry_ts=pos["entry_ts"],
        exit_ts=exit_ts,
        direction=pos["direction"],
        entry_price=pos["entry_price"],
        exit_price=exit_price,
        size_usd=pos["size_usd"],
        pnl_usd=pnl,
        exit_reason=reason,
        fees_usd=total_fee,
        slippage_usd=total_slip,
        funding_usd=pos["funding_accum"],
    )


def _check_exit(pos: dict, bar: pd.Series) -> Optional[tuple[str, float]]:
    """Return (reason, exit_price) if this bar hit a stop/target, else None.

    Conservative convention: when both levels lie in the bar's range, assume
    the stop hit first. Real fills would need tick data to resolve.
    """
    direction = pos["direction"]
    stop = pos["stop"]
    target = pos["target"]
    high, low = float(bar["high"]), float(bar["low"])

    if direction == "long":
        hit_stop = low <= stop
        hit_target = high >= target
        if hit_stop and hit_target:
            return "stop", stop
        if hit_stop:
            return "stop", stop
        if hit_target:
            return "target", target
    else:
        hit_stop = high >= stop
        hit_target = low <= target
        if hit_stop and hit_target:
            return "stop", stop
        if hit_stop:
            return "stop", stop
        if hit_target:
            return "target", target
    return None


def _accrue_funding(pos: dict, bar: pd.Series, funding_lookup: Callable[[pd.Timestamp], float]):
    """Accrue funding cost for the (pos.last_funding_ts -> bar.timestamp) span."""
    last_ts = pd.Timestamp(pos["last_funding_ts"])
    cur_ts = pd.Timestamp(bar["timestamp"])
    hours = max(0.0, (cur_ts - last_ts).total_seconds() / 3600.0)
    if hours <= 0:
        return
    rate = funding_lookup(cur_ts)
    # Long pays funding when rate > 0; short receives. Sign convention:
    #   cost to long = +rate; cost to short = -rate
    sign = 1.0 if pos["direction"] == "long" else -1.0
    pos["funding_accum"] += pos["size_usd"] * rate * hours * sign
    pos["last_funding_ts"] = cur_ts


def _slippage_per_unit(price: float, atr: float, base_bps: float, atr_frac: float) -> float:
    """Price-unit slippage: base bps of price + fraction of ATR."""
    return price * (base_bps / 10_000.0) + atr_frac * max(atr, 0.0)


def _atr_current(window: pd.DataFrame, period: int = 14) -> float:
    if len(window) < period + 2:
        return 0.0
    vals = _atr(
        window["high"].to_numpy(dtype=float),
        window["low"].to_numpy(dtype=float),
        window["close"].to_numpy(dtype=float),
        period=period,
    )
    return float(vals[-1]) if vals.size else 0.0


def _slice_before(df: Optional[pd.DataFrame], ts) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    return df[df["timestamp"] < ts]


def _build_funding_lookup(
    series: Optional[pd.DataFrame], fallback: float
) -> Callable[[pd.Timestamp], float]:
    if series is None or series.empty:
        return lambda _ts: fallback
    s = series.copy()
    s["timestamp"] = pd.to_datetime(s["timestamp"], utc=True)
    s = s.sort_values("timestamp").reset_index(drop=True)

    def lookup(ts):
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        idx = s["timestamp"].searchsorted(ts, side="right") - 1
        if idx < 0:
            return fallback
        return float(s["rate_hourly"].iloc[idx])

    return lookup


# ------------------------------------------------------------------
# Result summarization
# ------------------------------------------------------------------

def _summarize(
    *,
    trades: list[Trade],
    equity_curve: list[float],
    initial_capital: float,
    start_date: str,
    end_date: str,
    bars_per_year: int,
) -> BacktestResult:
    final_equity = equity_curve[-1] if equity_curve else initial_capital
    total_return_pct = (final_equity - initial_capital) / initial_capital

    # Per-bar log returns for Sharpe
    eq = np.array(equity_curve, dtype=float)
    if len(eq) >= 2:
        log_ret = np.diff(np.log(np.maximum(eq, 1e-9)))
        if log_ret.std(ddof=1) > 0:
            sharpe = float(log_ret.mean() / log_ret.std(ddof=1) * np.sqrt(bars_per_year))
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(eq) if len(eq) else np.array([initial_capital])
    dd = (running_max - eq) / running_max
    max_dd = float(dd.max()) if dd.size else 0.0

    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    wr = len(wins) / len(trades) if trades else 0.0
    gross_win = sum(t.pnl_usd for t in wins)
    gross_loss = abs(sum(t.pnl_usd for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else (gross_win if gross_win > 0 else 0.0)
    avg_win = np.mean([t.pnl_usd / t.size_usd for t in wins]) if wins else 0.0
    avg_loss = np.mean([t.pnl_usd / t.size_usd for t in losses]) if losses else 0.0

    return BacktestResult(
        trades=trades,
        equity_curve=list(eq),
        initial_capital=initial_capital,
        final_equity=float(final_equity),
        total_return_pct=total_return_pct,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
        win_rate=wr,
        profit_factor=pf,
        avg_win_pct=float(avg_win),
        avg_loss_pct=float(avg_loss),
        trades_count=len(trades),
        fees_total=sum(t.fees_usd for t in trades),
        slippage_total=sum(t.slippage_usd for t in trades),
        funding_total=sum(t.funding_usd for t in trades),
        start_date=start_date,
        end_date=end_date,
    )


# ------------------------------------------------------------------
# Purged walk-forward
# ------------------------------------------------------------------

@dataclass
class WalkForwardFold:
    in_sample_start: pd.Timestamp
    in_sample_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    in_sample_result: BacktestResult
    oos_result: BacktestResult


def walk_forward(
    candles_15m: pd.DataFrame,
    *,
    engine_factory: Callable[[], SnipeSignalEngine],
    in_sample_days: int = 60,
    out_of_sample_days: int = 14,
    embargo_days: int = 1,
    step_days: Optional[int] = None,
    backtest_kwargs: Optional[dict] = None,
) -> list[WalkForwardFold]:
    """
    Purged walk-forward: slide (in-sample | embargo | OOS) windows forward.

    `engine_factory()` must return a fresh engine per fold so tuning decisions
    don't leak across folds. `backtest_kwargs` passes through to
    run_snipe_backtest (fees, slippage, etc.).
    """
    if backtest_kwargs is None:
        backtest_kwargs = {}
    step_days = step_days or out_of_sample_days

    df = candles_15m.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    start = df["timestamp"].iloc[0]
    end = df["timestamp"].iloc[-1]

    folds: list[WalkForwardFold] = []
    cursor = start + pd.Timedelta(days=in_sample_days)
    while cursor + pd.Timedelta(days=embargo_days + out_of_sample_days) <= end:
        is_end = cursor
        is_start = is_end - pd.Timedelta(days=in_sample_days)
        oos_start = is_end + pd.Timedelta(days=embargo_days)
        oos_end = oos_start + pd.Timedelta(days=out_of_sample_days)

        is_df = df[(df["timestamp"] >= is_start) & (df["timestamp"] < is_end)]
        oos_df = df[(df["timestamp"] >= oos_start) & (df["timestamp"] < oos_end)]
        if len(is_df) < 500 or len(oos_df) < 50:
            cursor += pd.Timedelta(days=step_days)
            continue

        is_result = run_snipe_backtest(is_df, engine=engine_factory(), **backtest_kwargs)
        oos_result = run_snipe_backtest(oos_df, engine=engine_factory(), **backtest_kwargs)
        folds.append(WalkForwardFold(
            in_sample_start=is_start, in_sample_end=is_end,
            oos_start=oos_start, oos_end=oos_end,
            in_sample_result=is_result, oos_result=oos_result,
        ))
        cursor += pd.Timedelta(days=step_days)

    return folds

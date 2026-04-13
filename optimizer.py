"""
optimizer.py
Walk-forward strategy optimizer.

Triggers:
  - Every N trades (default 30 minimum) AND at least M days (default 14 max)
  - Emergency: rolling 7-day win rate drops below threshold (default 35%)
  - Cooldown: never retune more often than 7 days

Walk-forward logic:
  - In-sample window: last 60 days → grid search params
  - Out-of-sample window: last 14 days of that (held out during optimization)
  - Only apply new params if out-of-sample profit factor > 1.0
  - Log all changes with before/after metrics

Params tuned:
  - min_move_threshold
  - stop_pct (target_pct = stop_pct * rr_ratio, so only tune stop)
  - lookback_candles
  - MTF EMA fast/slow periods
"""

import itertools
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

OPTIMIZER_LOG = "optimizer_history.jsonl"


@dataclass
class OptimizeResult:
    timestamp: str
    trigger: str                   # "trade_count" | "time" | "emergency"
    trades_since_last: int
    days_since_last: float
    rolling_win_rate: float
    best_params: Dict[str, Any]
    in_sample_pf: float            # profit factor on in-sample data
    out_of_sample_pf: float        # profit factor on held-out data
    applied: bool                  # whether params were actually applied
    previous_params: Dict[str, Any]


@dataclass
class TradeRecord:
    timestamp: datetime
    won: bool
    net_pnl: float


class WalkForwardOptimizer:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()
        self._opt_cfg = self._config.get("optimizer", {})

        # Trigger thresholds
        self.min_trades = self._opt_cfg.get("min_trades_to_retune", 30)
        self.max_days = self._opt_cfg.get("max_days_to_retune", 14)
        self.emergency_wr = self._opt_cfg.get("emergency_win_rate_threshold", 0.35)
        self.cooldown_days = self._opt_cfg.get("min_days_between_retunes", 7)
        self.in_sample_days = self._opt_cfg.get("in_sample_days", 60)
        self.out_of_sample_days = self._opt_cfg.get("out_of_sample_days", 14)

        # State
        self._trades: List[TradeRecord] = []
        self._last_retune: Optional[datetime] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_trade(self, won: bool, net_pnl: float):
        """Record a completed paper/live trade."""
        with self._lock:
            self._trades.append(TradeRecord(
                timestamp=datetime.now(timezone.utc),
                won=won,
                net_pnl=net_pnl,
            ))
        logger.debug(f"Optimizer: trade recorded (won={won}, total={len(self._trades)})")

    def start_background(self):
        """Start optimizer check loop in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="optimizer")
        self._thread.start()
        logger.info("Optimizer background thread started")

    def stop(self):
        self._running = False

    def check_and_optimize(self, symbol: str, timeframe: str) -> Optional[OptimizeResult]:
        """
        Check if a retune is needed. If so, run walk-forward and optionally
        apply new params. Returns the result or None if no retune triggered.
        """
        with self._lock:
            trigger = self._get_trigger()
            if trigger is None:
                return None

            trades_since = self._trades_since_last_retune()
            days_since = self._days_since_last_retune()
            rolling_wr = self._rolling_win_rate(days=7)

        logger.info(
            f"Optimizer triggered: {trigger} | "
            f"trades={trades_since}, days={days_since:.1f}, "
            f"rolling_wr={rolling_wr:.1%}"
        )

        result = self._run_walk_forward(symbol, timeframe, trigger, trades_since, days_since, rolling_wr)

        with self._lock:
            self._last_retune = datetime.now(timezone.utc)

        self._log_result(result)
        return result

    # ------------------------------------------------------------------
    # Trigger logic
    # ------------------------------------------------------------------

    def _get_trigger(self) -> Optional[str]:
        """Determine if a retune should fire and why."""
        now = datetime.now(timezone.utc)

        # Cooldown check
        if self._last_retune is not None:
            days_since = (now - self._last_retune).total_seconds() / 86400
            if days_since < self.cooldown_days:
                return None

        trades_since = self._trades_since_last_retune()
        days_since = self._days_since_last_retune()
        rolling_wr = self._rolling_win_rate(days=7)

        # Emergency: rolling win rate crashed
        if trades_since >= 10 and rolling_wr < self.emergency_wr:
            logger.warning(f"Emergency retune: rolling win rate {rolling_wr:.1%} < {self.emergency_wr:.1%}")
            return "emergency"

        # Dual trigger: enough trades AND (time elapsed OR emergency)
        if trades_since >= self.min_trades:
            return "trade_count"

        if days_since >= self.max_days and trades_since >= 10:
            return "time"

        return None

    def _trades_since_last_retune(self) -> int:
        if self._last_retune is None:
            return len(self._trades)
        return sum(1 for t in self._trades if t.timestamp > self._last_retune)

    def _days_since_last_retune(self) -> float:
        if self._last_retune is None:
            if not self._trades:
                return 0.0
            return (datetime.now(timezone.utc) - self._trades[0].timestamp).total_seconds() / 86400
        return (datetime.now(timezone.utc) - self._last_retune).total_seconds() / 86400

    def _rolling_win_rate(self, days: int = 7) -> float:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent = [t for t in self._trades if t.timestamp >= cutoff]
        if not recent:
            return 1.0  # no recent trades = no degradation signal
        return sum(1 for t in recent if t.won) / len(recent)

    # ------------------------------------------------------------------
    # Walk-forward optimization
    # ------------------------------------------------------------------

    def _run_walk_forward(
        self,
        symbol: str,
        timeframe: str,
        trigger: str,
        trades_since: int,
        days_since: float,
        rolling_wr: float,
    ) -> OptimizeResult:
        """Run grid search on in-sample data, validate on out-of-sample."""
        config = self._load_config()
        previous_params = self._extract_tunable_params(config)

        # Date windows
        now = datetime.now(timezone.utc)
        in_sample_end = now - timedelta(days=self.out_of_sample_days)
        in_sample_start = in_sample_end - timedelta(days=self.in_sample_days - self.out_of_sample_days)
        out_of_sample_start = in_sample_end
        out_of_sample_end = now

        logger.info(
            f"Walk-forward windows:\n"
            f"  In-sample:      {in_sample_start.date()} → {in_sample_end.date()}\n"
            f"  Out-of-sample:  {out_of_sample_start.date()} → {out_of_sample_end.date()}"
        )

        # Fetch data
        try:
            from hyperliquid_feed import fetch_historical
            df_in = fetch_historical(
                symbol, timeframe,
                in_sample_start.strftime("%Y-%m-%d"),
                in_sample_end.strftime("%Y-%m-%d"),
                source=config["backtest"].get("data_source", "hyperliquid"),
            )
            df_out = fetch_historical(
                symbol, timeframe,
                out_of_sample_start.strftime("%Y-%m-%d"),
                out_of_sample_end.strftime("%Y-%m-%d"),
                source=config["backtest"].get("data_source", "hyperliquid"),
            )
        except Exception as e:
            logger.error(f"Optimizer data fetch failed: {e} — skipping retune")
            return OptimizeResult(
                timestamp=now.isoformat(),
                trigger=trigger,
                trades_since_last=trades_since,
                days_since_last=days_since,
                rolling_win_rate=rolling_wr,
                best_params=previous_params,
                in_sample_pf=0.0,
                out_of_sample_pf=0.0,
                applied=False,
                previous_params=previous_params,
            )

        # Grid search on in-sample
        param_grid = self._build_param_grid(config)
        logger.info(f"Grid search: {len(param_grid)} param combinations on {len(df_in)} in-sample candles")

        best_params, best_is_pf = self._grid_search(df_in, config, param_grid)
        logger.info(f"Best in-sample params: {best_params} (PF={best_is_pf:.2f})")

        # Validate on out-of-sample (held-out)
        oos_pf = self._score_params(df_out, config, best_params)
        logger.info(f"Out-of-sample profit factor: {oos_pf:.2f}")

        # Apply only if out-of-sample shows edge
        min_oos_pf = self._opt_cfg.get("min_out_of_sample_pf", 1.0)
        applied = False
        if oos_pf >= min_oos_pf:
            self._apply_params(config, best_params)
            applied = True
            logger.info(f"New params applied: {best_params}")
        else:
            logger.warning(
                f"Out-of-sample PF {oos_pf:.2f} < {min_oos_pf} threshold — "
                f"keeping current params"
            )

        return OptimizeResult(
            timestamp=now.isoformat(),
            trigger=trigger,
            trades_since_last=trades_since,
            days_since_last=days_since,
            rolling_win_rate=rolling_wr,
            best_params=best_params,
            in_sample_pf=best_is_pf,
            out_of_sample_pf=oos_pf,
            applied=applied,
            previous_params=previous_params,
        )

    def _build_param_grid(self, config: dict) -> List[Dict[str, Any]]:
        """Build the parameter grid to search over."""
        t = config["trading"]
        m = config["models"]
        mtf = config.get("mtf", {})

        # Param ranges — centred around current values
        stop = t["stop_pct"]
        threshold = t["min_move_threshold"]
        lookback = m["lookback_candles"]

        grid = {
            "min_move_threshold": [
                round(threshold * f, 4)
                for f in (0.5, 0.75, 1.0, 1.25, 1.5)
            ],
            "stop_pct": [
                round(stop * f, 4)
                for f in (0.5, 0.75, 1.0, 1.25, 1.5)
            ],
            "lookback_candles": [
                max(32, lookback + d)
                for d in (-16, 0, 16, 32)
            ],
            "ema_fast": mtf.get("ema_fast_search", [10, 20]),
            "ema_slow": mtf.get("ema_slow_search", [40, 50]),
        }

        # Deduplicate
        combos = list(itertools.product(*grid.values()))
        keys = list(grid.keys())
        return [dict(zip(keys, combo)) for combo in combos]

    def _grid_search(
        self, df: pd.DataFrame, config: dict, param_grid: List[Dict]
    ) -> Tuple[Dict, float]:
        """Find best params on in-sample data by profit factor."""
        best_pf = -1.0
        best_params = param_grid[0]

        for params in param_grid:
            try:
                pf = self._score_params(df, config, params)
                if pf > best_pf:
                    best_pf = pf
                    best_params = params
            except Exception as e:
                logger.debug(f"Param eval failed {params}: {e}")

        return best_params, best_pf

    def _score_params(self, df: pd.DataFrame, config: dict, params: Dict) -> float:
        """
        Run a mini-backtest with given params on df, return profit factor.
        Uses simplified signal logic to keep grid search fast.
        """
        from kronos_model import KronosModel
        from timesfm_model import TimesFMModel
        from signal_engine import SignalEngine
        from mtf_filter import MTFFilter

        t = config["trading"]
        r = config["risk"]
        b = config["backtest"]
        fee = b["fee_rate"]
        slip = b["slippage_rate"]
        rr = t["rr_ratio"]

        stop = params["stop_pct"]
        target = stop * rr
        lookback = params["lookback_candles"]

        # Lightweight models (use fallbacks for speed during grid search)
        kronos = KronosModel(model_size="mini", cache_dir=config["models"]["model_cache_dir"])
        timesfm = TimesFMModel(
            model_name=config["models"]["timesfm_model"],
            horizon=config["models"]["trend_horizon"],
            cache_dir=config["models"]["model_cache_dir"],
        )
        engine = SignalEngine(
            kronos=kronos,
            timesfm=timesfm,
            min_move_threshold=params["min_move_threshold"],
            stop_pct=stop,
            target_pct=target,
        )

        gross_wins = 0.0
        gross_losses = 0.0
        open_trade = None

        for i in range(lookback, len(df)):
            candle = df.iloc[i]
            window = df.iloc[i - lookback: i]

            if open_trade is not None:
                side = open_trade["side"]
                ep = open_trade["entry"]

                if side == "long":
                    if candle["low"] <= ep * (1 - stop):
                        pnl = -(stop + fee * 2 + slip * 2) * open_trade["size"]
                        gross_losses += abs(pnl)
                        open_trade = None
                    elif candle["high"] >= ep * (1 + target):
                        pnl = (target - fee * 2 - slip * 2) * open_trade["size"]
                        gross_wins += pnl
                        open_trade = None
                else:
                    if candle["high"] >= ep * (1 + stop):
                        pnl = -(stop + fee * 2 + slip * 2) * open_trade["size"]
                        gross_losses += abs(pnl)
                        open_trade = None
                    elif candle["low"] <= ep * (1 - target):
                        pnl = (target - fee * 2 - slip * 2) * open_trade["size"]
                        gross_wins += pnl
                        open_trade = None

            if open_trade is None and i + 1 < len(df):
                signal = engine.evaluate(window)
                if signal.action in ("long", "short"):
                    next_open = float(df.iloc[i + 1]["open"])
                    open_trade = {
                        "side": signal.action,
                        "entry": next_open,
                        "size": 1000.0,  # normalized size
                    }

        return gross_wins / gross_losses if gross_losses > 0 else (1.5 if gross_wins > 0 else 0.0)

    def _apply_params(self, config: dict, params: Dict):
        """Update config.yaml with new params."""
        config["trading"]["min_move_threshold"] = params["min_move_threshold"]
        config["trading"]["stop_pct"] = params["stop_pct"]
        config["trading"]["target_pct"] = round(params["stop_pct"] * config["trading"]["rr_ratio"], 4)
        config["models"]["lookback_candles"] = params["lookback_candles"]
        if "mtf" in config:
            config["mtf"]["ema_fast"] = params["ema_fast"]
            config["mtf"]["ema_slow"] = params["ema_slow"]

        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"config.yaml updated with new params")

    def _extract_tunable_params(self, config: dict) -> Dict[str, Any]:
        return {
            "min_move_threshold": config["trading"]["min_move_threshold"],
            "stop_pct": config["trading"]["stop_pct"],
            "lookback_candles": config["models"]["lookback_candles"],
            "ema_fast": config.get("mtf", {}).get("ema_fast", 20),
            "ema_slow": config.get("mtf", {}).get("ema_slow", 50),
        }

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_result(self, result: OptimizeResult):
        entry = asdict(result)
        with open(OPTIMIZER_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

        status = "APPLIED" if result.applied else "SKIPPED (OOS PF too low)"
        logger.info(
            f"\n{'='*60}\n"
            f"  OPTIMIZER RESULT [{result.trigger.upper()}]\n"
            f"  Trigger: {result.trades_since_last} trades, {result.days_since_last:.1f} days\n"
            f"  Rolling 7d win rate: {result.rolling_win_rate:.1%}\n"
            f"  In-sample PF:      {result.in_sample_pf:.2f}\n"
            f"  Out-of-sample PF:  {result.out_of_sample_pf:.2f}\n"
            f"  Status: {status}\n"
            f"  Params: {result.best_params}\n"
            f"{'='*60}"
        )

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _loop(self):
        """Background thread: check every 30 minutes if retune is needed."""
        config = self._load_config()
        symbol = config["trading"]["symbol"]
        timeframe = config["trading"]["timeframe"]

        while self._running:
            try:
                self.check_and_optimize(symbol, timeframe)
            except Exception as e:
                logger.error(f"Optimizer loop error: {e}")
            time.sleep(1800)  # check every 30 minutes

    def _load_config(self) -> dict:
        with open(self.config_path) as f:
            return yaml.safe_load(f)

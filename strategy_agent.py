"""
strategy_agent.py
LLM-driven strategy generation loop for Kronos Trader.

The loop:
  1. Ask Claude to generate N strategy Python modules following strategy_template.md
  2. AST-validate each module (import allowlist, no dangerous builtins)
  3. Run each variant in a subprocess against in-sample + OOS candle windows
  4. Feed structured metrics back to Claude for critique and refinement
  5. Repeat for K iterations, then ask Claude to write a final report

Usage:
    python strategy_agent.py                          # defaults, requires ANTHROPIC_API_KEY
    python strategy_agent.py --dry-run                # fixture variants, no API key needed
    python strategy_agent.py --iterations 3 --variants 5
    python strategy_agent.py --symbol ETH --timeframe 1h
    python main.py --mode agent --dry-run             # same, via main.py
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent
STRATEGIES_DIR = REPO_ROOT / "strategies" / "generated"
TEMPLATE_PATH = REPO_ROOT / "strategy_template.md"
TREND_ENGINE_PATH = REPO_ROOT / "trend_signal_engine.py"

ALLOWED_IMPORTS = {
    "pandas", "numpy", "math", "typing", "dataclasses",
    "collections", "enum", "functools", "itertools",
    "statistics", "abc", "numbers",
}
FORBIDDEN_BUILTINS = {
    "open", "exec", "eval", "compile", "__import__",
    "breakpoint", "input", "memoryview",
}
FORBIDDEN_MODULE_PREFIXES = (
    "os", "sys", "subprocess", "socket", "http", "urllib",
    "requests", "aiohttp", "websocket", "hyperliquid",
    "signal_engine", "backtest", "executor", "risk_manager",
)

SUBPROCESS_TIMEOUT = 120  # seconds per backtest window

OOS_MIN_PF = 1.0
OOS_MIN_TRADES = 5
OOS_MAX_DD = -0.30


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VariantResult:
    name: str
    iteration: int
    code_path: str
    in_sample: Optional[Dict] = None
    out_of_sample: Optional[Dict] = None
    accepted: bool = False
    rejection_reason: str = ""
    error: str = ""


@dataclass
class RunLog:
    run_id: str
    symbol: str
    timeframe: str
    in_sample_start: str
    in_sample_end: str
    oos_start: str
    oos_end: str
    model: str
    iterations: int
    variants_per_iteration: int
    results: List[VariantResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AST Validator
# ---------------------------------------------------------------------------

class ASTValidationError(Exception):
    pass


def ast_validate(source: str) -> None:
    """
    Parse `source` and walk the AST looking for forbidden patterns.
    Raises ASTValidationError describing the first violation found.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ASTValidationError(f"SyntaxError: {e}") from e

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in ALLOWED_IMPORTS:
                    raise ASTValidationError(
                        f"Forbidden import: '{alias.name}'. "
                        f"Only {sorted(ALLOWED_IMPORTS)} are allowed."
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            top = mod.split(".")[0]
            if top not in ALLOWED_IMPORTS:
                raise ASTValidationError(
                    f"Forbidden import from '{mod}'. "
                    f"Only {sorted(ALLOWED_IMPORTS)} are allowed."
                )
            if node.level and node.level > 0:
                raise ASTValidationError("Relative imports are not allowed.")

        # Check dangerous builtin calls
        elif isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in FORBIDDEN_BUILTINS:
                raise ASTValidationError(
                    f"Forbidden builtin call: '{name}()'."
                )

    # Must contain exactly one class named Strategy
    classes = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "Strategy"
    ]
    if not classes:
        raise ASTValidationError("Module must define a class named 'Strategy'.")
    if len(classes) > 1:
        raise ASTValidationError("Module must define exactly one 'Strategy' class.")

    # Strategy must have an evaluate method
    strat_cls = classes[0]
    methods = [n.name for n in ast.walk(strat_cls) if isinstance(n, ast.FunctionDef)]
    if "evaluate" not in methods:
        raise ASTValidationError("Strategy class must define an 'evaluate' method.")


# ---------------------------------------------------------------------------
# Subprocess backtest runner
# ---------------------------------------------------------------------------

def _run_backtest_subprocess(
    strategy_path: Path,
    candles_parquet: Path,
    sim_config: dict,
    timeout: int = SUBPROCESS_TIMEOUT,
) -> Dict:
    """
    Invoke strategy_runner.py in a subprocess. Returns the metrics dict.
    Raises RuntimeError on timeout, nonzero exit, or invalid JSON.
    """
    config_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    )
    try:
        json.dump(sim_config, config_file)
        config_file.flush()
        config_file.close()

        cmd = [
            sys.executable,
            str(REPO_ROOT / "strategy_runner.py"),
            str(strategy_path),
            str(candles_parquet),
            config_file.name,
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_ROOT),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"runner exited {proc.returncode}: {proc.stderr.strip()[:400]}"
            )
        try:
            return json.loads(proc.stdout.strip())
        except json.JSONDecodeError as e:
            raise RuntimeError(f"runner stdout not JSON: {e} — {proc.stdout[:200]}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"backtest timed out after {timeout}s")
    finally:
        try:
            os.unlink(config_file.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Candle data helpers
# ---------------------------------------------------------------------------

def _fetch_and_cache(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    cache_dir: Path,
) -> Path:
    """Fetch candles once and cache as parquet. Returns the parquet path."""
    slug = f"{symbol}_{timeframe}_{start}_{end}.parquet".replace("/", "-")
    path = cache_dir / slug
    if path.exists():
        logger.info(f"Candle cache hit: {path.name}")
        return path

    logger.info(f"Fetching {symbol}/{timeframe} {start} → {end}")
    # Import here to avoid slow SDK loads at module level
    from hyperliquid_feed import fetch_historical
    config_stub = {"backtest": {"data_source": "hyperliquid"}}
    df = fetch_historical(symbol, timeframe, start, end, source="hyperliquid")
    if df.empty:
        raise RuntimeError(f"No data returned for {symbol}/{timeframe} {start}→{end}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Cached {len(df)} candles → {path.name}")
    return path


def _make_synthetic_candles(n: int = 500) -> pd.DataFrame:
    """Generate a simple synthetic OHLCV DataFrame for dry-run testing."""
    import numpy as np
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 100.0 * (1 + rng.normal(0, 0.005, n)).cumprod()
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.uniform(1000, 50000, n)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _load_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _build_seed_prompt(
    symbol: str,
    timeframe: str,
    n_variants: int,
    template: str,
    trend_engine_src: str,
) -> str:
    return textwrap.dedent(f"""\
    You are an expert quantitative trading strategy developer.

    Your task is to generate {n_variants} DIVERSE Python strategy modules for
    the Kronos Trader system targeting {symbol} / {timeframe} crypto perpetuals
    on Hyperliquid.

    ## Contract (you MUST follow this exactly)

    {template}

    ## Reference implementation

    Here is the existing hand-written strategy you should try to BEAT:

    ```python
    {trend_engine_src}
    ```

    ## Your task

    Generate {n_variants} strategies. They must be meaningfully different from
    each other and from the reference. Good directions to explore:

    - Volatility-regime filters (trade only in trending markets, skip chop)
    - Volume confirmation (e.g. require above-average volume on signal bars)
    - Multi-EMA consensus (e.g. 3 EMAs must agree before entry)
    - Momentum/RSI overlay (enter only when momentum aligns)
    - Breakout-vs-pullback entries (rather than EMA crossovers)
    - Asymmetric R:R (vary stop_pct and target_pct by regime)
    - Combined trend + mean-reversion (different regimes, same module)

    For each strategy output ONLY the Python code in a fenced block:

    ```python
    # Strategy: <name>
    <full module code>
    ```

    Do NOT include explanatory prose between code blocks. Output {n_variants}
    code blocks and nothing else.
    """)


def _build_refinement_prompt(
    symbol: str,
    timeframe: str,
    n_variants: int,
    prior_results: List[Dict],
    iteration: int,
    template: str,
) -> str:
    summary = json.dumps(prior_results, indent=2)
    return textwrap.dedent(f"""\
    You are an expert quantitative trading strategy developer. This is
    iteration {iteration} of an automated strategy search for {symbol}/{timeframe}.

    ## Contract

    {template}

    ## Results so far

    {summary}

    ## Your task

    Based on the results above:
    1. Identify what worked and what failed (OOS profit_factor, drawdown, trade count).
    2. Generate {n_variants} REFINED strategies that improve on the best
       performers. You may also discard approaches that consistently failed.
    3. Aim to raise OOS profit_factor above 1.0 while keeping drawdown
       above {OOS_MAX_DD:.0%} and generating at least {OOS_MIN_TRADES} trades.

    Output {n_variants} Python code blocks exactly as specified in the contract.
    Do NOT include prose between blocks.
    """)


def _build_report_prompt(results: List[Dict]) -> str:
    accepted = [r for r in results if r.get("accepted")]
    rejected = [r for r in results if not r.get("accepted")]
    return textwrap.dedent(f"""\
    You are an expert quantitative trading analyst.

    Below are the results of an automated strategy search.

    Accepted variants ({len(accepted)}):
    {json.dumps(accepted, indent=2)}

    Rejected variants ({len(rejected)}):
    {json.dumps(rejected, indent=2)}

    Write a concise research report (markdown) covering:
    1. Top 3 strategies (by OOS profit factor) — what made them work
    2. Common failure modes among rejected variants
    3. Recommended next steps for further improvement
    4. Any warnings about potential overfitting

    Be specific. Reference exact metric values. Keep it under 600 words.
    """)


# ---------------------------------------------------------------------------
# Claude client
# ---------------------------------------------------------------------------

def _call_claude(prompt: str, model: str, max_tokens: int = 4096) -> str:
    """Call the Anthropic API and return the text response."""
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _extract_python_blocks(text: str) -> List[str]:
    """Pull all ```python ... ``` blocks out of an LLM response."""
    pattern = r"```python\s*\n(.*?)```"
    blocks = re.findall(pattern, text, re.DOTALL)
    return [b.strip() for b in blocks if b.strip()]


# ---------------------------------------------------------------------------
# Fixture strategies for --dry-run
# ---------------------------------------------------------------------------

FIXTURE_GOOD = """\
\"\"\"Simple EMA crossover variant for dry-run testing.\"\"\"
import pandas as pd
import numpy as np

class Strategy:
    def __init__(self, ema_fast: int = 9, ema_slow: int = 21,
                 atr_period: int = 14, atr_stop_mult: float = 1.5,
                 atr_target_mult: float = 3.0):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult

    def evaluate(self, candles: pd.DataFrame) -> dict:
        if len(candles) < self.ema_slow + self.atr_period + 2:
            return {"action": "flat", "stop_pct": 0.015, "target_pct": 0.03,
                    "confidence": 0.0, "skip_reason": "insufficient_bars"}
        closes = candles["close"].values.astype(float)
        highs = candles["high"].values.astype(float)
        lows = candles["low"].values.astype(float)

        def ema(arr, p):
            k = 2.0 / (p + 1)
            e = float(arr[0])
            for v in arr[1:]:
                e = v * k + e * (1 - k)
            return e

        ef_now  = ema(closes, self.ema_fast)
        es_now  = ema(closes, self.ema_slow)
        ef_prev = ema(closes[:-1], self.ema_fast)
        es_prev = ema(closes[:-1], self.ema_slow)

        tr = np.maximum(highs - lows,
             np.maximum(np.abs(highs - np.roll(closes, 1)),
                        np.abs(lows  - np.roll(closes, 1))))
        atr = float(np.mean(tr[-self.atr_period:]))
        if atr <= 0:
            return {"action": "flat", "stop_pct": 0.015, "target_pct": 0.03,
                    "confidence": 0.0, "skip_reason": "atr_zero"}

        crossed_up   = (ef_prev <= es_prev) and (ef_now > es_now)
        crossed_down = (ef_prev >= es_prev) and (ef_now < es_now)
        if not crossed_up and not crossed_down:
            return {"action": "flat", "stop_pct": 0.015, "target_pct": 0.03,
                    "confidence": 0.0, "skip_reason": "no_crossover"}

        action = "long" if crossed_up else "short"
        price  = float(closes[-1])
        stop_pct   = (self.atr_stop_mult * atr) / price
        target_pct = (self.atr_target_mult * atr) / price
        sep = abs(ef_now - es_now) / atr
        conf = min(sep / 2.0, 1.0)
        return {"action": action, "stop_pct": max(stop_pct, 0.002),
                "target_pct": max(target_pct, 0.002), "confidence": conf,
                "skip_reason": ""}
"""

FIXTURE_BAD_IMPORT = """\
\"\"\"This strategy tries to import os — should be rejected.\"\"\"
import os
import pandas as pd

class Strategy:
    def __init__(self):
        pass
    def evaluate(self, candles):
        return {"action": "flat", "stop_pct": 0.01, "target_pct": 0.02,
                "confidence": 0.0, "skip_reason": "test"}
"""

FIXTURE_CRASH = """\
\"\"\"This strategy raises on every call — should be caught.\"\"\"
import pandas as pd

class Strategy:
    def __init__(self):
        pass
    def evaluate(self, candles):
        raise ValueError("intentional crash for testing")
"""

FIXTURE_NO_STRATEGY_CLASS = """\
\"\"\"No Strategy class — should fail AST validation.\"\"\"
import pandas as pd

class BadName:
    def evaluate(self, candles):
        return {"action": "flat", "stop_pct": 0.01, "target_pct": 0.02,
                "confidence": 0.0, "skip_reason": "test"}
"""

DRY_RUN_FIXTURES: List[Tuple[str, str]] = [
    ("good_ema_crossover",      FIXTURE_GOOD),
    ("bad_import_os",           FIXTURE_BAD_IMPORT),
    ("crash_on_evaluate",       FIXTURE_CRASH),
    ("missing_strategy_class",  FIXTURE_NO_STRATEGY_CLASS),
]


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def run_agent_loop(config: dict, args: argparse.Namespace) -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = STRATEGIES_DIR / f"runs/{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = STRATEGIES_DIR / "candle_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    b = config["backtest"]
    sim_config = {
        "initial_capital": b.get("initial_capital", 10_000.0),
        "fee_rate":        b.get("fee_rate", 0.00035),
        "slippage_rate":   b.get("slippage_rate", 0.0002),
        "max_position_pct": config["risk"].get("max_position_pct", 0.10),
        "lookback": 75,
    }

    symbol    = args.symbol    or config["trading"]["symbol"]
    timeframe = args.timeframe or config["trading"]["timeframe"]
    in_start  = args.in_sample_start
    in_end    = args.in_sample_end
    oos_start = args.oos_start
    oos_end   = args.oos_end
    model     = args.model
    dry_run   = args.dry_run
    n_iter    = args.iterations
    n_var     = args.variants

    logger.info(f"Agent run {run_id} | {symbol}/{timeframe} | "
                f"IS={in_start}→{in_end} OOS={oos_start}→{oos_end}")
    logger.info(f"Output dir: {run_dir}")

    # --- Load / cache candles ---
    if dry_run:
        logger.info("DRY RUN: using synthetic candles (no network)")
        synth = _make_synthetic_candles(600)
        is_parquet  = cache_dir / "dry_run_is.parquet"
        oos_parquet = cache_dir / "dry_run_oos.parquet"
        synth.iloc[:450].reset_index(drop=True).to_parquet(is_parquet, index=False)
        synth.iloc[450:].reset_index(drop=True).to_parquet(oos_parquet, index=False)
    else:
        is_parquet  = _fetch_and_cache(symbol, timeframe, in_start,  in_end,  cache_dir)
        oos_parquet = _fetch_and_cache(symbol, timeframe, oos_start, oos_end, cache_dir)

    template         = _load_text(TEMPLATE_PATH)
    trend_engine_src = _load_text(TREND_ENGINE_PATH)

    log_path = run_dir / "run.jsonl"
    all_results: List[Dict] = []

    for iteration in range(1, n_iter + 1):
        logger.info(f"--- Iteration {iteration}/{n_iter} ---")

        # --- Generate code ---
        if dry_run:
            logger.info("DRY RUN: using fixture variants instead of Claude")
            code_blocks = [src for _, src in DRY_RUN_FIXTURES]
            names       = [name for name, _ in DRY_RUN_FIXTURES]
        else:
            if iteration == 1:
                prompt = _build_seed_prompt(
                    symbol, timeframe, n_var, template, trend_engine_src
                )
            else:
                prompt = _build_refinement_prompt(
                    symbol, timeframe, n_var, all_results, iteration, template
                )
            logger.info("Calling Claude...")
            response = _call_claude(prompt, model)
            code_blocks = _extract_python_blocks(response)
            if not code_blocks:
                logger.warning("Claude returned no code blocks — skipping iteration")
                continue
            code_blocks = code_blocks[:n_var]
            names = [f"iter{iteration:02d}_var{j+1:02d}" for j in range(len(code_blocks))]

        # --- Validate and backtest each variant ---
        for idx, (name, source) in enumerate(zip(names, code_blocks)):
            vr = VariantResult(
                name=name,
                iteration=iteration,
                code_path=str(run_dir / f"{name}.py"),
            )
            logger.info(f"  Variant: {name}")

            # 1. AST validation
            try:
                ast_validate(source)
            except ASTValidationError as e:
                vr.error = f"AST_ERROR: {e}"
                vr.rejection_reason = "ast_validation_failed"
                logger.warning(f"    REJECTED (AST): {e}")
                _log_result(log_path, vr)
                all_results.append(asdict(vr))
                continue

            # 2. Write to disk
            code_file = run_dir / f"{name}.py"
            code_file.write_text(source)
            vr.code_path = str(code_file)

            # 3. In-sample backtest
            try:
                vr.in_sample = _run_backtest_subprocess(
                    code_file, is_parquet, sim_config
                )
                logger.info(
                    f"    IS: trades={vr.in_sample['total_trades']} "
                    f"pf={vr.in_sample['profit_factor']:.2f} "
                    f"wr={vr.in_sample['win_rate']:.1%} "
                    f"dd={vr.in_sample['max_drawdown_pct']:.1%}"
                )
            except Exception as e:
                vr.error = f"IS_ERROR: {e}"
                vr.rejection_reason = "in_sample_failed"
                logger.warning(f"    REJECTED (IS error): {e}")
                _log_result(log_path, vr)
                all_results.append(asdict(vr))
                continue

            # 4. Out-of-sample backtest
            try:
                vr.out_of_sample = _run_backtest_subprocess(
                    code_file, oos_parquet, sim_config
                )
                logger.info(
                    f"    OOS: trades={vr.out_of_sample['total_trades']} "
                    f"pf={vr.out_of_sample['profit_factor']:.2f} "
                    f"wr={vr.out_of_sample['win_rate']:.1%} "
                    f"dd={vr.out_of_sample['max_drawdown_pct']:.1%}"
                )
            except Exception as e:
                vr.error = f"OOS_ERROR: {e}"
                vr.rejection_reason = "oos_backtest_failed"
                logger.warning(f"    REJECTED (OOS error): {e}")
                _log_result(log_path, vr)
                all_results.append(asdict(vr))
                continue

            # 5. OOS acceptance gate
            oos = vr.out_of_sample
            if oos["profit_factor"] < OOS_MIN_PF:
                vr.rejection_reason = f"oos_pf={oos['profit_factor']:.3f}<{OOS_MIN_PF}"
            elif oos["max_drawdown_pct"] < OOS_MAX_DD:
                vr.rejection_reason = f"oos_dd={oos['max_drawdown_pct']:.1%}<{OOS_MAX_DD:.0%}"
            elif oos["total_trades"] < OOS_MIN_TRADES:
                vr.rejection_reason = f"oos_trades={oos['total_trades']}<{OOS_MIN_TRADES}"
            else:
                vr.accepted = True
                logger.info(f"    ACCEPTED ✓")

            if not vr.accepted:
                logger.info(f"    REJECTED: {vr.rejection_reason}")

            _log_result(log_path, vr)
            all_results.append(asdict(vr))

    # --- Final report ---
    _write_report(run_dir, all_results, dry_run, model)
    logger.info(f"\nDone. Results in: {run_dir}")
    _print_summary(all_results, run_dir)


def _log_result(log_path: Path, vr: VariantResult) -> None:
    with open(log_path, "a") as f:
        f.write(json.dumps(asdict(vr)) + "\n")


def _write_report(run_dir: Path, results: List[Dict], dry_run: bool, model: str) -> None:
    if dry_run:
        accepted = [r for r in results if r.get("accepted")]
        rejected = [r for r in results if not r.get("accepted")]
        report = (
            "# Dry Run Report\n\n"
            f"Accepted: {len(accepted)}  |  Rejected: {len(rejected)}\n\n"
        )
        for r in accepted:
            oos = r.get("out_of_sample") or {}
            report += (
                f"## {r['name']}\n"
                f"- OOS PF: {oos.get('profit_factor', 'n/a'):.2f}  "
                f"WR: {oos.get('win_rate', 0):.1%}  "
                f"DD: {oos.get('max_drawdown_pct', 0):.1%}  "
                f"Trades: {oos.get('total_trades', 0)}\n\n"
            )
        for r in rejected:
            report += f"## {r['name']} — REJECTED: {r.get('rejection_reason') or r.get('error')}\n\n"
        (run_dir / "report.md").write_text(report)
        return

    prompt = _build_report_prompt(results)
    try:
        report_text = _call_claude(prompt, model, max_tokens=1500)
        (run_dir / "report.md").write_text(report_text)
        logger.info("report.md written (Claude-generated)")
    except Exception as e:
        logger.warning(f"Could not generate Claude report: {e}")
        (run_dir / "report.md").write_text(f"# Report\n\nCould not generate: {e}\n\n"
                                           f"Raw results:\n```json\n{json.dumps(results, indent=2)}\n```")


def _print_summary(results: List[Dict], run_dir: Path) -> None:
    accepted = [r for r in results if r.get("accepted")]
    rejected = [r for r in results if not r.get("accepted")]
    print(f"\n{'='*60}")
    print(f"  AGENT RUN COMPLETE")
    print(f"  Accepted: {len(accepted)}  |  Rejected: {len(rejected)}")
    print(f"{'='*60}")
    if accepted:
        print("\n  Accepted variants (by OOS profit factor):")
        for r in sorted(accepted, key=lambda x: x.get("out_of_sample", {}).get("profit_factor", 0), reverse=True):
            oos = r.get("out_of_sample") or {}
            print(
                f"    {r['name']:<30}  PF={oos.get('profit_factor', 0):.2f}  "
                f"WR={oos.get('win_rate', 0):.1%}  "
                f"DD={oos.get('max_drawdown_pct', 0):.1%}  "
                f"Trades={oos.get('total_trades', 0)}"
            )
    else:
        print("\n  No variants passed the OOS gate.")
        print("  Check report.md for failure analysis.")
    print(f"\n  Output dir: {run_dir}")
    print(f"  Log:        {run_dir / 'run.jsonl'}")
    print(f"  Report:     {run_dir / 'report.md'}")
    print()


# ---------------------------------------------------------------------------
# CLI (standalone, also called from main.py --mode agent)
# ---------------------------------------------------------------------------

def build_arg_parser(parent: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """Return an ArgumentParser for agent mode.

    If `parent` is given, arguments are added to it. Otherwise a new parser
    is created (used when running strategy_agent.py directly).
    """
    p = parent or argparse.ArgumentParser(description="Kronos LLM Strategy Agent")
    p.add_argument("--symbol",           default=None,           help="Asset symbol (e.g. SOL)")
    p.add_argument("--timeframe",        default=None,           help="Timeframe (e.g. 1h)")
    p.add_argument("--in-sample-start",  dest="in_sample_start", default="2024-01-01")
    p.add_argument("--in-sample-end",    dest="in_sample_end",   default="2024-10-01")
    p.add_argument("--oos-start",        dest="oos_start",       default="2024-10-01")
    p.add_argument("--oos-end",          dest="oos_end",         default="2025-04-01")
    p.add_argument("--iterations",       type=int, default=2,    help="LLM refinement iterations")
    p.add_argument("--variants",         type=int, default=4,    help="Variants per iteration")
    p.add_argument("--model",            default="claude-sonnet-4-6")
    p.add_argument("--dry-run",          action="store_true",
                   help="Use fixture variants + synthetic candles, no API key needed")
    return p


def main() -> None:
    parser = build_arg_parser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if not args.dry_run and not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set. Use --dry-run to test without an API key.")
        sys.exit(1)

    run_agent_loop(config, args)


if __name__ == "__main__":
    main()

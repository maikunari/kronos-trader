"""
setup_agent.py
LLM-driven setup-detector generation loop for Kronos Phase C.

The loop (parallel to strategy_agent.py's trend-engine loop, but evaluating
via the Phase C validation harness instead of a single-timeseries
backtest):

  1. Ask Claude to generate N setup detectors following setup_template.md
  2. AST-validate each module (import allowlist includes Phase C shared
     modules; stdlib-and-numeric otherwise)
  3. Run each variant in a subprocess through validation.report.run_validation
     against the 8-ticker probe
  4. Feed the recall / precision / capture stats back to Claude
  5. Gate acceptance on §9.5 graduation: recall ≥ 40%, precision ≥ 25%,
     capture ≥ 30% (for at least one direction)
  6. Repeat for K iterations, then ask Claude to write a final report

Usage:
    python setup_agent.py                   # defaults, requires ANTHROPIC_API_KEY
    python setup_agent.py --dry-run         # fixture variants, no API key
    python setup_agent.py --iterations 3 --variants 5
    python main.py --mode setup-agent --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from agent_infra import ASTValidationError, ASTValidator, SubprocessError, run_subprocess

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent
SETUPS_DIR = REPO_ROOT / "setups" / "generated"
TEMPLATE_PATH = REPO_ROOT / "setup_template.md"
BASELINE_PATH = REPO_ROOT / "setups" / "v_reversal.py"
CBS_TRADES_PATH = REPO_ROOT / "validation" / "cbs_trades.yaml"
RUNNER_PATH = REPO_ROOT / "setup_runner.py"


# ---------------------------------------------------------------------------
# Validator config
# ---------------------------------------------------------------------------

ALLOWED_IMPORTS = frozenset({
    # stdlib + numeric
    "pandas", "numpy", "math", "typing", "dataclasses",
    "collections", "enum", "functools", "itertools",
    "statistics", "abc", "numbers",
    # Phase C shared modules — detectors reuse these utilities
    "setups", "support_resistance", "pivot", "indicators",
})

FORBIDDEN_BUILTINS = frozenset({
    "open", "exec", "eval", "compile", "__import__",
    "breakpoint", "input", "memoryview",
})

VALIDATOR = ASTValidator(
    allowed_imports=ALLOWED_IMPORTS,
    forbidden_builtins=FORBIDDEN_BUILTINS,
    required_class="Setup",
    required_methods=("detect",),
)


# ---------------------------------------------------------------------------
# Probe + gate
# ---------------------------------------------------------------------------

PROBE_TICKERS = ["HYPE", "TAO", "ENA", "NEAR", "LINK", "AVAX", "ZRO", "WLD"]
PROBE_TIMEFRAME = "1h"
PROBE_START = "2025-04-01"
PROBE_END = "2026-04-01"
PROBE_THRESHOLD_PCT = 0.20
PROBE_WINDOW_HOURS = 72.0

# §9.5 graduation gate
GATE_RECALL_MIN = 0.40
GATE_PRECISION_MIN = 0.25
GATE_CAPTURE_MIN = 0.30

SUBPROCESS_TIMEOUT = 600  # Phase B validation is slow; one run can take minutes


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class VariantResult:
    name: str
    iteration: int
    code_path: str
    stats: Optional[Dict] = None   # output of setup_runner.run_setup
    accepted_direction: Optional[str] = None  # "long" | "short" | None
    rejection_reason: str = ""
    error: str = ""

    @property
    def accepted(self) -> bool:
        return self.accepted_direction is not None


@dataclass
class AgentRun:
    run_id: str
    model: str
    iterations: int
    variants_per_iteration: int
    probe_tickers: List[str]
    results: List[VariantResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Runner subprocess wrapper
# ---------------------------------------------------------------------------

def _run_detector_subprocess(detector_path: Path) -> Dict:
    """Invoke setup_runner.py in a subprocess against the probe config.
    Returns the stats dict. Raises SubprocessError on failure."""
    config = {
        "tickers": PROBE_TICKERS,
        "timeframe": PROBE_TIMEFRAME,
        "start": PROBE_START,
        "end": PROBE_END,
        "threshold_pct": PROBE_THRESHOLD_PCT,
        "window_hours": PROBE_WINDOW_HOURS,
    }
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    try:
        json.dump(config, tmp)
        tmp.flush()
        tmp.close()
        argv = [
            sys.executable,
            str(RUNNER_PATH),
            str(detector_path),
            tmp.name,
        ]
        return run_subprocess(
            argv, timeout=SUBPROCESS_TIMEOUT, cwd=REPO_ROOT, parse_json=True,
        )  # type: ignore[return-value]
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

def _check_gate(stats: Dict) -> Tuple[Optional[str], str]:
    """Return (accepted_direction_or_none, reason).

    Accepted if EITHER direction clears all three thresholds.
    """
    reasons: list[str] = []
    for direction in ("long", "short"):
        s = stats.get(direction)
        if s is None:
            reasons.append(f"{direction}:missing")
            continue
        r = s.get("recall", 0.0) or 0.0
        p = s.get("precision", 0.0) or 0.0
        c = s.get("median_capture_ratio", 0.0) or 0.0
        fails = []
        if r < GATE_RECALL_MIN:
            fails.append(f"recall={r*100:.1f}%<{GATE_RECALL_MIN*100:.0f}%")
        if p < GATE_PRECISION_MIN:
            fails.append(f"prec={p*100:.1f}%<{GATE_PRECISION_MIN*100:.0f}%")
        if c < GATE_CAPTURE_MIN:
            fails.append(f"cap={c*100:.1f}%<{GATE_CAPTURE_MIN*100:.0f}%")
        if not fails:
            return direction, ""
        reasons.append(f"{direction}:{'+'.join(fails)}")
    return None, " | ".join(reasons)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _extract_python_blocks(text: str) -> List[str]:
    pattern = r"```python\s*\n(.*?)```"
    blocks = re.findall(pattern, text, re.DOTALL)
    return [b.strip() for b in blocks if b.strip()]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_seed_prompt(n_variants: int) -> str:
    template = _load_text(TEMPLATE_PATH)
    baseline = _load_text(BASELINE_PATH)
    cbs = _load_text(CBS_TRADES_PATH)
    return textwrap.dedent(f"""\
    You are an expert quantitative trading developer working on the Kronos
    Phase C setup-detector framework.

    Your task is to generate {n_variants} DIVERSE setup detectors that plug
    into the existing validation harness and beat the current baseline on
    the 8-ticker probe.

    ## Contract (you MUST follow this exactly)

    {template}

    ## Baseline to beat: v_reversal detector

    ```python
    {baseline}
    ```

    ## CBS seed trade corpus (the 'ground truth' we're trying to replicate)

    ```yaml
    {cbs}
    ```

    ## Your task

    Generate {n_variants} setup detectors. They must be meaningfully different
    from each other AND from the baseline. Good directions:

    - Architecture §4.2 two-bar momentum continuation (AO two-bar rule)
    - Architecture §4.5 consolidation-under-resistance (leading)
    - Architecture §4.6 weak-bounce-from-support (shorts, leading)
    - Hybrid: combine two existing signals (e.g. divergence + consolidation)
    - Short-side specifically — all three baselines have 0% short recall

    Each detector should apply the four CBS-locked rules (wick-based stops,
    close-based breakout confirmation, 10% min-target-distance filter,
    binary outcomes) or justify in the docstring why it deviates.

    For each detector output ONLY the Python code in a fenced block:

    ```python
    # Detector: <descriptive name>
    <full module code>
    ```

    Do NOT include explanatory prose between code blocks. Output exactly
    {n_variants} code blocks and nothing else.
    """)


def _build_refinement_prompt(n_variants: int, prior_results: List[Dict], iteration: int) -> str:
    template = _load_text(TEMPLATE_PATH)
    # Drop bulky fields from the summary we send to Claude
    summary = [
        {
            "name": r.get("name"),
            "accepted": r.get("accepted"),
            "accepted_direction": r.get("accepted_direction"),
            "rejection_reason": r.get("rejection_reason") or r.get("error", "")[:200],
            "stats_long": (r.get("stats") or {}).get("long"),
            "stats_short": (r.get("stats") or {}).get("short"),
        }
        for r in prior_results
    ]
    return textwrap.dedent(f"""\
    You are an expert quantitative trading developer. Iteration {iteration}
    of the Kronos Phase C setup-detector search.

    ## Contract

    {template}

    ## Results so far

    {json.dumps(summary, indent=2, default=str)}

    ## Your task

    Based on the results above:

    1. Identify which approaches produced non-zero recall and why.
    2. Identify what kept precision low (too many false positives) vs
       what kept recall low (too many filters).
    3. Generate {n_variants} REFINED detectors. Aim for at least ONE
       direction (long OR short) to clear the graduation gate:
         recall    ≥ {GATE_RECALL_MIN*100:.0f}%
         precision ≥ {GATE_PRECISION_MIN*100:.0f}%
         capture   ≥ {GATE_CAPTURE_MIN*100:.0f}%

    Output {n_variants} Python code blocks. No prose between blocks.
    """)


def _build_report_prompt(results: List[Dict]) -> str:
    accepted = [r for r in results if r.get("accepted")]
    rejected = [r for r in results if not r.get("accepted")]
    return textwrap.dedent(f"""\
    You are an expert quantitative trading analyst.

    Below are results from an automated setup-detector search. The
    graduation gate is: recall ≥ {GATE_RECALL_MIN*100:.0f}%, precision ≥
    {GATE_PRECISION_MIN*100:.0f}%, capture ≥ {GATE_CAPTURE_MIN*100:.0f}%.

    Accepted ({len(accepted)}):
    {json.dumps(accepted, indent=2, default=str)}

    Rejected ({len(rejected)}):
    {json.dumps(rejected[:10], indent=2, default=str)}

    Write a concise research report (markdown, under 600 words):
    1. Top 3 detectors by accepted-direction capture% — what worked?
    2. Common failure modes — precision collapse vs recall collapse vs
       direction asymmetry (shorts).
    3. Which of the architecture §4.x setups are worth pursuing next?
    4. Overfitting warnings — 2 labeled pops is a tiny sample.
    """)


# ---------------------------------------------------------------------------
# Claude client
# ---------------------------------------------------------------------------

def _call_claude(prompt: str, model: str, max_tokens: int = 6000) -> str:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


# ---------------------------------------------------------------------------
# Fixture variants for --dry-run
# ---------------------------------------------------------------------------

FIXTURE_GOOD = '''\
"""Simple two-bar AO momentum fixture detector for dry-run testing."""
from typing import Optional

import pandas as pd
import numpy as np

from setups.base import MarketContext, Trigger, build_tp_ladder
from indicators.awesome_oscillator import two_bar_same_color


class Setup:
    """Fire long on two green AO bars, short on two red AO bars."""

    name: str = "fixture_two_bar_ao"

    def __init__(self, *, atr_period: int = 14, min_target_distance_pct: float = 0.05):
        self.atr_period = atr_period
        self.min_target_distance_pct = min_target_distance_pct

    def detect(self, ctx: MarketContext) -> Optional[Trigger]:
        if ctx.ao is None or len(ctx.candles) < self.atr_period + 3:
            return None
        for direction, color in (("long", "green"), ("short", "red")):
            if not two_bar_same_color(ctx.ao, color):
                continue
            entry = ctx.current_price
            if direction == "long":
                pivot_low = float(ctx.candles["low"].iloc[-2])
                stop = pivot_low * 0.997
            else:
                pivot_high = float(ctx.candles["high"].iloc[-2])
                stop = pivot_high * 1.003
            atr = self._atr(ctx.candles)
            ladder = build_tp_ladder(entry, direction, ctx.sr_zones, atr)
            if not ladder:
                continue
            if max(abs(tp.price - entry) / entry for tp in ladder) < self.min_target_distance_pct:
                continue
            return Trigger(
                ticker=ctx.ticker,
                timestamp=ctx.timestamp or pd.Timestamp.now(tz="UTC"),
                action="open_new",
                direction=direction,
                entry_price=entry,
                stop_price=stop,
                tp_ladder=ladder,
                setup=self.name,
                confidence=0.5,
                size_fraction=0.3,
                components={"fixture": True},
            )
        return None

    def _atr(self, candles: pd.DataFrame) -> float:
        high = candles["high"].astype(float).to_numpy()
        low = candles["low"].astype(float).to_numpy()
        close = candles["close"].astype(float).to_numpy()
        tr = np.maximum(high[1:] - low[1:],
             np.maximum(np.abs(high[1:] - close[:-1]),
                        np.abs(low[1:] - close[:-1])))
        if len(tr) < self.atr_period:
            return float(np.mean(tr)) if len(tr) else 0.0
        return float(np.mean(tr[-self.atr_period:]))
'''

FIXTURE_BAD_IMPORT = '''\
"""This detector imports os — should be rejected by the AST validator."""
import os
import pandas as pd
from setups.base import MarketContext

class Setup:
    name: str = "fixture_bad_import"
    def __init__(self):
        pass
    def detect(self, ctx: MarketContext):
        return None
'''

FIXTURE_CRASH = '''\
"""This detector raises on every call — should be caught in subprocess."""
from setups.base import MarketContext

class Setup:
    name: str = "fixture_crash"
    def __init__(self):
        pass
    def detect(self, ctx: MarketContext):
        raise ValueError("intentional crash for testing")
'''

FIXTURE_NO_SETUP_CLASS = '''\
"""No Setup class — should fail AST validation."""
from setups.base import MarketContext

class WrongName:
    name: str = "fixture_wrong_class"
    def detect(self, ctx: MarketContext):
        return None
'''

DRY_RUN_FIXTURES: List[Tuple[str, str]] = [
    ("good_two_bar_ao",       FIXTURE_GOOD),
    ("bad_import_os",         FIXTURE_BAD_IMPORT),
    ("crash_on_detect",       FIXTURE_CRASH),
    ("missing_setup_class",   FIXTURE_NO_SETUP_CLASS),
]


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def run_agent_loop(args: argparse.Namespace) -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = SETUPS_DIR / f"runs/{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Setup-agent run {run_id} | probe={len(PROBE_TICKERS)} tickers")
    logger.info(f"Output dir: {run_dir}")

    log_path = run_dir / "run.jsonl"
    all_results: List[Dict] = []

    for iteration in range(1, args.iterations + 1):
        logger.info(f"--- Iteration {iteration}/{args.iterations} ---")

        if args.dry_run:
            logger.info("DRY RUN: using fixture variants instead of Claude")
            code_blocks = [src for _, src in DRY_RUN_FIXTURES]
            names = [name for name, _ in DRY_RUN_FIXTURES]
        else:
            if iteration == 1:
                prompt = _build_seed_prompt(args.variants)
            else:
                prompt = _build_refinement_prompt(args.variants, all_results, iteration)
            logger.info("Calling Claude...")
            response = _call_claude(prompt, args.model)
            code_blocks = _extract_python_blocks(response)
            if not code_blocks:
                logger.warning("Claude returned no code blocks — skipping iteration")
                continue
            code_blocks = code_blocks[: args.variants]
            names = [f"iter{iteration:02d}_var{j+1:02d}" for j in range(len(code_blocks))]

        for name, source in zip(names, code_blocks):
            vr = VariantResult(
                name=name,
                iteration=iteration,
                code_path=str(run_dir / f"{name}.py"),
            )
            logger.info(f"  Variant: {name}")

            # 1. AST validation
            try:
                VALIDATOR.validate(source)
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

            # 3. Run validation harness in subprocess
            try:
                vr.stats = _run_detector_subprocess(code_file)
            except SubprocessError as e:
                vr.error = f"RUNNER_ERROR: {e}"
                vr.rejection_reason = "runner_failed"
                logger.warning(f"    REJECTED (runner error): {e}")
                _log_result(log_path, vr)
                all_results.append(asdict(vr))
                continue

            # 4. Gate check
            direction, reason = _check_gate(vr.stats)
            if direction is not None:
                vr.accepted_direction = direction
                logger.info(f"    ACCEPTED ({direction}) ✓")
            else:
                vr.rejection_reason = reason
                logger.info(f"    REJECTED: {reason}")

            _log_result(log_path, vr)
            all_results.append(asdict(vr))

    _write_report(run_dir, all_results, args.dry_run, args.model)
    _print_summary(all_results, run_dir)


def _log_result(log_path: Path, vr: VariantResult) -> None:
    with open(log_path, "a") as f:
        f.write(json.dumps(asdict(vr), default=str) + "\n")


def _write_report(run_dir: Path, results: List[Dict], dry_run: bool, model: str) -> None:
    accepted = [r for r in results if r.get("accepted")]
    rejected = [r for r in results if not r.get("accepted")]

    if dry_run:
        lines = [
            "# Setup-Agent Dry Run Report",
            "",
            f"Accepted: {len(accepted)}  |  Rejected: {len(rejected)}",
            "",
        ]
        for r in accepted:
            stats = r.get("stats") or {}
            d = r.get("accepted_direction")
            s = stats.get(d) if d else None
            if s:
                lines.append(
                    f"## {r['name']} — {d}\n"
                    f"- recall {s.get('recall', 0)*100:.1f}% | "
                    f"prec {s.get('precision', 0)*100:.1f}% | "
                    f"cap {s.get('median_capture_ratio', 0)*100:.1f}% | "
                    f"trigs {s.get('triggers', 0)} | TP {s.get('true_positives', 0)}\n"
                )
        for r in rejected:
            lines.append(
                f"## {r['name']} — REJECTED: "
                f"{r.get('rejection_reason') or r.get('error', '')[:200]}"
            )
        (run_dir / "report.md").write_text("\n".join(lines))
        return

    try:
        prompt = _build_report_prompt(results)
        report_text = _call_claude(prompt, model, max_tokens=1500)
        (run_dir / "report.md").write_text(report_text)
        logger.info("report.md written (Claude-generated)")
    except Exception as e:
        logger.warning(f"Could not generate Claude report: {e}")
        (run_dir / "report.md").write_text(
            f"# Setup-Agent Report\n\nCould not generate: {e}\n\n"
            f"Raw results:\n```json\n{json.dumps(results, indent=2, default=str)}\n```"
        )


def _print_summary(results: List[Dict], run_dir: Path) -> None:
    accepted = [r for r in results if r.get("accepted")]
    rejected = [r for r in results if not r.get("accepted")]
    print(f"\n{'='*60}")
    print(f"  SETUP-AGENT RUN COMPLETE")
    print(f"  Accepted: {len(accepted)}  |  Rejected: {len(rejected)}")
    print(f"{'='*60}")
    if accepted:
        print("\n  Accepted variants:")
        for r in accepted:
            d = r.get("accepted_direction")
            s = (r.get("stats") or {}).get(d) if d else None
            if s:
                print(
                    f"    {r['name']:<30} {d:<6} recall={s.get('recall', 0)*100:.1f}% "
                    f"prec={s.get('precision', 0)*100:.1f}% "
                    f"cap={s.get('median_capture_ratio', 0)*100:.1f}% "
                    f"trigs={s.get('triggers', 0)}"
                )
    else:
        print("\n  No variants passed the graduation gate.")
    print(f"\n  Output dir: {run_dir}")
    print(f"  Log:        {run_dir / 'run.jsonl'}")
    print(f"  Report:     {run_dir / 'report.md'}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser(parent: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    p = parent or argparse.ArgumentParser(description="Kronos Phase C Setup Agent")
    p.add_argument("--iterations", type=int, default=2, help="LLM refinement iterations")
    p.add_argument("--variants",   type=int, default=4, help="Variants per iteration")
    p.add_argument("--model",      default="claude-sonnet-4-6")
    p.add_argument("--dry-run",    action="store_true",
                   help="Use fixture variants — no API key needed")
    return p


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.dry_run and not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set. Use --dry-run to test without one.")
        return 1

    run_agent_loop(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

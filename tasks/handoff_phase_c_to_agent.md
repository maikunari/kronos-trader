# Handoff: Phase C complete, entering agent-loop era

**Date:** 2026-04-25
**Previous handoff:** [`handoff_phase_b_to_c.md`](handoff_phase_b_to_c.md)
**Read first:** this file (2 pages).

## TL;DR

- **Phase C setup library is complete** — three detectors land-and-tested:
  divergence, consolidation_breakout, v_reversal. Full suite 391 passing.
- **v_reversal is the best baseline** (defaults tuned in commit `c97ebc3`):
  8.3% recall, 3.4% precision, +3.41% mean realized return, +16.9% capture
  on the 8-ticker / 1h / 1-year probe.
- **Deterministic tuning is exhausted.** Two grid sweeps landed as negative
  evidence ([`tune_consolidation_breakout.md`](tune_consolidation_breakout.md),
  [`tune_v_reversal.md`](tune_v_reversal.md)). Nothing in parameter space
  hits the §9.5 graduation gate (recall ≥ 40%, precision ≥ 25%, capture ≥
  30%).
- **Two agentic loops are wired and ready** — the cloud-session trend-engine
  agent (PR #1, `strategy_agent.py`) and the Phase C setup-detector agent
  (`setup_agent.py`). Both share `agent_infra.sandbox` (AST validator +
  bounded-timeout subprocess runner). Both work end-to-end in `--dry-run`.
- **Next session: run the Phase C agent live** (spend API money on 8-16
  real Claude-generated detectors) AND grow `validation/cbs_trades.yaml`.

## Where we are

### Complete and tested

| Module | Purpose | Commit |
|---|---|---|
| `setups/divergence.py` | Divergence reversal (§4.1) | pre-session |
| `setups/consolidation_breakout.py` | Diagonal breakout (§4.3) | `23fde71` |
| `setups/v_reversal.py` | V-reversal at extended move (§4.4) | `a9bf37e` |
| `setups/base.py` | `build_tp_ladder` now lives here | `c103139` |
| `validation/cbs_replication.py` | Match detectors vs CBS trade log | `65961e2` |
| `agent_infra/sandbox.py` | Shared ASTValidator + subprocess runner | `1a80392` |
| `strategy_agent.py` | Trend-engine LLM loop (from PR #1) | `6b161f8` |
| `setup_agent.py` | Phase C LLM loop | `bf694f6` |
| `setup_runner.py` | Subprocess validator for generated detectors | `bf694f6` |
| `setup_template.md` | Contract doc for generated detectors | `bf694f6` |

### Validation runs committed

- [`phase_b_validation_v5.md`](phase_b_validation_v5.md) — all 3 detectors on probe
- [`cbs_replication_v3.md`](cbs_replication_v3.md) — all 3 detectors vs CBS seed trades (0/2 matched; HYPE window didn't catch, SPX unfetchable)
- [`tune_consolidation_breakout.md`](tune_consolidation_breakout.md), [`tune_breakout_threshold.md`](tune_breakout_threshold.md) — negative gate
- [`tune_v_reversal.md`](tune_v_reversal.md) — best-in-grid became new defaults

### Current probe numbers (from `phase_b_validation_v5.md`)

| setup | dir | trigs | TP | recall | prec | cap% | mean_ret |
|---|---|---|---|---|---|---|---|
| divergence | long | 12 | 1 | 4.2% | 8.3% | -10.3% | -2.14% |
| divergence | short | 14 | 0 | 0% | 0% | — | — |
| breakout | long | 3 | 0 | 0% | 0% | — | — |
| breakout | short | 1 | 0 | 0% | 0% | — | — |
| **v_reversal (old defaults)** | long | 133 | 2 | 8.3% | 1.5% | 16.9% | +3.41% |
| v_reversal | short | 119 | 0 | 0% | 0% | — | — |

After the `c97ebc3` tune, v_reversal long is now 59 triggers / 3.4% precision
(same 8.3% recall, same +3.41% return). **`phase_b_validation_v6.md` has not been re-run with the new defaults — worth doing in 60s as the first action next session.**

### The key structural findings to not re-debate

1. **Shorts are broken across all three detectors** (0 TP ever). This is a
   universe + labeler characteristic, not a parameter issue. A short-only
   detector that actually works on this universe would be genuine novel
   value — good prompt material for the agent.
2. **§4.3 consolidation-breakout fires cleanly but doesn't correlate with
   20%-pop labels.** Even at 10%-threshold pops, only 2/216 fire. Tuning
   exhausted — either skip this pattern or rethink labeling.
3. **§4.4 v_reversal is the first setup with positive economics.** Triggers
   make money on average (+3.41%). Problem is it fires too often (over-loose
   filter). Tightening helps 2.3× but can't reach the gate.

## What the next session should do

### Step 1: re-baseline with tuned v_reversal defaults (5 min)

```bash
./venv/bin/python validate_phase_b.py --out tasks/phase_b_validation_v6.md \
    --tickers HYPE TAO ENA NEAR LINK AVAX ZRO WLD
```

Confirms the precision lift from `c97ebc3` on the full 3-detector stack.

### Step 2: first live agent run (15-30 min, ~$2-5 API spend)

Assumes `ANTHROPIC_API_KEY` is exported.

```bash
./venv/bin/python main.py --mode setup-agent --iterations 2 --variants 4
```

2 iterations × 4 variants × ~90s validation per variant = ~12-15 min wall
time. Output lands in `setups/generated/runs/<timestamp>/` (gitignored).

**What to look for:**
- Does Claude generate syntactically valid detectors that pass the AST
  validator? (Fixtures already prove the plumbing works.)
- Do any variants hit the graduation gate? Probably no on iteration 1 —
  but we learn what shape the LLM reaches for first.
- Does iteration 2 meaningfully refine after seeing iteration 1's stats?
- Inspect `report.md` (Claude-authored critique) for qualitative takeaways.

If iteration 1 produces nothing useful, try `--iterations 4 --variants 6`
for more search volume. The bound is the $10-20 range per run.

### Step 3: grow `validation/cbs_trades.yaml` — highest-leverage data work

Corpus is still 2 trades. The priorities recorded in `cbs_trades.yaml`
(bottom of the file, lines 78-85) still apply:

1. **Closed trades with outcomes** (winners AND losers).
2. **Trades on tickers in our current universe** (HYPE, TAO, NEAR, LINK,
   AVAX, PENDLE) so we can replay our detector on the same candles.
3. **Older trades** (Jan/Feb/Mar 2026) with fully resolved outcomes.
4. **Losing trades** — show CBS's invalidation discipline.
5. **Short trades** — we have zero.

5-10 more trades materially improves confidence in both the replication
matcher and the agent's seed prompt (which includes `cbs_trades.yaml`
verbatim).

### Step 4: targeted structural questions the agent can't yet answer

These are not in the agent's prompt — consider adding them or iterating
manually:

- Should the pop labeler use a **lower threshold for specific setups**?
  Breakout triggers rarely produce 20% moves; maybe they should be
  evaluated against 8% pops, while divergence and v_reversal stay at 20%.
  The labeler is single-threshold today; per-setup thresholds would need
  a validation harness change.
- **Shorts on what universe?** Crypto shorts are structurally bad on most
  HL tickers (alts don't drop 20% in 72h nearly as often as they rise).
  Worth considering a short-only validation run on `XLM ADA DOT FET` —
  lower-beta, more mean-reverting — to see if the problem is universe or
  setup-class.

## Operational hygiene

- **Auto mode / plan mode:** user defaults to plan mode for non-trivial
  tasks per `~/.claude/CLAUDE.md`. Enter plan mode for any new
  architecture (e.g. adding a validation-harness feature) unless the user
  explicitly asks to skip planning.
- **Commit cadence:** per-step on this project. Don't batch.
- **Tests:** `./venv/bin/python -m pytest -q` — 391 tests pass in ~80s.
- **Secrets:** `ANTHROPIC_API_KEY` goes in `.env` (gitignored). Never
  print the key or commit it.
- **Subprocess timeout:** `setup_agent.py` pins 600s per variant (a full
  8-ticker / 1y validation run). Raise if the probe grows.

## Git state at handoff

- Branch: `main`, up to date with `origin/main`
- Last commit: `bf694f6` (Phase C agent)
- Working tree clean
- 391 tests passing
- Two stale uncommitted artifacts (`optimizer_*.json`) deleted this session

Good session. ~20 commits. The infrastructure is in place for the
interesting part — actually running the agent and growing the ground
truth.

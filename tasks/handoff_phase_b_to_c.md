# Handoff: Phase B complete, entering Phase C

**Date:** 2026-04-18
**Purpose:** orient the next session quickly. Read this first.

## TL;DR

- Infrastructure (universe, validation framework, divergence detector, data cache) is **done and tested** (321+ tests, full suite passes).
- Phase B signal validation ran 5 times across different configurations.
- **Divergence-only catches 5-12% of labeled pops across all configs.** That's the ceiling on a single-setup system, not a bug.
- CBS himself doesn't primarily use divergence in his recent posted trades — **he's trading consolidation breakouts and V-reversals**, which are Phase C setups we haven't built yet.
- Sample size is too small (1 TP in best run) to draw statistical conclusions from divergence alone.
- Next session: **build Phase C setups, starting with the ones CBS's actual trades are using.**

## Where we are

### Architecture is implemented up to Phase A + half of Phase B

What's built and tested:

| Module | Purpose | Status |
|--------|---------|--------|
| `pivot.py` | Fractal swing detection on close series | ✓ |
| `indicators/awesome_oscillator.py` | AO + bar colors + two-bar rule | ✓ |
| `indicators/rsi.py` | Wilder RSI + extreme zones | ✓ |
| `support_resistance.py` | Zones with CBS touch-count scoring (3-touch peak, decay after) | ✓ |
| `data_cache.py` | Local OHLCV cache with incremental fetch | ✓ |
| `universe_builder.py` | HL + CoinGecko filtered universe | ✓ |
| `setups/base.py` | Trigger / TPLevel / MarketContext / SetupDetector protocol | ✓ |
| `setups/divergence.py` | First detector — divergence reversal | ✓ |
| `validation/labeler.py` | Pop event labeling | ✓ |
| `validation/matcher.py` | Trigger-to-pop matching with recall/precision | ✓ |
| `validation/capture.py` | Single-trade capture simulator | ✓ |
| `validation/report.py` | Orchestrator + markdown/json output | ✓ |
| `validate_phase_b.py` | CLI wrapper | ✓ |

Still to build (Phase C and beyond):

| Module | Purpose | Priority |
|--------|---------|----------|
| `setups/consolidation_breakout.py` | Diagonal breakout from range (§4.3) | **HIGH — CBS HYPE trade** |
| `setups/v_reversal.py` | V-reversal at extended move (§4.4) | **HIGH — CBS SPX trade** |
| `setups/consolidation_under_resistance.py` | Leading pre-breakout setup (§4.5) | MED |
| `setups/two_bar_momentum.py` | Momentum continuation (§4.2) | MED |
| `setups/weak_bounce.py` | Failed support breakdown (§4.6) | LOW (shorts) |
| `validation/cbs_replication.py` | Match CBS trade log vs our trigger stream | **HIGH** |
| `position.py` | Pyramid-aware position model | Phase D |
| `scanner.py` | Multi-symbol orchestrator | Phase E |

## Key findings from Phase B runs

Five validation runs, summarized:

| Run | Universe | Filter | Stops | TPs | Recall | Prec | Cap% |
|-----|----------|--------|-------|-----|--------|------|------|
| v1 | Krillin 21 (w/ OMNI) | none | close-pivot | 4L/1S | 11.8%/9.1% | 5.3%/1.8% | +0.5%/-12.6% |
| v2a | Krillin 21 | 10% TP filter | close-pivot | 1L | 2.9% | 9.1% | +37.4% |
| **v2b** | **Krillin 20 (no OMNI)** | **10% TP filter** | **close-pivot** | **1L** | **5.0%** | **25.0%** | **+37.4%** |
| v3a | Core 47 | 10% TP | close-pivot | 1L | 4.0% | 8.3% | -3.9% |
| v3b | Core 47 | 10% TP | **wick-pivot** | 1L | 4.0% | 8.3% | -10.3% |

**v2b is the best result by a wide margin** — 25% precision, +37.4% capture on longs. That's a passing precision and passing capture; only recall (5%) fails the graduation gate.

The core-universe runs (47 tickers) actually **did worse** than the Krillin 20-ticker hand-curated list. Adding mid-cap majors (ADA, AVAX, LINK) and memes (FARTCOIN, PUMP) **diluted** the signal — divergence patterns on those ticker types don't behave like the mid-cap narrative alts Krillin curates.

## The alpha findings

### Finding 1: HYPE is CBS's most-traded market, was excluded by our filter

Our `$10B` MC ceiling filtered HYPE out (it's ~$13B MC). CBS explicitly said Apr 14: *"no real reason to trade anything other than $HYPE"*. We raised the ceiling to $30B this session but **the universe hasn't been rebuilt yet** (it's a network-dependent task, ~5 min).

**Action for next session:** rebuild universe:
```bash
./venv/bin/python universe_builder.py
```

Expected change: adds HYPE, SUI-adjacent mid-majors, maybe SOL. Doesn't add BTC/ETH (>$30B).

### Finding 2: CBS's actual trades are NOT divergences

Evidence from 2 documented trades (seeded in `validation/cbs_trades.yaml`):

- **HYPE long (Apr 5, 2026)**: consolidation breakout from 35.69 support zone, NOT a divergence. Setup matches §4.3 (diagonal breakout) and §4.5 (consolidation-under-resistance).
- **SPX long (Apr 5, 2026)**: V-reversal after a 10% drawdown, then break of local resistance at 6,640. Setup matches §4.4 (V-reversal at extended move) and §4.3.

**Neither trade would be caught by our current (divergence-only) detector.** This isn't a tuning issue — it's a structural gap. The architecture predicted this exactly: divergence is 1 of 6 setups, covering ~15% of patterns.

### Finding 3: Wick-based stop placement is correct, doesn't save bad trades

We corrected stop placement to use the pivot bar's wick (low for longs, high for shorts) rather than the close. Conceptually right and matches CBS methodology. In practice, on the one TP in the core-universe run, the wider wick-based stop just took a bigger loss when the setup genuinely invalidated. The fix doesn't compensate for a weak signal.

Commit: `b8d09e3` covers the wick-placement fix + all v2/v3 validation artifacts.

## Critical design decisions recorded so far

Don't re-debate these without fresh evidence:

1. **CBS's 3-touch sweet spot for S/R strength**: more touches = *weaker*, not stronger. Counter-intuitive vs naive algos. Test `test_touch_strength_3_is_strictly_greater_than_10` locks it.
2. **Stops fire on wicks, placement references wicks**: real exchange behavior. Placement uses pivot bar low/high, not close.
3. **Close-based breakout confirmation**: a wick through a level isn't a breakout — only a close is. `support_resistance.confirmed_breakout` requires N consecutive closes.
4. **Binary trade outcomes**: every trade exits on stop or full TP ladder fill — no timeout close. Architecture doc rule: *"trade either hits TP target or stops out, those are the only two possible outcomes."*
5. **10% minimum target distance filter**: reject triggers where the TP ladder has no target ≥ 10% away. Codifies CBS's "look left for meaningful resistance."

## What the next session should do (in order)

### Step 1: Rebuild universe with raised MC ceiling (~5 min)
```bash
./venv/bin/python universe_builder.py
# Expected: ~55 tickers now (adds HYPE + SOL + a few others)
git add markets/core_universe.yaml
git commit -m "chore(universe): rebuild with MC ceiling raised to $30B — adds HYPE"
```

### Step 2: Build CBS replication matcher (~1 day)
`validation/cbs_replication.py` — for each trade in `validation/cbs_trades.yaml`:
- Pull candles for `entry_date ± window` using `data_cache.get_candles`
- Run the detector library at each bar in the window
- Check: did any detector fire a matching-direction trigger within ±N hours of `entry_date`?
- Per-trade output: matched/unmatched, which detector, what was the gap

CLI wrapper: `validate_cbs_replication.py`. Output format: markdown table + per-trade diagnosis (which filters rejected us when we didn't match).

### Step 3: Build Phase C setups in evidence-driven order (~2-3 weeks)

Order dictated by CBS's posted trades, not architecture-doc order:

1. **`setups/consolidation_breakout.py`** — matches HYPE and SPX trades. Trigger: price has been in a tight range for N bars, then closes above the range high (or below range low for shorts). Use `support_resistance` zones to identify the range boundaries.

2. **`setups/v_reversal.py`** — matches SPX setup. Trigger: price made a significant extended move (X% over Y days), then reversed with higher-low structure. Distinct from divergence — no indicator divergence needed.

3. After each new setup: **run `validate_phase_b.py`** to see how recall grows. Target: combined recall ≥ 40% across all setups on the no-OMNI Krillin universe.

4. **Also run `validate_cbs_replication.py`** after each new setup. Target: growing overlap percentage with CBS's actual entries.

### Step 4: Ask operator for more trades continuously

Every new CBS trade the operator shares goes into `validation/cbs_trades.yaml`. Even 5-10 more trades with outcomes (winners AND losers) materially improves confidence. Priorities listed at bottom of that file.

## Decision points for next session

1. **Do we build V-reversal or consolidation-breakout first?** Both are in CBS's recent tapes. Consolidation-breakout is conceptually simpler (clearer trigger definition), so probably first. But V-reversal catches a different pattern class. Either is defensible.

2. **Do we keep the Krillin 20-ticker list as the primary validation universe?** v2b results were best on that list. Even with HYPE added to the core universe, the Krillin list may remain the "clean" validation target. Worth running both side-by-side.

3. **Shorts are completely broken in every run** (0 TPs across 5 runs). Either divergence-shorts genuinely don't work on this universe, or our extreme filter rejects them systematically. Phase C setups will give us more data to diagnose.

4. **Paper-trade graduation isn't even close yet.** The architecture doc requires 90 days paper + Sharpe ≥ 1.2 before live capital. We don't have signal that would survive paper-trading yet. **No live deployment until signal works.**

## File index for next session

Everything the next session needs to navigate:

- `tasks/shiller_bot_architecture.md` — the master spec (800 lines). Source of truth.
- `tasks/handoff_phase_b_to_c.md` — this file.
- `tasks/phase_b_validation_*.md` — five validation runs' reports.
- `validation/cbs_trades.yaml` — CBS documented trade log (seed = 2 trades).
- `markets/core_universe.yaml` — 47-ticker filtered universe (**needs rebuild** — step 1 above).
- `markets/krillin_watchlist.yaml` — 20-ticker hand-curated reference.
- `memory/feedback_commit_cadence.md` — the commit-per-step rule lives here.

## Operational hygiene

- **Commit after each step** is the durable preference (see `memory/feedback_commit_cadence.md`).
- **Python 3.11 via venv** at `./venv/` — all commands use `./venv/bin/python`.
- **Tests before committing**: `./venv/bin/python -m pytest -q` runs the full 321-test suite in ~70s.
- **Background long-running work**: use Bash with `run_in_background=true` so the chat isn't blocked. Poll via `tail` on the log file.
- **No live trading code paths have been exercised**. Paper-trading is the next real venue once signal validates.

## Git state at handoff

- Branch: `main`
- Commits ahead of origin: 39
- Working tree clean after this commit
- All 321 tests passing

Last commit before this handoff: `b8d09e3` (wick-based stop placement + validation artifacts).

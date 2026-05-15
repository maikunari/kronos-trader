# Kelly diagnostic: many-small-edges vs CBS-replication

**Date:** 2026-04-25
**Purpose:** decide whether to pivot from the CBS-replication thesis
(Track 2: Phase C setup detectors) to the many-small-edges philosophy
(Track 1: Phase 1 sniper), based on **per-trade log-growth rate G**
under Kelly-optimal sizing.

The single decisive metric: **G > 0 means capital grows in the long
run, G ≤ 0 means it does not.** Recall, precision, capture, Sharpe,
win rate — all interesting, all dominated by G as the survival
criterion.

## Phase C (CBS-replication) — verdict: **negative edge across the board**

From [`phase_b_validation_v6.md`](phase_b_validation_v6.md), 8-ticker
probe (HYPE TAO ENA NEAR LINK AVAX ZRO WLD), 1h, 1y, every trigger
simulated (TP + FP, not just matched-to-pop):

| setup | dir | n | win% | payoff | f* | G | G(f=1) |
|---|---|---|---|---|---|---|---|
| divergence | long  |  9 | 11.1% | 1.67 | 0 | 0 | -2.19% |
| divergence | short |  9 | 22.2% | 0.43 | 0 | 0 | -1.93% |
| breakout   | long  |  3 |  0.0% | 0.00 | 0 | 0 | -3.22% |
| v_reversal | long  | 21 | 23.8% | 1.40 | 0 | 0 | -2.38% |
| v_reversal | short | 32 | 18.8% | 1.28 | 0 | 0 | -3.51% |

Every detector: **Kelly fraction f* = 0**, meaning the math says don't
trade. Full-Kelly sizing (f=1) gives negative growth across all five
(detector × direction) cells. The +3.41% "mean realized return" we
had been celebrating for v_reversal was computed on 2 cherry-picked
TPs out of 59 triggers — the other 57 false positives were paying
stop-loss costs we weren't counting.

This **does not refute the CBS thesis** — it refutes our *current
implementation* of it. Possible reasons:
- CBS's edge isn't in his pattern recognition (our detectors); it's
  in his discretionary risk management, market timing, or pure luck.
- We're missing structural filters CBS uses (HTF trend alignment,
  capitulation gates, regime context).
- 2 documented trades is too few to calibrate against.
- The pop labeler (20% in 72h) doesn't correspond to CBS-trade
  payoffs.

Either way: **none of our setup detectors should be traded.**

## Phase 1 sniper (many-small-edges) — verdict: **negative edge, but close**

From [`sniper_kelly_diagnostic.md`](sniper_kelly_diagnostic.md), BTC /
ETH / SOL (HYPE: no data returned), 15m, same window:

| ticker | trades | win% | payoff | f* | G(f=1) | mean_ret | Sharpe | total_ret | max_dd |
|---|---|---|---|---|---|---|---|---|---|
| BTC  | 821 | 29.8% | 1.21 | 0 | -0.21% | -0.21% | -9.19 | -53.6% | 53.8% |
| ETH  | 795 | 34.0% | 1.47 | 0 | -0.16% | -0.15% | -4.06 | -33.6% | 38.7% |
| SOL  | 777 | 31.9% | 1.39 | 0 | -0.25% | -0.25% | -5.75 | -42.6% | 43.6% |
| HYPE |   0 |   —   |  —   | — |   —    |   —    |   —   |   —    |   —    |

**Pooled (2393 trades): win=31.9%, payoff=1.39, G(f=1)=-0.21%/trade,
f*=0.** Same verdict as Phase C — no positive Kelly edge — but the
shortfall is an order of magnitude smaller.

## Side-by-side comparison

| metric | Phase C (CBS) avg | Sniper (many-small) |
|---|---|---|
| trades / probe | ~12 per direction | ~800 per ticker |
| win rate | ~17.5% | 31.9% |
| payoff ratio | 1.0–1.7 | 1.39 |
| **mean return / trade** | **~-2.5%** | **-0.20%** |
| **G(f=1)** | **-2% to -4%** | **-0.21%** |
| distance-to-positive | wide | narrow |

The sniper is failing by ~20bps per trade. Round-trip fees alone are
~70bps (35bps × 2 sides) plus ~20bps slippage. **The sniper's signal
has roughly enough raw alpha to cover ~90% of its cost structure.**
A modest improvement on either side could push G > 0. The Phase C
setups are not even close — they're losing 12× as much per trade as
the sniper.

## Decision: drop CBS-replication, focus the sniper

The math is unambiguous:

1. **Phase C is a dead end as currently parameterized.** -2.5% per
   trade with low volume means slow burn + statistical
   indistinguishability from "no signal at all". The recall/precision/
   capture lens was hiding this; the Kelly lens makes it impossible
   to miss.

2. **The sniper is the better bet.** Volume (2393 trades), close to
   break-even (-0.20% / trade), and the gap is roughly the cost
   structure — not a fundamental signal problem. If we can:
     - Tune sniper params via `optimizer.py` (built-in walk-forward
       grid search) to lift win rate to ~35-40% OR improve payoff
       to ~1.8, **G crosses zero**.
     - Lower the cost model assumptions to match realistic HL fees
       and slippage (currently uses worst-case 35bps + ATR-frac).
     - Add a regime gate that suppresses signals in the highest-cost
       chop periods.

3. **Repurpose the agent loop.** `setup_agent.py` was targeting Phase
   C detectors that don't work. The same infrastructure pointed at
   the sniper's parameter space — generating diverse sniper variants,
   each evaluated with G as the gate — is a more productive use of
   API spend.

## Concrete next moves (in order)

### Move 1 — `optimizer.py` walk-forward sweep on the sniper (~30 min)

Already exists. Run it across a grid of sniper params on BTC/ETH/SOL
with G as the ranking metric (not Sharpe). Look for the cell with
positive OOS G. The optimizer was built for this exact purpose;
nobody's run it through a Kelly lens yet.

### Move 2 — cost-model audit (~1 hour)

The backtester uses `fee_rate=0.00035, slippage_base_bps=2.0,
slippage_atr_frac=0.05`. HL's actual maker fees are 0.01-0.02% (1-2bps),
taker 0.035% (3.5bps). If most fills are maker via the post-only-first
ExecutionPolicy, the effective cost is much lower than 35bps × 2. A
realistic fee model could shift G into positive territory without any
signal change.

### Move 3 — repoint the agent loop (~half day)

Modify `setup_agent.py` → `strategy_agent.py` to generate sniper-style
strategies (the contract already exists in `strategy_template.md`).
Evaluate each variant with G as the gate. The setup-agent infrastructure
moves to maintenance — kept in case we re-attempt CBS-replication later.

### What NOT to do

- Don't tune Phase C detectors further. The Kelly diagnostic shows
  the patterns themselves don't have edge; tweaking parameters won't
  fix that.
- Don't run the live `setup_agent.py` API spend. Save that for after
  the optimizer + cost-model work.
- Don't abandon the architecture. The data plumbing, validation
  framework, agent infrastructure all generalize cleanly to the
  many-small-edges philosophy.

## Decision branches

When the sniper diagnostic lands, the path forward is dictated by the
data:

### Branch A — sniper G > 0 on one or more tickers

The many-small-edges philosophy works on this universe. The math says
pivot to:
- Drop the CBS-replication research track (or park it for a v2 once
  we have 50+ CBS trades to calibrate against).
- Productionize the sniper as the primary strategy.
- Paper-trade for 90 days per architecture §9.5 paper-graduation rule.
- Repurpose the `setup_agent.py` infrastructure to generate diverse
  sniper-style strategies (single-symbol, fixed-stop, fixed-target
  contract), evaluating each with G as the gate.

### Branch B — sniper G ≤ 0 too

Neither track has positive edge as currently parameterized. Possible
moves:
- Tune sniper params (`optimizer.py` walk-forward grid search) to
  find a config with positive G.
- Try a wider universe / different timeframes.
- Question whether crypto perps have any retail-accessible alpha
  at all in 2026.
- Accept that the project may not have a tradeable edge and stop.

### Branch C — sniper G mixed (positive on some tickers, negative on others)

Per-ticker selection becomes the key parameter. Build a router that
picks instruments with stable positive G and trades them. This is
also viable but requires more care to avoid overfitting to whichever
ticker happened to perform well in-sample.

## What landed in the codebase regardless of branch

- `validation/kelly.py` — reusable growth-rate diagnostic.
- `SetupDirectionStats.growth_rate` and `.has_positive_edge`.
- `sniper_kelly_diagnostic.py` — apples-to-apples Kelly across both
  tracks.
- Per-trade-stream honesty: false positives now count toward the
  growth-rate computation, exposing edge-vs-noise without
  cherry-picking.

Whichever branch we take next, **G is now the gate** for any future
strategy evaluation. The §9.5 graduation gate (recall/precision/
capture) was hiding negative-edge strategies behind impressive-looking
pop-detection metrics; we replace it with the Kelly criterion.

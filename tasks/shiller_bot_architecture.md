# Shiller-Inspired Mid-Cap Bot — Architecture

**Status:** planning. Not yet built.
**Author:** synthesized from CBS notes, RSI Masterclass fragments, chart examples, and design conversation (2026-04).
**Scope:** new engine built on top of the existing Phase 1 infrastructure. See §11 for what's reused vs new.

---

## 1. Context

### 1.1 Why this pivot

Phase 1 built a classical Donchian/Keltner breakout sniper for BTC/ETH perps on Hyperliquid at 15m. The infrastructure is solid (150+ tests, regime detection, walk-forward backtest, risk layer), but tuning confirmed what the plan hypothesized:

- BTC/ETH at 5-15m with pure taker fills is structurally disadvantaged. Cost drag alone eats 40-50% of starting capital before strategy skill is considered.
- Classical breakouts on the majors have no remaining retail-accessible edge after decades of market-making attention.
- Full-year 2024 BTC backtest at default params: −71% return, PF 0.49. Optimizer grid sweeps showed monotone improvement with tighter regime gates but no combination crossed break-even.

### 1.2 What we're building instead

A pattern-recognition multi-setup scanner modeled on the documented approach of **ColdBloodedShiller (CBS)** and **Krillin**, two traders whose live-observed P&L and trade posts provided the rule base. Targets:

- **Universe:** 21 Hyperliquid mid-cap perps from Krillin's public watchlist (ENA, TAO, TON, JTO, ONDO, PENDLE, NEAR, DOGE, STRK, GALA, W, REZ, SAGA, NOT, DYM, OMNI, MAV, POLYX, AR, ENS, TNSR)
- **Timeframes:** primary 1h, context 4h, scalp variant 15m
- **Entries:** six distinct setup types, each detectable from pure price + indicators
- **Exits:** structural S/R ladder for TPs, leg-based trailing stops
- **Capital discipline:** pyramiding (start small, add on confirmation)
- **Validation:** event-based signal validation (recall/precision/capture) before any P&L backtest

The goal is to codify human discretionary edge while removing the three weaknesses the operator self-identified: capitulating during 3-day consolidations, missing the 20-30% pops, and TPing too early.

---

## 2. System overview

### 2.1 Data flow

```
                   ┌──────────────────────────────────┐
                   │  Hyperliquid WS + REST (candles, │
                   │  funding, OI) + Binance fallback │
                   └───────────────┬──────────────────┘
                                   │
                                   ▼
             ┌────────────────────────────────────────────┐
             │ Market data cache (21 tickers × {1h,4h,15m})│
             └─────────┬─────────────────────────┬────────┘
                       │                         │
            ┌──────────▼────────────┐  ┌─────────▼─────────┐
            │ S/R module            │  │ Narrative heat    │
            │ (pivots, zones, flips)│  │ (vol / OI / funding)│
            └──────────┬────────────┘  └─────────┬─────────┘
                       │                         │
                       ▼                         ▼
         ┌─────────────────────────────────────────┐
         │ Setup Detectors (6 modules, run/ticker) │
         │  1 Divergence reversal                  │
         │  2 Two-bar momentum continuation        │
         │  3 Diagonal breakout from consolidation │
         │  4 V-reversal at extended move          │
         │  5 Consolidation-under-resistance (pre) │
         │  6 Weak-bounce-from-support (pre)       │
         └──────────────────┬──────────────────────┘
                            │   (one or more triggers)
                            ▼
                  ┌────────────────────┐
                  │ Scanner / ranker   │
                  │ (per-bar best set) │
                  └─────────┬──────────┘
                            ▼
                  ┌────────────────────┐
                  │ Risk manager       │
                  │ (pyramid-aware)    │
                  └─────────┬──────────┘
                            ▼
                  ┌────────────────────┐
                  │ Executor + policy  │
                  │ (post-only first)  │
                  └─────────┬──────────┘
                            ▼
                  ┌────────────────────┐
                  │ Position manager   │
                  │ (legs, trails, TPs)│
                  └────────────────────┘
```

### 2.2 Cadence

- **Bar close (primary loop):** on each 1h close, re-run all setups × all tickers, rank new triggers, act.
- **Sub-cycle (position management):** on every 15m tick or 1h close, update trails, check TP fills, check stops.
- **Hourly:** recompute S/R zones per ticker (cached).
- **Weekly:** LLM-assisted S/R curation pass (offline). Operator reviews watchlist and updates.

---

## 3. Universe and market onboarding

### 3.1 Current universe

21 tickers from `markets/krillin_watchlist.yaml`. Rationale: two traders (CBS, Krillin) with different strategies but overlapping markets suggests selection-bias edge in the universe choice itself.

### 3.2 Filters

At load time:
- Must be currently listed on HL perps
- Minimum 30-day average daily volume of **$5M USD** (below this, our slippage model falls apart)
- Minimum **90 days of historical data** available on HL or Binance for warm-up

Tickers failing filters get quarantined but not deleted from the watchlist.

### 3.3 Onboarding new tickers

Manual, cadence weekly. Operator reviews paid-group content and posts, proposes additions, adds to `markets/krillin_watchlist.yaml` with dated commit. Git history tracks universe evolution.

### 3.4 Binance-only tickers

10 tickers on Krillin's list are not on HL (BONK, PEPE, AXL, BB, BEAM, CHZ, PDA, PORTAL, SSV, TOKEN). They are **used for signal research but not traded** — useful for validation data ("would our signal have fired on BONK?") but out of scope for live execution until/unless HL lists them.

---

## 4. Setup library

Six setups, each implemented as an isolated `SetupDetector` so they can be added/disabled independently. Each detector shares the same interface:

```python
class SetupDetector(Protocol):
    name: str

    def detect(self, ctx: MarketContext) -> Optional[Trigger]:
        ...
```

Where `Trigger` carries direction, entry price, stop price, initial TP ladder, setup type, and confidence.

### 4.1 Divergence reversal

**Source:** RSI Masterclass screenshots (bullish AO divergence, triple RSI divergence).
**When it fires:** swing pivot analysis shows AO and/or RSI making opposite extremes vs price at a point of price-extension.

```python
def detect(ctx):
    ao_pivots = recent_swing_pivots(ctx.ao, n=3)
    rsi_pivots = recent_swing_pivots(ctx.rsi, n=3)
    price_pivots = recent_swing_pivots(ctx.close, n=3)

    bullish = (
        price_pivots[-1].is_low
        and price_pivots[-1].value < price_pivots[-2].value       # LL price
        and (ao_pivots[-1].value > ao_pivots[-2].value            # HL AO
             or rsi_pivots[-1].value > rsi_pivots[-2].value)      # HL RSI (one suffices; both is ideal)
        and at_price_extreme(ctx)                                 # §4.1.1
        and two_bar_confirmation(ctx.ao, direction="bullish")     # §4.1.2
    )
    # symmetric bearish case

    if bullish: return Trigger(
        direction="long",
        entry=ctx.close,
        stop=price_pivots[-1].value * (1 - 0.003),   # just below divergent low
        tp_ladder=build_sr_ladder(ctx, "long"),      # §6.6
        setup="divergence_reversal",
        confidence=double_div_bonus(ao, rsi),
    )
```

**4.1.1 "At price extreme" filter.** Require one of: `close` outside 2σ Bollinger band (20-period); price > 2 ATR from 50-period SMA; price > 10% from 20-period close-mean. Any suffices; all three is high-conviction. Tuned during validation.

**4.1.2 Two-bar AO confirmation.** After divergence detected, wait for **two consecutive same-color AO bars** in the reversal direction before firing. Simple state machine:
```
DETECTED → WAITING_BAR_1 → WAITING_BAR_2 → FIRE
```
Invalidates if AO color flips back or price breaks the divergent extreme.

**Triple divergence boost.** Three-pivot divergence (instead of two-pivot) multiplies confidence by 1.5x and increases initial position fraction.

**Typical profile:** 1-3 day hold, 5-20R realized on runners, trigger rate 1-5/month per ticker.

### 4.2 Two-bar momentum continuation

**Source:** RSI Masterclass note ("two bars of a certain colour are very hard to reverse from").
**When it fires:** AO prints two consecutive same-color bars while HTF trend agrees and RSI is not in extremes.

```python
def detect(ctx):
    last_two_ao = ctx.ao.tail(2)
    all_green = all(last_two_ao > 0 and last_two_ao.diff() > 0)
    if all_green and htf_trend(ctx, "1h") == "up" and 35 < ctx.rsi.iloc[-1] < 65:
        return Trigger(
            direction="long",
            entry=ctx.close,
            stop=current_leg_bottom(ctx) * 0.998,    # just below leg low
            tp_ladder=build_sr_ladder(ctx, "long"),
            setup="two_bar_momentum",
            confidence=0.6,
        )
```

**Profile:** faster-moving than divergence; hours to 1 day; 2-5R targets.

### 4.3 Diagonal breakout from consolidation

**Source:** CBS "overeagerness" notes ("take diagonal breakouts that look like this — consolidation into a breakout, long the next leg").
**When it fires:** after a tight consolidation (low realized volatility, narrow range), price closes decisively above the consolidation high with HTF trend aligned.

```python
def detect(ctx):
    recent = ctx.bars.tail(20)
    consolidation_range = recent.high.max() - recent.low.min()
    if consolidation_range / recent.close.mean() < 0.03:  # < 3% range
        last_close = ctx.close
        breakout_level = recent.high.max()
        if last_close > breakout_level and htf_trend(ctx, "4h") == "up":
            return Trigger(
                direction="long",
                entry=last_close,
                stop=recent.low.min() * 0.998,
                tp_ladder=build_sr_ladder(ctx, "long"),
                setup="diagonal_breakout",
                confidence=0.7,
            )
```

**Profile:** hours to 2 days; 3-10R.

### 4.4 V-reversal at extended move

**Source:** S&P 500 4h chart (April 2026), PEPE-style mean-reversion setups.
**When it fires:** after a sustained move (10%+ decline or 15%+ rally over 5+ days), price shows capitulation followed by V-shaped reversal structure.

```python
def detect(ctx):
    move_size = extended_move_magnitude(ctx, lookback_bars=120)
    if move_size < 0.10:     # needs 10%+ prior adverse move
        return None
    capitulation = is_capitulation(ctx)    # largest-N-day volume + wide-range red candle
    if not capitulation:
        return None
    reversal = confirmed_v_structure(ctx)  # HL after initial bounce
    if reversal:
        return Trigger(
            direction="long",
            entry=ctx.close,
            stop=reversal.v_low * 0.998,   # just below the V
            tp_ladder=build_sr_ladder(ctx, "long"),
            setup="v_reversal",
            confidence=0.8,
        )
```

**Profile:** 2-5 day hold; 10-20R+ — the asymmetric-RR workhorse.

### 4.5 Consolidation-under-resistance (leading)

**Source:** CBS S/R notes ("tight consolidation under resistance helps us anticipate a breakout").
**When it fires:** price spends N bars within X% below a valid S/R resistance zone without meaningful drop-away. Leading signal — fires **before** the breakout.

```python
def detect(ctx):
    for zone in ctx.sr_zones:
        if zone.side != "resistance" or zone.strength < 0.6:
            continue
        if tight_consolidation_under(ctx.bars.tail(10), zone,
                                      max_distance_pct=0.02,
                                      max_drawdown_pct=0.03):
            return Trigger(
                direction="long",
                entry=ctx.close,
                stop=ctx.bars.tail(10).low.min() * 0.998,
                tp_ladder=build_sr_ladder(ctx, "long", skip_first=zone.price),
                setup="consolidation_under_resistance",
                confidence=0.55,        # leading = lower initial confidence
                initial_fraction=0.3,   # smaller first entry; add on breakout
            )
```

First entry at 30% of target position during consolidation; `add_on` signal fires on close above `zone.price` for remaining size.

**Profile:** entry in consolidation → breakout → runner; 5-15R if the breakout lands.

### 4.6 Weak-bounce-from-support (leading)

**Source:** CBS S/R notes ("weak bounces from support suggest a coming breakdown").
**When it fires:** recent bounces off a support zone are materially weaker than historical bounces at the same level.

```python
def detect(ctx):
    for zone in ctx.sr_zones:
        if zone.side != "support" or zone.touches < 3:
            continue
        bounce_heights = [bounce_height_after(t, ctx) for t in zone.touch_list]
        if len(bounce_heights) >= 3 and bounce_heights[-1] < 0.5 * np.median(bounce_heights[:-1]):
            return Trigger(
                direction="short",
                entry=ctx.close,
                stop=bounce_heights[-1].bounce_high * 1.002,
                tp_ladder=build_sr_ladder(ctx, "short", skip_first=zone.price),
                setup="weak_bounce",
                confidence=0.55,
                initial_fraction=0.3,
            )
```

Symmetric structure to §4.5 but for shorts. First entry small; add on close below support.

**Profile:** 3-10R.

---

## 5. Support / Resistance module

### 5.1 Detection algorithm

Runs per-ticker, cached per hour. Key design decisions follow CBS rules:

1. **Close-based, not wick-based.** Pivots detected on the `close` series (effectively a "line chart" view). Wick noise filtered out.
2. **Swing fractal detection:** pivot = a close that's higher (or lower) than all closes in a window (default 5 bars each side).
3. **Cluster merging:** pivots within a small band (default 0.5% for liquid, 1.5% for thin tickers) merge into one zone.
4. **Touch scoring follows CBS:** 3 touches is the sweet spot; more touches *decrease* strength (order flow exhaustion).

```python
def touch_strength(n_touches: int) -> float:
    if n_touches < 2: return 0.0
    if n_touches == 2: return 0.6
    if n_touches == 3: return 1.0                       # CBS ideal
    return max(0.3, 1.0 - (n_touches - 3) * 0.15)       # decay after 3
```

5. **Age decay:** zones older than N bars get their strength scaled down proportionally (default: linear decay after 500 bars, zero after 2000).
6. **Volume weighting:** touches with above-median volume score higher.

### 5.2 Flip detection

A zone flips role (resistance → support, or vice versa) after:

1. **Decisive break:** 2+ consecutive closes on the other side (closes only per CBS).
2. **Consolidation confirmation:** held on the new side for ≥ `min_hold_bars` (default 5).
3. **Retest:** price returned within `retest_tolerance_pct` of the level (default 1%).
4. **Respect:** didn't close through on retest.

Flipped zones are tagged with `role: "flipped_support"` / `"flipped_resistance"` and become high-conviction entry zones for continuation trades.

### 5.3 Breakout confirmation

Throughout the system: breakouts are confirmed on **close**, not high/low. A wick through a level is not a breakout. Single close above is provisional; second close above is confirmation.

### 5.4 S/R exports

Per ticker, exposes a list of `SRZone` dataclasses:
```
SRZone {
    price_mid, price_low, price_high,   # band
    side: "support" | "resistance" | "flipped_support" | "flipped_resistance",
    strength: 0..1,
    touch_count: int,
    last_touch_bar: int,
    avg_volume_at_touch: float,
    touch_list: list[TouchEvent],       # for weak-bounce analysis
}
```

### 5.5 LLM curation (offline, weekly)

Layer on top of algorithmic detection. Once per week:
- Render chart + algorithmic S/R zones for each ticker
- Send to LLM with prompt: "flag zones that appear stale, weakly supported, or likely to have flipped"
- Operator reviews flagged zones, applies accept/override
- Curated map saved as `cache/sr_curated_{ticker}.json`
- Runtime bot loads curated map if present, else falls back to auto

**Runtime behavior remains 100% deterministic** — the LLM only touches offline curation, never the live trade decision path.

---

## 6. Position management

### 6.1 Position model

Pyramid-aware (CBS's "attacking mindset"):

```python
@dataclass
class Position:
    ticker: str
    direction: str           # "long" | "short"
    legs: list[Leg]          # (entry_price, entry_time, size_usd, stop_at_entry)
    current_stop: float
    tp_ladder: list[TPLevel] # [(price, fraction, filled_bool)]
    avg_entry_price: float
    total_size_usd: float
    filled_tp_pnl: float
    setup_origin: str        # "divergence", "diagonal", etc.

@dataclass
class Leg:
    entry_price: float
    entry_time: pd.Timestamp
    size_usd: float
    stop_at_entry: float

@dataclass
class TPLevel:
    price: float
    fraction: float     # 0.33 → 33% of total_size_usd closes here
    filled: bool
    fill_price: float = 0.0
    fill_time: Optional[pd.Timestamp] = None
```

### 6.2 Entry logic

Triggers come in two flavors:
- `action="open_new"` → creates a new Position
- `action="add_on"` → appends a Leg to an existing Position

Each entry has its own `size_fraction` (of target total). Default pattern per setup:

| Setup | First entry | Add-on triggers | Final |
|-------|-------------|-----------------|-------|
| Divergence reversal | 40% | +40% on 2-bar confirm | +20% on first TP |
| Two-bar momentum | 60% | +40% on HTF confirm | — |
| Diagonal breakout | 50% | +50% on successful retest | — |
| V-reversal | 30% | +40% on higher-high confirm | +30% on break of recent high |
| Consolidation-under-res | 30% | +70% on close above resistance | — |
| Weak-bounce | 30% | +70% on close below support | — |

### 6.3 Stop management (leg-based trailing)

Per CBS: "move stop up to the bottom of current leg each time."

```python
def update_trailing_stop(position, new_bar):
    leg_bottom = current_leg_bottom(position.ticker_candles_since_entry)
    if position.direction == "long":
        candidate = leg_bottom * (1 - 0.002)   # small buffer
        position.current_stop = max(position.current_stop, candidate)  # monotone up
    else:
        candidate = leg_top * 1.002
        position.current_stop = min(position.current_stop, candidate)  # monotone down
```

A "leg" = the most recent impulse swing in the position-favorable direction. Detected via fractal analysis on closes since entry.

**Alternative: ChandelierTrail.** For setup types where leg-based is too aggressive (e.g., V-reversal early stage where legs aren't yet formed), fall back to `ChandelierTrail(direction, atr_mult=3.0)` from existing `atr_engine.py`. Configurable per setup.

### 6.4 Take-profit ladder

Per CBS: "look left for TP, target previous resistance, TP 2 or 3 times on the way up."

On entry, build `tp_ladder` from S/R zones in the direction of the trade:

```python
def build_sr_ladder(ctx, direction, max_targets=3):
    zones = [z for z in ctx.sr_zones if in_direction(z, direction)]
    zones = sorted(zones, key=distance_from_price)[:max_targets]
    if not zones:
        # Fallback: Fibonacci extensions (1.272, 1.618, 2.0) or ATR-based
        return atr_based_ladder(ctx)
    fractions = [0.33, 0.33, 0.34]
    return [TPLevel(price=z.price_mid, fraction=f, filled=False)
            for z, f in zip(zones, fractions)]
```

### 6.5 Runner logic

After all laddered TPs fill, the residual position is a "runner" — held with a tightened trailing stop (default: `ChandelierTrail(atr_mult=2.0)` — tighter than the 3.0 during active add-ons).

Runner exits when:
- Stop hit
- Manual intervention (operator flag)
- **Never** via time-based exit

### 6.6 Stop migration after TPs

Classic risk-management pattern:
- After first TP fills: move stop to break-even on remaining size
- After second TP fills: move stop to first-TP level (lock in partial profit on runner)
- Chandelier continues to ratchet from there

---

## 7. Risk management

### 7.1 Reuse existing `risk_manager.py`

The Phase 1 `RiskManager` already has:
- Volatility targeting
- Fractional Kelly overlay
- Daily loss limit
- Consecutive loss cool-off
- Weekly drawdown halving
- Manual halt + tripwire

### 7.2 Required changes

**Pyramid awareness:**
- `approve_trade` must handle `action="add_on"` distinctly from `action="open_new"`.
- Total risk across all legs of a single Position must not exceed a per-position cap (default: 3% of equity at initial stop distance).
- Maximum concurrent Positions across universe: 3-5 (vs Phase 1's 1).
- Correlation cap: positions on highly-correlated tickers (e.g., two AI-narrative tokens) share exposure budget.

**Setup-specific confidence scaling:**
- Position sizing receives `confidence` from the trigger. Higher confidence = larger target size, within the hard cap.
- Triple divergence: 1.5× normal.
- Leading setups (consolidation-under-res, weak-bounce): 0.7× normal until confirmation.

### 7.3 Per-ticker caps

No single ticker gets more than 40% of the concurrent-position budget. Prevents all-in on one narrative.

---

## 8. Scanner orchestrator

### 8.1 Loop

```
Every 1h bar close:
    for ticker in universe:
        ctx = build_market_context(ticker)   # candles + S/R + narrative_heat
        for detector in SETUP_DETECTORS:
            trigger = detector.detect(ctx)
            if trigger: candidates.append((ticker, trigger))

    ranked = rank_candidates(candidates)     # by confidence × narrative_heat
    for (ticker, trigger) in ranked:
        approval = risk_manager.approve_trade(trigger, ...)
        if approval.approved:
            executor.execute(trigger, approval)
            position_manager.open_or_add(ticker, trigger, approval)

Every tick (or at least every 15m):
    for position in open_positions:
        update_trailing_stop(position)
        check_tp_fills(position)
        check_stop_hit(position)
```

### 8.2 Ranking

When multiple triggers fire on the same bar, rank by:
1. `confidence × narrative_heat_z_score`
2. Tiebreaker: setup precedence (divergence > v-reversal > diagonal > two-bar > leading setups)
3. Tiebreaker: ticker precedence (one trade per ticker per bar)

### 8.3 Capital allocation

Available capital (per `risk_manager`) distributes across approved triggers proportionally to their `target_size × confidence`, with per-position and per-ticker caps enforced.

---

## 9. Signal validation framework

Critical step. Must pass before any live capital.

### 9.1 Ground-truth labeler

For each ticker's historical candles, identify "pops":

```python
def label_pops(candles, threshold_pct=0.20, window_hours=72):
    for i, bar in enumerate(candles):
        forward = candles.iloc[i : i + window_hours]
        if forward.high.max() / bar.close >= 1 + threshold_pct:
            yield POP(ts=bar.timestamp, direction="long", magnitude=..., time_to_peak=...)
        if forward.low.min() / bar.close <= 1 - threshold_pct:
            yield POP(ts=bar.timestamp, direction="short", magnitude=..., time_to_peak=...)
```

Tunable thresholds: 15%, 20%, 30%. Run all three to see sensitivity.

### 9.2 Trigger emitter

At every historical bar, for every ticker × every setup, run `detector.detect(ctx)` and log the full trigger stream.

### 9.3 Matcher

For each trigger, look forward up to `max_hold_hours` (default 72). Did a matching-direction pop occur? Classify:

| Outcome | Label |
|---------|-------|
| Pop + matching trigger within lookback | True positive (caught) |
| Pop + no matching trigger within lookback | False negative (missed) |
| Trigger + no pop in forward window | False positive (faded) |

### 9.4 Capture simulator

For true positives, simulate the single trade:
- Enter at trigger price
- Apply initial stop + TP ladder + trail
- Run forward until exit
- Compute realized P&L as fraction of pop magnitude

### 9.5 Output

Per setup × per direction:

```
DIVERGENCE_REVERSAL / LONG
  Pops labeled: 42
  Caught: 27 (64% recall)
  Triggers emitted: 103
  Led to pop: 27 (26% precision)
  Avg pop magnitude: 28%
  Avg capture ratio: 41% (realized 11.5% avg)
  Median lead time: 4.2 hrs before pop peak
  Exit breakdown: target=18, trail=6, stop=3
```

Thresholds for "pass to live":
- **Recall ≥ 40%** (miss less than 60% of pops)
- **Precision ≥ 25%** (at least 1 in 4 triggers leads to a pop)
- **Capture ≥ 30%** (bank at least a third of the move)

All three must clear for a setup to graduate to live.

### 9.6 Replication fidelity (if CBS trades are accessible)

If 3-6 months of CBS's posted entries are available: compute **entry overlap** between his trades and our bot's triggers on the same tickers/timeframes. Target: >60% replication for high-conviction setups.

---

## 10. Narrative heat

Per §2 of the build discussion. Free signal composite:

```python
narrative_heat = z_score(
    0.4 * volume_anomaly(ticker, current_bar)     # vs 30d baseline
  + 0.4 * oi_growth(ticker, lookback_3_bars)      # free from Binance for most
  + 0.2 * funding_extreme_bias(ticker, direction) # HL fundingHistory
)
```

Feeds into:
1. Trigger ranking (§8.2): tickers with high heat ranked first
2. Position sizing (§7.2): heat acts as a confidence multiplier
3. Validation (§9): check whether high-heat triggers outperform low-heat

LunarCrush Galaxy Score ($25/mo) added only if validation shows free signal is insufficient.

---

## 11. Relationship to existing Phase 1 code

### 11.1 Reused as-is

| Module | Role |
|--------|------|
| `hyperliquid_feed.py` | WS + REST + liquidation stream |
| `execution_policy.py` | Post-only first, taker fallback, iceberg |
| `executor.py` | HL order placement |
| `alerts.py` | Discord/Slack webhooks |
| `regime.py` | ADX + Hurst + RV (for optional regime-gate on setups) |
| `atr_engine.py` | ATR math + ChandelierTrail (for V-reversal early stage) |
| `derivatives_feed.py` | Funding rate history |
| Test scaffolding (`tests/`, `conftest.py`, fixtures) | Unchanged |

### 11.2 Needs extension

| Module | Change |
|--------|--------|
| `risk_manager.py` | Add `action="add_on"` path; correlation cap; per-ticker cap |
| `main.py` | Replace single-symbol loop with multi-symbol scanner |

### 11.3 New modules

| Module | Purpose |
|--------|---------|
| `pivot.py` | Swing detection primitives (fractals, swing_high/low) |
| `indicators/awesome_oscillator.py` | AO calculation |
| `indicators/rsi.py` | RSI (wrap `ta` library) |
| `support_resistance.py` | §5 — zones, flips, touch scoring, exports |
| `sr_curation.py` | §5.5 — offline LLM curation tool |
| `setups/divergence.py` | §4.1 |
| `setups/two_bar_momentum.py` | §4.2 |
| `setups/diagonal_breakout.py` | §4.3 |
| `setups/v_reversal.py` | §4.4 |
| `setups/consolidation_under_resistance.py` | §4.5 |
| `setups/weak_bounce.py` | §4.6 |
| `position.py` | §6 — pyramid-aware Position model |
| `narrative_heat.py` | §10 |
| `scanner.py` | §8 — multi-symbol orchestrator |
| `validation/labeler.py` | §9.1 |
| `validation/matcher.py` | §9.3 |
| `validation/capture.py` | §9.4 |
| `validation/report.py` | §9.5 |

### 11.4 Retired

| Module | Fate |
|--------|------|
| `snipe_signal_engine.py` | Superseded by six-setup library |
| `backtest.py` | Rewritten as multi-symbol event-based validation + (later) portfolio backtest |
| `optimizer.py` | Rewritten to tune per-setup parameters against validation metrics |
| `run_periods.py` | Still useful; update to take new scanner |

---

## 12. Build sequence

Phased. Each phase ends with a shippable, testable deliverable.

### Phase A — Primitives (1 week)
- `pivot.py` with tests
- `indicators/awesome_oscillator.py`, `indicators/rsi.py` with tests
- `support_resistance.py` core detection (no flips yet) with tests
- Data caching layer for 21 tickers (1h + 4h + 15m)

**Deliverable:** command-line tool that outputs S/R zones for a ticker as of a given timestamp.

### Phase B — First setup + validation framework (1 week)
- `setups/divergence.py` (most documented, highest-conviction)
- `validation/labeler.py`, `matcher.py`, `capture.py`, `report.py`
- Run first validation report on 12 months of 21 tickers

**Deliverable:** validation report PDF/markdown with recall/precision/capture per direction for divergence setup only. This tells us if the approach has any signal at all before building more.

### Phase C — Remaining setups (2 weeks, one per ~2 days)
- Two-bar momentum
- Diagonal breakout
- V-reversal
- Consolidation-under-resistance
- Weak-bounce
- Re-run validation after each setup; iterate parameters

**Deliverable:** validation report with per-setup metrics; keep setups that pass §9.5 thresholds.

### Phase D — Position management (1 week)
- `position.py` with pyramiding
- TP ladder builder
- Leg-based trailing stop
- S/R flip detection
- Narrative heat module

**Deliverable:** extend validation capture sim with full pyramid/ladder/trail logic; compare realized capture ratios with vs without.

### Phase E — Scanner + live paper (1 week)
- `scanner.py` orchestrator
- `risk_manager.py` pyramid extensions
- `main.py` rewrite for multi-symbol scanner
- Paper trading against live data for 30 days

**Deliverable:** paper-running bot on 21 tickers; dashboard shows per-ticker per-setup state.

### Phase F — Graduation criteria (30+ days paper)
- Minimum 90 calendar days paper
- Paper Sharpe ≥ 1.2
- Every kill switch tripped at least once (validated)
- Live fill quality within 20% of validation simulation
- Only then: 10% live sizing for 30 days, step up if matches paper

**Total: ~6-7 weeks of development + 90+ days of paper validation before real capital.**

---

## 13. Open questions

Items we explicitly don't know and need to resolve during build:

1. **Exact CBS RSI thresholds.** Guess: 70/30 for overbought/oversold, but could be 80/20. Resolved during parameter tuning with validation.
2. **Exact "extreme of price action" criterion.** Three candidates (Bollinger, SMA+ATR, %-from-close-mean). Try all, pick best via validation.
3. **Scalp vs swing threshold.** How does CBS decide which to take? Likely context-dependent (recent volatility, HTF trend strength). Initial guess: scalps on 15m when 4h strongly trending; swings on 1h when HTF breakout. Iterate.
4. **Short hit-rate vs long.** Mid-caps in bull regimes have long bias. Likely short setups have lower recall but higher capture-per-fire. Validation report will show directly.
5. **Pyramid fraction per setup.** Numbers in §6.2 are initial guesses. Tune against realized R on validation data.
6. **Binance-only tickers.** Do we include them for signal research only, or find a Binance execution path? Defer to Phase E.
7. **LLM curation ROI.** Does the weekly LLM review measurably improve S/R detection accuracy over pure algorithmic? Measurable after 4-6 weeks.
8. **Correlation cap specifics.** Need to decide correlation window (30d? 90d?) and threshold (>0.7?). Standard retail quant numbers; validate empirically.

---

## 14. Glossary

| Term | Meaning |
|------|---------|
| **AO** | Awesome Oscillator: 5-period SMA minus 34-period SMA of `(high+low)/2` |
| **CBS** | ColdBloodedShiller, source trader for rule base |
| **Divergence** | Price and indicator moving in opposite directions at pivots |
| **Flip** | S/R level that changes role (resistance becomes support or vice versa) |
| **Leg** | Most recent impulse move in favorable direction within an open position |
| **HTF** | Higher timeframe (relative to current bar's TF) |
| **Ladder** | Structured sequence of partial TPs at distinct S/R levels |
| **Look-left** | CBS's phrase for using historical S/R as TP targets |
| **Narrative heat** | Composite of volume anomaly + OI growth + funding extreme |
| **Pop** | Significant directional move (default: 20%+ over 72h) used for validation labels |
| **Pyramid** | Entering full position in multiple staged entries |
| **R:R** | Risk-to-reward ratio |
| **Runner** | Residual position after all laddered TPs filled |
| **SRZone** | Detected support/resistance zone with role, strength, touches |
| **Two-bar rule** | CBS: two consecutive same-color AO bars confirm momentum direction |
| **V-reversal** | Reversal of an extended move via V-shaped price structure |

---

## 15. Appendix: why this approach, briefly

Three things justify the scope:

1. **Every rule is codifiable.** The six setups are pattern-matching on price + indicators + S/R. No subjective judgment in the hot path (LLM curation only in offline tooling).

2. **Validation-first order of operations.** Before any P&L backtest, we measure whether the pattern detectors fire on real pops. Most trading-bot failures happen because the signal never had statistical edge and no one checked. We check first.

3. **Replicates documented trader edge.** Operator has months of observed CBS trades — this is not "build a new strategy and hope." This is "replicate a proven-working discretionary trader with systematic discipline the human lacks (3-day patience, no-premature-TP, disciplined pyramid entry)."

The hardest parts to implement (pivot detection, S/R clustering, pyramid-aware position management) are well-studied problems with standard solutions. The novel parts (combining into the specific CBS ruleset) are where the edge lives.

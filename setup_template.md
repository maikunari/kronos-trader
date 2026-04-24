# Kronos Generated Setup Detector Contract

Generated setup detectors plug into the Phase C framework alongside the
hand-written baselines (`DivergenceReversalDetector`,
`ConsolidationBreakoutDetector`, `VReversalDetector`). The agent's AST
validator will reject modules that import forbidden names or call unsafe
builtins. The runner will reject modules that raise, time out, or fail
the graduation gate on the multi-ticker probe.

## File layout

- One Python file per setup.
- Top-level docstring describing the pattern in 1-3 sentences.
- Exactly one class named `Setup` with a `.detect(ctx)` method.
- A class attribute `.name` (short string, used as the `setup` label on
  emitted `Trigger` objects and in validation reports).

## Allowed imports

Stdlib and numeric scaffolding:

- `pandas`, `numpy`, `math`, `typing`, `dataclasses`, `collections`,
  `enum`, `functools`, `itertools`, `statistics`, `abc`, `numbers`

Kronos Phase C modules (the generated detector is explicitly allowed to
reuse these utilities — prefer this to reinventing AO/RSI/pivot/S-R logic):

- `setups.base` — `Direction`, `MarketContext`, `Trigger`, `TPLevel`,
  `build_tp_ladder`, `SetupDetector`
- `support_resistance` — `SRZone`, `detect_sr_zones`, `confirmed_breakout`
- `pivot` — `Pivot`, `PivotKind`, `find_pivots`
- `indicators.awesome_oscillator` — `awesome_oscillator`,
  `two_bar_same_color`
- `indicators.rsi` — `rsi`

Forbidden: `os`, `sys`, `subprocess`, `socket`, `http`, `urllib`,
`requests`, `aiohttp`, `websocket`, `hyperliquid`, `snipe_signal_engine`,
`backtest`, `executor`, `risk_manager`, `open()`, `exec()`, `eval()`,
`compile()`, `__import__`, relative imports.

The detector has no file system, no network, no access to the live
runtime.

## Class contract

```python
from typing import Optional
from setups.base import MarketContext, Trigger, build_tp_ladder

class Setup:
    """One-sentence description of the pattern."""

    name: str = "your_setup_name"   # copied onto the Trigger.setup label

    # All tunable parameters MUST be keyword-only with defaults.
    def __init__(
        self,
        *,
        lookback_bars: int = 50,
        # ... any other tunables ...
    ):
        self.lookback_bars = lookback_bars
        # ...

    def detect(self, ctx: MarketContext) -> Optional[Trigger]:
        """
        Evaluate the setup on the current bar.

        `ctx` is a MarketContext with:
          * ctx.candles — pd.DataFrame with [timestamp, open, high, low,
            close, volume], sorted oldest -> newest. Current bar is
            ctx.candles.iloc[-1].
          * ctx.ao, ctx.rsi — precomputed pd.Series (or None if the
            window is too short)
          * ctx.sr_zones — list of SRZone objects or None
          * ctx.current_price, ctx.current_bar_index, ctx.timestamp

        Return a Trigger (see setups/base.py) or None.
        """
        ...
```

## Building a Trigger

Use the constructor directly:

```python
return Trigger(
    ticker=ctx.ticker,
    timestamp=ctx.timestamp,
    action="open_new",
    direction=direction,                      # "long" or "short"
    entry_price=ctx.current_price,
    stop_price=stop,                          # wick-based, see below
    tp_ladder=build_tp_ladder(entry, direction, ctx.sr_zones, atr),
    setup=self.name,
    confidence=0.65,                          # 0.0 .. 1.0
    size_fraction=0.4,                        # fraction of target position
    components={"diagnostic": "info"},
)
```

## Locked design rules (CBS-calibrated)

These are the rules the three baseline detectors all follow. Deviate
only with good reason — the graduation gate will catch mistakes.

1. **Wick-based stop placement.** Longs: `stop = pivot_low * (1 - buffer)`,
   `buffer ≈ 0.003`. Shorts: `stop = pivot_high * (1 + buffer)`. CBS
   places stops beyond the WICK, not the close, because exchanges fire
   stops on wicks.

2. **Close-based breakout confirmation.** A wick through a level isn't a
   breakout — only a close is. `support_resistance.confirmed_breakout`
   already encodes this; prefer it to rolling your own check.

3. **10% min-target-distance filter.** After building the TP ladder,
   reject the trigger if no TP is at least 10% from entry. CBS's "look
   left for meaningful resistance" rule. All three baselines apply it.

4. **Binary outcomes.** Every trade exits on stop or full TP ladder fill
   — no timeout close. The validation framework enforces this; just
   structure your TP ladder with enough runway.

## Hard rules (AST validator rejects)

1. Exactly one top-level class, named `Setup`.
2. `__init__` accepts only keyword-only arguments with default values
   (use `def __init__(self, *, param: int = 5): ...`).
3. No `while True` without a bounded counter; no recursion.
4. No file I/O, no top-level `print(...)`, no global mutable state.
5. `detect(ctx)` must return in under 250 ms on a 500-candle context.

## Out-of-sample gate (architecture §9.5 graduation)

On the 8-ticker probe (`HYPE TAO ENA NEAR LINK AVAX ZRO WLD`, 1h, 1y),
a generated detector is accepted only if, for at least one direction:

- `recall >= 0.40` — catches 40%+ of labeled 20% pops
- `precision >= 0.25` — 25%+ of triggers are real
- `median_capture_ratio >= 0.30` — median captured 30%+ of pop magnitude

Failures are kept on disk but flagged `rejected` in the run log.

## Baselines to beat

Current best (as of 2026-04-20): `VReversalDetector` with defaults from
commit `c97ebc3` on the 8-ticker probe:

- **long:** 59 triggers, 2 TP, 8.3% recall, 3.4% precision,
  16.9% capture, +3.41% mean realized return
- **short:** 52 triggers, 0 TP, 0% recall (same blind spot across all
  baselines)

Other baselines:

- `DivergenceReversalDetector`: 12 long triggers, 4.2% recall, 8.3%
  precision, -10.3% capture (loses money)
- `ConsolidationBreakoutDetector`: 3 long triggers, 0 TP

None hit the graduation gate. **Your job is to find setups that clear it.**

## Good directions to explore

- **§4.2 two-bar momentum continuation** — after two consecutive same-colour
  AO bars with price above/below the 21-EMA.
- **§4.5 consolidation-under-resistance** (leading) — price spends N bars
  within X% below a strong resistance zone without meaningful drop-away.
  Fires BEFORE the breakout.
- **§4.6 weak-bounce-from-support** (leading, shorts) — recent bounces
  off a support zone are materially weaker than historical ones.
- **Hybrid gates** — divergence triggers that ALSO require a prior
  consolidation context; v_reversal triggers gated by capitulation
  volume.
- **Short-side specifically** — all three baselines have 0% short recall.
  A short-only detector that actually works would be novel value.

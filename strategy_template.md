# Kronos Generated Strategy Contract

Generated strategies must conform exactly to this contract. The agent's AST
validator will reject modules that import forbidden names or call unsafe
builtins. The agent's subprocess runner will reject modules that raise, time
out, or return malformed signals.

## File layout

- One Python file per strategy.
- Top-level docstring describing the idea in 1-3 sentences.
- Exactly one class named `Strategy` with a `.evaluate(candles)` method.

## Allowed imports

Only these modules may be imported, at the top of the file:

- `pandas`
- `numpy`
- `math`
- `typing`
- `dataclasses`
- `collections`

Anything else (`os`, `sys`, `subprocess`, `socket`, `requests`, `open`,
`__import__`, `exec`, `eval`, `compile`, relative imports, trading SDKs,
other Kronos modules) is rejected. The strategy has no file system, no
network, no access to the live runtime.

## Class contract

```python
class Strategy:
    """One-sentence description of the idea."""

    # All tunable parameters MUST be declared as constructor arguments
    # with default values. The agent uses these defaults to run the
    # first backtest; future iterations may suggest different defaults.
    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        atr_period: int = 14,
        atr_stop_mult: float = 1.5,
        atr_target_mult: float = 3.0,
        # ... any other params your strategy needs
    ):
        self.ema_fast = ema_fast
        # ...

    def evaluate(self, candles: "pd.DataFrame") -> dict:
        """
        `candles` is a pandas DataFrame with columns
            [timestamp, open, high, low, close, volume]
        sorted oldest -> newest. The most recent candle is candles.iloc[-1].

        Must return a dict with exactly these keys:
            {
                "action": "long" | "short" | "flat",
                "stop_pct": float,     # fraction of entry price, 0.002..0.10
                "target_pct": float,   # fraction of entry price, 0.002..0.20
                "confidence": float,   # 0.0..1.0, informational only
                "skip_reason": str,    # required when action == "flat"
            }

        - `stop_pct` and `target_pct` MUST be strictly positive even when
          action == "flat" (use any valid defaults, e.g. 0.01 / 0.02).
        - The runner applies the stop/target as percentages of the actual
          entry price (next-bar open after your signal), so do NOT include
          entry_price in the output.
        - The runner enforces one position at a time. Do not model
          multi-leg trades.
        """
```

## Hard rules (validator will reject the variant)

1. Exactly one top-level class, named `Strategy`.
2. `__init__` accepts only keyword arguments with default values (no
   positional-only, no `*args`, no `**kwargs`).
3. No `while True` without a bounded counter; no recursion.
4. No file I/O, no `print(...)` at module top-level (inside methods is fine
   but discouraged), no global mutable state.
5. `evaluate` must return in under 250 ms on a 500-candle window.

## Soft guidelines (for scoring, not validation)

- Prefer explicit skip_reasons over silent flats. The agent critiques these.
- Parameters that are wildly out of range (e.g. `ema_fast=1000`) will be
  backtested anyway; the critic will call them out.
- Diversity across variants is rewarded. Two near-identical strategies
  count as one.
- The current hand-written baseline is `TrendSignalEngine` in
  `trend_signal_engine.py` — your strategies should beat it on the
  out-of-sample window, not merely match it.

## Out-of-sample gate

After backtesting a variant on both the in-sample and out-of-sample
windows, the runner accepts it only if:

- `out_of_sample.profit_factor >= 1.0`
- `out_of_sample.max_drawdown_pct >= -0.30`  (i.e. drawdown <= 30%)
- `out_of_sample.total_trades >= 5`

Variants failing any of these are kept on disk but flagged `rejected` in
the run log and excluded from the final report shortlist.

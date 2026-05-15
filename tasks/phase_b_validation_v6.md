# Phase B signal validation

- Timeframe: **1h**
- Window: 2025-04-01 00:00:00+00:00 → 2026-04-01 00:00:00+00:00
- Pop threshold: 20% in 72.0h
- Detectors: divergence_reversal, consolidation_breakout, v_reversal
- Tickers: 8

**Totals: 36 pops labeled, 141 triggers emitted**

## Results per setup × direction

| setup | dir | pops | trigs | TP | FN | FP | recall | prec | cap% | mean_ret% | lead_h | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| divergence_reversal | long | 24 | 12 | 1 | 23 | 11 | 4.2% | 8.3% | -10.3% | -2.14% | 15.0 | ✗ |
| divergence_reversal | short | 12 | 14 | 0 | 12 | 14 | 0.0% | 0.0% | 0.0% | +0.00% | — | ✗ |
| consolidation_breakout | long | 24 | 3 | 0 | 24 | 3 | 0.0% | 0.0% | 0.0% | +0.00% | — | ✗ |
| consolidation_breakout | short | 12 | 1 | 0 | 12 | 1 | 0.0% | 0.0% | 0.0% | +0.00% | — | ✗ |
| v_reversal | long | 24 | 59 | 2 | 22 | 57 | 8.3% | 3.4% | 16.9% | +3.41% | 24.0 | ✗ |
| v_reversal | short | 12 | 52 | 0 | 12 | 52 | 0.0% | 0.0% | 0.0% | +0.00% | — | ✗ |

## Pops per ticker

| ticker | pops |
|---|---|
| LINK | 13 |
| AVAX | 11 |
| TAO | 5 |
| HYPE | 3 |
| ENA | 2 |
| NEAR | 2 |
| ZRO | 0 |
| WLD | 0 |

## Kelly diagnostics (all triggers, TP+FP)

G = per-trade log-growth rate at the Kelly-optimal sizing f*. G > 0 → capital grows in the long run regardless of recall.

| setup | dir | n_trades | win% | payoff | f* | G | G(f=1) | edge |
|---|---|---|---|---|---|---|---|---|
| divergence_reversal | long | 9 | 11.1% | 1.67 | 0.000 | +0.00000 | -0.02188 | ✗ |
| divergence_reversal | short | 9 | 22.2% | 0.43 | 0.000 | +0.00000 | -0.01931 | ✗ |
| consolidation_breakout | long | 3 | 0.0% | 0.00 | 0.000 | +0.00000 | -0.03223 | ✗ |
| consolidation_breakout | short | 0 | 0.0% | 0.00 | 0.000 | +0.00000 | +0.00000 | ✗ |
| v_reversal | long | 21 | 23.8% | 1.40 | 0.000 | +0.00000 | -0.02379 | ✗ |
| v_reversal | short | 32 | 18.8% | 1.28 | 0.000 | +0.00000 | -0.03511 | ✗ |

## Exit-reason breakdown (true positives only)

- **divergence_reversal / long**: stop=1
- **v_reversal / long**: unresolved=2

## Graduation gate (§9.5)

Thresholds: recall ≥ 40%, precision ≥ 25%, median capture ≥ 30%.


**Failing (6):**
- divergence_reversal / long: recall 4.2%, precision 8.3%, capture -10.3%
- divergence_reversal / short: recall 0.0%, precision 0.0%, capture 0.0%
- consolidation_breakout / long: recall 0.0%, precision 0.0%, capture 0.0%
- consolidation_breakout / short: recall 0.0%, precision 0.0%, capture 0.0%
- v_reversal / long: recall 8.3%, precision 3.4%, capture 16.9%
- v_reversal / short: recall 0.0%, precision 0.0%, capture 0.0%
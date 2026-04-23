# Phase B signal validation

- Timeframe: **1h**
- Window: 2025-04-01 00:00:00+00:00 → 2026-04-01 00:00:00+00:00
- Pop threshold: 20% in 72.0h
- Detectors: divergence_reversal, consolidation_breakout, v_reversal
- Tickers: 8

**Totals: 36 pops labeled, 282 triggers emitted**

## Results per setup × direction

| setup | dir | pops | trigs | TP | FN | FP | recall | prec | cap% | mean_ret% | lead_h | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| divergence_reversal | long | 24 | 12 | 1 | 23 | 11 | 4.2% | 8.3% | -10.3% | -2.14% | 15.0 | ✗ |
| divergence_reversal | short | 12 | 14 | 0 | 12 | 14 | 0.0% | 0.0% | 0.0% | +0.00% | — | ✗ |
| consolidation_breakout | long | 24 | 3 | 0 | 24 | 3 | 0.0% | 0.0% | 0.0% | +0.00% | — | ✗ |
| consolidation_breakout | short | 12 | 1 | 0 | 12 | 1 | 0.0% | 0.0% | 0.0% | +0.00% | — | ✗ |
| v_reversal | long | 24 | 133 | 2 | 22 | 131 | 8.3% | 1.5% | 16.9% | +3.41% | 24.0 | ✗ |
| v_reversal | short | 12 | 119 | 0 | 12 | 119 | 0.0% | 0.0% | 0.0% | +0.00% | — | ✗ |

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
- v_reversal / long: recall 8.3%, precision 1.5%, capture 16.9%
- v_reversal / short: recall 0.0%, precision 0.0%, capture 0.0%
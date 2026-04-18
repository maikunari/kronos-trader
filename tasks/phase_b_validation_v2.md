# Phase B signal validation

- Timeframe: **1h**
- Window: 2025-04-01 00:00:00+00:00 → 2026-04-01 00:00:00+00:00
- Pop threshold: 20% in 72.0h
- Detectors: divergence_reversal
- Tickers: 21

**Totals: 45 pops labeled, 16 triggers emitted**

## Results per setup × direction

| setup | dir | pops | trigs | TP | FN | FP | recall | prec | cap% | mean_ret% | lead_h | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| divergence_reversal | long | 34 | 11 | 1 | 33 | 10 | 2.9% | 9.1% | 37.4% | +7.87% | -3.0 | ✗ |
| divergence_reversal | short | 11 | 5 | 0 | 11 | 5 | 0.0% | 0.0% | 0.0% | +0.00% | — | ✗ |

## Pops per ticker

| ticker | pops |
|---|---|
| OMNI | 21 |
| TAO | 5 |
| AR | 3 |
| JTO | 3 |
| REZ | 3 |
| ENA | 2 |
| MAV | 2 |
| NEAR | 2 |
| POLYX | 2 |
| GALA | 1 |
| SAGA | 1 |
| DOGE | 0 |
| DYM | 0 |
| ENS | 0 |
| NOT | 0 |
| ONDO | 0 |
| PENDLE | 0 |
| STRK | 0 |
| TNSR | 0 |
| TON | 0 |
| W | 0 |

## Exit-reason breakdown (true positives only)

- **divergence_reversal / long**: target=1

## Graduation gate (§9.5)

Thresholds: recall ≥ 40%, precision ≥ 25%, median capture ≥ 30%.


**Failing (2):**
- divergence_reversal / long: recall 2.9%, precision 9.1%
- divergence_reversal / short: recall 0.0%, precision 0.0%, capture 0.0%
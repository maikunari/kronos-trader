# Phase B signal validation

- Timeframe: **1h**
- Window: 2025-04-01 00:00:00+00:00 → 2026-04-01 00:00:00+00:00
- Pop threshold: 20% in 72.0h
- Detectors: divergence_reversal
- Tickers: 47

**Totals: 36 pops labeled, 24 triggers emitted**

## Results per setup × direction

| setup | dir | pops | trigs | TP | FN | FP | recall | prec | cap% | mean_ret% | lead_h | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| divergence_reversal | long | 25 | 12 | 1 | 24 | 11 | 4.0% | 8.3% | -10.3% | -2.14% | 15.0 | ✗ |
| divergence_reversal | short | 11 | 12 | 0 | 11 | 12 | 0.0% | 0.0% | 0.0% | +0.00% | — | ✗ |

## Pops per ticker

| ticker | pops |
|---|---|
| LINK | 13 |
| AVAX | 11 |
| TAO | 5 |
| JTO | 3 |
| NEAR | 2 |
| ENA | 2 |
| ZEC | 0 |
| FARTCOIN | 0 |
| XPL | 0 |
| LIT | 0 |
| PAXG | 0 |
| MON | 0 |
| SUI | 0 |
| PUMP | 0 |
| VVV | 0 |
| AAVE | 0 |
| WLD | 0 |
| ZRO | 0 |
| ADA | 0 |
| XMR | 0 |
| WLFI | 0 |
| TRUMP | 0 |
| ASTER | 0 |
| BCH | 0 |
| CRV | 0 |
| FET | 0 |
| ALGO | 0 |
| ARB | 0 |
| DOT | 0 |
| PENGU | 0 |
| VIRTUAL | 0 |
| ORDI | 0 |
| LTC | 0 |
| APT | 0 |
| TON | 0 |
| UNI | 0 |
| MORPHO | 0 |
| STABLE | 0 |
| JUP | 0 |
| WIF | 0 |
| LDO | 0 |
| ONDO | 0 |
| DASH | 0 |
| SPX | 0 |
| RENDER | 0 |
| AERO | 0 |
| XLM | 0 |

## Exit-reason breakdown (true positives only)

- **divergence_reversal / long**: stop=1

## Graduation gate (§9.5)

Thresholds: recall ≥ 40%, precision ≥ 25%, median capture ≥ 30%.


**Failing (2):**
- divergence_reversal / long: recall 4.0%, precision 8.3%, capture -10.3%
- divergence_reversal / short: recall 0.0%, precision 0.0%, capture 0.0%
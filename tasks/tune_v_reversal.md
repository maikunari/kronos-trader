# v_reversal parameter sweep

- Probe tickers: HYPE, TAO, ENA, NEAR, LINK, AVAX, ZRO, WLD
- 1h, 2025-04-01 → 2026-04-01 (same as phase_b_validation_v5.md)
- Ranked by long precision; recall as tiebreak.

| move% | bounce% | hl | trigs L | TP L | recall L | prec L | cap% L | mean_ret L | trigs S | TP S | recall S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 15% | 5% | 2 | 59 | 2 | 8.3% | 3.4% | 16.9% | +3.41% | 52 | 0 | 0.0% |
| 20% | 5% | 2 | 31 | 1 | 4.2% | 3.2% | 28.9% | +5.79% | 38 | 0 | 0.0% |
| 15% | 5% | 5 | 66 | 2 | 8.3% | 3.0% | 16.9% | +3.41% | 60 | 0 | 0.0% |
| 20% | 5% | 5 | 35 | 1 | 4.2% | 2.9% | 28.9% | +5.79% | 45 | 0 | 0.0% |
| 20% | 3% | 2 | 39 | 1 | 4.2% | 2.6% | 28.9% | +5.79% | 49 | 0 | 0.0% |
| 15% | 3% | 2 | 83 | 2 | 8.3% | 2.4% | 16.9% | +3.41% | 76 | 0 | 0.0% |
| 10% | 5% | 2 | 90 | 2 | 8.3% | 2.2% | 16.9% | +3.41% | 74 | 0 | 0.0% |
| 20% | 3% | 5 | 45 | 1 | 4.2% | 2.2% | 28.9% | +5.79% | 55 | 0 | 0.0% |
| 15% | 3% | 5 | 93 | 2 | 8.3% | 2.2% | 16.9% | +3.41% | 83 | 0 | 0.0% |
| 10% | 5% | 5 | 101 | 2 | 8.3% | 2.0% | 16.9% | +3.41% | 87 | 0 | 0.0% |
| 10% | 3% | 2 | 119 | 2 | 8.3% | 1.7% | 16.9% | +3.41% | 106 | 0 | 0.0% |
| 10% | 3% | 5 | 133 | 2 | 8.3% | 1.5% | 16.9% | +3.41% | 119 | 0 | 0.0% |

## Decision gate

Any config with **precision ≥ 10% AND recall ≥ 5%** qualifies as a new default. If the top row is below that bar, deterministic tuning is exhausted and the agentic loop is the path forward.
# Consolidation-breakout: threshold diagnostic

- Probe tickers: HYPE, TAO, ENA, NEAR, LINK, AVAX, ZRO, WLD
- Same 8-ticker / 1h / 1-year window as the earlier sweep.
- Question: does loosening the pop threshold from 20% → 10% reveal recall that was hidden by move-magnitude filtering?

| range% | N | thr | pops | trigs L | TP L | recall L | prec L | cap% L | mean_ret L | trigs S | TP S | recall S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3% | 2 | 10% | 216 | 3 | 0 | 0.0% | 0.0% | 0.0% | +0.00% | 1 | 0 | 0.0% |
| 3% | 2 | 20% | 36 | 3 | 0 | 0.0% | 0.0% | 0.0% | +0.00% | 1 | 0 | 0.0% |
| 4% | 1 | 10% | 216 | 6 | 1 | 0.9% | 16.7% | -19.9% | -2.00% | 12 | 0 | 0.0% |
| 4% | 1 | 20% | 36 | 6 | 0 | 0.0% | 0.0% | 0.0% | +0.00% | 12 | 0 | 0.0% |
| 5% | 1 | 10% | 216 | 14 | 2 | 1.8% | 14.3% | -0.0% | +0.00% | 21 | 0 | 0.0% |
| 5% | 1 | 20% | 36 | 14 | 0 | 0.0% | 0.0% | 0.0% | +0.00% | 21 | 0 | 0.0% |

## Reading

Compare each (range, N) pair's 10% vs 20% rows. If the 10% row has non-zero TP/recall while the 20% row is zero, the detector was catching real moves — just smaller than the 20% label cut-off.
# Sniper Kelly diagnostic

- Timeframe: 15m
- Window: 2025-04-01 → 2026-04-01
- Tickers: BTC, ETH, SOL, HYPE
- Total trades pooled: 2393

## Per-ticker

| ticker | trades | win% | payoff | f* | G | G(f=1) | mean_ret | Sharpe | total_ret | max_dd |
|---|---|---|---|---|---|---|---|---|---|---|
| BTC | 821 | 29.8% | 1.21 | 0.000 | +0.00000 | -0.00209 | -0.21% | -9.19 | -53.6% | 53.8% |
| ETH | 795 | 34.0% | 1.47 | 0.000 | +0.00000 | -0.00158 | -0.15% | -4.06 | -33.6% | 38.7% |
| SOL | 777 | 31.9% | 1.39 | 0.000 | +0.00000 | -0.00254 | -0.25% | -5.75 | -42.6% | 43.6% |
| HYPE | 0 | — | — | — | — | — | — | — | — | — |

## Universe-pooled Kelly

- n_trades:          **2393**
- win_rate:          **31.9%**
- avg_win:           +1.18%
- avg_loss:          -0.85%
- payoff_ratio:      1.39
- Kelly fraction f*: **0.000**
- growth rate G:     **+0.00000** per trade
- G(f=1.0):          -0.00207
- mean return:       -0.200%

**Verdict:** ✗ Sniper has NO positive edge under Kelly — capital decays under any positive sizing fraction. Either retune or abandon.
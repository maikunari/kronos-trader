# Sniper Kelly diagnostic
### Scenario: **Realistic HL costs (mixed maker/taker, post-only-first)**

- Timeframe: 15m
- Window: 2025-04-01 → 2026-04-01
- Tickers: BTC, ETH, SOL
- Total trades pooled: 2393
- Cost model: fee=3.00bps/side · slippage_base=1.50bps/side · slippage_atr_frac=0.030

## Per-ticker

| ticker | trades | win% | payoff | f* | G | G(f=1) | mean_ret | Sharpe | total_ret | max_dd |
|---|---|---|---|---|---|---|---|---|---|---|
| BTC | 821 | 30.1% | 1.42 | 0.000 | +0.00000 | -0.00152 | -0.15% | -6.95 | -43.2% | 43.6% |
| ETH | 795 | 34.6% | 1.64 | 0.000 | +0.00000 | -0.00083 | -0.08% | -2.18 | -19.6% | 27.8% |
| SOL | 777 | 32.4% | 1.55 | 0.000 | +0.00000 | -0.00173 | -0.16% | -3.94 | -31.3% | 32.7% |

## Universe-pooled Kelly

- n_trades:          **2393**
- win_rate:          **32.3%**
- avg_win:           +1.24%
- avg_loss:          -0.78%
- payoff_ratio:      1.58
- Kelly fraction f*: **0.000**
- growth rate G:     **+0.00000** per trade
- G(f=1.0):          -0.00136
- mean return:       -0.130%

**Verdict:** ✗ Sniper has NO positive edge under Kelly — capital decays under any positive sizing fraction. Either retune or abandon.
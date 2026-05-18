# Sniper Kelly diagnostic
### Scenario: **Optimistic HL costs (mostly maker, HYPE staking + referral)**

- Timeframe: 15m
- Window: 2025-04-01 → 2026-04-01
- Tickers: BTC, ETH, SOL
- Total trades pooled: 2400
- Cost model: fee=1.00bps/side · slippage_base=0.50bps/side · slippage_atr_frac=0.015

## Per-ticker

| ticker | trades | win% | payoff | f* | G | G(f=1) | mean_ret | Sharpe | total_ret | max_dd |
|---|---|---|---|---|---|---|---|---|---|---|
| BTC | 825 | 32.8% | 1.73 | 0.000 | +0.00000 | -0.00051 | -0.05% | -2.50 | -18.1% | 19.8% |
| ETH | 797 | 36.0% | 1.91 | 1.000 | +0.00029 | +0.00029 | +0.04% | 0.75 | +7.8% | 10.9% |
| SOL | 778 | 33.7% | 1.81 | 0.000 | +0.00000 | -0.00056 | -0.05% | -1.14 | -10.2% | 13.2% |

## Universe-pooled Kelly

- n_trades:          **2400**
- win_rate:          **34.2%**
- avg_win:           +1.28%
- avg_loss:          -0.69%
- payoff_ratio:      1.84
- Kelly fraction f*: **0.000**
- growth rate G:     **+0.00000** per trade
- G(f=1.0):          -0.00026
- mean return:       -0.020%

**Verdict:** ✗ Sniper has NO positive edge under Kelly — capital decays under any positive sizing fraction. Either retune or abandon.
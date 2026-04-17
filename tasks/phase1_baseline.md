# Phase 1 Pre-Tuning Baseline

Date: 2026-04-17 (initial pipeline completion)

## Run

```
python main.py --mode backtest --symbol BTC --start 2024-01-01 --end 2024-12-31
```

Default `config.yaml` parameters. Real BTC 15m data from Binance (HL
returned empty for that range).

## Results

| Metric           | Value       |
|------------------|-------------|
| Return           | -71.22%     |
| Sharpe           | -11.41      |
| Max DD           | 71.32%      |
| Trades           | 1,179       |
| Win rate         | 28.3%       |
| Profit factor    | 0.49        |
| Avg win          | +0.89%      |
| Avg loss         | -0.73%      |
| Fees total       | $2,013.16   |
| Slippage total   | $2,223.46   |
| Funding total    | $0.00       |

## Reading

The strategy at default parameters is **not profitable** on BTC 2024.
This is an expected Phase 1 state, not a regression — it shows:

1. **Regime gate still too permissive.** 1,179 trades in a year = ~3.2/day.
   Target is well under 1/day for a 15m sniper. ADX threshold (20) and
   Hurst threshold (0.50) need to be tightened.
2. **Cost drag dominates.** $4,236 in fees + slippage on a $10k
   account — 42% of starting capital. This alone would kill even a
   profitable strategy. Two possible fixes:
   - Fewer, more-selective trades (from #1)
   - Post-only entries to capture maker rebates instead of taker fees
     (ExecutionPolicy exists but isn't wired through main.py's
     position-open path yet — see "Known gaps" below)
3. **Win rate × payoff is unfavorable.** 28.3% WR × 1.22 payoff ratio
   (avg_win / |avg_loss|) = expected value ~0.35 vs. 1.0 breakeven.
   Either WR or payoff needs to move materially.

## Next steps (operator)

Use the optimizer to find better parameters via purged walk-forward:

```bash
python optimizer.py \
    --symbol BTC --timeframe 15m \
    --start 2024-01-01 --end 2024-12-31 \
    --grid grids/donchian_basic.yaml \
    --out optimizer_results.json
```

Then view leaderboard:

```bash
streamlit run dashboard.py
```

Suggested parameter spaces to explore (edit the grid file):
- `adx_threshold`: [25, 28, 30, 33]  ← tighter regime gate
- `donchian_period`: [20, 30, 40, 50] ← fewer false breakouts on longer lookback
- `stop_atr_mult`: [1.0, 1.5, 2.0, 2.5] ← wider stops against noise
- `composite_threshold`: [0.0, 0.25, 0.5] ← require microstructure confirmation

Stress windows to run via `run_periods.py`:
- 2022-05 Terra: trending down hard
- 2022-11 FTX: range-bound with gap risk
- 2023-10 through 2024-04: BTC ETF run-up (trending up)
- 2024 full year (shown above)
- 2025 YTD (sideways)

## Known gaps (to address before live)

1. **ExecutionPolicy not wired into main.py's entry path.** The policy
   is built and unit-tested, but `run_live` still calls
   `executor.market_open` directly rather than `executor.place_entry`.
   Wiring it requires a live book snapshot source (HL WebSocket L2
   subscription), which is the next live-mode milestone.
2. **Microstructure features not populated in live loop.** The
   MarketContext is built with only 15m candles; funding rate, OI,
   basis, and liquidation clusters are plumbed but not fetched. This
   means composite score is always 0 in practice today (neutral,
   which passes the default threshold=0 but misses the potential edge
   from confirmation).
3. **Paper trading has not run 90 days yet.** Graduation criteria are
   documented in README; no live capital until met.

## Integration-test baseline (for regression detection)

The numbers above are year-long; for CI, a 30-day March-2025 slice
is frozen in `tests/fixtures/btc_15m_2025_03.csv` and asserted
exactly in `tests/test_integration.py`:

- trades = 80
- final_equity = $9,048.29
- sharpe = -13.36
- max_dd = 9.55%
- fees = $217.58, slippage = $240.45

Update these deliberately only when the pipeline changes semantics.

## Deferred ideas (Phase 2+)

### X Cashtags / social sentiment

X launched iPhone Cashtags (Apr 2026) surfacing stock/crypto chatter
with inline price charts. Not useful for Phase 1 because:

- BTC and ETH are too institutional — retail chatter is a lagging
  indicator, dominated by funding-rate and book positioning which
  we already consume.
- 15m bars are too fast for social sentiment signal; by the time
  chatter volume builds, price has moved.
- No Cashtag-specific API. Pulling equivalent data needs X API v2
  ($100–$5,000/mo) or aggregators (LunarCrush, Santiment). All
  add dependency + cost.
- Social-from-alpha is a crowded trade on liquid majors.

Where it may matter later:
- **Phase 2 LightGBM** — one of ~100 features, with appropriate lag,
  could contribute bps of edge in ensemble.
- **Phase 3 portfolio expansion to meme / small-caps** — retail-
  chatter-driven price is real on HYPE, PEPE, etc.
- **Circuit breaker** — anomalous chatter spike across finance-X
  could trigger defensive flatten. Narrow, not alpha.

Default plumbing when this becomes worth building: LunarCrush or
Santiment REST API (aggregates X + Reddit + Discord), not X direct.

# Kronos Trader — Phase 1 Sniper

Short-timeframe trend sniper for Hyperliquid perps. Classical
breakouts + microstructure confirmation, no ML in the hot path. Built
around honest backtesting, realistic execution, and a strict regime gate.

> The previous Kronos/TimesFM ML system was removed because the models
> were pretrained on off-distribution data (stocks / long-horizon macro),
> not crypto short-TF. This is the replacement; a LightGBM vetoer can
> layer on top in Phase 2 once the classical baseline is validated.

## Pipeline

At each 15-minute bar close:

1. **Regime gate** — `RegimeDetector` requires `ADX(14) ≥ 20` **and**
   `Hurst(200) ≥ 0.50`. Skip-don't-trade in chop.
2. **Breakout** — Donchian(20) break of prior-N-bar high/low, or
   Keltner(20, 2×ATR) channel break. Fresh-break logic: previous close
   must be on the opposite side of the band.
3. **1H SuperTrend trend gate** — must agree with breakout direction.
4. **4H veto** — `MTFFilter.vetoes(direction, which="4h")`: reject
   entries directly opposing the 4H EMA trend (no AND-gate).
5. **Funding-rate veto** — reject longs when funding > +0.03%/hr,
   shorts when funding < −0.03%/hr (paying carry into the move).
6. **Microstructure composite score** — signed contributions from
   OI delta, perp-spot basis expansion, CVD slope, and liquidation
   cluster proximity. Require composite ≥ 0.
7. **Risk sizing** — `RiskManager` vol-targets size against realized
   annualized vol, clipped to a hard per-trade cap. Optional
   fractional-Kelly overlay.
8. **Entry + stops** — ATR-based initial stop (1.5×) and target (3×) →
   2:1 R:R. Chandelier trail ratchets the stop once in position.
9. **Execution** — `ExecutionPolicy` submits post-only-first; falls
   through to taker after `taker_timeout_ms`; skips if spread is
   anomalously wide; icebergs large intents by top-of-book depth.

## Setup

```bash
git clone <repo>
cd kronos-trader

python3.11 -m venv venv
./venv/bin/pip install -r requirements.txt

# Credentials (only needed for live mode)
cp .env.example .env   # then edit
# HYPERLIQUID_WALLET_ADDRESS=
# HYPERLIQUID_PRIVATE_KEY=
# COINGLASS_API_KEY=    # optional, for liquidation clusters
# ALERT_WEBHOOK_URL=    # optional, Discord/Slack
```

## Usage

### Backtest

```bash
./venv/bin/python main.py --mode backtest --symbol BTC --start 2024-01-01 --end 2024-12-31
```

Fetches 15m + 1H + 4H candles from Hyperliquid (Binance fallback for
15m), runs the full pipeline, prints a metrics summary. Edit
`config.yaml` to change parameters.

### Multi-period comparison

Side-by-side backtests across named windows (Terra, FTX, 2024, YTD):

```bash
./venv/bin/python run_periods.py --symbol BTC
./venv/bin/python run_periods.py --symbol ETH --periods "bull:2023-10-01:2024-04-30"
```

### Walk-forward grid search

Tunes parameters honestly with purged walk-forward and ranks by
median OOS Sharpe:

```bash
./venv/bin/python optimizer.py \
    --symbol BTC --timeframe 15m \
    --start 2024-01-01 --end 2024-12-31 \
    --grid grids/donchian_basic.yaml \
    --out optimizer_results.json
```

### Paper / live

Paper mode runs the full loop against live data without placing orders:

```bash
./venv/bin/python main.py --mode paper --symbol BTC
```

Live mode requires `.env` credentials and an explicit `yes` confirmation:

```bash
./venv/bin/python main.py --mode live --symbol BTC
```

### Dashboard

```bash
./venv/bin/streamlit run dashboard.py --server.port 8502
```

Panels: backtest result viewer (equity curve + cost attribution),
optimizer leaderboard, live-state stub.

## Module map

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point wiring the pipeline end-to-end |
| `snipe_signal_engine.py` | Regime-gated breakout + microstructure composite |
| `regime.py` | ADX + Hurst + realized-vol classification |
| `microstructure.py` | Pure-math primitives: OI delta, basis, CVD, liq proximity |
| `mtf_filter.py` | 1H/4H EMA trend bias with `.vetoes()` / `.confirms()` helpers |
| `atr_engine.py` | ATR math + `ChandelierTrail` ratcheting exit |
| `risk_manager.py` | Vol-targeting, Kelly overlay, kill switches |
| `execution_policy.py` | Post-only → taker, iceberg, spread anomaly skip |
| `executor.py` | Hyperliquid SDK wrapper (paper + live modes) |
| `hyperliquid_feed.py` | WS candle + liquidation streams, historical REST |
| `coinglass_client.py` | Aggregated liquidation heatmap fetcher |
| `backtest.py` | Canonical backtester + purged walk-forward |
| `optimizer.py` | Grid search on top of walk-forward |
| `run_periods.py` | Multi-period side-by-side backtest comparison |
| `dashboard.py` | Streamlit UI |
| `alerts.py` | Discord/Slack webhook alerting |

Legacy, retained only for the derivatives engine: `trend_signal_engine.py`,
`trend_backtest.py`, `derivatives_*`. Removed at end of Phase 1.

## Paper → live graduation criteria

Before any live capital, all must hold:

1. ≥ 90 calendar days paper, ≥ 150 trades total (BTC + ETH combined).
2. Paper Sharpe ≥ 1.2 (daily-resampled), max drawdown ≤ 10%.
3. At least one trigger of every kill switch in paper — verified behavior.
4. Live fill quality within 20% of backtest slippage assumption during
   a two-week paper-next-to-live shadow test.
5. Start live at 10% of target sizing for 30 days; step up only if
   live P&L tracks paper within 25%.

## Tests

```bash
./venv/bin/python -m pytest -q
```

Suite runs headlessly (no network), ~60s. Covers every pure-math
module, every kill switch, every execution-policy branch, and the
backtest + walk-forward flow on synthetic data.

## Roadmap

- **Phase 1 (current)** — classical breakouts + microstructure + honest
  backtest + kill-switch-heavy risk layer. Paper-trade 90 days.
- **Phase 2** — LightGBM classifier on crypto features (funding deltas,
  OI, basis, CVD, liquidations, cross-asset). Integrated as a *vetoer*
  on Phase 1 signals, not a primary predictor. Triple-barrier labels,
  purged walk-forward CV, weekly retraining, drift monitor.
- **Phase 3** — Portfolio of uncorrelated strategies (trend,
  funding-squeeze, mean-reversion) across BTC/ETH/SOL/HYPE, with a
  rolling-Sharpe meta-allocator.

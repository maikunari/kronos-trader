# Kronos Trader

A systematic crypto trading bot for Hyperliquid perps with three distinct strategy modules:

1. **Kronos/TimesFM** — ML prediction-based scalper (original system)
2. **Derivatives Engine** — Counter-leverage squeeze plays using funding rate extremes
3. **Trend Engine** — EMA crossover trend-following with ATR-based stops

---

## Architecture

```
Hyperliquid WebSocket / REST
         │
  OHLCV + Funding Rate data
         │
  ┌──────┴───────────────────────┐
  │                              │
Kronos/TimesFM Engine    Derivatives Engine      Trend/ATR Engine
(ML prediction scalper)  (funding rate squeeze)  (EMA + ATR trend follow)
  │                              │                     │
  └──────────────────┬───────────┘─────────────────────┘
                     │
              Signal Engine
              (D1 MTF gate)
                     │
              Risk Manager
              (size, daily loss limit)
                     │
               Executor
             (paper / live)
                     │
              Hyperliquid
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/maikunari/kronos-trader
cd kronos-trader
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Kronos from GitHub (optional, for ML engine)

```bash
pip install git+https://github.com/shiyu-coder/Kronos.git
```

> If Kronos is not installed, the bot falls back to a statistical baseline model. Backtesting works either way.

### 3. Configure

Edit `config.yaml` to set your target symbol, timeframe, strategy, and risk parameters.

### 4. For live/paper trading: set credentials

```bash
cp .env.example .env
# edit .env with your Hyperliquid wallet address and private key
```

---

## Strategies

### 1. Kronos/TimesFM Scalper (`signal_engine.py`)

- **Kronos** predicts next candle OHLCV → direction + predicted move %
- **TimesFM** forecasts next 10 candles → macro trend direction
- Trade if both agree AND predicted move ≥ `min_move_threshold` (default 0.3%)
- Default R:R: 1:2 (stop 0.2%, target 0.4%)

### 2. Derivatives / Funding Rate Engine (`derivatives_signal_engine.py`)

Counter-leverage squeeze plays riding forced liquidations:
- Extreme positive funding + D1 downtrend → **SHORT** the over-leveraged longs
- Extreme negative funding + D1 uptrend → **LONG** the over-leveraged shorts
- Funding threshold: 0.05%/hr (Hyperliquid convention). 0.1%/hr = extreme.
- D1 MTF gate: only trade the squeeze if the daily trend agrees

### 3. Trend / ATR Engine (`trend_signal_engine.py`, `atr_engine.py`)

EMA crossover trend-following with ATR-based dynamic stops:
- Entry: EMA fast/slow crossover confirmed by D1 EMA(20/50) trend gate
- Stop: 1.5 × ATR(14) from entry
- Target: 3.0 × ATR(14) from entry → always 2:1 R:R
- Default params: EMA fast=9, slow=21, ATR period=14

---

## Backtesting

No API keys required — fetches historical data from Hyperliquid (Binance fallback).

```bash
# Default (SOL, 15m)
./run_backtest.sh

# Custom symbol and timeframe
./run_backtest.sh ETH 5m
./run_backtest.sh BTC 15m

# Direct with date range
python backtest.py --config config.yaml --symbol SOL --timeframe 15m --start 2024-01-01 --end 2024-12-31

# Derivatives engine backtest
python derivatives_backtest.py

# Trend engine backtest
python trend_backtest.py
```

### Batch & Comparison Tools

```bash
# Run across multiple symbols/timeframes
./run_batch.sh

# Compare across three market periods (2024 bull / 2025 drawdown / 2026 YTD)
python run_periods.py

# Compare M15 vs M5 vs M1 side-by-side (90-day window)
python compare_tf.py

# Compare MTF require_both=True vs 4H-only gate
python compare_mtf.py
```

---

## Running the Bot

### Paper mode (live feed, no real orders)

```bash
python main.py --mode paper
```

### Live mode (real trades on Hyperliquid)

```bash
python main.py --mode live
```

Requires `.env` with `HYPERLIQUID_PRIVATE_KEY` and `HYPERLIQUID_WALLET_ADDRESS`.

### Live Dashboard (Streamlit)

```bash
venv/bin/streamlit run dashboard.py --server.port 8502
```

Displays live paper trade P&L, recent signals, equity curve, and position status.

---

## Risk Management

- Max position size: 10% of account per trade
- Daily loss limit: -3% of account equity → halts trading for the rest of the day
- Max 1 concurrent position

---

## Key Files

| File | Purpose |
|------|---------|
| `config.yaml` | All configuration — no hardcoded values |
| `main.py` | Entry point for live/paper trading |
| `signal_engine.py` | Kronos + TimesFM signal logic |
| `derivatives_signal_engine.py` | Funding rate extreme + MTF gate signals |
| `derivatives_feed.py` | Funding rate + OHLCV data for derivatives engine |
| `derivatives_backtest.py` | Backtester for derivatives strategy |
| `trend_signal_engine.py` | EMA crossover + D1 gate signals |
| `atr_engine.py` | ATR-based volatility and position sizing |
| `trend_backtest.py` | Backtester for trend strategy |
| `backtest.py` | Core backtester (Kronos/TimesFM strategy) |
| `kronos_model.py` | Kronos wrapper (lazy-loaded, stat fallback) |
| `timesfm_model.py` | TimesFM wrapper (lazy-loaded, SMA fallback) |
| `hyperliquid_feed.py` | WebSocket live feed + REST historical data |
| `risk_manager.py` | Position sizing and daily loss kill switch |
| `executor.py` | Hyperliquid order placement (live + paper) |
| `dashboard.py` | Streamlit live paper trading dashboard |
| `mtf_filter.py` | Multi-timeframe trend gate |
| `daily_report.py` | Daily P&L summary |
| `compare_tf.py` | Side-by-side timeframe comparison |
| `compare_mtf.py` | MTF gate on/off comparison |
| `run_periods.py` | Backtest across 2024 / 2025 / 2026 YTD |
| `run_batch.sh` | Batch backtest runner |
| `run_backtest.sh` | Quick backtest shortcut |

---

## Supported Assets

Any Hyperliquid perp: SOL, BTC, ETH, AVAX, LINK, XRP, etc.

Recommended starting point: **SOL-PERP** (deep liquidity, good volatility).

---

## Notes

- Backtest results ≠ live performance. Paper trade for at least a month before going live.
- Hyperliquid taker fee: 0.035% per side. Break-even at 1:2 R:R requires ~48% win rate.
- The statistical fallback models (no Kronos/TimesFM installed) are simpler — install the real models for meaningful results.

# Kronos Trader

A systematic crypto scalping bot combining **Kronos** (candlestick OHLCV prediction) and **TimesFM** (macro trend filter) for trading on Hyperliquid perps.

Target: SOL-PERP on 15m candles. Break-even win rate ~48% at 1:2 R:R.

```
Architecture:

  Hyperliquid WebSocket / REST
           │
    OHLCV candle history
           │
    ┌──────┴──────┐
    │             │
  Kronos      TimesFM
  (next OHLCV  (macro trend
   prediction)  direction)
    │             │
    └──────┬──────┘
           │
     Signal Engine
     (agree? move > threshold?)
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
git clone https://github.com/yourname/kronos-trader
cd kronos-trader
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Kronos from GitHub

```bash
pip install git+https://github.com/shiyu-coder/Kronos.git
```

> **Note:** If Kronos is not installed, the bot falls back to a statistical baseline model. Backtesting works either way — install Kronos for real ML predictions.

### 3. Configure

Edit `config.yaml` to set your target symbol, timeframe, risk parameters, etc.

Default config targets `SOL` on `15m` candles.

### 4. For live/paper trading only: set credentials

```bash
cp .env.example .env
# edit .env with your Hyperliquid wallet address and private key
```

---

## Running a Backtest

Backtesting requires **no API keys**. It fetches historical data from Hyperliquid (or Binance as fallback).

```bash
# Default (SOL, 15m, 2024 full year)
./run_backtest.sh

# Custom symbol and timeframe
./run_backtest.sh ETH 5m
./run_backtest.sh BTC 15m

# Via Python directly with date override
python backtest.py --config config.yaml --symbol SOL --timeframe 15m --start 2024-06-01 --end 2024-12-31
```

Output includes:
- Console report (win rate, profit factor, Sharpe, max drawdown)
- `results_SOL_15m.png` — equity curve + drawdown chart

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

You'll be asked to confirm. Requires `.env` with `HYPERLIQUID_PRIVATE_KEY` and `HYPERLIQUID_WALLET_ADDRESS`.

---

## Signal Logic

1. **Kronos** predicts next candle OHLCV → direction (long/short) + predicted move %
2. **TimesFM** forecasts next 10 candles of close prices → macro trend direction
3. **Trade** if both agree AND predicted move ≥ `min_move_threshold` (default 0.3%)
4. **Flat** if they disagree

Default R:R: **1:2** (stop 0.2%, target 0.4%)

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
| `backtest.py` | Standalone backtester |
| `main.py` | Entry point for live/paper |
| `signal_engine.py` | Combines Kronos + TimesFM into trade signals |
| `kronos_model.py` | Kronos wrapper (lazy-loaded, stat fallback if not installed) |
| `timesfm_model.py` | TimesFM wrapper (lazy-loaded, SMA fallback if not installed) |
| `hyperliquid_feed.py` | WebSocket live feed + REST historical data |
| `risk_manager.py` | Position sizing and daily loss kill switch |
| `executor.py` | Hyperliquid order placement (live + paper) |

---

## Important Notes

- **Backtest results ≠ live performance.** These models are research artifacts.
- Paper trade for at least a month before going live.
- The statistical fallback models (used when Kronos/TimesFM not installed) are simpler — install the real models for meaningful results.
- Hyperliquid taker fee: 0.035% per side. Break-even math at 1:2 R:R requires ~48% win rate.

---

## Supported Assets

Any Hyperliquid perp: SOL, BTC, ETH, AVAX, LINK, XRP, etc.

Recommended starting point: **SOL-PERP** (deep liquidity, good volatility, Kronos likely has heavy training data).

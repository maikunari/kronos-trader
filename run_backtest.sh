#!/usr/bin/env bash
# run_backtest.sh — Quick backtest launcher
# Usage: ./run_backtest.sh [symbol] [timeframe]
# Examples:
#   ./run_backtest.sh
#   ./run_backtest.sh SOL 15m
#   ./run_backtest.sh ETH 5m

set -e
cd "$(dirname "$0")"

SYMBOL=${1:-SOL}
TIMEFRAME=${2:-15m}
PLOT="results_${SYMBOL}_${TIMEFRAME}.png"

echo "Running backtest: $SYMBOL / $TIMEFRAME"
echo "Output plot: $PLOT"
echo ""

./venv/bin/python backtest.py \
  --config config.yaml \
  --symbol "$SYMBOL" \
  --timeframe "$TIMEFRAME" \
  --plot "$PLOT"

echo ""
echo "Done. Equity curve saved to $PLOT"

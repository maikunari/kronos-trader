#!/bin/bash
cd ~/clawd/projects/kronos-trader
source venv/bin/activate

echo "Running 2024 (Jan 1 - Dec 31)..."
python3 backtest.py --config config.yaml --symbol SOL --timeframe 4h --start 2024-01-01 --end 2025-01-01 --plot results_SOL_4h_2024.png > /tmp/bt_2024.log 2>&1
echo "2024 complete."

echo "Running 2025 (Jan 1 - Dec 31)..."
python3 backtest.py --config config.yaml --symbol SOL --timeframe 4h --start 2025-01-01 --end 2026-01-01 --plot results_SOL_4h_2025.png > /tmp/bt_2025.log 2>&1
echo "2025 complete."

echo "Running 2026 (Jan 1 - Apr 14)..."
python3 backtest.py --config config.yaml --symbol SOL --timeframe 4h --start 2026-01-01 --end 2026-04-14 --plot results_SOL_4h_2026.png > /tmp/bt_2026.log 2>&1
echo "2026 complete."

#!/usr/bin/env python3
"""
Quick verification that RS debug is enabled by default
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from backtester_core.backtest_engine import BacktestEngine
import pandas as pd

# Create minimal test data
print("\n" + "="*80)
print("VERIFYING RS DEBUG IS ENABLED BY DEFAULT")
print("="*80)

# Minimal config (no debug settings provided)
config = {
    'ema_short': 10,
    'ema_long': 20,
    'supertrend_stop_pct': 10.0,
    'max_holdings': 5
}

# Create dummy data
dates = pd.date_range('2024-01-01', periods=100, freq='D')
stock_data = pd.DataFrame({
    'symbol': 'TRENT',
    'date': dates,
    'open': [100]*100,
    'high': [105]*100,
    'low': [95]*100,
    'close': [102]*100,
    'adj_close': [102 + i*0.5 for i in range(100)],
    'volume': [1000000]*100,
    'turnover_cr': [15]*100
})

index_data = pd.DataFrame({
    'date': dates,
    'open': [24000]*100,
    'high': [24500]*100,
    'low': [23500]*100,
    'close': [24000]*100,
    'adj_close': [24000 + i*10 for i in range(100)],
    'volume': [10000000]*100
})

# Initialize engine
engine = BacktestEngine(stock_data, index_data, config)

# Check debug settings
print(f"\n✅ Configuration Check:")
print(f"   debug_rs enabled: {engine.debug_rs}")
print(f"   debug_rs_symbol: {engine.debug_rs_symbol}")

if engine.debug_rs and engine.debug_rs_symbol == 'TRENT':
    print(f"\n✅ SUCCESS! RS Debug is ENABLED by default for TRENT")
    print(f"\n📊 When you run a backtest, you will see detailed RS calculation output!")
else:
    print(f"\n❌ FAILED! Debug settings not as expected")

print("="*80 + "\n")


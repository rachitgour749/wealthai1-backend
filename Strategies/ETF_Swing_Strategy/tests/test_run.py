import sys
import os
import json
import pandas as pd
from datetime import datetime

# Add root project path for imports
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_path)

from Strategies.ETF_Swing_Strategy.services.backtester import ETFSwingBacktester

def test_strategy_logic():
    print("Starting Strategy Logic Test...")
    
    # Initialize backtester
    backtester = ETFSwingBacktester()
    
    # Sample parameters
    tickers = ["NIFTYBEES", "BANKBEES", "GOLDBEES"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    initial_capital = 1000000
    
    print(f"Running backtest for {len(tickers)} tickers from {start_date} to {end_date}")
    
    try:
        results = backtester.run_backtest(tickers, start_date, end_date, initial_capital)
        
        if "error" in results:
            print(f"❌ Backtest failed with error: {results['error']}")
            return

        metrics = results.get("metrics", {})
        print("\n✅ Backtest completed successfully!")
        print("-" * 30)
        print(f"Total Return: {metrics.get('Total Return (%)')}%")
        print(f"CAGR: {metrics.get('CAGR (%)')}%")
        print(f"Max Drawdown: {metrics.get('Max Drawdown (%)')}%")
        print(f"Final Capital: ₹{metrics.get('Final Capital'):,.2f}")
        print(f"Total Trades: {metrics.get('Total Trades')}")
        print("-" * 30)
        
        # Check transaction log
        log = results.get("transaction_log", [])
        if log:
            print(f"\nSample Trades (First 3):")
            for trade in log[:3]:
                print(f" - {trade['date']} | {trade['action']} {trade['symbol']} at ₹{trade['price']:.2f} (Qty: {trade['qty']})")
        else:
            print("\nNo trades executed during the period.")
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_strategy_logic()

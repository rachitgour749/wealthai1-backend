import sys
import os
import json
import pandas as pd
from datetime import datetime

# Add root project path for imports
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_path)

from Strategies.ETF_Swing_Strategy.services.backtester import ETFSwingBacktester
from Databases.app_data_db_connection import create_connection

def test_strategy_logic():
    print("Starting Strategy Logic Test...")
    
    # Initialize Database Connection
    if not create_connection():
        print("X Failed to connect to database. Check DATABASE_STRING in .env")
        return
    
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
            print(f"X Backtest failed with error: {results['error']}")
            return

        metrics = results.get("metrics", {})
        print("\nOK Backtest completed successfully!")
        
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n" + "="*60 + "\n")
            f.write(f"{'Metric':<20} | {'Strategy':<15} | {'Nifty 50':<15}\n")
            f.write("-" * 60 + "\n")
            
            comp_metrics = ["Return (%)", "CAGR (%)", "Max DD (%)", "Final Value"]
            for metric in comp_metrics:
                strat_val = metrics.get("Strategy", {}).get(metric, "N/A")
                bench_val = metrics.get("Benchmark (Nifty 50)", {}).get(metric, "N/A")
                f.write(f"{metric:<20} | {strat_val:<15} | {bench_val:<15}\n")
            
            f.write("="*60 + "\n")
            
            # Check transaction log
            log = results.get("transaction_log", [])
            if log:
                f.write(f"\nSample Trades (First 5):\n")
                for trade in log[:5]:
                    f.write(f" - {trade['date']} | {trade['action']} {trade['symbol']} at ₹{trade['price']:.2f} (Qty: {trade['qty']})\n")
            else:
                f.write("\nNo trades executed during the period.\n")
        
        print(f"Results written to: {output_file}")
            
    except Exception as e:
        print(f"X Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_strategy_logic()

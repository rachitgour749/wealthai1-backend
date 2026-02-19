import sys
import os
import pandas as pd
from datetime import datetime

# Add project root to path
working_dir = r"c:\Users\Lenovo\Desktop\Broker Integration\Broker Integration\wealthai-backend-v2"
if working_dir not in sys.path:
    sys.path.insert(0, working_dir)

from Strategies.ETF_Swing_Strategy.services.backtester import ETFSwingBacktester

def test_sma_validation():
    print("Testing SMA Validation Logic...")
    
    # Initialize backtester
    backtester = ETFSwingBacktester()
    
    # Set high SMA lookback
    backtester.strategy.sma_lookback = 600
    
    # Mock data with only 500 days
    dates = pd.date_range(end=datetime.now(), periods=500)
    mock_df = pd.DataFrame({
        'open': [100] * 500,
        'high': [105] * 500,
        'low': [95] * 500,
        'close': [102] * 500,
        'volume': [1000] * 500
    }, index=dates)
    
    # Override load_data to return mock data
    def mock_load_data(tickers, start_date, end_date):
        return {
            "NIFTYBEES.NS": mock_df,
            "BENCHMARK_NIFTY50": mock_df
        }
    
    backtester.load_data = mock_load_data
    
    # Run backtest
    result = backtester.run_backtest(
        tickers=["NIFTYBEES.NS"],
        start_date="2024-02-28",
        end_date="2026-02-18",
        initial_capital=1000000
    )
    
    # Verify result
    if "error" in result:
        print(f"Test Passed: Caught expected error - '{result['error']}'")
        return True
    else:
        print("Test Failed: Backtest ran despite insufficient data.")
        return False

if __name__ == "__main__":
    success = test_sma_validation()
    sys.exit(0 if success else 1)

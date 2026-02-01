
import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(r'd:\WEALTHAI_V2\wealthai-backend-v2')
sys.path.append(r'd:\WEALTHAI_V2\wealthai-backend-v2\Strategies') # For imports inside backtester

from Strategies.Rotation_International_ETF.services.backtester import InternationalETFRotationBacktester
from Databases.app_data_db_connection import create_connection

def verify_backtester_loading():
    if not create_connection():
        print("Failed to initialize DB connection")
        return

    try:
        print("Instantiating Backtester...")
        backtester = InternationalETFRotationBacktester()
        backtester.set_verbose(True)
        
        tickers = ['QQQ', 'SPY'] 
        start_date = '2024-01-01'
        end_date = '2024-01-31'
        
        print(f"Loading data for {tickers} from {start_date} to {end_date}...")
        data = backtester.load_data_from_database(tickers, start_date, end_date)
        
        if data:
            print("Successfully loaded data keys:", data.keys())
            if 'close' in data and not data['close'].empty:
                print("SUCCESS: 'close' data frame is present and not empty.")
                print(data['close'].head())
            else:
                print("FAILURE: 'close' data missing or empty")
        else:
            print("FAILURE: No data returned")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_backtester_loading()

import sys
import os
import json
from datetime import datetime, timedelta

# Add project root to sys.path
sys.path.append(os.getcwd())

from Services.scheduler.generators.etf_swing_generator import _process_instance
from Databases.app_data_db_connection import create_connection

def test_live_logic():
    print("Initializing Database Connection...")
    create_connection()
    
    # Test cases: One India ETF, one US Stock
    test_instances = [
        {
            "id": "TEST_INDIA_ETF",
            "user_id": "test_user@wealthai.com",
            "user_code": "TESTIN",
            "tickers": ["NIFTYBEES", "JUNIORBEES"],
            "strategies_parameters": {
                "market": "INDIA",
                "asset_type": "ETF",
                "sma_lookback": 50,
                "number_of_slots": 5
            }
        },
        {
            "id": "TEST_US_STOCK",
            "user_id": "test_user@wealthai.com",
            "user_code": "TESTUS",
            "tickers": ["AAPL", "MSFT", "TSLA"],
            "strategies_parameters": {
                "market": "US",
                "asset_type": "STOCK",
                "sma_lookback": 50,
                "number_of_slots": 2
            }
        }
    ]

    signal_date = datetime.now()
    print(f"\nRunning Signal Generation Simulation for Date: {signal_date.date()}")
    print("="*60)

    for instance in test_instances:
        market = instance['strategies_parameters']['market']
        asset_type = instance['strategies_parameters']['asset_type']
        print(f"\n>>> TESTING: {market} {asset_type} (Tickers: {instance['tickers']})")
        
        try:
            # This triggers the actual generalized logic
            signals = _process_instance(instance, signal_date)
            
            if not signals:
                print(f"Result: No signals generated (likely due to market conditions or missing data for {instance['tickers']})")
            else:
                for s in signals:
                    print(f"Result: {s.side} Signal for {s.symbol} at {s.price}")
                    print(f"Metadata: {s.strategy_metadata}")
                    
        except Exception as e:
            print(f"Error during simulation: {e}")

    print("\n" + "="*60)
    print("Simulation Complete. Check the logs in Logs/INDIA_ETF_Swing_Strategy.log and Logs/US_STOCK_Swing_Strategy.log")

if __name__ == "__main__":
    test_live_logic()

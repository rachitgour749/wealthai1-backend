import sys
import os
import json
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.getcwd())

from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

def test_api_generalization():
    print("\n" + "="*60)
    print("TESTING API GENERALIZATION (INDIA & US)")
    print("="*60)

    # 1. Test India ETF Backtest
    print("\n--- Sending India ETF Backtest Request ---")
    india_request = {
        "strategy_type": "ETF_Swing_Strategy",
        "start_date": "2024-01-01",
        "end_date": "2024-02-01",
        "tickers": ["NIFTYBEES", "BANKBEES"],
        "initial_capital": 100000,
        "market": "INDIA",
        "asset_type": "ETF",
        "sma_lookback": 50,
        "number_of_slots": 2
    }
    
    try:
        response = client.post("/api/run_backtest", json=india_request)
        if response.status_code == 200:
            data = response.json()
            print(f"[SUCCESS] Strategy: {data.get('strategy_type')}")
            metrics = data.get('metrics', {})
            print(f"Metrics: CAGR={metrics.get('cagr')}%, Total Return={metrics.get('total_return_pct')}%")
        else:
            print(f"[FAILED] Status: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"[ERROR] Exception during request: {e}")

    # 2. Test US Stock Backtest
    print("\n--- Sending US US Stock Backtest Request ---")
    us_request = {
        "strategy_type": "ETF_Swing_Strategy",
        "start_date": "2024-01-01",
        "end_date": "2024-02-01",
        "tickers": ["AAPL", "MSFT"],
        "initial_capital": 100000,
        "market": "US",
        "asset_type": "STOCK",
        "sma_lookback": 50,
        "number_of_slots": 2
    }
    
    try:
        response = client.post("/api/run_backtest", json=us_request)
        if response.status_code == 200:
            data = response.json()
            print(f"[SUCCESS] Strategy: {data.get('strategy_type')}")
            metrics = data.get('metrics', {})
            print(f"Metrics: CAGR={metrics.get('cagr')}%, Total Return={metrics.get('total_return_pct')}%")
        else:
            print(f"[FAILED] Status: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"[ERROR] Exception during request: {e}")

    print("\n" + "="*60)
    print("API TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_api_generalization()

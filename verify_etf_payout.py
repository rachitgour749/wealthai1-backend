import requests
import json
from datetime import datetime

def test_etf_payout_backtest():
    url = "http://localhost:8000/api/run_backtest"
    
    payload = {
        "strategy_type": "ETF_Payout",
        "start_date": "2022-01-01",
        "end_date": "2023-12-31",
        "tickers": ["NIFTYBEES.NS", "BANKBEES.NS", "GOLDBEES.NS"],
        "capital_per_week": 50000,
        "accumulation_weeks": 52,
        "brokerage_percent": 0.1,
        "compounding_enabled": False,
        "risk_free_rate": 8.0,
        "withdraw_amount": 10000,
        "payout_start_week": 53
    }
    
    print(f"Sending backtest request for ETF_Payout (with .NS tickers)...")
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            print("\n✅ Backtest successful!")
            
            metrics = data.get("metrics", {})
            performance_data = data.get("performance_data", {})
            
            # Check Benchmark Data
            benchmark_buyhold = performance_data.get("benchmark_buyhold", [])
            print(f"Benchmark Buy-Hold data points: {len(benchmark_buyhold)}")
            if benchmark_buyhold:
                print("✅ benchmark_buyhold is populated")
            else:
                print("❌ benchmark_buyhold is EMPTY")
            
            # Check Benchmark Metrics
            benchmark_metrics = metrics.get("benchmark_metrics", {})
            if benchmark_metrics:
                print("✅ benchmark_metrics is populated")
                print(f"   Debug - benchmark_metrics keys: {list(benchmark_metrics.keys())}")
                cagr = benchmark_metrics.get('cagr_pct')
                print(f"   Benchmark CAGR: {cagr}%")
                
                # Check for other normalized keys
                expected_keys = ['total_return_pct', 'cagr_pct', 'max_drawdown_pct', 'sharpe_ratio']
                missing_keys = [k for k in expected_keys if k not in benchmark_metrics]
                if not missing_keys:
                    print("✅ All expected normalized keys present")
                else:
                    print(f"❌ Missing normalized keys: {missing_keys}")
            else:
                print("❌ benchmark_metrics is EMPTY")

            # Check Withdrawal Data
            cumulative_withdrawn = performance_data.get("cumulative_withdrawn", [])
            print(f"Cumulative Withdrawn data points: {len(cumulative_withdrawn)}")
            if cumulative_withdrawn:
                print("✅ cumulative_withdrawn is populated")
                print(f"   Final cumulative withdrawn: {cumulative_withdrawn[-1]}")
            else:
                print("❌ cumulative_withdrawn is EMPTY")
                
        else:
            print(f"\n❌ Backtest failed: {data.get('error')}")
            
    except Exception as e:
        print(f"\n❌ Error during request: {e}")

if __name__ == "__main__":
    test_etf_payout_backtest()

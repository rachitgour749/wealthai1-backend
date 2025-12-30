
import sys
import os
import pandas as pd
from datetime import datetime
import json
import codecs

# Force UTF-8 for stdout/stderr to handle currency symbols
if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, 'strict')

# Add project root to path
backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(backend_root)

# Import necessary modules from API routes file logic (simulating the API behavior)
from Strategies.CustomStrategies.Rotation_ETF_Payout.backtester import RotationETFPayoutBacktester

def sanitize_data(data):
    """Sanitize float values to avoid NaN/Infinity for JSON serialization"""
    if isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(i) for i in data]
    elif isinstance(data, float):
        if pd.isna(data) or pd.isinf(data):
            return 0.0
        return data
    return data

def mock_get_etf_transaction_log(portfolio_log):
    """
    Simulated version of the API function to test week normalization logic
    """
    transaction_log = []
    
    # ... (Replicating the loop logic from API for constructing transaction_log) ...
    # Simplified loop for verification
    for log in portfolio_log:
        transaction_log.append({
            'week': log.get('week', 0),
            'execution_date': log.get('execution_date'),
            'action': log.get('action', 'NONE'),
            'total_raised': log.get('total_raised', 0),
            'withdrawal_amount': log.get('withdrawal_amount', 0)
        })

    # === THE FIX BEING TESTED ===
    # Normalize Week Numbers
    if transaction_log:
        min_week = min(log['week'] for log in transaction_log)
        week_offset = min_week - 1
        
        if week_offset > 0:
            print(f"Normalizing log weeks by offset: -{week_offset} (First week was {min_week})")
            for log in transaction_log:
                log['week'] = log['week'] - week_offset
    # ============================
    
    return transaction_log

def verify_fix():
    print("Initializing Backtester...")
    backtester = RotationETFPayoutBacktester()
    
    # Set explicit parameters
    capital_per_week = 50000
    withdraw_amount = 25000
    
    backtester.withdraw_amount = withdraw_amount
    backtester.accumulation_per_week = capital_per_week
    
    print("Running Backtest...")
    results = backtester.run_backtest(
        tickers=['NIFTYBEES', 'GOLDBEES', 'ITBEES'],
        start_date='2020-01-01',
        end_date='2020-06-01', # Shorter period for quick check
        capital_per_week=capital_per_week,
        accumulation_weeks=5,
        brokerage_percent=0.03,
        compounding_enabled=True
    )
    
    if not backtester.portfolio_log:
        print("No logs generated")
        return

    print(f"\nOriginal First Log Week: {backtester.portfolio_log[0].get('week')}")
    
    # Run the simulated API logic
    normalized_logs = mock_get_etf_transaction_log(backtester.portfolio_log)
    
    print("\n--- Normalized Logs Verification ---")
    if normalized_logs:
        first_log = normalized_logs[0]
        print(f"First Log Week Number: {first_log['week']}")
        if first_log['week'] == 1:
            print("✅ SUCCESS: Week normalized to 1")
        else:
            print(f"❌ FAILURE: Week is {first_log['week']}, expected 1")
    
    # Verify withdrawals are present in churn logs
    churn_logs = [l for l in normalized_logs if l['action'] == 'churn']
    if churn_logs:
        first_churn = churn_logs[0]
        print(f"\n--- Withdrawal Verification ---")
        print(f"Withdrawal Amount in Log: {first_churn.get('withdrawal_amount')}")
        if first_churn.get('withdrawal_amount') == withdraw_amount:
             print("✅ SUCCESS: Withdrawal amount present in log")
        else:
             print("❌ FAILURE: Withdrawal amount mismatch or missing")
    else:
        print("No churn logs in this short run (expected if < accumulation weeks)")

if __name__ == "__main__":
    verify_fix()

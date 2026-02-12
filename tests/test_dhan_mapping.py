
import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

from Broker.Dhan.mapping import get_security_id

def test_mapping():
    test_cases = [
        ("SBIN", "NSE", "3045"),
        ("SBIN-EQ", "NSE", "3045"),
        ("INFY", "NSE", "1594"),
        ("RELIANCE", "NSE", "2885"),
        ("RELIANCE", "BSE", "500325"), # Just a guess for BSE, let's see
        ("SBIN", "NFO", "3045"), # Should map to NSE if not specific F&O symbol
    ]
    
    print(f"{'Symbol':<15} | {'Exchange':<10} | {'Expected':<10} | {'Got':<10} | {'Status'}")
    print("-" * 60)
    
    for symbol, exchange, expected in test_cases:
        got = get_security_id(symbol, exchange)
        status = "PASS" if got == expected or expected == "?" else "FAIL"
        if expected == "?":
             status = "CHECK"
        print(f"{symbol:<15} | {exchange:<10} | {expected:<10} | {got:<10} | {status}")

if __name__ == "__main__":
    test_mapping()

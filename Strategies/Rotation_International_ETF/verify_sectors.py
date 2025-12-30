
import sys
import os
import pandas as pd

# Add project root to path
# File is at: .../WEALTH_AI_BACKEND/Strategies/Rotation_International_ETF/verify_sectors.py
# 1. Rotation_International_ETF
# 2. Strategies
# 3. WEALTH_AI_BACKEND
backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(backend_root)

# Import the backtester class
try:
    from Strategies.Rotation_International_ETF.services.backtester import InternationalETFRotationBacktester
except ImportError as e:
    print(f"Import failed: {e}")
    # Fallback: try adding Strategies directly if running from backend root context
    strategies_path = os.path.join(backend_root, 'Strategies')
    sys.path.append(strategies_path)
    try:
        from Rotation_International_ETF.services.backtester import InternationalETFRotationBacktester
    except ImportError as e2:
        print(f"Fallback import failed: {e2}")
        raise

def verify_sectors():
    print("Initializing Backtester (mocking DB connection)...")
    
    # We can instantiate without DB connection if we mock the connection function or just handle the error
    # Actually, the constructor calls create_market_data_connection. 
    # Let's try to minimal init or just use the method if it was static (it's instance method).
    # We will try to instantiate. If DB fails, we might need a workaround.
    
    try:
        backtester = InternationalETFRotationBacktester()
    except Exception as e:
        print(f"Initialization failed (expected if DB not reachable): {e}")
        print("Patching DB init to bypass...")
        
        # Patch init to skip DB calls for this test
        original_init = InternationalETFRotationBacktester.__init__
        
        def mock_init(self, db_path=None):
            self.etf_metadata = {}
            self._verbose = False
            self.logger = None
            
        InternationalETFRotationBacktester.__init__ = mock_init
        backtester = InternationalETFRotationBacktester()
        # Restore for safety? Not needed in script process.

    print("\n--- Verifying Sector Classifications ---")
    
    test_tickers = [
        'SPY', 'QQQ', 'XLE', 'XLF', 'XLK', 'XLV', 
        'GLD', 'EEM', 'AGG', 'VNQ', 'XLY', 'XLP',
        'UNKNOWN_TICKER'
    ]
    
    results = []
    
    for ticker in test_tickers:
        sector = backtester.get_asset_sector_classification(ticker)
        print(f"{ticker:<15} -> {sector}")
        results.append((ticker, sector))

    # Basic validations
    expected_map = {
        'SPY': 'Other', # As per map 'Broad Market'/'Other'
        'QQQ': 'Technology',
        'XLE': 'Energy',
        'XLF': 'Financial',
        'XLK': 'Technology',
        'XLV': 'Healthcare', 
        'GLD': 'Commodities',
        'AGG': 'Bonds'
    }

    failed = False
    for ticker, expected in expected_map.items():
        actual = next(r[1] for r in results if r[0] == ticker)
        if actual != expected:
            # SPY mapped to 'Other' in my map, check if that matches expectation
            if ticker == 'SPY' and actual == 'Other': continue
            
            print(f"❌ Mismatch for {ticker}: Expected {expected}, Got {actual}")
            failed = True
            
    if not failed:
        print("\n✅ All critical sector mappings verified successfully!")
    else:
        print("\n❌ Verification failed for some sectors.")

if __name__ == "__main__":
    verify_sectors()

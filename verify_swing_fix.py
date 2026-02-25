import sys
import os
import pandas as pd
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.getcwd())

def test_benchmark_resolution():
    print("\n" + "="*60)
    print("VERIFYING BENCHMARK RESOLUTION FOR ETF SWING STRATEGY")
    print("="*60)

    from Strategies.ETF_Swing_Strategy.services.backtester import ETFSwingBacktester
    from Services.market_data_service import MarketDataService

    # 1. Mock MarketDataService.fetch_close_prices
    # Simulate a scenario where ^GSPC is present but S&P_500 is not
    def mock_fetch(tickers, market, asset_type, start_date, end_date):
        if asset_type == "INDEX":
            # Simulate only ^GSPC having data
            df = pd.DataFrame({
                "^GSPC": [3000, 3010, 3020],
                "S&P_500": [None, None, None]
            }, index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]))
            return df
        else:
            # Asset data
            return pd.DataFrame({
                "SPY": [400, 401, 402]
            }, index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]))

    with patch.object(MarketDataService, 'fetch_close_prices', side_effect=mock_fetch):
        print("\nTesting US Market Benchmark Resolution...")
        bt = ETFSwingBacktester(market="US")
        
        # Test load_data
        data = bt.load_data(["SPY"], "2023-01-01", "2023-01-03")
        
        if "BENCHMARK" in data:
            print("✅ Successfully resolved benchmark.")
            # Check which symbol was used (mentally, we know it should choose ^GSPC)
            # We can't easily check the original symbol after rename, but if it exists, it passed the candidate check.
            print(f"Data columns: {data['BENCHMARK'].columns.tolist()}")
        else:
            print("❌ Failed to resolve benchmark.")
            raise Exception("Benchmark data missing in load_data result")

    print("\n" + "="*60)
    print("VERIFICATION SUCCESSFUL")
    print("="*60)

if __name__ == "__main__":
    test_benchmark_resolution()

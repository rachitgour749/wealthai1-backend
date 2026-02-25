
import sys
import os
from sqlalchemy import text
from datetime import datetime

# Add project root to path
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)

from Databases.app_data_db_connection import create_connection, get_session

def check_data():
    if not create_connection():
        print("Failed to connect to database")
        return

    session = get_session()
    tickers = ['GLD', 'IWM', 'SLV', 'QQQ', 'VGT']
    benchmark_candidates = ["^GSPC", "S&P_500", "SPY", "SPX"]
    
    print("Checking US ETF Market Data:")
    for ticker in tickers:
        count = session.execute(text("SELECT count(*) FROM us_etf_market WHERE symbol = :symbol"), {"symbol": ticker}).scalar()
        if count > 0:
            date_range = session.execute(text("SELECT min(date), max(date) FROM us_etf_market WHERE symbol = :symbol"), {"symbol": ticker}).fetchone()
            print(f"  {ticker}: {count} rows, from {date_range[0]} to {date_range[1]}")
        else:
            print(f"  {ticker}: 0 rows")

    print("\nChecking US Benchmark Data:")
    for bench in benchmark_candidates:
        count = session.execute(text("SELECT count(*) FROM s_p_500_index_market WHERE symbol = :symbol"), {"symbol": bench}).scalar()
        if count > 0:
            date_range = session.execute(text("SELECT min(date), max(date) FROM s_p_500_index_market WHERE symbol = :symbol"), {"symbol": bench}).fetchone()
            print(f"  {bench}: {count} rows, from {date_range[0]} to {date_range[1]}")
        else:
            print(f"  {bench}: 0 rows")
    
    # Also check if there are ANY symbols in us_etf_market
    total_etfs = session.execute(text("SELECT count(DISTINCT symbol) FROM us_etf_market")).scalar()
    print(f"\nTotal unique symbols in us_etf_market: {total_etfs}")
    if total_etfs > 0:
        sample_symbols = session.execute(text("SELECT DISTINCT symbol FROM us_etf_market LIMIT 5")).fetchall()
        print(f"Sample symbols: {[s[0] for s in sample_symbols]}")

    session.close()

if __name__ == "__main__":
    check_data()

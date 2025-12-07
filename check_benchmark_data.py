import sys
sys.path.insert(0, r'c:\Users\Lenovo\Desktop\WEALTHAI_PROD\wealthai1-backend')

from Databases.market_data_db_connection import create_connection, get_session
from sqlalchemy import text

# Initialize connection
if not create_connection():
    print("Failed to connect to database")
    sys.exit(1)

session = get_session()

# Check which benchmark symbols exist and have data
benchmark_symbols = [
    'NIFTYBEES.NS', 'NIFTYBEES', 'NIFTY50', 'SENSEX',
    'BANKBEES.NS', 'BANKBEES', 'JUNIORBEES.NS', 'JUNIORBEES',
    'ITBEES.NS', 'ITBEES'
]

print("Checking benchmark symbols in etf_data table:\n")

for symbol in benchmark_symbols:
    query = text("""
        SELECT 
            COUNT(*) as total_rows,
            MIN(date) as start_date,
            MAX(date) as end_date
        FROM etf_data
        WHERE symbol = :symbol
    """)
    
    result = session.execute(query, {"symbol": symbol})
    row = result.fetchone()
    
    if row and row[0] > 0:
        print(f"✅ {symbol:20s}: {row[0]:6d} rows, {row[1]} to {row[2]}")
    else:
        print(f"❌ {symbol:20s}: NO DATA")

# Check for the specific date range used in backtest
print("\n" + "="*70)
print("Checking data availability for 2021-07-19 to 2025-12-01:\n")

for symbol in benchmark_symbols:
    query = text("""
        SELECT COUNT(*) as count
        FROM etf_data
        WHERE symbol = :symbol
        AND date >= '2021-07-19'
        AND date <= '2025-12-01'
    """)
    
    result = session.execute(query, {"symbol": symbol})
    count = result.fetchone()[0]
    
    if count > 0:
        print(f"✅ {symbol:20s}: {count:6d} rows in date range")

session.close()

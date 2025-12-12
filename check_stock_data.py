
import os
import sys
import pandas as pd
from sqlalchemy import text

# Add database path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'Databases')))

try:
    from app_data_db_connection import get_engine
    
    # Try using the market_data_db_connection if available, as that's what the backtester uses
    try:
        sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'Strategies', 'Rotation_Stocks', 'services')))
        from market_data_db_connection import create_connection, get_session
        print("Using market_data_db_connection")
        
        if not create_connection():
            print("Failed to create connection")
            sys.exit(1)
            
        session = get_session()
        
        # Check stock_data count
        print("Checking stock_data table...")
        result = session.execute(text("SELECT count(*) FROM stock_data"))
        count = result.scalar()
        print(f"Total rows in stock_data: {count}")
        
        if count > 0:
            # Check distinct symbols
            result = session.execute(text("SELECT DISTINCT symbol FROM stock_data LIMIT 20"))
            symbols = [row[0] for row in result.fetchall()]
            print(f"Sample symbols: {symbols}")
            
            # Check date range for a sample symbol
            if symbols:
                symbol = symbols[0]
                result = session.execute(text("SELECT min(date), max(date) FROM stock_data WHERE symbol = :symbol"), {"symbol": symbol})
                row = result.fetchone()
                print(f"Date range for {symbol}: {row[0]} to {row[1]}")
        
        session.close()
        
    except ImportError as e:
        print(f"Could not import market_data_db_connection: {e}")
        # Fallback to general app DB if needed, but backtester triggers likely use market DB
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

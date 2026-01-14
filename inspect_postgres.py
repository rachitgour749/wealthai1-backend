import sys
import os
from sqlalchemy import text

# Add parent directory to path to import Databases
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Databases.market_data_db_connection import create_connection, get_session

def inspect_postgres(symbol='NIFTYBEES'):
    if not create_connection():
        print("Failed to connect to PostgreSQL")
        return
        
    session = get_session()
    try:
        query = text("SELECT COUNT(*), MIN(date), MAX(date) FROM etf_data WHERE symbol = :symbol")
        result = session.execute(query, {"symbol": symbol}).fetchone()
        print(f"--- etf_data for {symbol} ---")
        print(f"Count: {result[0]}")
        print(f"Min Date: {result[1]}")
        print(f"Max Date: {result[2]}")
        
        # Check index_data too
        query = text("SELECT symbol, COUNT(*), MIN(date), MAX(date) FROM index_data GROUP BY symbol")
        print("\n--- index_data Summary ---")
        for row in session.execute(query):
            print(f"Symbol: {row[0]}, Count: {row[1]}, Range: {row[2]} to {row[3]}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    inspect_postgres()

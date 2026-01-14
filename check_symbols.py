import sys
import os
from sqlalchemy import text
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Databases.market_data_db_connection import create_connection, get_session

def check_etf_data(symbol):
    if not create_connection():
        print("Failed to connect to database")
        return
    
    session = get_session()
    try:
        query = text(f"SELECT COUNT(*), MIN(date), MAX(date) FROM etf_data WHERE symbol = :symbol")
        result = session.execute(query, {"symbol": symbol})
        count, min_date, max_date = result.fetchone()
        print(f"Stats for {symbol}: Count={count}, Range={min_date} to {max_date}")
        
    finally:
        session.close()

if __name__ == "__main__":
    check_etf_data("NIFTYBEES.NS")
    check_etf_data("NIFTYBEES")
    check_etf_data("BANKBEES.NS")
    check_etf_data("BANKBEES")

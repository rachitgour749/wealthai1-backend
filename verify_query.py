
import sys
import os
import logging
from datetime import datetime, timedelta
from sqlalchemy import select

# Add project root to path
sys.path.append(os.getcwd())

from Databases.app_data_db_connection import get_session, create_connection, ETFMarket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_query():
    if not create_connection():
        print("Failed DB connection")
        return

    import sqlalchemy
    print(f"SQLAlchemy Version: {sqlalchemy.__version__}")

    session = get_session()
    try:
        tickers = ['NIFTYBEES', 'GOLDBEES', 'MON100']
        
        signal_date = datetime.now()
        lookback_start = signal_date - timedelta(days=400)
        
        print("\n--- Test 1: Single Value Filter ---")
        one_etf = session.query(ETFMarket).filter(ETFMarket.symbol == 'NIFTYBEES').first()
        print(f"Single filter result: {one_etf.symbol if one_etf else 'None'}")

        print("\n--- Test 2: List IN Filter ---")
        data = session.query(ETFMarket).filter(
            ETFMarket.symbol.in_(tickers),
            ETFMarket.date >= lookback_start
        ).limit(5).all()
        print(f"Success! Found {len(data)} rows.")
        for d in data:
            print(f" - {d.symbol} {d.date}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()

if __name__ == "__main__":
    verify_query()

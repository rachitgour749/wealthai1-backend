from Databases.app_data_db_connection import get_session, Nifty50IndexMarket, SP500IndexMarket, create_connection
from sqlalchemy import func

def check_symbols():
    if not create_connection():
        print("Failed to connect to database")
        return
        
    session = get_session()
    try:
        nifty_symbols = session.query(Nifty50IndexMarket.symbol).distinct().all()
        print(f"Nifty 50 Symbols: {[s[0] for s in nifty_symbols]}")
        
        sp500_symbols = session.query(SP500IndexMarket.symbol).distinct().all()
        print(f"S&P 500 Symbols: {[s[0] for s in sp500_symbols]}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    check_symbols()

import os
import sys
from sqlalchemy import text
from dotenv import load_dotenv

# Add project root to path
project_root = 'd:\\WEALTHAI_V2\\wealthai-backend-v2'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Databases.app_data_db_connection import create_connection, get_session

def check_strategies():
    if not create_connection():
        print("Failed to connect to database")
        return

    session = get_session()
    try:
        result = session.execute(text("SELECT DISTINCT strategy_name, strategy_type FROM saved_instances;"))
        rows = result.fetchall()
        print("Strategy Name | Strategy Type")
        print("-" * 50)
        for row in rows:
            print(f"{row[0]} | {row[1]}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    check_strategies()

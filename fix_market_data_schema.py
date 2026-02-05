
import sys
import os
import logging
from sqlalchemy import text, inspect

# Add project root to path
sys.path.append(os.getcwd())

from Databases.app_data_db_connection import get_engine, create_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_schema():
    if not create_connection():
        print("Failed to connect")
        return

    engine = get_engine()
    inspector = inspect(engine)
    
    tables_to_check = [
        'etf_market', 
        'stock_market', 
        'nifty_50_index_market',
        's_p_500_index_market',
        'us_etf_market'
    ]
    
    with engine.connect() as conn:
        for table in tables_to_check:
            if not inspector.has_table(table):
                print(f"Table {table} does not exist.")
                continue
                
            columns = [c['name'] for c in inspector.get_columns(table)]
            if 'created_at' not in columns:
                print(f"⚠️ Missing 'created_at' in {table}. Adding it...")
                try:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN created_at TIMESTAMP DEFAULT NOW()"))
                    conn.commit()
                    print(f"✅ Added 'created_at' to {table}")
                except Exception as e:
                    print(f"❌ Failed to add column: {e}")
            else:
                print(f"✅ {table} has 'created_at'")

if __name__ == "__main__":
    fix_schema()

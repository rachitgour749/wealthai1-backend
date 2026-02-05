import sys
import os
from sqlalchemy import inspect
from sqlalchemy import text

# Add project root to path
sys.path.append(os.getcwd())

from Databases.app_data_db_connection import get_session, Base, create_connection, get_engine
from Databases.signal_models import TradingSignal

def verify_schema():
    if not create_connection():
        print("Failed to connect to database.")
        sys.exit(1)
        
    engine = get_engine()
    inspector = inspect(engine)
    table_name = 'trading_signals'
    
    if inspector.has_table(table_name):
        print(f"Table '{table_name}' EXISTS.")
        columns = [c['name'] for c in inspector.get_columns(table_name)]
        print(f"Existing columns: {columns}")
        
        # Check for new required columns
        required_columns = ['run_id', 'high_52', 'low_52', 'strategy_type']
        missing = [c for c in required_columns if c not in columns]
        
        if missing:
            print(f"⚠️ MISSING columns: {missing}")
            print("Running migration: Dropping and Recreating table 'trading_signals'...")
            try:
                # Drop specific table
                TradingSignal.__table__.drop(engine)
                print("Table dropped.")
                
                # Recreate
                Base.metadata.create_all(engine)
                print("✅ Table 'trading_signals' recreated successfully with new schema.")
            except Exception as e:
                print(f"❌ Migration failed: {e}")
        else:
            print("✅ All required columns are present.")
    else:
        print(f"Table '{table_name}' does NOT exist. Creating it now...")
        Base.metadata.create_all(engine)
        print(f"✅ Table '{table_name}' created successfully.")

if __name__ == "__main__":
    verify_schema()

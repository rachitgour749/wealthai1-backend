#!/usr/bin/env python3
"""
Database setup script for nifty500_data_with_metadata.sqlite
Creates all required tables for the RS Strategy platform
"""

import sqlite3
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import engine, Base, StockData, IndexData, Nifty500Constituents, StrategyConfig, BacktestResult, TradeLog, PortfolioSnapshot, RSLiveSignal
from sqlalchemy import inspect

def setup_database():
    """Set up all required tables in the database"""
    
    print("🚀 Setting up RS Strategy Database")
    print("=" * 50)
    print(f"Database: nifty500_data_with_metadata.sqlite")
    
    try:
        # Create all tables
        print("\n📊 Creating database tables...")
        Base.metadata.create_all(bind=engine)
        print("✅ All tables created successfully!")
        
        # Verify tables were created
        print("\n🔍 Verifying table creation...")
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        expected_tables = [
            'stock_data',
            'index_data', 
            'nifty500_metadata',
            'strategy_config',
            'backtest_results',
            'trade_logs',
            'portfolio_snapshots',
            'rs_live'
        ]
        
        print(f"\n📋 Found {len(tables)} tables:")
        for table in tables:
            status = "✅" if table in expected_tables else "⚠️"
            print(f"  {status} {table}")
        
        # Check for missing tables
        missing_tables = set(expected_tables) - set(tables)
        if missing_tables:
            print(f"\n❌ Missing tables: {missing_tables}")
        else:
            print(f"\n🎉 All {len(expected_tables)} required tables are present!")
        
        # Show table schemas
        print("\n📋 Table Schemas:")
        print("=" * 50)
        
        for table_name in expected_tables:
            if table_name in tables:
                columns = inspector.get_columns(table_name)
                print(f"\n🔹 {table_name}:")
                for col in columns:
                    col_type = col['type']
                    nullable = "NULL" if col['nullable'] else "NOT NULL"
                    default = f" DEFAULT {col['default']}" if col['default'] is not None else ""
                    print(f"   - {col['name']}: {col_type} {nullable}{default}")
        
        print("\n" + "=" * 50)
        print("✅ Database setup complete!")
        print("\n📝 Next steps:")
        print("1. Populate stock_data with Nifty 500 price data")
        print("2. Populate index_data with Nifty 50 index data")
        print("3. Populate nifty500_metadata with stock metadata")
        print("4. Create strategy configurations")
        print("5. Run backtests")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        return False

def check_existing_data():
    """Check if there's existing data in the database"""
    
    print("\n🔍 Checking existing data...")
    
    try:
        # Connect to database
        conn = sqlite3.connect('nifty500_data_with_metadata.sqlite')
        cursor = conn.cursor()
        
        # Check each table
        tables_to_check = [
            'stock_data',
            'index_data', 
            'nifty500_metadata',
            'strategy_config',
            'backtest_results',
            'trade_logs',
            'portfolio_snapshots',
            'rs_live'
        ]
        
        for table in tables_to_check:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table};")
                count = cursor.fetchone()[0]
                status = "📊" if count > 0 else "📭"
                print(f"  {status} {table}: {count:,} records")
            except sqlite3.OperationalError:
                print(f"  ❌ {table}: Table does not exist")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking data: {e}")

if __name__ == "__main__":
    success = setup_database()
    if success:
        check_existing_data()

#!/usr/bin/env python3
"""
Database setup script for PostgreSQL MarketData database
Creates all required tables for the RS Strategy platform
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from market data database connection
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
databases_path = os.path.join(parent_dir, 'Databases')
sys.path.insert(0, databases_path)

from app_data_db_connection import (
    create_connection,
    get_engine,
    init_database,
    Base,
    StockMarket as StockData,
    Nifty50IndexMarket as IndexData
)

# Import strategy-specific models from local database.py
from database import (
    StrategyConfig,
    BacktestResult,
    TradeLog,
    PortfolioSnapshot,
    RSLiveSignal
)

from sqlalchemy import inspect

def setup_database():
    """Set up all required tables in the PostgreSQL MarketData database"""
    
    print("🚀 Setting up RS Strategy Database")
    print("=" * 50)
    print(f"Database: PostgreSQL ApplicationData (Neon)")
    
    try:
        # Ensure connection is established
        if not create_connection():
            print("❌ Failed to connect to PostgreSQL ApplicationData database")
            return False
        
        # Initialize all tables
        print("\n📊 Creating database tables...")
        if not init_database():
            print("❌ Failed to initialize database tables")
            return False
        
        # Create all tables using Base metadata
        engine = get_engine()
        Base.metadata.create_all(bind=engine)
        print("✅ All tables created successfully!")
        
        # Verify tables were created
        print("\n🔍 Verifying table creation...")
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        # Market data tables (in ApplicationData database)
        market_data_tables = [
            'stock_market',
            'etf_market',
            'nifty_50_index_market', 
            's_p_500_index_market',
            'us_etf_market'
        ]
        
        # Strategy tables (also in MarketData database)
        strategy_tables = [
            'strategy_config',
            'backtest_results',
            'trade_logs',
            'portfolio_snapshots',
            'rs_live'
        ]
        
        expected_tables = market_data_tables + strategy_tables
        
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
                    col_type = str(col['type'])
                    nullable = "NULL" if col['nullable'] else "NOT NULL"
                    default = f" DEFAULT {col['default']}" if col['default'] is not None else ""
                    print(f"   - {col['name']}: {col_type} {nullable}{default}")
        
        print("\n" + "=" * 50)
        print("✅ Database setup complete!")
        print("\n📝 Next steps:")
        print("1. Populate stock_data with Nifty 500 price data")
        print("2. Populate stock_metadata with stock metadata")
        print("3. Populate index_data with Nifty 50 index data")
        print("4. Populate nifty500_metadata with Nifty 500 metadata")
        print("5. Create strategy configurations")
        print("6. Run backtests")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_existing_data():
    """Check if there's existing data in the PostgreSQL MarketData database"""
    
    print("\n🔍 Checking existing data...")
    
    try:
        from app_data_db_connection import get_session
        from sqlalchemy import text
        
        session = get_session()
        
        # Check each table
        tables_to_check = [
            'stock_market',
            'etf_market',
            'nifty_50_index_market', 
            's_p_500_index_market',
            'us_etf_market',
            'strategy_config',
            'backtest_results',
            'trade_logs',
            'portfolio_snapshots',
            'rs_live'
        ]
        
        for table in tables_to_check:
            try:
                result = session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.scalar()
                status = "📊" if count > 0 else "📭"
                print(f"  {status} {table}: {count:,} records")
            except Exception as e:
                print(f"  ❌ {table}: Error - {str(e)}")
        
        session.close()
        
    except Exception as e:
        print(f"❌ Error checking data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    success = setup_database()
    if success:
        check_existing_data()

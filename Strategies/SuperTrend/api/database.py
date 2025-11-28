"""
Database connection and session management
Migrated from SQLite to PostgreSQL
"""
from typing import Optional
import pandas as pd
import sys
import os

# Add Databases path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
databases_path = os.path.join(parent_dir, 'Databases')
sys.path.insert(0, databases_path)

from Databases.app_data_db_connection import get_session, create_connection, init_database
from Databases.market_data_db_connection import get_session as get_market_data_session, create_connection as create_market_data_connection
from Databases.strategy_models import (
    SuperTrendStrategyConfig, SuperTrendBacktestResult, 
    SuperTrendCurrentPosition, SuperTrendCandidate
)
from sqlalchemy import text


def get_connection():
    """Get PostgreSQL session (kept for compatibility, returns session)"""
    return get_session()


def execute_query(query: str, params: Optional[tuple] = None) -> pd.DataFrame:
    """Execute a query and return results as DataFrame (PostgreSQL)"""
    # Determine which database to use based on table name
    query_lower = query.lower()
    if 'stock_data' in query_lower or 'index_data' in query_lower:
        # Use MarketData database for market data
        session = get_market_data_session()
        try:
            # Convert SQLite placeholders (?) to PostgreSQL (%s)
            pg_query = query.replace('?', '%s')
            if params:
                result = session.execute(text(pg_query), params)
            else:
                result = session.execute(text(pg_query))
            rows = result.fetchall()
            columns = list(result.keys())
            df = pd.DataFrame(rows, columns=columns)
            return df
        finally:
            session.close()
    else:
        # Use ApplicationData database for strategy tables
        session = get_session()
        try:
            # Convert SQLite placeholders (?) to PostgreSQL (%s)
            pg_query = query.replace('?', '%s')
            if params:
                result = session.execute(text(pg_query), params)
            else:
                result = session.execute(text(pg_query))
            rows = result.fetchall()
            columns = list(result.keys())
            df = pd.DataFrame(rows, columns=columns)
            return df
        finally:
            session.close()


def execute_write(query: str, params: Optional[tuple] = None):
    """Execute a write query (INSERT, UPDATE, DELETE) - PostgreSQL"""
    # Determine which database to use based on table name
    if 'stock_data' in query.lower() or 'index_data' in query.lower():
        # Use MarketData database for market data
        session = get_market_data_session()
        try:
            # Convert SQLite placeholders (?) to PostgreSQL (%s)
            pg_query = query.replace('?', '%s')
            if params:
                session.execute(text(pg_query), params)
            else:
                session.execute(text(pg_query))
            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
    else:
        # Use ApplicationData database for strategy tables
        session = get_session()
        try:
            # Convert SQLite placeholders (?) to PostgreSQL (%s)
            pg_query = query.replace('?', '%s')
            if params:
                session.execute(text(pg_query), params)
            else:
                session.execute(text(pg_query))
            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()


def init_database():
    """Initialize database tables in PostgreSQL"""
    # Ensure connections are established
    if not create_connection():
        print("Failed to connect to ApplicationData database")
        return False
    
    if not create_market_data_connection():
        print("Failed to connect to MarketData database")
        return False
    
    # Initialize ApplicationData tables (strategy-specific)
    if not init_database():
        print("Failed to initialize ApplicationData database tables")
        return False
    
    # Note: stock_data and index_data tables are in MarketData database
    # and are managed by market_data_db_connection
    
    # Insert default config if not exists
    session = get_session()
    try:
        config_count = session.query(SuperTrendStrategyConfig).count()
        if config_count == 0:
            default_config = SuperTrendStrategyConfig(
                ema_short=10,
                ema_long=20,
                supertrend_period=10,
                supertrend_stop_pct=10.0,
                max_holdings=5,
                buffer_pct=10.0,
                price_floor=50.0,
                liquidity_cr=10.0,
                rs_window_1=5,
                rs_window_2=21,
                rs_window_3=63,
                benchmark='NIFTY50',
                universe='NIFTY200'
            )
            session.add(default_config)
            session.commit()
            print("Default SuperTrend strategy config created")
    finally:
        session.close()
    
    print("SuperTrend database initialized successfully in PostgreSQL")
    return True


if __name__ == "__main__":
    init_database()


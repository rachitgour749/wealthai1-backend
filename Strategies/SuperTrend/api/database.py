"""
Database connection and session management
"""
import sqlite3
from pathlib import Path
from typing import Optional
import pandas as pd

# Database path
DB_PATH = Path(__file__).parent.parent / "strategy_data.sqlite"


def get_connection():
    """Get SQLite connection"""
    return sqlite3.connect(str(DB_PATH))


def execute_query(query: str, params: Optional[tuple] = None) -> pd.DataFrame:
    """Execute a query and return results as DataFrame"""
    conn = get_connection()
    try:
        if params:
            df = pd.read_sql_query(query, conn, params=params)
        else:
            df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()


def execute_write(query: str, params: Optional[tuple] = None):
    """Execute a write query (INSERT, UPDATE, DELETE)"""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        conn.commit()
    finally:
        conn.close()


def init_database():
    """Initialize database tables"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create stock_data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adj_close REAL,
            volume INTEGER,
            UNIQUE(symbol, date)
        )
    """)
    
    # Create index_data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS index_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL UNIQUE,
            adj_close REAL
        )
    """)
    
    # Create strategy_config table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS strategy_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ema_short INTEGER DEFAULT 10,
            ema_long INTEGER DEFAULT 20,
            supertrend_period INTEGER DEFAULT 10,
            supertrend_stop_pct REAL DEFAULT 10.0,
            max_holdings INTEGER DEFAULT 5,
            buffer_pct REAL DEFAULT 10.0,
            price_floor REAL DEFAULT 50.0,
            liquidity_cr REAL DEFAULT 10.0,
            rs_window_1 INTEGER DEFAULT 5,
            rs_window_2 INTEGER DEFAULT 21,
            rs_window_3 INTEGER DEFAULT 63,
            benchmark TEXT DEFAULT 'NIFTY50',
            universe TEXT DEFAULT 'NIFTY200'
        )
    """)
    
    # Add buffer_pct column if it doesn't exist (for existing databases)
    try:
        cursor.execute("ALTER TABLE strategy_config ADD COLUMN buffer_pct REAL DEFAULT 10.0")
        conn.commit()
    except sqlite3.OperationalError:
        # Column already exists
        pass
    
    # Create backtest_results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TEXT,
            trade_date TEXT,
            symbol TEXT,
            action TEXT,
            price REAL,
            quantity INTEGER,
            position_value REAL,
            portfolio_value REAL,
            cash REAL
        )
    """)
    
    # Create current_positions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS current_positions (
            symbol TEXT PRIMARY KEY,
            entry_date TEXT,
            entry_price REAL,
            quantity INTEGER,
            current_price REAL,
            current_value REAL,
            pnl REAL,
            pnl_pct REAL
        )
    """)
    
    # Create candidates table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            symbol TEXT PRIMARY KEY,
            date TEXT,
            adj_close REAL,
            ema10 REAL,
            ema20 REAL,
            supertrend TEXT,
            rs_score REAL,
            rank INTEGER,
            eligible INTEGER
        )
    """)
    
    # Insert default config if not exists
    cursor.execute("SELECT COUNT(*) FROM strategy_config")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO strategy_config 
            (ema_short, ema_long, supertrend_period, supertrend_stop_pct, max_holdings, buffer_pct,
             price_floor, liquidity_cr, rs_window_1, rs_window_2, rs_window_3, benchmark, universe)
            VALUES (10, 20, 10, 10.0, 5, 10.0, 50.0, 10.0, 5, 21, 63, 'NIFTY50', 'NIFTY200')
        """)
    
    conn.commit()
    conn.close()
    
    print("Database initialized successfully")


if __name__ == "__main__":
    init_database()


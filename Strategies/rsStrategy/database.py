from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import asyncio
import time
import logging
from contextlib import contextmanager

# Database URL
DATABASE_URL = "sqlite:///./Strategies/rsStrategy/nifty500_data_with_metadata.sqlite"

# Enhanced SQLite configuration to prevent locking issues
engine = create_engine(
    DATABASE_URL, 
    connect_args={
        "check_same_thread": False,
        "timeout": 30,  # 30 second timeout
        "isolation_level": None  # Autocommit mode to reduce locking
    },
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600,   # Recycle connections every hour
    echo=False
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency to get database session with enhanced error handling
def get_db():
    db = None
    try:
        db = SessionLocal()
        yield db
    except Exception as e:
        logging.error(f"Database session error: {e}")
        if db:
            db.rollback()
        raise
    finally:
        if db:
            try:
                db.close()
            except Exception as e:
                logging.error(f"Error closing database session: {e}")

# Stock data model
class StockData(Base):
    __tablename__ = "stock_data"
    
    symbol = Column(String, primary_key=True, index=True)
    date = Column(DateTime, primary_key=True, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Integer)
    
# Index data model
class IndexData(Base):
    __tablename__ = "index_data"
    
    symbol = Column(String, primary_key=True, index=True)  # e.g., "^NSEI" for Nifty 50
    date = Column(DateTime, primary_key=True, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Integer)

# Nifty 500 metadata
class Nifty500Constituents(Base):
    __tablename__ = "nifty500_metadata"
    
    symbol = Column(String, primary_key=True, index=True)
    start_date = Column(String)  # DATE as string
    end_date = Column(String)    # DATE as string
    total_records = Column(Integer)
    last_updated = Column(String)  # DATE as string
    data_source = Column(String)
    years_available = Column(Float)

# Strategy configuration
class StrategyConfig(Base):
    __tablename__ = "strategy_config"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)  # User-specific configuration
    config_name = Column(String)  # Removed unique constraint to allow same name per user
    main_index = Column(String, default="NIFTY50")  # Nifty 50
    stock_universe = Column(String)  # Add missing stock_universe field
    lookback_weeks = Column(Integer, default=5)
    lookback_months = Column(Integer, default=20)
    lookback_quarters = Column(Integer, default=60)
    max_positions = Column(Integer, default=20)
    position_size_pct = Column(Float, default=5.0)  # 5% per position
    buffer_capital_pct = Column(Float, default=10.0)  # 10% buffer
    total_capital = Column(Float, default=1000000)  # 10 Lakh
    stop_loss_pct = Column(Float, default=15.0)  # 15% stop loss
    capital_reset_threshold_pct = Column(Float, default=25.0)  # 25% reset threshold
    max_holding_period = Column(Integer, default=52)  # 52 weeks
    min_turnover = Column(Float, default=1000000)  # Min daily turnover
    min_price = Column(Float, default=10.0)  # Min price filter
    transaction_cost_pct = Column(Float, default=0.1)  # 0.1% transaction cost
    is_active = Column(Boolean, default=True)
    created_at = Column(String)  # Store as string to match database schema
    updated_at = Column(String)  # Store as string to match database schema

# Backtest results
class BacktestResult(Base):
    __tablename__ = "backtest_results"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)  # User-specific results
    config_id = Column(Integer, index=True, nullable=True)  # Allow None for custom config backtests
    start_date = Column(DateTime, index=True)
    end_date = Column(DateTime, index=True)
    total_return_pct = Column(Float)
    annualized_return_pct = Column(Float)
    cagr_pct = Column(Float)
    xirr_pct = Column(Float)
    max_drawdown_pct = Column(Float)
    sharpe_ratio = Column(Float)
    beta = Column(Float)
    treynor_ratio = Column(Float)
    calmar_ratio = Column(Float)
    win_rate_pct = Column(Float)
    total_trades = Column(Integer)
    avg_holding_period = Column(Float)
    final_capital = Column(Float)
    created_at = Column(DateTime)

# Trade logs
class TradeLog(Base):
    __tablename__ = "trade_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)  # User-specific trades
    backtest_id = Column(Integer, index=True)
    date = Column(DateTime, index=True)
    symbol = Column(String, index=True)
    action = Column(String)  # BUY, SELL
    quantity = Column(Integer)
    price = Column(Float)
    amount = Column(Float)
    reason = Column(String)  # Entry, Exit, Stop Loss, etc.
    rs_score = Column(Float)
    rs_rank = Column(Integer)
    
    # Detailed transaction cost breakdown
    transaction_value = Column(Float)  # Base transaction value (quantity * price)
    brokerage = Column(Float)  # Brokerage charges
    stt = Column(Float)  # Securities Transaction Tax (sell only)
    stamp_duty = Column(Float)  # Stamp duty (buy only)
    exchange_charges = Column(Float)  # Exchange charges
    sebi_charges = Column(Float)  # SEBI charges
    gst = Column(Float)  # GST on brokerage
    total_costs = Column(Float)  # Total transaction costs
    net_amount = Column(Float)  # Net amount after all costs

# Portfolio snapshots
class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)  # User-specific snapshots
    backtest_id = Column(Integer, index=True)
    date = Column(DateTime, index=True)
    total_value = Column(Float)
    cash_balance = Column(Float)
    positions_json = Column(Text)  # JSON string of positions
    daily_pnl = Column(Float)
    cumulative_pnl = Column(Float)
    drawdown_pct = Column(Float)

# RS Live signals
class RSLiveSignal(Base):
    __tablename__ = "rs_live"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)  # User-specific signals
    signal_date = Column(DateTime, index=True)
    symbol = Column(String, index=True)
    action = Column(String)  # BUY, SELL, HOLD
    rs_score = Column(Float)
    rs_rank = Column(Integer)
    price = Column(Float)
    reason = Column(String)
    config_id = Column(Integer, index=True)
    run_id = Column(String, index=True)  # NEW: Foreign key to saved_rs_strategies
    webhook = Column(String)  # NEW: Webhook URL
    client_information = Column(String)  # NEW: Client details (JSON)
    buy_symbol_json = Column(String)  # NEW: JSON array of buy symbols
    sell_symbol_json = Column(String)  # NEW: JSON array of sell symbols
    update_price_json = Column(String)  # NEW: JSON with price updates
    created_at = Column(DateTime, default=datetime.now)

# Backtest database setup with enhanced configuration
BACKTEST_DATABASE_URL = "sqlite:///./Strategies/rsStrategy/nifty500_data_with_metadata.sqlite"
backtest_engine = create_engine(
    BACKTEST_DATABASE_URL, 
    connect_args={
        "check_same_thread": False,
        "timeout": 30,  # 30 second timeout
        "isolation_level": None  # Autocommit mode to reduce locking
    },
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600,   # Recycle connections every hour
    echo=False
)
BacktestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=backtest_engine)

# Utility functions
async def execute_with_retry(func, max_retries=3, delay=0.5):
    """Execute a function with retry logic for database operations"""
    for attempt in range(max_retries):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            else:
                return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            logging.warning(f"Database operation failed (attempt {attempt + 1}/{max_retries}): {e}")
            await asyncio.sleep(delay)
            delay *= 2  # Exponential backoff

def check_database_health():
    """Check if database is accessible"""
    try:
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        return True
    except Exception as e:
        logging.error(f"Database health check failed: {e}")
        return False

async def save_backtest_result_safely(result_data, max_retries=5, retry_delay=2):
    """Save backtest result with enhanced error handling and retry logic"""
    db = None
    for attempt in range(max_retries):
        try:
            db = BacktestSessionLocal()
            result = BacktestResult(**result_data)
            db.add(result)
            db.commit()
            db.refresh(result)
            result_id = result.id
            db.close()
            db = None
            logging.info(f"Successfully saved backtest result with ID: {result_id}")
            return result
        except Exception as e:
            if db:
                try:
                    db.rollback()
                    db.close()
                except Exception as rollback_error:
                    logging.error(f"Error during rollback: {rollback_error}")
                finally:
                    db = None
            
            if attempt == max_retries - 1:
                logging.error(f"Failed to save backtest result after {max_retries} attempts: {e}")
                # Force unlock database before final failure
                force_unlock_database()
                raise
            else:
                logging.warning(f"Failed to save backtest result (attempt {attempt + 1}/{max_retries}): {e}")
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5  # Gradual backoff

async def save_additional_data_safely(trade_logs=None, portfolio_snapshots=None, max_retries=3, retry_delay=1):
    """Save trade logs and portfolio snapshots with enhanced error handling and retry logic"""
    db = None
    for attempt in range(max_retries):
        try:
            db = BacktestSessionLocal()
            
            if trade_logs:
                for log_data in trade_logs:
                    log = TradeLog(**log_data)
                    db.add(log)
            
            if portfolio_snapshots:
                for snapshot_data in portfolio_snapshots:
                    snapshot = PortfolioSnapshot(**snapshot_data)
                    db.add(snapshot)
            
            db.commit()
            db.close()
            db = None
            logging.info(f"Successfully saved additional data: {len(trade_logs or [])} trades, {len(portfolio_snapshots or [])} snapshots")
            return True
            
        except Exception as e:
            if db:
                try:
                    db.rollback()
                    db.close()
                except Exception as rollback_error:
                    logging.error(f"Error during rollback: {rollback_error}")
                finally:
                    db = None
            
            if attempt == max_retries - 1:
                logging.error(f"Failed to save additional data after {max_retries} attempts: {e}")
                # Force unlock database before final failure
                force_unlock_database()
                raise
            else:
                logging.warning(f"Failed to save additional data (attempt {attempt + 1}/{max_retries}): {e}")
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5  # Gradual backoff

def reset_database_connections():
    """Reset database connections"""
    try:
        engine.dispose()
        backtest_engine.dispose()
        logging.info("Database connections reset successfully")
    except Exception as e:
        logging.error(f"Failed to reset database connections: {e}")

def force_unlock_database():
    """Force unlock database by resetting connections and clearing lock files"""
    try:
        # Reset all database connections
        reset_database_connections()
        
        # Wait for connections to close
        time.sleep(0.5)
        
        # Try to remove any SQLite lock files (if they exist)
        db_path = "./Strategies/rsStrategy/nifty500_data_with_metadata.sqlite"
        lock_files = [f"{db_path}-wal", f"{db_path}-shm", f"{db_path}-journal"]
        
        for lock_file in lock_files:
            try:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
                    logging.info(f"Removed lock file: {lock_file}")
            except Exception as e:
                logging.warning(f"Could not remove lock file {lock_file}: {e}")
        
        logging.info("Database unlocked successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to unlock database: {e}")
        return False

@contextmanager
def get_backtest_session():
    """Context manager for backtest database sessions with proper cleanup"""
    db = None
    try:
        db = BacktestSessionLocal()
        yield db
    except Exception as e:
        if db:
            db.rollback()
        logging.error(f"Database session error: {e}")
        raise
    finally:
        if db:
            try:
                db.close()
            except Exception as e:
                logging.error(f"Error closing database session: {e}")

@contextmanager
def get_main_session():
    """Context manager for main database sessions with proper cleanup"""
    db = None
    try:
        db = SessionLocal()
        yield db
    except Exception as e:
        if db:
            db.rollback()
        logging.error(f"Database session error: {e}")
        raise
    finally:
        if db:
            try:
                db.close()
            except Exception as e:
                logging.error(f"Error closing database session: {e}")

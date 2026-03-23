from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import sys
import asyncio
import time
import logging
from contextlib import contextmanager

# Import market data database connection for market data tables
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
databases_path = os.path.join(parent_dir, 'Databases')
sys.path.insert(0, databases_path)

from app_data_db_connection import (
    create_connection as create_app_data_connection,
    get_session as get_app_data_session,
    get_engine as get_app_data_engine,
    init_database as init_app_data_database,
    Base,
    StockMarket as StockData,
    Nifty50IndexMarket as IndexData
)

# Use application data connection for all tables
# Initialize connection
if not create_app_data_connection():
    raise RuntimeError("Failed to connect to ApplicationData database")

engine = get_app_data_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base is imported from app_data_db_connection

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

# Stock data model - using models from app_data_db_connection
# StockData (StockMarket) and IndexData (Nifty50IndexMarket) are imported from app_data_db_connection


# Backtest results
class BacktestResult(Base):
    __tablename__ = "backtest_results"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)  # User-specific backtest
    config_id = Column(Integer, index=True, nullable=True) # Linked config
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
    avg_holding_period = Column(Float) # in weeks
    final_capital = Column(Float)
    created_at = Column(DateTime)

# Trade logs
class TradeLog(Base):
    __tablename__ = "trade_logs"
    __table_args__ = {'extend_existing': True}
    
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
    __table_args__ = {'extend_existing': True}
    
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

# Saved RS Strategy
class SavedRSStrategy(Base):
    __tablename__ = "saved_rs_strategies"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    strategy_name = Column(String, index=True)
    strategy_type = Column(String, default="RS Strategy")
    user_id = Column(String, index=True)
    start_date = Column(DateTime, index=True)
    end_date = Column(DateTime, index=True)
    stock_universe = Column(String)
    backtest_results = Column(Text)  # JSON string of backtest results
    strategy_config = Column(Text)  # JSON string of strategy configuration
    created_at = Column(DateTime, default=datetime.now)
    is_active = Column(Boolean, default=True)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Additional fields for strategy management
    last_run_date = Column(DateTime, nullable=True)
    next_run_date = Column(DateTime, nullable=True)
    run_frequency = Column(String, default="daily")  # daily, weekly, monthly
    status = Column(String, default="deploy")   # deploy, paused, stopped
    run_id = Column(String, index=True, nullable=True)
    webhook_url = Column(String, nullable=True)
    client_information_json = Column(Text, nullable=True)

# RS Live signals
class RSLiveSignal(Base):
    __tablename__ = "rs_live"
    __table_args__ = {'extend_existing': True}
    
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

backtest_engine = None  # Will be initialized lazily
BacktestSessionLocal = None  # Will be initialized lazily

def _init_engine_and_session():
    """Initialize engine and SessionLocal if not already done"""
    global engine, SessionLocal
    if engine is None:
        engine = get_app_data_engine()
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def _init_backtest_session():
    """Initialize backtest engine and session if not already done"""
    global backtest_engine, BacktestSessionLocal
    if backtest_engine is None:
        _init_engine_and_session()  # Ensure main engine is initialized first
        backtest_engine = engine
        BacktestSessionLocal = SessionLocal

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
    _init_backtest_session()  # Ensure backtest session is initialized
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
            logging.info(f"Successfully saved RS backtest result with ID: {result_id}")
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
                force_unlock_database()
                raise
            else:
                logging.warning(f"Failed to save backtest result (attempt {attempt + 1}/{max_retries}): {e}")
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5

async def save_additional_data_safely(trade_logs=None, portfolio_snapshots=None, max_retries=3, retry_delay=1):
    """Save trade logs and portfolio snapshots with enhanced error handling and retry logic"""
    _init_backtest_session()  # Ensure backtest session is initialized
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
                force_unlock_database()
                raise
            else:
                logging.warning(f"Failed to save additional data (attempt {attempt + 1}/{max_retries}): {e}")
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5

def reset_database_connections():
    """Reset database connections"""
    try:
        global engine, backtest_engine
        if engine is not None:
            engine.dispose()
        if backtest_engine is not None:
            backtest_engine.dispose()
        logging.info("Database connections reset successfully")
    except Exception as e:
        logging.error(f"Failed to reset database connections: {e}")

def force_unlock_database():
    """Force unlock database by resetting connections"""
    try:
        # Reset all database connections
        reset_database_connections()
        logging.info("Database connections reset successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to reset database connections: {e}")
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
    _init_engine_and_session()  # Ensure connection is initialized
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


# Alias for backward compatibility
StrategyConfig = SavedRSStrategy

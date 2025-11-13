from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import asyncio
import time
import logging
from contextlib import contextmanager

# Database URL - Using MarketData.sqlite
DATABASE_URL = "sqlite:///./MarketData.sqlite"

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

# ETF data model - using etf_data table
class ETFData(Base):
    __tablename__ = "etf_data"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, primary_key=True, index=True)
    date = Column(DateTime, primary_key=True, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    adjusted_close = Column(Float)  # Column name in MarketData.sqlite
    created_at = Column(DateTime)

# Index data model (can use same index_data table or create separate)
class IndexData(Base):
    __tablename__ = "index_data"
    
    symbol = Column(String, primary_key=True, index=True)
    date = Column(DateTime, primary_key=True, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adjusted_close = Column(Float)  # Column name in MarketData.sqlite
    volume = Column(Integer)

# Strategy configuration
class StrategyConfig(Base):
    __tablename__ = "rs_etf_strategy_config"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    config_name = Column(String)
    main_index = Column(String, default="ETF")
    etf_universe = Column(String)  # ETF universe instead of stock_universe
    lookback_weeks = Column(Integer, default=5)
    lookback_months = Column(Integer, default=20)
    lookback_quarters = Column(Integer, default=60)
    max_positions = Column(Integer, default=8)  # Lower for ETFs
    position_size_pct = Column(Float, default=12.5)  # Higher per position for ETFs
    buffer_capital_pct = Column(Float, default=10.0)
    total_capital = Column(Float, default=1000000)
    stop_loss_pct = Column(Float, default=15.0)  # Tighter stop loss for ETFs
    capital_reset_threshold_pct = Column(Float, default=25.0)
    max_holding_period = Column(Integer, default=120)  # 120 days
    min_turnover = Column(Float, default=1000000)
    min_price = Column(Float, default=10.0)
    transaction_cost_pct = Column(Float, default=0.1)
    is_active = Column(Boolean, default=True)
    created_at = Column(String)
    updated_at = Column(String)

# Backtest results
class BacktestResult(Base):
    __tablename__ = "rs_etf_backtest_results"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    config_id = Column(Integer, index=True, nullable=True)
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
    __tablename__ = "rs_etf_trade_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    backtest_id = Column(Integer, index=True)
    date = Column(DateTime, index=True)
    symbol = Column(String, index=True)
    action = Column(String)  # BUY, SELL
    quantity = Column(Integer)
    price = Column(Float)
    amount = Column(Float)
    reason = Column(String)
    rs_score = Column(Float)
    rs_rank = Column(Integer)
    
    # Detailed transaction cost breakdown
    transaction_value = Column(Float)
    brokerage = Column(Float)
    stt = Column(Float)
    stamp_duty = Column(Float)
    exchange_charges = Column(Float)
    sebi_charges = Column(Float)
    gst = Column(Float)
    total_costs = Column(Float)
    net_amount = Column(Float)

# Portfolio snapshots
class PortfolioSnapshot(Base):
    __tablename__ = "rs_etf_portfolio_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    backtest_id = Column(Integer, index=True)
    date = Column(DateTime, index=True)
    total_value = Column(Float)
    cash_balance = Column(Float)
    positions_json = Column(Text)
    daily_pnl = Column(Float)
    cumulative_pnl = Column(Float)
    drawdown_pct = Column(Float)

# Saved ETF Strategy
class SavedETFStrategy(Base):
    __tablename__ = "rs_etf_saved_strategies"
    
    id = Column(Integer, primary_key=True, index=True)
    strategy_name = Column(String, index=True)
    strategy_type = Column(String, default="RS ETF Strategy")
    user_id = Column(String, index=True)
    start_date = Column(DateTime, index=True)
    end_date = Column(DateTime, index=True)
    rs_etf_universe = Column(String)  # ALL_ETFS or custom list
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

# Backtest database setup
BACKTEST_DATABASE_URL = "sqlite:///./MarketData.sqlite"
backtest_engine = create_engine(
    BACKTEST_DATABASE_URL, 
    connect_args={
        "check_same_thread": False,
        "timeout": 30,
        "isolation_level": None
    },
    pool_pre_ping=True,
    pool_recycle=3600,
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
            delay *= 2

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
            logging.info(f"Successfully saved RS ETF backtest result with ID: {result_id}")
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
        engine.dispose()
        backtest_engine.dispose()
        logging.info("Database connections reset successfully")
    except Exception as e:
        logging.error(f"Failed to reset database connections: {e}")

def force_unlock_database():
    """Force unlock database by resetting connections and clearing lock files"""
    try:
        reset_database_connections()
        time.sleep(0.5)
        
        db_path = "./MarketData.sqlite"
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


"""
Neon PostgreSQL Database Connection Module

This module provides connection to Neon PostgreSQL database for all backend storage
including strategy configurations, live signals, execution tracking, and consolidated
market data (ETF, Stock, and Index prices).
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, UniqueConstraint
from sqlalchemy.sql import func

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neon PostgreSQL Database URL

NEON_DATABASE_URL = os.getenv("DATABASE_STRING")

if not NEON_DATABASE_URL:
    raise ValueError("DATABASE_STRING environment variable is not set")

# Base for SQLAlchemy models
Base = declarative_base()


# ============================================================================
# CONSOLIDATED MARKET DATA MODELS
# ============================================================================

class ETFMarket(Base):
    """Model for Indian ETF market data"""
    __tablename__ = "etf_market"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    adj_close = Column(Float)
    created_at = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_etf_market_symbol_date'),
    )


class StockMarket(Base):
    """Model for Indian Stock market data"""
    __tablename__ = "stock_market"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_stock_market_symbol_date'),
    )


class Nifty50IndexMarket(Base):
    """Model for Nifty 50 Index market data (Benchmark)"""
    __tablename__ = "nifty_50_index_market"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_nifty_50_index_market_symbol_date'),
    )


class SP500IndexMarket(Base):
    """Model for S&P 500 Index market data (US Benchmark)"""
    __tablename__ = "s_p_500_index_market"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_s_p_500_index_market_symbol_date'),
    )


class USETFMarket(Base):
    """Model for US ETF market data"""
    __tablename__ = "us_etf_market"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    adj_close = Column(Float)
    created_at = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_us_etf_market_symbol_date'),
    )

# Global engine and session maker
engine = None
SessionLocal = None


def create_connection():
    """
    Create and test connection to Neon PostgreSQL database.
    Returns True if successful, False otherwise.
    If connection already exists, returns True without recreating it.
    """
    global engine, SessionLocal
    
    # If connection already exists, verify it's still working
    if engine is not None and SessionLocal is not None:
        try:
            # Test if existing connection is still valid
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            logger.debug("Using existing PostgreSQL database connection")
            return True
        except Exception as e:
            logger.warning(f"Existing connection invalid, recreating: {e}")
            # Connection is invalid, reset and recreate
            try:
                engine.dispose()
            except:
                pass
            engine = None
            SessionLocal = None
    
    try:
        # Create engine with connection pooling
        engine = create_engine(
            NEON_DATABASE_URL,
            poolclass=QueuePool,
            pool_size=20,        # Increased from 5 to 20
            max_overflow=40,     # Increased from 10 to 40
            pool_pre_ping=True,  # Verify connections before using them
            pool_recycle=1800,   # Recycle connections after 30 mins (Neon likes this)
            pool_timeout=60,     # Wait up to 60s for a connection
            echo=False,          # Set to True for SQL query logging
            connect_args={
                "connect_timeout": 15,  # 15 second connection timeout
                "sslmode": "require"
            }
        )
        
        # Test the connection
        with engine.connect() as connection:
            result = connection.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            logger.info(f"Successfully connected to PostgreSQL database")
            logger.info(f"PostgreSQL version: {version[:50]}...")
        
        # Create session maker
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        return True
        
    except Exception as e:
        logger.error(f"Error connecting to Neon database: {str(e)}")
        import traceback
        traceback.print_exc()
        engine = None
        SessionLocal = None
        return False


def get_session():
    """
    Get a database session for manual use. 
    IMPORTANT: You must call session.close() when done.
    """
    if SessionLocal is None:
        raise RuntimeError("Database connection not initialized. Call create_connection() first.")
    return SessionLocal()


def get_db():
    """
    FastAPI dependency that provides a database session and ensures it's closed.
    Automatically rolls back failed transactions to prevent "transaction is aborted" errors.
    Usage:
        @app.get("/")
        def index(db: Session = Depends(get_db)):
            ...
    """
    if SessionLocal is None:
        raise RuntimeError("Database connection not initialized. Call create_connection() first.")
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        # Rollback the transaction if any error occurs
        logger.error(f"Database transaction error, rolling back: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def get_engine():
    """
    Get the database engine.
    """
    if engine is None:
        raise RuntimeError("Database connection not initialized. Call create_connection() first.")
    return engine


def init_database():
    """
    Initialize database by creating all tables defined in Base models.
    This should be called after all models are imported.
    """
    if engine is None:
        raise RuntimeError("Database connection not initialized. Call create_connection() first.")
    
    try:
        # Import strategy models to register them with Base
        from Databases import strategy_models  # noqa: F401
        from Databases import broker_models # noqa: F401
        from Databases import signal_models  # noqa: F401  # Trading signals for all strategies
        
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully")
        logger.info("Strategy and signal tables are now using PostgreSQL")
        return True
    except Exception as e:
        logger.error(f"Error initializing database tables: {e}")
        return False


def test_connection():
    """
    Test the database connection and print status.
    Returns True if connection successful, False otherwise.
    """
    print("\n" + "="*60)
    print("Testing Neon PostgreSQL Database Connection...")
    print("="*60)
    
    try:
        success = create_connection()
        
        if success:
            print("\n[OK] Database connected successfully!")
            print("[OK] Connection to Neon PostgreSQL established")
            
            # Get some database info
            try:
                with engine.connect() as connection:
                    # Get database name
                    result = connection.execute(text("SELECT current_database();"))
                    db_name = result.fetchone()[0]
                    
                    # Get current user
                    result = connection.execute(text("SELECT current_user;"))
                    db_user = result.fetchone()[0]
                    
                    print(f"\nDatabase Name: {db_name}")
                    print(f"Connected as: {db_user}")
                    print(f"Database URL: {NEON_DATABASE_URL.split('@')[1].split('/')[0]}")  # Hide credentials
            except Exception as e:
                logger.warning(f"Could not fetch database info: {e}")
            
            print("\n" + "="*60)
            return True
        else:
            print("\n[FAIL] Database connection failed!")
            print("[FAIL] Please check your connection string and network access")
            print("\n" + "="*60)
            return False
            
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("\n" + "="*60)
        return False


if __name__ == "__main__":
    # When run directly, test the connection
    success = test_connection()
    sys.exit(0 if success else 1)


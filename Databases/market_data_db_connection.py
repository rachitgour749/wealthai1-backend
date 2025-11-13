"""
Neon PostgreSQL Database Connection Module for Market Data

This module provides connection to Neon PostgreSQL MarketData database for ETF and Stock market data.
"""

import os
import sys
from sqlalchemy import create_engine, text, Column, Integer, String, Float, DateTime, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy.sql import func
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neon PostgreSQL MarketData Database URL
# Format: postgresql://user:password@host/database?sslmode=require&channel_binding=require
MARKET_DATA_DATABASE_URL = "postgresql://neondb_owner:npg_WgVhOYtnP12l@ep-solitary-silence-a1yoj91r.ap-southeast-1.aws.neon.tech/MarketData?sslmode=require&channel_binding=require"

# Base for SQLAlchemy models
Base = declarative_base()


# ETF Unified Data Model (DEPRECATED - use ETFData instead)
# Kept for backward compatibility during migration
class ETFUnified(Base):
    """Model for ETF unified market data (DEPRECATED - use ETFData)"""
    __tablename__ = "etf_unified"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False, index=True)
    date = Column(String, nullable=False, index=True)  # Stored as YYYY-MM-DD string
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Integer)
    adj_close = Column(Float)
    created_at = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_etf_unified_symbol_date'),
    )


# ETF Metadata Model
class ETFMetadata(Base):
    """Model for ETF metadata"""
    __tablename__ = "etf_metadata"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False, unique=True, index=True)
    start_date = Column(String)  # YYYY-MM-DD
    end_date = Column(String)  # YYYY-MM-DD
    years_available = Column(Float)
    total_records = Column(Integer)
    data_source = Column(String, default='database')
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


# Stock Data Model (for stock_data table)
class StockData(Base):
    """Model for stock market data"""
    __tablename__ = "stock_data"
    
    symbol = Column(String, primary_key=True, nullable=False, index=True)
    date = Column(DateTime, primary_key=True, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Integer)
    
    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_stock_data_symbol_date'),
    )


# Stock Metadata Model
class StockMetadata(Base):
    """Model for stock metadata"""
    __tablename__ = "stock_metadata"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False, unique=True, index=True)
    start_date = Column(String)  # YYYY-MM-DD
    end_date = Column(String)  # YYYY-MM-DD
    years_available = Column(Float)
    total_records = Column(Integer)
    data_source = Column(String, default='database')
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


# ETF Data Model (PRIMARY - use this for all ETF data operations)
class ETFData(Base):
    """Model for ETF market data - PRIMARY table for ETF data"""
    __tablename__ = "etf_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    adjusted_close = Column(Float)
    created_at = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_etf_data_symbol_date'),
    )


# Index Data Model
class IndexData(Base):
    """Model for index data"""
    __tablename__ = "index_data"
    
    symbol = Column(String, primary_key=True, nullable=False, index=True)
    date = Column(DateTime, primary_key=True, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Integer)
    
    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_index_data_symbol_date'),
    )


# Nifty 500 Metadata Model
class Nifty500Metadata(Base):
    """Model for Nifty 500 metadata"""
    __tablename__ = "nifty500_metadata"
    
    symbol = Column(String, primary_key=True, nullable=False, index=True)
    start_date = Column(String)  # DATE as string
    end_date = Column(String)    # DATE as string
    total_records = Column(Integer)
    last_updated = Column(String)  # DATE as string
    data_source = Column(String)
    years_available = Column(Float)


# Global engine and session maker
engine = None
SessionLocal = None


def create_connection():
    """
    Create and test connection to Neon PostgreSQL MarketData database.
    Returns True if successful, False otherwise.
    """
    global engine, SessionLocal
    
    try:
        # Create engine with connection pooling
        engine = create_engine(
            MARKET_DATA_DATABASE_URL,
            poolclass=QueuePool,
            pool_size=10,  # Larger pool for market data operations
            max_overflow=20,
            pool_pre_ping=True,  # Verify connections before using them
            pool_recycle=3600,   # Recycle connections after 1 hour
            echo=False,          # Set to True for SQL query logging
            connect_args={
                "connect_timeout": 10,  # 10 second connection timeout
                "sslmode": "require"
            }
        )
        
        # Test the connection
        with engine.connect() as connection:
            result = connection.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            logger.info(f"Successfully connected to PostgreSQL MarketData database")
            logger.info(f"PostgreSQL version: {version[:50]}...")
        
        # Create session maker
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        return True
        
    except Exception as e:
        logger.error(f"Error connecting to MarketData database: {str(e)}")
        engine = None
        SessionLocal = None
        return False


def get_session():
    """
    Get a database session.
    Usage:
        session = get_session()
        try:
            # Your database operations
            pass
        finally:
            session.close()
    """
    if SessionLocal is None:
        raise RuntimeError("Database connection not initialized. Call create_connection() first.")
    return SessionLocal()


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
        Base.metadata.create_all(bind=engine)
        logger.info("MarketData database tables initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing MarketData database tables: {e}")
        return False


def test_connection():
    """
    Test the database connection and print status.
    Returns True if connection successful, False otherwise.
    """
    print("\n" + "="*60)
    print("Testing Neon PostgreSQL MarketData Database Connection...")
    print("="*60)
    
    try:
        success = create_connection()
        
        if success:
            print("\n✓ Database connected successfully!")
            print("✓ Connection to Neon PostgreSQL MarketData established")
            
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
                    print(f"Database URL: {MARKET_DATA_DATABASE_URL.split('@')[1].split('/')[0]}")  # Hide credentials
            except Exception as e:
                logger.warning(f"Could not fetch database info: {e}")
            
            print("\n" + "="*60)
            return True
        else:
            print("\n✗ Database connection failed!")
            print("✗ Please check your connection string and network access")
            print("\n" + "="*60)
            return False
            
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("\n" + "="*60)
        return False


if __name__ == "__main__":
    # When run directly, test the connection
    success = test_connection()
    if success:
        # Initialize tables
        init_database()
    sys.exit(0 if success else 1)


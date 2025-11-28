"""
Neon PostgreSQL Database Connection Module

This module provides connection to Neon PostgreSQL database for all backend storage
including strategy configurations, live signals, and execution tracking.
Market data (ETF/Stock prices) is also stored in PostgreSQL.
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neon PostgreSQL Database URL
# Format: postgresql://user:password@host/database?sslmode=require&channel_binding=require
NEON_DATABASE_URL = "postgresql://neondb_owner:npg_WgVhOYtnP12l@ep-solitary-silence-a1yoj91r.ap-southeast-1.aws.neon.tech/ApplicationData?sslmode=require&channel_binding=require"

# Base for SQLAlchemy models
Base = declarative_base()

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
            pool_size=5,
            max_overflow=10,
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
        # Import strategy models to register them with Base
        from Databases import strategy_models  # noqa: F401
        
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
            print("\n✓ Database connected successfully!")
            print("✓ Connection to Neon PostgreSQL established")
            
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
    sys.exit(0 if success else 1)


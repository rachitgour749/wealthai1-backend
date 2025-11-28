# app/database.py
"""Database connection and session management"""
import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from ChatAI1.chatai1_config import settings

logger = logging.getLogger(__name__)

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    echo=False,  # Set to True for SQL query logging
    # pool_pre_ping=True,  # Verify connections before using - DISABLED for debugging
    connect_args={"ssl": "require"},  # SSL configuration for asyncpg
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Base class for ORM models
Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting database session

    Yields:
        AsyncSession instance
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Initialize database - create tables if they don't exist"""
    logger.info("Initializing ChatAI1 database...")
    try:
        async with engine.begin() as conn:
            logger.info("Connected to database, creating tables...")
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Tables created successfully")
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        import traceback
        traceback.print_exc()
        raise


async def close_db():
    """Close database connections"""
    await engine.dispose()
    logger.info("Database connections closed")
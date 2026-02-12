"""
SQLAlchemy Models for Strategy and Signal Tables
Migrated from SQLite to PostgreSQL (Neon)

This module contains all models for:
- Strategy configuration tables (ETF, Stock, RS ETF, Custom)
- Live signal tables (ETF and Stock signals)
- Deployment and execution tables
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, UniqueConstraint, Index
from sqlalchemy.sql import func
from Databases.app_data_db_connection import Base


# ============================================================================
# STRATEGY CONFIGURATION TABLES
# ============================================================================

class CustomStrategy(Base):
    """Custom Strategy Configuration"""
    __tablename__ = 'custom_strategies'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_email = Column(String(255), nullable=False)
    user_phone = Column(String(50), nullable=False)
    strategy_description = Column(Text, nullable=False)
    ai_analysis_json = Column(Text, nullable=False)  # JSON
    strategy_rating = Column(Integer)
    status = Column(String(50), default='pending')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_custom_strategies_user_email', 'user_email'),
    )



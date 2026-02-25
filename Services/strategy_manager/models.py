"""SQLAlchemy ORM models for centralized strategy management system"""
from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, JSON, Float
from sqlalchemy.sql import func
from Databases.app_data_db_connection import Base

class SavedInstance(Base):
    """Database model for saved strategy instances"""
    __tablename__ = "saved_instances"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, index=True)
    strategy_name = Column(String(255), nullable=True) # Added to match user payload
    strategy_type = Column(String(100), nullable=False)
    tickers = Column(Text, nullable=True)
    start_date = Column(String(50), nullable=True)
    end_date = Column(String(50), nullable=True)
    strategies_parameters = Column(JSON, nullable=True)
    use_custom_date = Column(Boolean, default=False)
    run_id = Column(String(100), unique=True, index=True)
    client_info = Column(JSON, nullable=True)
    webhook_url = Column(Text, nullable=True)
    webhook_key = Column(String(100), nullable=True)
    status = Column(String(50), default='not deploy', index=True)
    reference_capital = Column(Float, nullable=True)
    last_execution_date = Column(DateTime(timezone=True), nullable=True)
    next_execution_date = Column(DateTime(timezone=True), nullable=True)
    email_notification = Column(Boolean, default=False)
    telegram_notification = Column(Boolean, default=False)
    user_code = Column(String(100), nullable=True)
    rem_exe_count = Column(Integer, default=0) # Added as per user request
    source = Column(String(50), default='internal') # internal or other
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

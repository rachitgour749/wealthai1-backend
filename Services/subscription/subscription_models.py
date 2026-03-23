# Services/Subscription/subscription_models.py
"""SQLAlchemy ORM models for subscription database tables"""
from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, Enum as SQLEnum, JSON, Numeric
from sqlalchemy.sql import func
from datetime import datetime
from Databases.app_data_db_connection import Base
from .subscription_schemas import SubscriptionStatus, SubscriptionPlan, ProductCode, SubscriptionType


class UserDetails(Base):
    """Database model for user details"""
    __tablename__ = "user_details"
    
    user_email = Column(String(255), primary_key=True)
    user_name = Column(String(255), nullable=True)
    phone_no = Column(String(50), nullable=True)
    active_token_hash = Column(String(64), nullable=True)  # SHA256 Hash of the active JWT/Token
    status = Column(String(20), default="TRIAL")  # TRIAL/PAID
    role = Column(String(50), nullable=True, default="CLIENT")  # Admin, RM, Client, etc.
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    total_credits = Column(Integer, default=0)
    used_credits = Column(Integer, default=0)


class ProductManager(Base):
    """Database model for product manager - manages product subscriptions"""
    __tablename__ = "product_manager"
    
    id = Column(String(36), primary_key=True)
    user_email = Column(String(255), nullable=False, index=True)
    # Use String instead of SQLEnum to handle both short codes (A, M, T) and full enum values
    product_code = Column(String(50), nullable=False)
    subscription_type = Column(SQLEnum(SubscriptionType), nullable=False, default=SubscriptionType.TRIAL)
    status = Column(SQLEnum(SubscriptionStatus), nullable=False, default=SubscriptionStatus.TRIAL)
    
    # Subscription dates
    subscription_start_date = Column(DateTime(timezone=True), nullable=True)
    subscription_end_date = Column(DateTime(timezone=True), nullable=True)

    plan_name = Column(String(100), nullable=True)
    
    # ChatAI Token Tracking Fields
    chatai_key = Column(String(100), nullable=True)
    # Map to existing PostgreSQL column names: total_tokens, used_tokens, remaining_tokens
    total_token = Column("total_tokens", Integer, default=0)
    used_token = Column("used_tokens", Integer, default=0)
    remaining_token = Column("remaining_tokens", Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())




class UserHierarchy(Base):
    """Database model for user hierarchy - parent-child relationships"""
    __tablename__ = "user_hierarchy"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_email = Column(String(255), nullable=False, unique=True, index=True)
    parent_email = Column(String(255), nullable=True, index=True)
    hierarchy_level = Column(Integer, nullable=False, default=0)
    created_by = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)

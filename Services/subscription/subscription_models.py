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
    status = Column(String(20), default="TRIAL")  # TRIAL/PAID
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())


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
    
    # ChatAI Token Tracking Fields
    chatai_key = Column(String(100), nullable=True)
    # Map to existing PostgreSQL column names: total_tokens, used_tokens, remaining_tokens
    total_token = Column("total_tokens", Integer, default=0)
    used_token = Column("used_tokens", Integer, default=0)
    remaining_token = Column("remaining_tokens", Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())


class Subscription(Base):
    """Database model for user subscriptions - matches Neon PostgreSQL schema"""
    __tablename__ = "subscription"
    
    # Primary key is user_email
    user_email = Column(String, primary_key=True)
    user_name = Column(String)
    
    # Subscription dates
    subscription_start_date = Column(DateTime(timezone=True))
    subscription_end_date = Column(DateTime(timezone=True))
    
    # Metadata timestamps
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    
    # Plan and trial info
    plan_code = Column(Numeric, nullable=True)  # Numeric plan code
    is_trial = Column(Boolean, default=False)  # Boolean trial flag


class ProductSubscription(Base):
    """Database model for product-specific subscriptions - using Neon PostgreSQL"""
    __tablename__ = "product_subscriptions"
    
    id = Column(String, primary_key=True)
    user_email = Column(String(255), nullable=False, index=True)
    product_code = Column(SQLEnum(ProductCode), nullable=False)
    subscription_type = Column(SQLEnum(SubscriptionType), nullable=False, default=SubscriptionType.TRIAL)
    status = Column(SQLEnum(SubscriptionStatus), nullable=False, default=SubscriptionStatus.TRIAL)
    plan_code = Column(String(50), nullable=False, default="FREE")
    
    # Simplified subscription date fields (matches actual database schema)
    subscription_start_date = Column(DateTime(timezone=True), nullable=True)
    subscription_end_date = Column(DateTime(timezone=True), nullable=True)
    
    # Legacy trial period fields (for backward compatibility)
    trial_start_date = Column(DateTime(timezone=True), nullable=True)
    trial_end_date = Column(DateTime(timezone=True), nullable=True)
    trial_duration_days = Column(Integer, default=7)  # Default 7 days trial
    
    # Legacy paid subscription fields (for backward compatibility)
    paid_start_date = Column(DateTime(timezone=True), nullable=True)
    paid_end_date = Column(DateTime(timezone=True), nullable=True)
    payment_id = Column(String(100), nullable=True)
    payment_status = Column(String(50), nullable=True)
    
    # Bundle information
    bundle_id = Column(String(36), nullable=True)  # Links products in same bundle
    is_bundle_subscription = Column(Boolean, default=False)
    
    # ChatAI Token Tracking Fields
    chatai_key = Column(String(100), nullable=True)  # Unique key for ChatAI subscription
    total_tokens = Column(Integer, default=0)  # Total tokens allocated for this subscription
    used_tokens = Column(Integer, default=0)  # Tokens used so far
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    product_metadata = Column(JSON, nullable=True)  # Store additional product-specific data


class ProductAccessLog(Base):
    """Database model for tracking product access attempts - using Neon PostgreSQL"""
    __tablename__ = "product_access_logs"
    
    id = Column(String, primary_key=True)
    user_email = Column(String(255), nullable=False, index=True)
    product_code = Column(SQLEnum(ProductCode), nullable=False)
    access_attempted_at = Column(DateTime(timezone=True), default=func.now())
    access_granted = Column(Boolean, nullable=False)
    access_type = Column(String(20), nullable=True)  # trial, paid, bundle
    subscription_id = Column(String(36), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)

"""
Trading Signal Models

Generic signal table that supports all WealthAI strategies
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Index
from sqlalchemy.sql import func
from datetime import datetime
from Databases.app_data_db_connection import Base


class TradingSignal(Base):
    """
    Generic trading signal model for all strategies
    
    Supports: ETF Rotation, Stock Rotation, International ETF, RS ETF, RS Stocks, etc.
    """
    __tablename__ = 'trading_signals'
    
    # Primary identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String(50), unique=True, nullable=False)  # UUID for tracking
    
    # Strategy and user information
    strategy_name = Column(String(100), index=True, nullable=False)  # ETF_Rotation, Rotation_Stocks, etc.
    user_id = Column(String(100), index=True, nullable=False)
    user_code = Column(String(100))
    instance_id = Column(Integer)  # Reference to saved_instances
    
    # Signal details
    signal_type = Column(String(20), nullable=False)  # BUY, SELL, HOLD
    symbol = Column(String(50), nullable=False)
    quantity = Column(Integer, nullable=True)
    
    # Pricing and metrics (strategy-specific)
    price = Column(Float)
    score = Column(Float, nullable=True)  # Strategy-specific score
    strategy_metadata = Column(JSON)  # Flexible field for strategy-specific data
    # Example strategy_metadata for rotation strategies:
    # {
    #   '52w_high': 1234.56,
    #   '52w_low': 987.65,
    #   'distance_from_low': 5.2,
    #   'distance_from_high': 15.8,
    #   'current_price': 1050.00
    # }
    
    # Execution details
    client_info = Column(JSON)  # Client IDs for order placement
    webhook_url = Column(Text, nullable=True)
    
    # Timing
    signal_date = Column(DateTime, index=True, nullable=False)  # When signal was generated
    execution_date = Column(DateTime, nullable=True)  # When to execute
    expiry_date = Column(DateTime, nullable=True)
    
    # Status tracking
    status = Column(String(20), default='pending', index=True)  # pending, executed, failed, expired
    execution_result = Column(JSON, nullable=True)
    
    # Audit
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, onupdate=func.now())
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_strategy_status', 'strategy_name', 'status'),
        Index('idx_user_signal_date', 'user_id', 'signal_date'),
        Index('idx_signal_date_status', 'signal_date', 'status'),
    )
    
    def __repr__(self):
        return f"<TradingSignal(id={self.id}, strategy={self.strategy_name}, symbol={self.symbol}, type={self.signal_type}, status={self.status})>"
    
    def to_dict(self):
        """Convert signal to dictionary"""
        return {
            'id': self.id,
            'signal_id': self.signal_id,
            'strategy_name': self.strategy_name,
            'user_id': self.user_id,
            'user_code': self.user_code,
            'instance_id': self.instance_id,
            'signal_type': self.signal_type,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'price': self.price,
            'score': self.score,
            'strategy_metadata': self.strategy_metadata,
            'client_info': self.client_info,
            'webhook_url': self.webhook_url,
            'signal_date': self.signal_date.isoformat() if self.signal_date else None,
            'execution_date': self.execution_date.isoformat() if self.execution_date else None,
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date else None,
            'status': self.status,
            'execution_result': self.execution_result,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

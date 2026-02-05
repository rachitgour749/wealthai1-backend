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
    user_id = Column(String(100), index=True, nullable=False)
    run_id = Column(String(100), nullable=True) # Added as requested
    user_code = Column(String(100))
    strategy_name = Column(String(100), index=True, nullable=False)
    strategy_type = Column(String(100), nullable=True) # Added as requested
    
    # Signal details
    order_side = Column(String(20), nullable=False) # Renamed from signal_type
    symbol_name = Column(String(50), nullable=False) # Renamed from symbol
    
    # Execution details
    client_json = Column(JSON) # Renamed from client_info
    webhook_url = Column(Text, nullable=True)
    
    # Timing
    signal_date = Column(DateTime, index=True, nullable=False)
    
    # Metrics
    score = Column(Float, nullable=True)
    price = Column(Float)
    high_52 = Column(Float, nullable=True) # Added as requested
    low_52 = Column(Float, nullable=True) # Added as requested
    
    # Status tracking
    created_at = Column(DateTime, default=func.now(), nullable=False)
    executed_at = Column(DateTime, nullable=True) # Renamed from execution_date
    execution_status = Column(String(20), default='pending', index=True) # Renamed from status
    
    # Fields not explicitly requested but likely needed for compatibility or audit (keeping minimum)
    expiry_date = Column(DateTime, nullable=True)
    execution_result = Column(JSON, nullable=True)
    updated_at = Column(DateTime, onupdate=func.now())
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_strategy_status', 'strategy_name', 'execution_status'),
        Index('idx_user_signal_date', 'user_id', 'signal_date'),
    )
    
    def __repr__(self):
        return f"<TradingSignal(id={self.id}, strategy={self.strategy_name}, symbol={self.symbol_name}, side={self.order_side}, status={self.execution_status})>"
    
    def to_dict(self):
        """Convert signal to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'run_id': self.run_id,
            'user_code': self.user_code,
            'strategy_name': self.strategy_name,
            'strategy_type': self.strategy_type,
            'order_side': self.order_side,
            'symbol_name': self.symbol_name,
            'client_json': self.client_json,
            'webhook_url': self.webhook_url,
            'signal_date': self.signal_date.isoformat() if self.signal_date else None,
            'score': self.score,
            'price': self.price,
            'high_52': self.high_52,
            'low_52': self.low_52,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'executed_at': self.executed_at.isoformat() if self.executed_at else None,
            'execution_status': self.execution_status,
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date else None,
            'execution_result': self.execution_result
        }

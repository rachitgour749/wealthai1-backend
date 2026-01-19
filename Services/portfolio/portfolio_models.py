"""SQLAlchemy models for portfolio tracking"""
from sqlalchemy import Column, Integer, String, Numeric, Date, DateTime, Index
from sqlalchemy.sql import func
from Databases.app_data_db_connection import Base


class PortfolioTrade(Base):
    """Portfolio trades table - stores all executed trades"""
    __tablename__ = 'portfolio_trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Strategy Identification
    user_email = Column(String(255), nullable=False, index=True)
    run_id = Column(String(255), nullable=False, index=True)
    strategy_name = Column(String(255), nullable=False)
    strategy_type = Column(String(100), nullable=False)
    
    # Client Identification
    client_code = Column(String(255), nullable=False, index=True)
    
    # Trade Details
    trade_date = Column(Date, nullable=False, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # 'BUY' or 'SELL'
    quantity = Column(Integer, nullable=False)
    price = Column(Numeric(15, 4), nullable=False)
    
    # Costs
    brokerage = Column(Numeric(10, 2), default=0)
    taxes = Column(Numeric(10, 2), default=0)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_trades_composite', 'run_id', 'client_code', 'symbol', 'trade_date', 'side', unique=True),
    )

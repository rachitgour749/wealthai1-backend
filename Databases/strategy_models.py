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

class ETFSavedStrategy(Base):
    """ETF Saved Strategy Configuration"""
    __tablename__ = 'etf_saved_strategy'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(255), nullable=False)
    strategy_type = Column(String(100), nullable=False)
    user_id = Column(String(255), nullable=False)  # User email
    tickers = Column(Text, nullable=False)  # JSON array
    start_date = Column(String(50), nullable=False)
    end_date = Column(String(50), nullable=False)
    capital_per_week = Column(Float, nullable=False)
    accumulation_weeks = Column(Integer, nullable=False)
    brokerage_percent = Column(Float, nullable=False)
    compounding_enabled = Column(Boolean, nullable=False)
    risk_free_rate = Column(Float, nullable=False)
    use_custom_dates = Column(Boolean, nullable=False)
    backtest_results = Column(Text, nullable=False)  # JSON
    created_at = Column(String(50), nullable=False)
    created_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    config_id = Column(Integer)
    backtest_id = Column(Integer)
    etf_universe = Column(String(100), default='NIFTY_ETFS')
    strategy_config = Column(Text)  # JSON
    run_id = Column(String(255), unique=True)
    client_information_json = Column(Text)  # JSON
    webhook_url = Column(Text)
    status = Column(String(50), default='deploy')
    execution_date = Column(String(50))
    ltp = Column(Float)
    reference_capital = Column(Text)
    deployment_data = Column(Text)  # JSON
    etf_count = Column(Integer)
    etf_names = Column(Text)  # JSON array
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_etf_saved_strategy_user_id', 'user_id'),
        Index('idx_etf_saved_strategy_run_id', 'run_id'),
    )


class StockSavedStrategy(Base):
    """Stock Saved Strategy Configuration"""
    __tablename__ = 'stock_saved_strategy'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(255), nullable=False)
    strategy_type = Column(String(100), nullable=False)
    user_id = Column(String(255), nullable=False)  # User email
    tickers = Column(Text, nullable=False)  # JSON array
    start_date = Column(String(50), nullable=False)
    end_date = Column(String(50), nullable=False)
    capital_per_week = Column(Float, nullable=False)
    accumulation_weeks = Column(Integer, nullable=False)
    brokerage_percent = Column(Float, nullable=False)
    compounding_enabled = Column(Boolean, nullable=False)
    risk_free_rate = Column(Float, nullable=False)
    use_custom_dates = Column(Boolean, nullable=False)
    backtest_results = Column(Text, nullable=False)  # JSON
    created_at = Column(String(50), nullable=False)
    created_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    config_id = Column(Integer)
    backtest_id = Column(Integer)
    stock_universe = Column(String(100), default='NIFTY500')
    strategy_config = Column(Text)  # JSON
    run_id = Column(String(255), unique=True)
    client_information_json = Column(Text)  # JSON
    webhook_url = Column(Text)
    status = Column(String(50), default='deploy')
    execution_date = Column(String(50))
    reference_capital = Column(Text)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_stock_saved_strategy_user_id', 'user_id'),
        Index('idx_stock_saved_strategy_run_id', 'run_id'),
    )


class RSEtFSavedStrategy(Base):
    """RS ETF Saved Strategy Configuration"""
    __tablename__ = 'rs_etf_saved_strategies'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(255), nullable=False)
    strategy_type = Column(String(100), nullable=False, default='RS ETF Strategy')
    user_id = Column(String(255), nullable=False)  # User email
    start_date = Column(String(50))
    end_date = Column(String(50))
    rs_etf_universe = Column(Text)
    backtest_results = Column(Text)  # JSON
    strategy_config = Column(Text)  # JSON
    created_at = Column(String(50))
    is_active = Column(Boolean, default=True)
    updated_at = Column(String(50))
    last_run_date = Column(String(50))
    next_run_date = Column(String(50))
    run_frequency = Column(String(50), default='daily')
    status = Column(String(50), default='deploy')
    run_id = Column(String(255))
    webhook_url = Column(Text)
    client_information_json = Column(Text)  # JSON
    
    __table_args__ = (
        UniqueConstraint('strategy_name', 'user_id', name='uq_rs_etf_strategy_name_user'),
        Index('idx_rs_etf_saved_strategies_user_id', 'user_id'),
        Index('idx_rs_etf_saved_strategies_run_id', 'run_id'),
    )


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


# ============================================================================
# LIVE SIGNAL TABLES
# ============================================================================

class LiveSignal(Base):
    """ETF Live Trading Signals"""
    __tablename__ = 'live_signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(255), nullable=False)
    user_id = Column(String(255))  # User email
    strategy_version = Column(String(100), nullable=False)
    signal_date = Column(String(50), nullable=False)
    signal_symbol = Column(String(50), nullable=False)
    etf_name = Column(String(255))
    side = Column(String(10), nullable=False)  # 'BUY' or 'SELL'
    score = Column(Float, nullable=False)
    reason = Column(Text, nullable=False)
    payload_json = Column(Text)  # JSON
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        UniqueConstraint('run_id', 'signal_symbol', 'side', name='uq_live_signals_run_symbol_side'),
        Index('idx_live_signals_run_id', 'run_id'),
        Index('idx_live_signals_date', 'signal_date'),
        Index('idx_live_signals_user_id', 'user_id'),
    )


class LiveRun(Base):
    """ETF Signal Generation Runs"""
    __tablename__ = 'live_runs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(255), nullable=False, unique=True)
    strategy_version = Column(String(100), nullable=False)
    signal_date = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String(50), default='generated')  # 'generated', 'webhook_sent', 'failed'
    webhook_attempts = Column(Integer, default=0)
    last_error = Column(Text)
    summary_json = Column(Text)  # JSON
    
    __table_args__ = (
        Index('idx_live_runs_date', 'signal_date'),
        Index('idx_live_runs_status', 'status'),
    )


class LiveStockSignal(Base):
    """Stock Live Trading Signals"""
    __tablename__ = 'live_stock_signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(255), nullable=False)
    symbol = Column(String(50), nullable=False)
    side = Column(String(10), nullable=False)  # 'BUY' or 'SELL'
    score = Column(Float, nullable=False)
    reason = Column(Text)
    payload = Column(Text)  # JSON
    signal_date = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_live_stock_signals_run_id', 'run_id'),
        Index('idx_live_stock_signals_date', 'signal_date'),
    )


class LiveStockRun(Base):
    """Stock Signal Generation Runs"""
    __tablename__ = 'live_stock_runs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(255), nullable=False)
    strategy_type = Column(String(100), nullable=False)
    strategy_config = Column(Text)  # JSON
    signals_count = Column(Integer, default=0)
    buy_count = Column(Integer, default=0)
    sell_count = Column(Integer, default=0)
    run_date = Column(String(50), nullable=False)
    duration_seconds = Column(Float)
    status = Column(String(50), default='completed')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_live_stock_runs_run_id', 'run_id'),
        Index('idx_live_stock_runs_date', 'run_date'),
    )


# ============================================================================
# DEPLOYMENT & EXECUTION TABLES
# ============================================================================

class ExecutedDetail(Base):
    """Execution Tracking Records"""
    __tablename__ = 'executed_details'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False)  # User email
    run_id = Column(String(255), nullable=False)
    signal_symbol = Column(String(50), nullable=False)
    side = Column(String(10), nullable=False)  # 'BUY' or 'SELL'
    success = Column(Boolean, nullable=False)
    error_message = Column(Text)
    response_data = Column(Text)  # JSON
    webhook_url = Column(Text)
    payload_sent = Column(Text)  # JSON
    execution_date = Column(String(50))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_executed_details_user_id', 'user_id'),
        Index('idx_executed_details_run_id', 'run_id'),
        Index('idx_executed_details_execution_date', 'execution_date'),
    )


class Strategy(Base):
    """Webhook Strategy Configuration"""
    __tablename__ = 'strategies'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(255), nullable=False)
    user_email = Column(String(255))
    webhook = Column(Text, nullable=False)
    reference_capital = Column(Text)
    client_ids = Column(Text)  # JSON array
    capitals = Column(Text)  # JSON array
    execution_date = Column(String(50))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String(50), default='active')


# ============================================================================
# SUPERTREND STRATEGY TABLES
# ============================================================================

class SuperTrendStrategyConfig(Base):
    """SuperTrend Strategy Configuration"""
    __tablename__ = 'strategy_config'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ema_short = Column(Integer, default=10)
    ema_long = Column(Integer, default=20)
    supertrend_period = Column(Integer, default=10)
    supertrend_stop_pct = Column(Float, default=10.0)
    max_holdings = Column(Integer, default=5)
    buffer_pct = Column(Float, default=10.0)
    price_floor = Column(Float, default=50.0)
    liquidity_cr = Column(Float, default=10.0)
    rs_window_1 = Column(Integer, default=5)
    rs_window_2 = Column(Integer, default=21)
    rs_window_3 = Column(Integer, default=63)
    benchmark = Column(String(50), default='NIFTY50')
    universe = Column(String(50), default='NIFTY200')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class SuperTrendBacktestResult(Base):
    """SuperTrend Backtest Results"""
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_date = Column(String(50))
    trade_date = Column(String(50))
    symbol = Column(String(50))
    action = Column(String(10))
    price = Column(Float)
    quantity = Column(Integer)
    position_value = Column(Float)
    portfolio_value = Column(Float)
    cash = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_supertrend_backtest_run_date', 'run_date'),
        Index('idx_supertrend_backtest_trade_date', 'trade_date'),
    )


class SuperTrendCurrentPosition(Base):
    """SuperTrend Current Positions"""
    __tablename__ = 'current_positions'
    
    symbol = Column(String(50), primary_key=True)
    entry_date = Column(String(50))
    entry_price = Column(Float)
    quantity = Column(Integer)
    current_price = Column(Float)
    current_value = Column(Float)
    pnl = Column(Float)
    pnl_pct = Column(Float)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class SuperTrendCandidate(Base):
    """SuperTrend Stock Candidates"""
    __tablename__ = 'candidates'
    
    symbol = Column(String(50), primary_key=True)
    date = Column(String(50))
    adj_close = Column(Float)
    ema10 = Column(Float)
    ema20 = Column(Float)
    supertrend = Column(String(10))
    rs_score = Column(Float)
    rank = Column(Integer)
    eligible = Column(Integer)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_supertrend_candidates_date', 'date'),
        Index('idx_supertrend_candidates_eligible', 'eligible'),
    )


class SaveJson(Base):
    """JSON Data Storage for Deployments"""
    __tablename__ = 'savejson'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_email = Column(String(255), nullable=False)
    json_data = Column(Text, nullable=False)  # JSON
    strategy_name = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DeployDetail(Base):
    """Deployment Details (Legacy/Compatibility Table)"""
    __tablename__ = 'deploy_details'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(255), nullable=False)
    user_email = Column(String(255), nullable=False)
    strategy_type = Column(String(100))  # 'ETF Strategy' or 'Stock Strategy'
    etf_count = Column(Integer)
    stock_count = Column(Integer)
    status = Column(String(50), default='running')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_deploy_details_run_id', 'run_id'),
        Index('idx_deploy_details_status', 'status'),
    )


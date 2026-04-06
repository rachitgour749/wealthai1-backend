"""
Unified Schemas for Centralized Backtest API
Supports all strategy types with a single request model
"""
from pydantic import BaseModel, field_validator, Field, ConfigDict
from typing import List, Optional, Literal, Dict, Any, Union
from datetime import datetime


class UnifiedBacktestRequest(BaseModel):
    """
    Unified backtest request supporting all strategy types.
    
    Strategy Types:
    - ETF_Rotation: Weekly SIP-based ETF rotation
    - RS_ETF_Rotation: Relative Strength ETF strategy with lump sum
    - International_ETF_Rotation: Weekly SIP for international ETFs
    - Rotation_Stocks: Weekly SIP-based stock rotation
    - ETF_Payout: ETF rotation with periodic withdrawals
    - SuperTrend: Technical indicator-based strategy
    """
    
    # ============================================================================
    # REQUIRED FOR ALL STRATEGIES
    # ============================================================================
    strategy_type: Literal[
        "ETF_Rotation", 
        "RS_ETF_Rotation", 
        "RS_Stocks",
        "International_ETF_Rotation",
        "Stock_Rotation",
        "ETF_Payout",
        "SuperTrend",
        "ETF_Buy_on_Dip",
        "ETF_Swing_Strategy"
    ] = Field(..., description="Type of strategy to run")
    
    start_date: str = Field(..., description="Backtest start date (YYYY-MM-DD or ISO format)")
    end_date: str = Field(..., description="Backtest end date (YYYY-MM-DD or ISO format)")
    
    # ============================================================================
    # ASSET SELECTION (strategy-dependent)
    # ============================================================================
    tickers: Optional[List[str]] = Field(None, description="Asset tickers for Rotation strategies")
    symbols: Optional[List[str]] = Field(None, description="Stock symbols for SuperTrend")
    custom_etfs: Optional[List[str]] = Field(None, description="Custom ETF selection for RS_ETF")
    custom_stocks: Optional[List[str]] = Field(None, description="Custom stock selection for RS_Stocks")
    
    # ============================================================================
    # CAPITAL PARAMETERS (mutually exclusive based on strategy)
    # ============================================================================
    # For Rotation strategies (weekly SIP model)
    capital_per_week: Optional[float] = Field(None, description="Weekly investment amount")
    accumulation_weeks: Optional[int] = Field(None, description="Number of weeks to invest")
    capital_per_month: Optional[float] = Field(None, description="Monthly base capital for Buy-on-Dip")
    
    # For RS strategies (lump sum model)
    total_capital: Optional[float] = Field(None, description="Total capital for RS strategies")
    
    # For SuperTrend (lump sum model)
    initial_capital: Optional[float] = Field(None, description="Initial capital for SuperTrend")
    
    # ============================================================================
    # COMMON PARAMETERS
    # ============================================================================
    brokerage_percent: Optional[float] = Field(None, description="Brokerage percentage for Rotation strategies")
    brokerage_pct: Optional[float] = Field(None, description="Brokerage percentage for SuperTrend")
    risk_free_rate: Optional[float] = Field(8.0, description="Risk-free rate for metrics calculation")
    
    # ============================================================================
    # ROTATION-SPECIFIC PARAMETERS
    # ============================================================================
    compounding_enabled: Optional[bool] = Field(False, description="Enable compounding for Rotation strategies")
    
    # ============================================================================
    # ETF PAYOUT SPECIFIC
    # ============================================================================
    withdraw_amount: Optional[float] = Field(0.0, description="Withdrawal amount per churning week")
    payout_start_week: Optional[int] = Field(1, description="Week number to start payouts")
    
    # ============================================================================
    # RS STRATEGY SPECIFIC
    # ============================================================================
    main_index: Optional[str] = Field("^NSEI", description="Main benchmark index")
    etf_universe: Optional[str] = Field(None, description="ETF universe (ALL_ETFS, etc.)")
    stock_universe: Optional[str] = Field(None, description="Stock universe (NIFTY50, NIFTY500, etc.)")
    max_positions: Optional[int] = Field(None, description="Maximum number of positions")
    position_size_pct: Optional[float] = Field(None, description="Position size percentage (auto-calculated if None)")
    stop_loss_pct: Optional[float] = Field(None, description="Stop loss percentage")
    buffer_capital_pct: Optional[float] = Field(None, description="Buffer capital percentage")
    capital_reset_threshold_pct: Optional[float] = Field(None, description="Capital reset threshold")
    max_holding_period: Optional[int] = Field(None, description="Maximum holding period in weeks")
    transaction_cost_pct: Optional[float] = Field(None, description="Transaction cost percentage")
    min_price: Optional[float] = Field(None, description="Minimum stock/ETF price filter")
    min_turnover: Optional[float] = Field(None, description="Minimum turnover filter")
    lookback_weeks: Optional[int] = Field(None, description="RS lookback period in weeks")
    lookback_months: Optional[int] = Field(None, description="RS lookback period in months")
    lookback_quarters: Optional[int] = Field(None, description="RS lookback period in quarters")
    
    # ============================================================================
    # SUPERTREND SPECIFIC
    # ============================================================================
    buffer_pct: Optional[float] = Field(None, description="Buffer percentage for SuperTrend")
    ema_short: Optional[int] = Field(None, description="Short EMA period")
    ema_long: Optional[int] = Field(None, description="Long EMA period")
    supertrend_period: Optional[int] = Field(None, description="SuperTrend period")
    supertrend_stop_pct: Optional[float] = Field(None, description="SuperTrend stop percentage")
    max_holdings: Optional[int] = Field(None, description="Maximum holdings for SuperTrend")
    
    # ============================================================================
    # ETF SWING STRATEGY SPECIFIC
    # ============================================================================
    sma_lookback: Optional[int] = Field(None, description="SMA lookback period")
    profit_threshold_pct: Optional[float] = Field(None, description="Profit threshold percentage for exit")
    number_of_slots: Optional[int] = Field(None, description="Number of concurrent positions (slots)")
    
    # ============================================================================
    # VALIDATORS
    # ============================================================================
    
    @field_validator('strategy_type')
    @classmethod
    def validate_strategy_type(cls, v: str) -> str:
        """Validate strategy type"""
        valid_types = [
            "ETF_Rotation", "RS_ETF_Rotation", "RS_Stocks",
            "International_ETF_Rotation", "Stock_Rotation",
            "ETF_Payout", "SuperTrend", "ETF_Buy_on_Dip",
            "ETF_Swing_Strategy"
        ]
        if v not in valid_types:
            raise ValueError(f"Invalid strategy_type. Must be one of: {valid_types}")
        return v
    
    @field_validator('*', mode='before')
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Any:
        """Convert empty strings to None"""
        if v == '':
            return None
        return v
    
    model_config = ConfigDict(
        extra='allow',  # Allow extra fields for legacy parameters
        json_schema_extra={
            "example": {
                "strategy_type": "ETF_Rotation",
                "start_date": "2020-01-01",
                "end_date": "2023-12-31",
                "tickers": ["NIFTYBEES.NS", "BANKBEES.NS", "GOLDBEES.NS"],
                "capital_per_week": 50000,
                "accumulation_weeks": 52,
                "brokerage_percent": 0.1,
                "compounding_enabled": False,
                "risk_free_rate": 8.0
            }
        }
    )


class UnifiedBacktestResponse(BaseModel):
    """Unified response for all strategy backtests"""
    success: bool
    strategy_type: str
    metrics: Dict[str, Any]
    performance_data: Optional[Dict[str, Any]] = None
    transaction_log: Optional[List[Dict[str, Any]]] = None
    cost_breakdown: Optional[Dict[str, Any]] = None
    portfolio_snapshots: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "strategy_type": "ETF_Rotation",
                "metrics": {
                    "total_return": 45.5,
                    "cagr_pct": 12.3,
                    "sharpe_ratio": 1.8,
                    "max_drawdown": -15.2
                },
                "performance_data": {
                    "dates": ["2020-01-01", "2020-01-08"],
                    "strategy_value": [1000000, 1050000]
                }
            }
        }
    )


# ============================================================================
# CENTRALIZED STRATEGY API SCHEMAS
# ============================================================================

class DateRangeRequest(BaseModel):
    """Request for calculating available date range"""
    strategy_type: Literal[
        "ETF_Rotation", 
        "RS_ETF_Rotation", 
        "RS_Stocks",
        "International_ETF_Rotation",
        "Stock_Rotation",
        "ETF_Payout",
        "SuperTrend",
        "ETF_Buy_on_Dip"
    ]
    
    # For Rotation strategies
    tickers: Optional[List[str]] = None
    
    # For RS strategies
    etf_universe: Optional[str] = None
    stock_universe: Optional[str] = None
    custom_etfs: Optional[List[str]] = None
    custom_stocks: Optional[List[str]] = None
    main_index: Optional[str] = "^NSEI"
    
    # RS lookback periods
    lookback_weeks: Optional[int] = None
    lookback_months: Optional[int] = None
    lookback_quarters: Optional[int] = None


class SaveStrategyRequest(BaseModel):
    """Request for saving a strategy configuration"""
    strategy_type: Literal[
        "ETF_Rotation", 
        "RS_ETF_Rotation", 
        "International_ETF_Rotation",
        "Stock_Rotation",
        "ETF_Payout",
        "ETF_Buy_on_Dip",
        "ETF_Swing_Strategy"
    ]
    user_id: str
    strategy_name: str
    config: Dict[str, Any]
    webhook_url: Optional[str] = None
    clients: Optional[List[Dict[str, Any]]] = None

"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date


class StrategyConfig(BaseModel):
    ema_short: int = Field(default=10, description="Short EMA period")
    ema_long: int = Field(default=20, description="Long EMA period")
    supertrend_period: int = Field(default=10, description="Supertrend period")
    supertrend_stop_pct: float = Field(default=10.0, description="Supertrend stop percentage")
    max_holdings: int = Field(default=5, description="Maximum number of holdings")
    buffer_pct: float = Field(default=10.0, description="Buffer reserve % for P&L absorption (5-20%)")
    price_floor: float = Field(default=50.0, description="Minimum stock price")
    liquidity_cr: float = Field(default=10.0, description="Minimum liquidity in crores")
    rs_window_1: int = Field(default=5, description="RS window 1 (days)")
    rs_window_2: int = Field(default=21, description="RS window 2 (days)")
    rs_window_3: int = Field(default=63, description="RS window 3 (days)")
    benchmark: str = Field(default="NIFTY50", description="Benchmark index")
    universe: str = Field(default="NIFTY200", description="Stock universe")
    debug_rs: bool = Field(default=False, description="Enable RS calculation debug output")
    debug_rs_symbol: str = Field(default=None, description="Symbol to debug (e.g., 'UCOBANK')")


class BacktestRequest(BaseModel):
    start_date: str = Field(..., description="Backtest start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Backtest end date (YYYY-MM-DD)")
    initial_capital: float = Field(default=1000000.0, description="Initial capital")
    brokerage_pct: float = Field(default=0.1, description="Brokerage % (e.g., 0.03 for 0.03%)")
    buffer_pct: float = Field(default=10.0, description="Buffer reserve % for P&L absorption")
    ema_short: Optional[int] = Field(default=None, description="Override for short EMA period")
    ema_long: Optional[int] = Field(default=None, description="Override for long EMA period")
    max_holdings: Optional[int] = Field(default=None, description="Override for maximum holdings")
    symbols: Optional[List[str]] = Field(default=None, description="Custom list of stock symbols to include")


class BacktestResponse(BaseModel):
    status: str
    message: str
    metrics: Optional[dict] = None
    equity_curve: Optional[List[dict]] = None
    trades: Optional[List[dict]] = None
    cost_analysis: Optional[dict] = None
    total_costs: Optional[float] = None
    selected_symbols: Optional[List[str]] = None


class EODRunResponse(BaseModel):
    status: str
    message: str
    run_date: str
    candidates_count: int
    positions_count: int


class Candidate(BaseModel):
    symbol: str
    date: str
    adj_close: float
    ema10: float
    ema20: float
    supertrend: str
    rs_score: Optional[float] = None
    rank: int
    eligible: int


class Position(BaseModel):
    symbol: str
    entry_date: str
    entry_price: float
    quantity: int
    current_price: float
    current_value: float
    pnl: float
    pnl_pct: float


class CandidatesResponse(BaseModel):
    status: str
    candidates: List[Candidate]


class PositionsResponse(BaseModel):
    status: str
    positions: List[Position]
    total_value: float
    total_pnl: float
    total_pnl_pct: float


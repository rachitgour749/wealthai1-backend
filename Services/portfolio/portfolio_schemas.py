"""Pydantic schemas for portfolio API"""
from pydantic import BaseModel
from typing import List, Dict, Optional


class WebhookPayload(BaseModel):
    """Incoming webhook payload from external system"""
    run_id: str
    exchange: str
    symbol: str
    user_id: str
    order_side: str  # 'BUY' or 'SELL'
    product_type: str
    clients: Dict[str, str]  # {"SRKH1512": "1", "VEDF1515": "5"}
    trade_date: Optional[str] = None  # Optional: YYYY-MM-DD format, defaults to today


class HoldingResponse(BaseModel):
    """Single holding in portfolio"""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    market_value: float
    total_cost: float
    unrealized_pnl: float
    return_pct: float


class PortfolioResponse(BaseModel):
    """Portfolio response"""
    client_code: Optional[str] = None
    holdings: List[HoldingResponse]
    total_value: float
    total_invested: float
    unrealized_pnl: float
    total_return_pct: float
    holdings_count: int


class EquityCurvePoint(BaseModel):
    """Single point in equity curve"""
    date: str
    portfolio_value: float
    total_invested: float
    return_pct: float
    benchmark_value: Optional[float] = 0.0
    benchmark_return_pct: Optional[float] = 0.0


class ClientDetail(BaseModel):
    """Individual client details within a strategy"""
    client_code: str
    capital: float  # Allocated capital for this strategy


class UserStrategySummary(BaseModel):
    """Summary of a single strategy for a user"""
    run_id: str
    strategy_name: str
    strategy_type: str
    total_invested: float
    market_value: float
    pnl: float
    return_pct: float
    holdings_count: int
    running_status: Optional[str] = "deploy"
    owner_email: Optional[str] = None
    owner_name: Optional[str] = None
    clients: List['ClientDetail'] = []


class UserPortfolioResponse(BaseModel):
    """Overall portfolio summary for a user"""
    user_email: str
    total_invested: float
    total_value: float
    total_pnl: float
    total_return_pct: float
    strategies_count: int
    strategies: List[UserStrategySummary]

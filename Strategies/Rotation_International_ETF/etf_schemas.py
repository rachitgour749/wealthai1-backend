from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class BacktestRequest(BaseModel):
    tickers: List[str]
    start_date: str
    end_date: str
    capital_per_week: float
    accumulation_weeks: int
    brokerage_percent: float
    compounding_enabled: bool = False
    risk_free_rate: float = 8.0

class ETFMetadata(BaseModel):
    ticker: str
    name: str
    category: str
    expense_ratio: float
    aum: float

class BacktestResult(BaseModel):
    success: bool
    total_investment: float
    final_portfolio_value: float
    total_return: float
    total_brokerage: float
    error: Optional[str] = None

class BacktestResults(BaseModel):
    total_return: Optional[str] = None
    cagr: Optional[str] = None
    sharpe_ratio: Optional[str] = None
    max_drawdown: Optional[str] = None

class SaveETFStrategyRequest(BaseModel):
    strategy_name: str
    strategy_type: str
    user_id: str
    tickers: List[str]
    start_date: str
    end_date: str
    capital_per_week: float
    accumulation_weeks: int
    brokerage_percent: float
    compounding_enabled: bool
    risk_free_rate: float
    use_custom_dates: bool
    backtest_results: BacktestResults
    created_at: str

class SavedETFStrategy(BaseModel):
    id: int
    strategy_name: str
    strategy_type: str
    user_id: str
    tickers: List[str]
    start_date: str
    end_date: str
    capital_per_week: float
    accumulation_weeks: int
    brokerage_percent: float
    compounding_enabled: bool
    risk_free_rate: float
    use_custom_dates: bool
    backtest_results: Dict[str, Any]
    created_at: str

import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session

class WealthAIBase(ABC):
    """
    Level 1: Core Foundation
    
    The absolute foundation of the WealthAI backend.
    Handles universal logic applicable to any trading system:
    - Logging configuration
    - Database session management
    - Mathematical calculations (CAGR, Sharpe, Drawdown)
    - Abstract definitions for data loading
    """
    
    def __init__(self, db_session: Session = None):
        """
        Initialize the WealthAI base class.
        
        Args:
            db_session: SQLAlchemy database session
        """
        self.db_session = db_session
        self.logger = self.setup_logging()
        
        # Portfolio containers
        self.trades: List[Dict] = []
        self.equity_curve: pd.Series = pd.Series(dtype=float)
        self.daily_nav: List[Dict] = []
        
    def setup_logging(self) -> logging.Logger:
        """
        Configures the standardized logging format for WealthAI.
        """
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def get_db_session(self) -> Session:
        """
        Returns the active database session.
        """
        if not self.db_session:
            raise ValueError("Database session not initialized")
        return self.db_session

    @abstractmethod
    def load_data(self, start_date: datetime, end_date: datetime) -> Any:
        """
        Abstract method to fetch OHLCV data.
        Must be implemented by child classes.
        """
        pass

    def calculate_cagr(self, start_value: float, end_value: float, years: float) -> float:
        """
        Computes Compound Annual Growth Rate.
        
        Formula: (End / Start) ^ (1 / Years) - 1
        """
        if start_value <= 0 or years <= 0:
            return 0.0
        return ((end_value / start_value) ** (1 / years) - 1) * 100

    def calculate_volatility(self, returns: pd.Series, annualization_factor: int = 52) -> float:
        """
        Computes annualized volatility.
        
        Args:
            returns: Series of percentage returns
            annualization_factor: 52 for weekly, 252 for daily
        """
        if returns.empty:
            return 0.0
        return returns.std() * np.sqrt(annualization_factor) * 100

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.07, 
                             annualization_factor: int = 52) -> float:
        """
        Computes risk-adjusted returns (Sharpe Ratio).
        
        Args:
            returns: Series of percentage returns
            risk_free_rate: Annual risk-free rate (default 7%)
            annualization_factor: 52 for weekly, 252 for daily
        """
        if returns.empty or returns.std() == 0:
            return 0.0
        
        # Adjust risk-free rate for the period
        period_rf = risk_free_rate / annualization_factor
        excess_returns = returns - period_rf
        
        return (excess_returns.mean() / returns.std()) * np.sqrt(annualization_factor)

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Computes the maximum peak-to-valley loss.
        """
        if equity_curve.empty:
            return 0.0
            
        cumulative = equity_curve / equity_curve.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100

    def generate_trade_log(self) -> List[Dict]:
        """
        Returns the list of executed trades.
        """
        return self.trades

    def generate_equity_curve(self) -> pd.Series:
        """
        Returns the equity curve.
        """
        return self.equity_curve

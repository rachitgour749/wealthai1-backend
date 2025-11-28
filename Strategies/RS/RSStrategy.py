from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass
import math

from Segments.EquitySegment import EquitySegment

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

@dataclass
class DynamicParams:
    """Dynamic parameters that adjust based on market conditions"""
    rs_threshold: float
    position_size_pct: float
    stop_loss_pct: float
    max_positions: int
    rebalance_frequency_days: int
    volatility_threshold: float

class RSStrategy(EquitySegment):
    """
    Base class for Relative Strength (RS) strategies.
    Inherits from EquitySegment to leverage Indian market specifics and equity trading logic.
    """
    def __init__(self, db_session=None):
        super().__init__(db_session)
        
        # Strategy specific attributes
        self.lookback_weeks = 5
        self.lookback_months = 20
        self.lookback_quarters = 60
        
        # Dynamic parameters
        self.current_regime = MarketRegime.BULL
        self.dynamic_params = self.get_default_dynamic_params()
        
        # State tracking
        self.positions = {}
        self.regime_history = []
        
    def get_default_dynamic_params(self) -> DynamicParams:
        """Get default dynamic parameters"""
        return DynamicParams(
            rs_threshold=0.1,
            position_size_pct=0.05,  # 5%
            stop_loss_pct=0.15,      # 15%
            max_positions=20,
            rebalance_frequency_days=7,
            volatility_threshold=0.5
        )

    def calculate_rs_score(self, asset_prices: pd.Series, index_prices: pd.Series, 
                          current_date: datetime) -> Optional[float]:
        """
        Calculate Relative Strength score for an asset against an index.
        Uses a composite score of weekly, monthly, and quarterly RS.
        """
        try:
            # Get available trading dates
            available_asset_dates = sorted(asset_prices.index)
            available_index_dates = sorted(index_prices.index)

            # Find current date index
            try:
                current_asset_index = available_asset_dates.index(current_date) 
                current_index_index = available_index_dates.index(current_date) 
            except ValueError:
                return None

            # Get current prices
            current_asset_price = asset_prices.loc[current_date]
            current_index_price = index_prices.loc[current_date]
            
            # Calculate required lookback periods
            max_lookback = max(self.lookback_weeks, self.lookback_months, self.lookback_quarters)
            
            # Check history
            if current_asset_index < max_lookback or current_index_index < max_lookback:
                return None

            # Get historical dates
            week_ago_date = available_asset_dates[current_asset_index - self.lookback_weeks]
            month_ago_date = available_asset_dates[current_asset_index - self.lookback_months]
            quarter_ago_date = available_asset_dates[current_asset_index - self.lookback_quarters]

            # Verify index dates (using index positions corresponding to asset lookbacks)
            # Note: This assumes aligned trading days or sufficient overlap. 
            # A more robust approach matches dates exactly, but index-based lookback is standard in original code.
            week_ago_index_date = available_index_dates[current_index_index - self.lookback_weeks]
            month_ago_index_date = available_index_dates[current_index_index - self.lookback_months]
            quarter_ago_index_date = available_index_dates[current_index_index - self.lookback_quarters]

            # Calculate RS for each period
            rs_w = self.calculate_single_rs(
                current_asset_price, asset_prices.loc[week_ago_date],
                current_index_price, index_prices.loc[week_ago_index_date]      
            )
            rs_m = self.calculate_single_rs(
                current_asset_price, asset_prices.loc[month_ago_date],
                current_index_price, index_prices.loc[month_ago_index_date]     
            )
            rs_q = self.calculate_single_rs(
                current_asset_price, asset_prices.loc[quarter_ago_date],        
                current_index_price, index_prices.loc[quarter_ago_index_date]   
            )
            
            if any(rs is None for rs in [rs_w, rs_m, rs_q]):
                return None
            
            # Composite RS score
            rs_score = (rs_w + rs_m + rs_q) / 3
            return rs_score
            
        except (KeyError, IndexError, ValueError, ZeroDivisionError):
            return None

    def calculate_single_rs(self, asset_current: float, asset_past: float,
                           index_current: float, index_past: float) -> Optional[float]:
        """Calculate single period RS"""
        try:
            if asset_past == 0 or index_past == 0:
                return None
            rs = (asset_current / asset_past) / (index_current / index_past) - 1
            return rs
        except (ZeroDivisionError, ValueError):
            return None

    def detect_market_regime(self, index_data: pd.DataFrame, current_date: datetime) -> MarketRegime:
        """
        Detect current market regime based on index moving averages and volatility.
        Placeholder implementation - to be enhanced.
        """
        # TODO: Implement robust regime detection logic
        return MarketRegime.BULL

    def apply_dynamic_stops(self, current_price: float, buy_price: float, 
                           volatility: float = 0) -> float:
        """
        Calculate dynamic stop loss price based on market conditions/volatility.
        """
        # Basic implementation using fixed percentage from dynamic params
        stop_pct = self.dynamic_params.stop_loss_pct
        
        # Adjust based on volatility if provided (example logic)
        if volatility > self.dynamic_params.volatility_threshold:
            stop_pct *= 1.5  # Widen stops in volatile markets
            
        return buy_price * (1 - stop_pct)

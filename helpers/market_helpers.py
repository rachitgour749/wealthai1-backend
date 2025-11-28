"""
Market Helper Utilities

Utility functions for market data calculations and momentum indicators.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def compute_52_week_high_low(df: pd.DataFrame, current_date: datetime) -> pd.DataFrame:
    """
    Calculate rolling 52-week high/low for each ticker at signal date.
    
    This is a core momentum indicator used in rotation strategies.
    
    Args:
        df: DataFrame with price data (index should be dates, columns are tickers)
        current_date: Current date for calculation
        
    Returns:
        DataFrame with columns:
        - 52_week_high: Highest price in last 52 weeks
        - 52_week_low: Lowest price in last 52 weeks
        - current_price: Current price at signal date
        - distance_from_high_pct: Distance from 52-week high (%)
    """
    lookback_start = current_date - timedelta(weeks=52)
    
    # Filter data for 52-week lookback period
    mask = (df.index >= lookback_start) & (df.index <= current_date)
    lookback_df = df[mask]
    
    if lookback_df.empty:
        return pd.DataFrame()
    
    # Calculate 52-week high and low for each ticker
    high_52w = lookback_df.max()
    low_52w = lookback_df.min()
    current_price = df.loc[current_date] if current_date in df.index else df.iloc[-1]
    
    # Calculate distance from high (momentum indicator)
    distance_from_high = ((current_price - high_52w) / high_52w * 100)
    
    result_df = pd.DataFrame({
        '52_week_high': high_52w,
        '52_week_low': low_52w,
        'current_price': current_price,
        'distance_from_high_pct': distance_from_high
    })
    
    return result_df

"""
Date Helper Utilities

Pure utility functions for handling trading day calculations.
These are stateless functions that don't depend on any class state.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


def get_last_trading_day(close_df: pd.DataFrame, target_date: datetime, 
                         day: str = 'Friday', max_lookback: int = 7) -> Optional[datetime]:
    """
    Return the nearest available trading day (e.g., Friday or fallback Thursday).
    
    Args:
        close_df: DataFrame with close prices (index should be dates)
        target_date: Target date to find trading day for
        day: Preferred day of week (default: 'Friday') - currently not used
        max_lookback: Maximum number of days to look back (default: 7)
        
    Returns:
        Nearest trading day or None if not found
    """
    for offset in range(max_lookback):
        check_date = target_date - timedelta(days=offset)
        if check_date in close_df.index:
            return check_date
    
    return None


def get_next_trading_day(open_df: pd.DataFrame, target_date: datetime,
                         day: str = 'Monday', max_lookforward: int = 7) -> Optional[datetime]:
    """
    Return Monday's open or fallback to Tuesday if holiday.
    
    Args:
        open_df: DataFrame with open prices (index should be dates)
        target_date: Target date to find trading day for
        day: Preferred day of week (default: 'Monday') - currently not used
        max_lookforward: Maximum number of days to look forward (default: 7)
        
    Returns:
        Next trading day or None if not found
    """
    for offset in range(max_lookforward):
        check_date = target_date + timedelta(days=offset)
        if check_date in open_df.index:
            return check_date
    
    return None

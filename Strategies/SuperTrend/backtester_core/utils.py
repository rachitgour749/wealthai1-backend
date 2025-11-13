"""
Utility functions
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict


def validate_date_format(date_str: str) -> bool:
    """
    Validate date format (YYYY-MM-DD)
    
    Args:
        date_str: Date string
    
    Returns:
        True if valid, False otherwise
    """
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def get_trading_days(start_date: str, end_date: str) -> List[str]:
    """
    Get list of trading days between dates
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        List of date strings
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    return [d.strftime('%Y-%m-%d') for d in dates]


def format_currency(value: float) -> str:
    """
    Format value as Indian currency
    
    Args:
        value: Numeric value
    
    Returns:
        Formatted string
    """
    if value >= 10000000:  # >= 1 crore
        return f"₹{value/10000000:.2f} Cr"
    elif value >= 100000:  # >= 1 lakh
        return f"₹{value/100000:.2f} L"
    else:
        return f"₹{value:,.2f}"


def format_percentage(value: float) -> str:
    """
    Format value as percentage
    
    Args:
        value: Numeric value
    
    Returns:
        Formatted string
    """
    return f"{value:.2f}%"


def calculate_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate drawdown series
    
    Args:
        equity_curve: Portfolio value series
    
    Returns:
        Drawdown series (%)
    """
    cummax = equity_curve.cummax()
    drawdown = ((equity_curve - cummax) / cummax) * 100
    return drawdown


def get_date_range_stats(df: pd.DataFrame, date_col: str = 'date') -> Dict:
    """
    Get date range statistics
    
    Args:
        df: DataFrame with date column
        date_col: Name of date column
    
    Returns:
        Dictionary with min_date, max_date, total_days
    """
    df[date_col] = pd.to_datetime(df[date_col])
    
    return {
        'min_date': df[date_col].min().strftime('%Y-%m-%d'),
        'max_date': df[date_col].max().strftime('%Y-%m-%d'),
        'total_days': (df[date_col].max() - df[date_col].min()).days
    }


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division fails
    
    Returns:
        Division result or default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default


def filter_valid_data(df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    """
    Filter rows with valid data in required columns
    
    Args:
        df: DataFrame
        required_cols: List of required columns
    
    Returns:
        Filtered DataFrame
    """
    df = df.copy()
    
    # Drop rows with NaN in required columns
    df = df.dropna(subset=required_cols)
    
    # Drop rows with inf values
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required_cols)
    
    return df


def calculate_transaction_costs(transaction_value: float, brokerage_pct: float, 
                                  is_buy: bool = True) -> dict:
    """
    Calculate Indian market transaction costs
    
    Args:
        transaction_value: Total transaction amount (quantity * price)
        brokerage_pct: Brokerage percentage (e.g., 0.03 for 0.03%)
        is_buy: True for buy, False for sell
    
    Returns:
        Dictionary with cost breakdown
    """
    # Base rates (as percentages)
    STT_RATE = 0.10 / 100          # 0.001% on sell only
    STAMP_DUTY_RATE = 0.015 / 100   # 0.005% on buy only
    EXCHANGE_RATE = 0.00297 / 100   # 0.00345% on both
    SEBI_RATE = 0.0001 / 100        # 0.0001% on both
    GST_RATE = 18 / 100             # 18% on brokerage
    
    # Calculate individual costs
    brokerage = transaction_value * (brokerage_pct / 100)
    gst = brokerage * GST_RATE
    exchange_charges = (transaction_value * EXCHANGE_RATE) * 0.18
    sebi_charges = (transaction_value * SEBI_RATE) * 0.18
    
    if is_buy:
        stt = transaction_value * 0.10 / 100
        stamp_duty = transaction_value * STAMP_DUTY_RATE
    else: # SELL
        stt = transaction_value * STT_RATE
        stamp_duty = 0 # No stamp duty on sell
    
    # Total costs
    total_costs = brokerage + gst + stt + stamp_duty + exchange_charges + sebi_charges
    
    return {
        'brokerage': brokerage,
        'gst': gst,
        'stt': stt,
        'stamp_duty': stamp_duty,
        'exchange_charges': exchange_charges,
        'sebi_charges': sebi_charges,
        'total_costs': total_costs,
        'transaction_value': transaction_value,
        'net_amount': transaction_value + total_costs if is_buy else transaction_value - total_costs
    }
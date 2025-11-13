"""
Entry and Exit Signal Logic
"""
import pandas as pd
import numpy as np


def apply_hygiene_filters(df: pd.DataFrame, price_floor: float = 50.0, 
                          liquidity_cr: float = 10.0) -> pd.DataFrame:
    """
    Apply hygiene filters to stock data
    
    Filters:
    - Price >= price_floor
    - Median 60-day turnover >= liquidity_cr crores
    
    Args:
        df: Stock dataframe
        price_floor: Minimum stock price
        liquidity_cr: Minimum liquidity in crores
    
    Returns:
        Filtered dataframe
    """
    df = df.copy()
    
    # Price filter
    df = df[df['adj_close'] >= price_floor]
    
    # Calculate turnover (price * volume)
    df['turnover'] = df['adj_close'] * df['volume']
    
    # Calculate 60-day median turnover per symbol
    df['turnover_60d_median'] = df.groupby('symbol')['turnover'].transform(
        lambda x: x.rolling(window=60, min_periods=30).median()
    )
    
    # Convert to crores (assuming turnover is in rupees, 1 crore = 10 million)
    df['turnover_cr'] = df['turnover_60d_median'] / 10000000
    
    # Apply liquidity filter
    df = df[df['turnover_cr'] >= liquidity_cr]
    
    return df


def calculate_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate eligibility filter
    
    Eligibility Rule:
    - adj_close > EMA20 AND EMA10 > EMA20
    
    Args:
        df: DataFrame with indicators
    
    Returns:
        DataFrame with 'eligible' column
    """
    df = df.copy()
    
    df['eligible'] = (
        (df['adj_close'] > df['ema20']) & 
        (df['ema10'] > df['ema20'])
    ).astype(int)
    
    return df


def calculate_entry_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate entry signals
    
    Entry Rule:
    - Supertrend = Green AND Eligible = True
    
    Args:
        df: DataFrame with indicators and eligibility
    
    Returns:
        DataFrame with 'entry_signal' column
    """
    df = df.copy()
    
    df['entry_signal'] = (
        (df['supertrend_color'] == 'green') & 
        (df['eligible'] == 1)
    ).astype(int)
    
    return df


def calculate_exit_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate exit signals
    
    Exit Rule:
    - Supertrend = Red OR EMA10 < EMA20 OR adj_close < EMA20
    
    Args:
        df: DataFrame with indicators
    
    Returns:
        DataFrame with 'exit_signal' column
    """
    df = df.copy()
    
    df['exit_signal'] = (
        (df['supertrend_color'] == 'red') | 
        (df['ema10'] < df['ema20']) | 
        (df['adj_close'] < df['ema20'])
    ).astype(int)
    
    return df


def add_all_signals(df: pd.DataFrame, price_floor: float = 50.0,
                    liquidity_cr: float = 10.0) -> pd.DataFrame:
    """
    Add all signals to dataframe
    
    Args:
        df: DataFrame with indicators
        price_floor: Minimum price
        liquidity_cr: Minimum liquidity
    
    Returns:
        DataFrame with all signals
    """
    # Apply hygiene filters
    df = apply_hygiene_filters(df, price_floor, liquidity_cr)
    
    # Calculate eligibility
    df = calculate_eligibility(df)
    
    # Calculate entry signals
    df = calculate_entry_signals(df)
    
    # Calculate exit signals
    df = calculate_exit_signals(df)
    
    return df


if __name__ == "__main__":
    # Test with sample data
    import numpy as np
    
    test_data = pd.DataFrame({
        'symbol': ['TEST'] * 100,
        'date': pd.date_range('2024-01-01', periods=100),
        'adj_close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 100),
        'ema10': np.random.randn(100).cumsum() + 99,
        'ema20': np.random.randn(100).cumsum() + 98,
        'supertrend_color': np.random.choice(['green', 'red'], 100)
    })
    
    result = add_all_signals(test_data)
    print(result[['date', 'adj_close', 'ema10', 'ema20', 'supertrend_color', 
                  'eligible', 'entry_signal', 'exit_signal']].tail(10))


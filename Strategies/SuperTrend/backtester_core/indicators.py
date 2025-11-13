"""
Technical indicators: EMA-based strategy (EMA15 and EMA21)
"""
import pandas as pd
import numpy as np


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average
    
    Args:
        data: Price series (typically adj_close)
        period: EMA period
    
    Returns:
        EMA series
    """
    return data.ewm(span=period, adjust=False).mean()


def _calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Average True Range based on adjusted close
    """
    high = df['high']
    low = df['low']
    close = df['adj_close']

    prev_close = close.shift(1)

    tr_components = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ],
        axis=1
    )

    true_range = tr_components.max(axis=1)
    atr = true_range.rolling(window=period, min_periods=period).mean()
    return atr


def _calculate_supertrend(
    df: pd.DataFrame,
    period: int,
    stop_pct: float
) -> pd.DataFrame:
    """
    Calculate Supertrend using FIXED percentage offset (NO ATR).
    
    Fixed 10% Stop Logic:
    - hl2 = (high + low) / 2
    - offset = hl2 * 0.10
    - upperband = hl2 + offset
    - lowerband = hl2 - offset

    Args:
        df: DataFrame with 'high', 'low', 'adj_close'
        period: Not used (kept for API compatibility)
        stop_pct: Fixed stop percentage (e.g. 10.0 for 10%)
    """
    result = df.copy()

    multiplier = stop_pct / 100.0 if stop_pct > 5 else stop_pct

    # Calculate HL2 (average of high and low)
    hl2 = (result['high'] + result['low']) / 2.0

    # Fixed percentage offset (NO ATR)
    basic_ub = hl2 + (hl2 * multiplier)
    basic_lb = hl2 - (hl2 * multiplier)

    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()

    for i in range(1, len(result)):
        close_prev = result.loc[result.index[i - 1], 'adj_close']

        if close_prev <= final_ub.iloc[i - 1]:
            final_ub.iloc[i] = min(basic_ub.iloc[i], final_ub.iloc[i - 1])
        else:
            final_ub.iloc[i] = basic_ub.iloc[i]

        if close_prev >= final_lb.iloc[i - 1]:
            final_lb.iloc[i] = max(basic_lb.iloc[i], final_lb.iloc[i - 1])
        else:
            final_lb.iloc[i] = basic_lb.iloc[i]

    supertrend = pd.Series(np.nan, index=result.index, dtype=float)
    trend_direction = pd.Series(0, index=result.index, dtype=int)

    for i in range(len(result)):
        close_price = result.iloc[i]['adj_close']

        # Initialize on first row
        if i == 0:
            if close_price > final_ub.iloc[i]:
                trend_direction.iloc[i] = 1
                supertrend.iloc[i] = final_lb.iloc[i]
            else:
                trend_direction.iloc[i] = -1
                supertrend.iloc[i] = final_ub.iloc[i]
            continue

        # Determine trend direction
        if close_price > final_ub.iloc[i]:
            trend_direction.iloc[i] = 1
        elif close_price < final_lb.iloc[i]:
            trend_direction.iloc[i] = -1
        else:
            trend_direction.iloc[i] = trend_direction.iloc[i - 1]

        # Set supertrend value based on direction
        if trend_direction.iloc[i] >= 0:
            supertrend.iloc[i] = final_lb.iloc[i]
            trend_direction.iloc[i] = 1
        else:
            supertrend.iloc[i] = final_ub.iloc[i]
            trend_direction.iloc[i] = -1

    result['basic_ub'] = basic_ub
    result['basic_lb'] = basic_lb
    result['final_ub'] = final_ub
    result['final_lb'] = final_lb
    result['supertrend'] = supertrend
    result['supertrend_direction'] = trend_direction
    result['supertrend_color'] = np.where(trend_direction > 0, 'green',
                                          np.where(trend_direction < 0, 'red', np.nan))

    return result


def add_indicators(
    df: pd.DataFrame,
    ema_short: int = 15,
    ema_long: int = 21,
    supertrend_period: int = 10,
    supertrend_stop_pct: float = 10.0
) -> pd.DataFrame:
    """
    Add EMA and Supertrend indicators to the dataframe
    
    Strategy Logic:
    - Eligibility: adj_close > EMA20 AND EMA10 > EMA20
    - Entry: Supertrend = Green AND Eligible = True
    - Exit: Supertrend = Red OR EMA10 < EMA20 OR adj_close < EMA20
    
    Supertrend: Fixed 10% offset (NO ATR)
    - hl2 = (high + low) / 2
    - upperband = hl2 + (hl2 * 0.10)
    - lowerband = hl2 - (hl2 * 0.10)
    
    Args:
        df: Stock price dataframe with 'high', 'low', 'adj_close'
        ema_short: Short EMA period (default: 15, mapped to ema10)
        ema_long: Long EMA period (default: 21, mapped to ema20)
        supertrend_period: Not used (kept for API compatibility)
        supertrend_stop_pct: Fixed stop percentage (default: 10.0 for 10%)
    
    Returns:
        DataFrame with EMA and Supertrend indicators
    """
    df = df.copy()

    # Calculate EMAs
    df['ema15'] = calculate_ema(df['adj_close'], ema_short)
    df['ema21'] = calculate_ema(df['adj_close'], ema_long)

    # Keep legacy column names for backward compatibility
    df['ema10'] = df['ema15']  # Map to old name
    df['ema20'] = df['ema21']  # Map to old name

    # Calculate Supertrend
    df = _calculate_supertrend(df, supertrend_period, supertrend_stop_pct)

    return df


if __name__ == "__main__":
    # Test with sample data
    test_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'high': np.random.randn(100).cumsum() + 100,
        'low': np.random.randn(100).cumsum() + 95,
        'close': np.random.randn(100).cumsum() + 98,
        'adj_close': np.random.randn(100).cumsum() + 98
    })
    test_data['high'] = test_data[['high', 'close']].max(axis=1)
    test_data['low'] = test_data[['low', 'close']].min(axis=1)
    
    result = add_indicators(test_data)
    print(result[['date', 'close', 'ema10', 'ema20', 'supertrend_color']].tail(10))


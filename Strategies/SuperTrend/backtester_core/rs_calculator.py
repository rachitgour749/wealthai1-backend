"""
Relative Strength (RS) Calculator vs Benchmark
"""
import pandas as pd
import numpy as np
from typing import List, Tuple


def calculate_returns(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate percentage returns over a window
    
    Args:
        prices: Price series
        window: Lookback window in days
    
    Returns:
        Returns series
    """
    return prices.pct_change(periods=window, fill_method=None) * 100


def calculate_rs_score(stock_df: pd.DataFrame, index_df: pd.DataFrame, 
                       windows: List[int] = [5, 21, 63],
                       debug: bool = False,
                       debug_symbol: str = None) -> pd.DataFrame:
    """
    Calculate Relative Strength score vs benchmark
    
    Logic:
    - RS_w = stock_return(window) - index_return(window)
    - RS_Score = average of all RS windows (skip NaN values)
    
    Args:
        stock_df: Stock dataframe with 'date' and 'adj_close'
        index_df: Index dataframe with 'date' and 'adj_close'
        windows: List of lookback windows [5D, 21D, 63D]
        debug: Enable detailed debug output
        debug_symbol: Symbol to debug (only shows debug for this symbol)
    
    Returns:
        DataFrame with RS components and final score
    """
    # Get symbol name if available
    symbol = stock_df['symbol'].iloc[0] if 'symbol' in stock_df.columns else 'UNKNOWN'
    
    # Merge stock and index data on date
    merged = stock_df[['date', 'adj_close']].copy()
    merged = merged.merge(index_df[['date', 'adj_close']], on='date', how='left', 
                         suffixes=('_stock', '_index'))
    
    # Calculate returns for each window
    rs_components = []
    
    for i, window in enumerate(windows, 1):
        stock_return = calculate_returns(merged['adj_close_stock'], window)
        index_return = calculate_returns(merged['adj_close_index'], window)
        rs = stock_return - index_return
        merged[f'rs_{window}d'] = rs
        rs_components.append(f'rs_{window}d')
    
    # Calculate average RS score - skip NaN values (insufficient data)
    merged['rs_score'] = merged[rs_components].mean(axis=1, skipna=True)
    
    # For rows where all RS components are NaN (insufficient data), set to None
    # This indicates insufficient historical data for RS calculation
    all_nan_mask = merged[rs_components].isna().all(axis=1)
    merged.loc[all_nan_mask, 'rs_score'] = None
    
    # DEBUG OUTPUT - Show detailed calculation for last available date
    if debug and (debug_symbol is None or symbol == debug_symbol):
        # Find last valid row with RS score
        last_valid_idx = merged['rs_score'].last_valid_index()
        if last_valid_idx is not None:
            print(f"\n{'='*80}")
            print(f"DEBUG RS Calculation for {symbol}")
            print(f"{'='*80}")
            
            current_date = merged.loc[last_valid_idx, 'date']
            current_stock_price = merged.loc[last_valid_idx, 'adj_close_stock']
            current_index_price = merged.loc[last_valid_idx, 'adj_close_index']
            
            print(f"    Current Date: {current_date}")
            print(f"    Current stock index: {last_valid_idx}")
            print(f"    Max lookback required: {max(windows)}")
            print(f"    Lookback periods: week={windows[0]}, month={windows[1]}, quarter={windows[2]}")
            print(f"    Current prices:")
            print(f"      Stock: ₹{current_stock_price:.2f}, Index: ₹{current_index_price:.2f}")
            
            print(f"    Historical dates and prices:")
            
            # Calculate historical positions
            window_labels = ['Week', 'Month', 'Quarter']
            rs_values = {}
            
            for window, label in zip(windows, window_labels):
                lookback_idx = last_valid_idx - window
                if lookback_idx >= 0:
                    hist_date = merged.loc[lookback_idx, 'date']
                    hist_stock = merged.loc[lookback_idx, 'adj_close_stock']
                    hist_index = merged.loc[lookback_idx, 'adj_close_index']
                    
                    # Calculate returns
                    stock_return = ((current_stock_price - hist_stock) / hist_stock)
                    index_return = ((current_index_price - hist_index) / hist_index)
                    rs_val = stock_return - index_return
                    rs_values[label.lower()] = rs_val
                    
                    print(f"      {label} ago ({window}d): {hist_date} - Stock: ₹{hist_stock:.2f}, Index: ₹{hist_index:.2f}")
                else:
                    rs_values[label.lower()] = None
                    print(f"      {label} ago ({window}d): [Insufficient data]")
            
            # Show RS values
            print(f"    RS values: ", end="")
            rs_parts = [f"{label}={val:.3f}" if val is not None else f"{label}=N/A" 
                       for label, val in rs_values.items()]
            print(", ".join(rs_parts))
            
            # Final RS score
            final_rs = merged.loc[last_valid_idx, 'rs_score']
            print(f"    Final RS score: {final_rs:.3f}")
            print(f"    {symbol}: RS Score = {final_rs:.3f}")
            print(f"{'='*80}\n")
    
    # Add back to original dataframe
    result = stock_df.copy()
    result = result.merge(merged[['date'] + rs_components + ['rs_score']], 
                         on='date', how='left')
    
    return result


def calculate_rs_for_universe(stock_data: pd.DataFrame, index_data: pd.DataFrame,
                               windows: List[int] = [5, 21, 63],
                               debug: bool = False,
                               debug_symbol: str = None) -> pd.DataFrame:
    """
    Calculate RS scores for all stocks in universe
    
    Args:
        stock_data: DataFrame with all stocks (must have 'symbol', 'date', 'adj_close')
        index_data: DataFrame with benchmark index ('date', 'adj_close')
        windows: RS calculation windows
        debug: Enable detailed debug output
        debug_symbol: Specific symbol to debug (e.g., 'UCOBANK')
    
    Returns:
        DataFrame with RS scores for all stocks
    """
    results = []
    
    symbols = stock_data['symbol'].unique()
    
    for symbol in symbols:
        stock_df = stock_data[stock_data['symbol'] == symbol].copy()
        stock_df = stock_df.sort_values('date').reset_index(drop=True)
        
        # Calculate RS for this stock
        stock_with_rs = calculate_rs_score(stock_df, index_data, windows,
                                           debug=debug, debug_symbol=debug_symbol)
        stock_with_rs['symbol'] = symbol
        
        results.append(stock_with_rs)
    
    # Combine all results
    all_stocks = pd.concat(results, ignore_index=True)
    
    return all_stocks


def rank_by_rs(df: pd.DataFrame, date: str, top_n: int = 5) -> pd.DataFrame:
    """
    Rank stocks by RS score for a given date
    
    Args:
        df: DataFrame with RS scores
        date: Date to rank on
        top_n: Number of top stocks to return
    
    Returns:
        Top N stocks sorted by RS score
    """
    date_df = df[df['date'] == date].copy()
    date_df = date_df.sort_values('rs_score', ascending=False).reset_index(drop=True)
    date_df['rank'] = range(1, len(date_df) + 1)
    
    return date_df.head(top_n)


if __name__ == "__main__":
    # Test with sample data
    from datetime import datetime, timedelta
    
    dates = pd.date_range('2024-01-01', periods=100)
    
    # Sample stock data
    stock_data = pd.DataFrame({
        'symbol': ['STOCK1'] * 100 + ['STOCK2'] * 100,
        'date': list(dates) * 2,
        'adj_close': list(np.random.randn(100).cumsum() + 100) + list(np.random.randn(100).cumsum() + 100)
    })
    
    # Sample index data
    index_data = pd.DataFrame({
        'date': dates,
        'adj_close': np.random.randn(100).cumsum() + 100
    })
    
    # Calculate RS
    result = calculate_rs_for_universe(stock_data, index_data)
    print(result[['symbol', 'date', 'adj_close', 'rs_score']].tail(10))
    
    # Rank for latest date
    latest_date = dates[-1].strftime('%Y-%m-%d')
    top_stocks = rank_by_rs(result, latest_date, top_n=2)
    print(f"\nTop stocks on {latest_date}:")
    print(top_stocks[['symbol', 'rs_score', 'rank']])


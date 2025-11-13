"""
Data ingestion service for loading data from SQLite
"""
import pandas as pd
from typing import Optional, List
from ..api.database import execute_query, get_connection


def load_stock_data(symbols: Optional[List[str]] = None, 
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load stock data from database
    
    Args:
        symbols: List of symbols to load (None = all)
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)
    
    Returns:
        DataFrame with stock data
    """
    query = "SELECT * FROM stock_data WHERE 1=1"
    params = []
    
    if symbols:
        placeholders = ','.join('?' * len(symbols))
        query += f" AND symbol IN ({placeholders})"
        params.extend(symbols)
    
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    
    query += " ORDER BY symbol, date"
    
    if params:
        df = execute_query(query, tuple(params))
    else:
        df = execute_query(query)
    
    return df


def load_index_data(start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load index data from database
    
    Args:
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)
    
    Returns:
        DataFrame with index data
    """
    query = "SELECT * FROM index_data WHERE 1=1"
    params = []
    
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    
    query += " ORDER BY date"
    
    if params:
        df = execute_query(query, tuple(params))
    else:
        df = execute_query(query)
    
    return df


def get_available_symbols() -> List[str]:
    """
    Get list of available symbols in database
    
    Returns:
        List of symbols
    """
    df = execute_query("SELECT DISTINCT symbol FROM stock_data ORDER BY symbol")
    return df['symbol'].tolist()


def get_date_range() -> dict:
    """
    Get available date range in database
    
    Returns:
        Dictionary with min_date and max_date
    """
    stock_range = execute_query("""
        SELECT MIN(date) as min_date, MAX(date) as max_date 
        FROM stock_data
    """)
    
    index_range = execute_query("""
        SELECT MIN(date) as min_date, MAX(date) as max_date 
        FROM index_data
    """)
    
    return {
        'stock_min_date': stock_range.iloc[0]['min_date'] if not stock_range.empty else None,
        'stock_max_date': stock_range.iloc[0]['max_date'] if not stock_range.empty else None,
        'index_min_date': index_range.iloc[0]['min_date'] if not index_range.empty else None,
        'index_max_date': index_range.iloc[0]['max_date'] if not index_range.empty else None,
    }


def check_data_availability() -> dict:
    """
    Check data availability in database
    
    Returns:
        Dictionary with data statistics
    """
    stock_count = execute_query("SELECT COUNT(*) as count FROM stock_data")
    index_count = execute_query("SELECT COUNT(*) as count FROM index_data")
    symbol_count = execute_query("SELECT COUNT(DISTINCT symbol) as count FROM stock_data")
    
    date_range = get_date_range()
    
    return {
        'stock_records': int(stock_count.iloc[0]['count']) if not stock_count.empty else 0,
        'index_records': int(index_count.iloc[0]['count']) if not index_count.empty else 0,
        'symbols_count': int(symbol_count.iloc[0]['count']) if not symbol_count.empty else 0,
        **date_range
    }


if __name__ == "__main__":
    # Test data loading
    stats = check_data_availability()
    print("Data Availability:")
    print(stats)


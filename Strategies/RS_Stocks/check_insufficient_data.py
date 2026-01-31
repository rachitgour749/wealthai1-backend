"""
Diagnostic Script: Check Stocks with Insufficient Data for RS Strategy

This script identifies stocks that don't have enough historical data
for RS strategy backtesting (need at least 90 weeks buffer).
"""

import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
databases_path = os.path.join(parent_dir, 'Databases')
sys.path.insert(0, databases_path)

from app_data_db_connection import (
    create_connection as create_app_data_connection,
    get_session as get_app_data_session,
    init_database as init_app_data_database
)
from sqlalchemy import text
import pandas as pd

# RS Strategy buffer requirements
BUFFER_WEEKS = 90  # Same as in rs_backtester_core.py
BUFFER_DAYS = BUFFER_WEEKS * 7  # 630 calendar days
MIN_REQUIRED_DAYS = BUFFER_DAYS + 365  # Buffer + at least 1 year for backtesting


def check_stock_data_availability() -> Dict:
    """
    Check all stocks in the database and identify those with insufficient data.
    
    Returns:
        Dictionary with:
        - total_stocks: Total number of stocks in database
        - stocks_with_data: Number of stocks with any data
        - stocks_sufficient: Number of stocks with sufficient data
        - stocks_insufficient: List of stocks with insufficient data
        - stocks_no_data: List of stocks with no data
    """
    # Initialize database connection
    if not create_app_data_connection():
        raise RuntimeError("Failed to connect to ApplicationData database")
    
    init_app_data_database()
    session = get_app_data_session()
    
    try:
        # Get all unique stock symbols
        query_all_symbols = text("""
            SELECT DISTINCT symbol
            FROM stock_market
            ORDER BY symbol
        """)
        
        all_symbols_result = session.execute(query_all_symbols).fetchall()
        all_symbols = [row[0] for row in all_symbols_result]
        
        print(f"📊 Total stocks in database: {len(all_symbols)}")
        print(f"🔍 Checking data availability for each stock...")
        print(f"📏 Required: {BUFFER_DAYS} days buffer + 365 days minimum = {MIN_REQUIRED_DAYS} days total\n")
        
        stocks_with_data = []
        stocks_insufficient = []
        stocks_no_data = []
        stocks_sufficient = []
        
        # Check each stock
        for symbol in all_symbols:
            query_dates = text("""
                SELECT 
                    MIN(date) as min_date, 
                    MAX(date) as max_date,
                    COUNT(*) as record_count
                FROM stock_market
                WHERE symbol = :symbol
            """)
            
            result = session.execute(query_dates, {"symbol": symbol}).fetchone()
            
            if not result or result.min_date is None:
                stocks_no_data.append({
                    'symbol': symbol,
                    'reason': 'No data found'
                })
                continue
            
            min_date = pd.to_datetime(result.min_date)
            max_date = pd.to_datetime(result.max_date)
            record_count = result.record_count
            
            # Calculate total days available
            total_days = (max_date - min_date).days
            
            # Calculate if buffer + minimum backtest period fits
            strategy_start_with_buffer = min_date + timedelta(days=BUFFER_DAYS)
            days_available_after_buffer = (max_date - strategy_start_with_buffer).days
            
            stock_info = {
                'symbol': symbol,
                'start_date': min_date.strftime('%Y-%m-%d'),
                'end_date': max_date.strftime('%Y-%m-%d'),
                'total_days': total_days,
                'record_count': record_count,
                'years_available': total_days / 365.25,
                'strategy_start_with_buffer': strategy_start_with_buffer.strftime('%Y-%m-%d'),
                'days_available_after_buffer': days_available_after_buffer,
                'years_after_buffer': days_available_after_buffer / 365.25 if days_available_after_buffer > 0 else 0
            }
            
            stocks_with_data.append(stock_info)
            
            # Check if sufficient
            if days_available_after_buffer < 365:  # Less than 1 year after buffer
                stocks_insufficient.append(stock_info)
            else:
                stocks_sufficient.append(stock_info)
        
        # Sort insufficient stocks by days available (worst first)
        stocks_insufficient.sort(key=lambda x: x['days_available_after_buffer'])
        
        return {
            'total_stocks': len(all_symbols),
            'stocks_with_data': len(stocks_with_data),
            'stocks_sufficient': len(stocks_sufficient),
            'stocks_insufficient': stocks_insufficient,
            'stocks_no_data': stocks_no_data,
            'stocks_sufficient_list': stocks_sufficient
        }
        
    finally:
        session.close()


def print_report(results: Dict):
    """Print a formatted report of the results"""
    print("=" * 100)
    print("📋 STOCK DATA AVAILABILITY REPORT")
    print("=" * 100)
    print(f"\n📊 Summary:")
    print(f"   Total stocks in database: {results['total_stocks']}")
    print(f"   Stocks with data: {results['stocks_with_data']}")
    print(f"   ✅ Stocks with SUFFICIENT data: {results['stocks_sufficient']}")
    print(f"   ⚠️  Stocks with INSUFFICIENT data: {len(results['stocks_insufficient'])}")
    print(f"   ❌ Stocks with NO data: {len(results['stocks_no_data'])}")
    
    if results['stocks_no_data']:
        print(f"\n❌ Stocks with NO DATA ({len(results['stocks_no_data'])}):")
        print("-" * 100)
        for stock in results['stocks_no_data']:
            print(f"   {stock['symbol']}: {stock['reason']}")
    
    if results['stocks_insufficient']:
        print(f"\n⚠️  STOCKS WITH INSUFFICIENT DATA ({len(results['stocks_insufficient'])}):")
        print("-" * 100)
        print(f"{'Symbol':<15} {'Start Date':<12} {'End Date':<12} {'Total Days':<12} {'Years':<8} {'Days After Buffer':<18} {'Years After Buffer':<18}")
        print("-" * 100)
        
        for stock in results['stocks_insufficient']:
            status = "❌" if stock['days_available_after_buffer'] < 0 else "⚠️"
            print(f"{stock['symbol']:<15} {stock['start_date']:<12} {stock['end_date']:<12} "
                  f"{stock['total_days']:<12} {stock['years_available']:<8.1f} "
                  f"{stock['days_available_after_buffer']:<18} {stock['years_after_buffer']:<18.2f} {status}")
        
        print("\n" + "=" * 100)
        print("📝 DETAILED INSUFFICIENT DATA LIST:")
        print("=" * 100)
        for i, stock in enumerate(results['stocks_insufficient'], 1):
            print(f"\n{i}. {stock['symbol']}")
            print(f"   Data Range: {stock['start_date']} to {stock['end_date']}")
            print(f"   Total Days: {stock['total_days']} days ({stock['years_available']:.2f} years)")
            print(f"   Record Count: {stock['record_count']} records")
            print(f"   Strategy Start (with {BUFFER_WEEKS}-week buffer): {stock['strategy_start_with_buffer']}")
            print(f"   Days Available After Buffer: {stock['days_available_after_buffer']} days")
            print(f"   Years Available After Buffer: {stock['years_after_buffer']:.2f} years")
            
            if stock['days_available_after_buffer'] < 0:
                shortage = abs(stock['days_available_after_buffer'])
                print(f"   ❌ SHORTAGE: {shortage} days missing (buffer exceeds data range)")
            elif stock['days_available_after_buffer'] < 365:
                shortage = 365 - stock['days_available_after_buffer']
                print(f"   ⚠️  SHORTAGE: {shortage} days short of minimum 1-year backtest period")
    
    print(f"\n✅ Stocks with SUFFICIENT data: {results['stocks_sufficient']} stocks")
    print(f"   (These stocks have at least {BUFFER_DAYS} days buffer + 365 days for backtesting)")
    
    print("\n" + "=" * 100)


def export_to_csv(results: Dict, filename: str = "insufficient_data_stocks.csv"):
    """Export insufficient data stocks to CSV"""
    if not results['stocks_insufficient']:
        print("No insufficient data stocks to export.")
        return
    
    df = pd.DataFrame(results['stocks_insufficient'])
    df = df[[
        'symbol', 'start_date', 'end_date', 'total_days', 
        'years_available', 'strategy_start_with_buffer',
        'days_available_after_buffer', 'years_after_buffer', 'record_count'
    ]]
    
    filepath = os.path.join(os.path.dirname(__file__), filename)
    df.to_csv(filepath, index=False)
    print(f"\n💾 Exported insufficient data stocks to: {filepath}")


if __name__ == "__main__":
    print("🔍 Checking stock data availability for RS Strategy...")
    print(f"📏 Buffer requirement: {BUFFER_WEEKS} weeks ({BUFFER_DAYS} days)")
    print(f"📏 Minimum backtest period: 365 days")
    print(f"📏 Total required: {MIN_REQUIRED_DAYS} days\n")
    
    try:
        results = check_stock_data_availability()
        print_report(results)
        
        # Export to CSV
        if results['stocks_insufficient']:
            export_to_csv(results)
        
        print(f"\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


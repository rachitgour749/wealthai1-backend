#!/usr/bin/env python3
"""
Script to fetch missing EOD data for RS Strategy on specific dates
Dates: 6, 7, 8, 9, 10, 13, 14, 15, 16, 17 of October 2025
"""

import os
import sys
import sqlite3
import pandas as pd
from datetime import datetime, date
import yfinance as yf
import time
import logging

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, current_dir)

# Import RS EOD data fetcher
try:
    from rs_eod_data_fetcher import RSEODDataFetcher
except ImportError as e:
    print(f"Error importing RSEODDataFetcher: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(parent_dir, 'logs', 'missing_dates_fetch.log')),
        logging.StreamHandler()
    ]
)

def get_missing_dates():
    """Get the list of missing dates"""
    return [
        date(2025, 10, 6),   # Monday
        date(2025, 10, 7),   # Tuesday
        date(2025, 10, 8),   # Wednesday
        date(2025, 10, 9),   # Thursday
        date(2025, 10, 10),  # Friday
        date(2025, 10, 13),  # Monday
        date(2025, 10, 14),  # Tuesday
        date(2025, 10, 15),  # Wednesday
        date(2025, 10, 16),  # Thursday
        date(2025, 10, 17),  # Friday
    ]

def check_existing_data(db_path, target_date):
    """Check if data already exists for a specific date"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check stock_data table
        cursor.execute('''
            SELECT COUNT(*) FROM stock_data 
            WHERE date = ?
        ''', (target_date.strftime('%Y-%m-%d'),))
        
        stock_count = cursor.fetchone()[0]
        
        # Check index_data table
        cursor.execute('''
            SELECT COUNT(*) FROM index_data 
            WHERE date = ?
        ''', (target_date.strftime('%Y-%m-%d'),))
        
        index_count = cursor.fetchone()[0]
        
        conn.close()
        
        return stock_count, index_count
        
    except Exception as e:
        logging.error(f"Error checking existing data: {e}")
        return 0, 0

def fetch_data_for_date(fetcher, target_date):
    """Fetch data for a specific date"""
    try:
        logging.info(f"Fetching data for {target_date.strftime('%Y-%m-%d')} ({target_date.strftime('%A')})")
        
        # Check existing data first
        stock_count, index_count = check_existing_data(fetcher.db_path, target_date)
        logging.info(f"Existing data - Stocks: {stock_count}, Index: {index_count}")
        
        if stock_count > 0 and index_count > 0:
            logging.info(f"Data already exists for {target_date.strftime('%Y-%m-%d')}")
            return True
        
        # Fetch historical data for the specific date
        logging.info(f"Fetching historical data for {target_date.strftime('%Y-%m-%d')}")
        
        # Calculate start and end dates for yfinance
        start_date = target_date - pd.Timedelta(days=5)  # Get a few days before
        end_date = target_date + pd.Timedelta(days=2)    # Get a few days after
        
        # Fetch stock data
        stock_symbols = fetcher.stock_symbols
        fetched_stocks = 0
        failed_stocks = 0
        
        for symbol in stock_symbols:  # Fetch all 500 stocks
            try:
                # Add .NS suffix for NSE stocks
                yf_symbol = f"{symbol}.NS"
                ticker = yf.Ticker(yf_symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if not data.empty:
                    # Find data for the target date
                    target_data = data[data.index.date == target_date]
                    
                    if not target_data.empty:
                        latest = target_data.iloc[-1]
                        
                        # Save to database
                        fetcher.save_stock_data([{
                            'symbol': symbol,
                            'date': target_date.strftime('%Y-%m-%d'),
                            'open': float(latest['Open']),
                            'high': float(latest['High']),
                            'low': float(latest['Low']),
                            'close': float(latest['Close']),
                            'volume': int(latest['Volume']),
                            'adj_close': float(latest['Close'])
                        }])
                        fetched_stocks += 1
                        logging.info(f"SUCCESS {symbol}: {latest['Close']:.2f}")
                    else:
                        failed_stocks += 1
                        logging.warning(f"FAILED {symbol}: No data for {target_date}")
                else:
                    failed_stocks += 1
                    logging.warning(f"FAILED {symbol}: No data available")
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                failed_stocks += 1
                logging.error(f"ERROR {symbol}: {e}")
        
        # Fetch index data (Nifty 50)
        try:
            nifty_ticker = yf.Ticker("^NSEI")
            nifty_data = nifty_ticker.history(start=start_date, end=end_date)
            
            if not nifty_data.empty:
                target_data = nifty_data[nifty_data.index.date == target_date]
                
                if not target_data.empty:
                    latest = target_data.iloc[-1]
                    
                    fetcher.save_index_data([{
                        'symbol': '^NSEI',
                        'date': target_date.strftime('%Y-%m-%d'),
                        'open': float(latest['Open']),
                        'high': float(latest['High']),
                        'low': float(latest['Low']),
                        'close': float(latest['Close']),
                        'volume': int(latest['Volume']),
                        'adj_close': float(latest['Close'])
                    }])
                    logging.info(f"SUCCESS Nifty 50: {latest['Close']:.2f}")
                else:
                    logging.warning(f"FAILED Nifty 50: No data for {target_date}")
            else:
                logging.warning(f"FAILED Nifty 50: No data available")
                
        except Exception as e:
            logging.error(f"ERROR Nifty 50: {e}")
        
        # Verify data was saved
        final_stock_count, final_index_count = check_existing_data(fetcher.db_path, target_date)
        logging.info(f"Final data - Stocks: {final_stock_count}, Index: {final_index_count}")
        
        if final_stock_count > 0 and final_index_count > 0:
            logging.info(f"SUCCESS: Successfully fetched data for {target_date.strftime('%Y-%m-%d')}")
            return True
        else:
            logging.error(f"FAILED: Failed to fetch data for {target_date.strftime('%Y-%m-%d')}")
            return False
            
    except Exception as e:
        logging.error(f"ERROR: Error fetching data for {target_date}: {e}")
        return False

def main():
    """Main function to fetch missing dates"""
    print("=" * 80)
    print("RS Strategy - Missing Dates EOD Data Fetch")
    print("=" * 80)
    print()
    
    # Initialize fetcher
    db_path = os.path.join(parent_dir, 'Strategies', 'rsStrategy', 'nifty500_data_with_metadata.sqlite')
    fetcher = RSEODDataFetcher(db_path)
    
    # Get missing dates
    missing_dates = get_missing_dates()
    
    print(f"Fetching data for {len(missing_dates)} dates:")
    for d in missing_dates:
        print(f"   - {d.strftime('%Y-%m-%d')} ({d.strftime('%A')})")
    print()
    
    # Fetch data for each date
    success_count = 0
    failed_count = 0
    
    for i, target_date in enumerate(missing_dates, 1):
        print(f"\n{'='*60}")
        print(f"Processing Date {i}/{len(missing_dates)}: {target_date.strftime('%Y-%m-%d')} ({target_date.strftime('%A')})")
        print(f"{'='*60}")
        
        success = fetch_data_for_date(fetcher, target_date)
        
        if success:
            success_count += 1
            print(f"SUCCESS: Data fetched for {target_date.strftime('%Y-%m-%d')}")
        else:
            failed_count += 1
            print(f"FAILED: Could not fetch data for {target_date.strftime('%Y-%m-%d')}")
        
        # Wait between dates to avoid rate limiting
        if i < len(missing_dates):
            print("Waiting 5 seconds before next date...")
            time.sleep(5)
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"SUCCESS: Successfully fetched: {success_count} dates")
    print(f"FAILED: Failed to fetch: {failed_count} dates")
    print(f"TOTAL: Total processed: {len(missing_dates)} dates")
    
    if success_count > 0:
        print(f"\nData fetch completed! Check your database for the new data.")
        print(f"Database: {db_path}")
        print(f"Logs: {os.path.join(parent_dir, 'logs', 'missing_dates_fetch.log')}")
    
    if failed_count > 0:
        print(f"\nSome dates failed. Check the logs for details.")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()

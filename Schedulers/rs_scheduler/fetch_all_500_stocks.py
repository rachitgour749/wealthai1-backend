#!/usr/bin/env python3
"""
Script to fetch ALL 500 stocks for RS Strategy on specific dates
Dates: 6, 7, 8, 9, 10, 13, 14, 15, 16, 17 of October 2025
This will fetch ALL 500 stocks, not just 50
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
        logging.FileHandler(os.path.join(parent_dir, 'logs', 'all_500_stocks_fetch.log')),
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

def clear_existing_data_for_dates(db_path, dates):
    """Clear existing data for specific dates to allow fresh fetch"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        date_strings = [d.strftime('%Y-%m-%d') for d in dates]
        placeholders = ','.join(['?' for _ in date_strings])
        
        # Clear stock data for these dates
        cursor.execute(f'''
            DELETE FROM stock_data 
            WHERE date IN ({placeholders})
        ''', date_strings)
        
        # Clear index data for these dates
        cursor.execute(f'''
            DELETE FROM index_data 
            WHERE date IN ({placeholders})
        ''', date_strings)
        
        conn.commit()
        conn.close()
        
        logging.info(f"Cleared existing data for {len(dates)} dates")
        return True
        
    except Exception as e:
        logging.error(f"Error clearing existing data: {e}")
        return False

def fetch_data_for_date(fetcher, target_date):
    """Fetch data for a specific date - ALL 500 stocks"""
    try:
        logging.info(f"Fetching data for {target_date.strftime('%Y-%m-%d')} ({target_date.strftime('%A')})")
        
        # Calculate start and end dates for yfinance
        start_date = target_date - pd.Timedelta(days=5)  # Get a few days before
        end_date = target_date + pd.Timedelta(days=2)    # Get a few days after
        
        # Fetch stock data - ALL 500 stocks
        stock_symbols = fetcher.stock_symbols
        fetched_stocks = 0
        failed_stocks = 0
        
        logging.info(f"Starting to fetch data for {len(stock_symbols)} stocks...")
        
        for i, symbol in enumerate(stock_symbols, 1):
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
                        
                        if i % 50 == 0:  # Log progress every 50 stocks
                            logging.info(f"Progress: {i}/{len(stock_symbols)} stocks processed")
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
        
        # Final summary for this date
        logging.info(f"Date {target_date.strftime('%Y-%m-%d')} completed:")
        logging.info(f"  - Successfully fetched: {fetched_stocks} stocks")
        logging.info(f"  - Failed to fetch: {failed_stocks} stocks")
        logging.info(f"  - Success rate: {(fetched_stocks/(fetched_stocks+failed_stocks)*100):.1f}%")
        
        return fetched_stocks > 0
            
    except Exception as e:
        logging.error(f"ERROR: Error fetching data for {target_date}: {e}")
        return False

def main():
    """Main function to fetch missing dates"""
    print("=" * 80)
    print("RS Strategy - ALL 500 STOCKS Missing Dates EOD Data Fetch")
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
    
    # Clear existing data for these dates first
    print("Clearing existing data for these dates...")
    clear_existing_data_for_dates(db_path, missing_dates)
    print()
    
    # Fetch data for each date
    success_count = 0
    failed_count = 0
    total_stocks_fetched = 0
    
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
            print("Waiting 10 seconds before next date...")
            time.sleep(10)
    
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
        print(f"Logs: {os.path.join(parent_dir, 'logs', 'all_500_stocks_fetch.log')}")
        print(f"\nExpected records per date: ~500 stocks + 1 index = ~501 records")
        print(f"Total expected records: {success_count * 501}")
    
    if failed_count > 0:
        print(f"\nSome dates failed. Check the logs for details.")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()

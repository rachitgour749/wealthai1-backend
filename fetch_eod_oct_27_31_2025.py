#!/usr/bin/env python3
"""
Script to fetch EOD data for ETF, RS, and STOCK strategies
Dates: October 27, 28, 29, 30, 31, 2025
"""

import os
import sys
import sqlite3
import pandas as pd
from datetime import datetime, date
import yfinance as yf
import time
import logging
import warnings

# Suppress yfinance error logging for missing/delisted symbols
yfinance_logger = logging.getLogger('yfinance')
yfinance_logger.setLevel(logging.CRITICAL)  # Only show critical errors

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*possibly delisted.*")
warnings.filterwarnings("ignore", message=".*delisted.*")
warnings.filterwarnings("ignore", message=".*not found.*")

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'Schedulers', 'rs_scheduler'))
sys.path.insert(0, os.path.join(current_dir, 'Schedulers', 'etf_stock_scheduler'))

# Import fetchers
try:
    from rs_eod_data_fetcher import RSEODDataFetcher
except ImportError as e:
    print(f"Error importing RSEODDataFetcher: {e}")
    RSEODDataFetcher = None

try:
    from scheduler import IndianMarketDataFetcher
except ImportError as e:
    print(f"Error importing IndianMarketDataFetcher: {e}")
    IndianMarketDataFetcher = None

# Setup logging
logs_dir = os.path.join(current_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'eod_fetch_oct_27_31_2025.log')),
        logging.StreamHandler()
    ]
)

def get_target_dates():
    """Get the list of target dates"""
    return [
        date(2025, 10, 27),   # Monday
        date(2025, 10, 28),   # Tuesday
        date(2025, 10, 29),   # Wednesday
        date(2025, 10, 30),   # Thursday
        date(2025, 10, 31),   # Friday
    ]

def fetch_rs_data_for_date(fetcher, target_date):
    """Fetch RS strategy data for a specific date"""
    if not fetcher:
        logging.warning("RS fetcher not available")
        return False
    
    try:
        logging.info(f"[RS] Fetching data for {target_date.strftime('%Y-%m-%d')} ({target_date.strftime('%A')})")
        
        # Calculate date range for yfinance
        start_date = target_date - pd.Timedelta(days=5)
        end_date = target_date + pd.Timedelta(days=2)
        
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Fetch stock data
        stock_symbols = fetcher.stock_symbols
        fetched_stocks = 0
        failed_stocks = 0
        
        logging.info(f"[RS] Fetching data for {len(stock_symbols)} stocks...")
        
        for i, symbol in enumerate(stock_symbols, 1):
            try:
                yf_symbol = f"{symbol}.NS"
                
                # Suppress yfinance warnings and errors for missing/delisted symbols
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Temporarily set yfinance logger level to ERROR
                    old_level = yfinance_logger.level
                    yfinance_logger.setLevel(logging.ERROR)
                    try:
                        ticker = yf.Ticker(yf_symbol)
                        data = ticker.history(start=start_date_str, end=end_date_str, raise_errors=False)
                    finally:
                        yfinance_logger.setLevel(old_level)
                
                if not data.empty:
                    target_data = data[data.index.date == target_date]
                    
                    if not target_data.empty:
                        latest = target_data.iloc[-1]
                        
                        fetcher.save_stock_data([{
                            'symbol': symbol,
                            'date': target_date,
                            'open': float(latest['Open']),
                            'high': float(latest['High']),
                            'low': float(latest['Low']),
                            'close': float(latest['Close']),
                            'volume': int(latest['Volume']),
                            'adj_close': float(latest['Close'])
                        }])
                        fetched_stocks += 1
                        
                        if i % 50 == 0:
                            logging.info(f"[RS] Progress: {i}/{len(stock_symbols)} stocks processed ({fetched_stocks} successful, {failed_stocks} failed)")
                    else:
                        failed_stocks += 1
                        # Only log if it's not a weekend/holiday (some failures are expected)
                        if i % 100 == 0:  # Log every 100th failure to reduce noise
                            logging.debug(f"[RS] No data for {symbol} on {target_date.strftime('%Y-%m-%d')}")
                else:
                    failed_stocks += 1
                    if i % 100 == 0:
                        logging.debug(f"[RS] No data available for {symbol}")
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                failed_stocks += 1
                # Only log errors that aren't about missing/delisted symbols
                error_msg = str(e).lower()
                if 'delisted' not in error_msg and 'not found' not in error_msg and '404' not in error_msg:
                    logging.warning(f"[RS] Error for {symbol}: {e}")
        
        # Fetch index data (Nifty 50)
        try:
            # Suppress yfinance warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                old_level = yfinance_logger.level
                yfinance_logger.setLevel(logging.ERROR)
                try:
                    nifty_ticker = yf.Ticker("^NSEI")
                    nifty_data = nifty_ticker.history(start=start_date_str, end=end_date_str, raise_errors=False)
                finally:
                    yfinance_logger.setLevel(old_level)
            
            if not nifty_data.empty:
                target_data = nifty_data[nifty_data.index.date == target_date]
                
                if not target_data.empty:
                    latest = target_data.iloc[-1]
                    
                    fetcher.save_index_data([{
                        'symbol': '^NSEI',
                        'date': target_date,
                        'open': float(latest['Open']),
                        'high': float(latest['High']),
                        'low': float(latest['Low']),
                        'close': float(latest['Close']),
                        'volume': int(latest['Volume']),
                        'adj_close': float(latest['Close'])
                    }])
                    logging.info(f"[RS] SUCCESS Nifty 50: {latest['Close']:.2f}")
                else:
                    logging.warning(f"[RS] FAILED Nifty 50: No data for {target_date}")
            else:
                logging.warning(f"[RS] FAILED Nifty 50: No data available")
                
        except Exception as e:
            logging.error(f"[RS] ERROR Nifty 50: {e}")
        
        success_rate = (fetched_stocks / len(stock_symbols)) * 100 if stock_symbols else 0
        logging.info(f"[RS] Completed: {fetched_stocks}/{len(stock_symbols)} stocks fetched ({success_rate:.1f}%), {failed_stocks} failed")
        
        # Return True if we got at least 80% of stocks (some failures are normal for delisted/missing symbols)
        return fetched_stocks >= (len(stock_symbols) * 0.8)
        
    except Exception as e:
        logging.error(f"[RS] ERROR: Error fetching data for {target_date}: {e}")
        return False

def fetch_etf_stock_data_for_date(fetcher, target_date):
    """Fetch ETF and Stock data for a specific date"""
    if not fetcher:
        logging.warning("ETF/Stock fetcher not available")
        return False
    
    try:
        logging.info(f"[ETF/STOCK] Fetching data for {target_date.strftime('%Y-%m-%d')} ({target_date.strftime('%A')})")
        
        # Calculate date range
        start_date = target_date - pd.Timedelta(days=5)
        end_date = target_date + pd.Timedelta(days=2)
        
        fetched_count = 0
        failed_count = 0
        
        all_symbols = fetcher.all_symbols
        logging.info(f"[ETF/STOCK] Fetching data for {len(all_symbols)} symbols (ETF + Stock)...")
        
        for i, symbol in enumerate(all_symbols, 1):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if not data.empty:
                    target_data = data[data.index.date == target_date]
                    
                    if not target_data.empty:
                        latest = target_data.iloc[-1]
                        
                        symbol_data = {
                            'symbol': symbol,
                            'success': True,
                            'data': {
                                'date': target_date.strftime('%Y-%m-%d'),
                                'open': float(latest['Open']),
                                'high': float(latest['High']),
                                'low': float(latest['Low']),
                                'close': float(latest['Close']),
                                'volume': int(latest['Volume']),
                                'adj_close': float(latest['Close'])
                            }
                        }
                        
                        fetcher.save_to_database(symbol_data)
                        fetched_count += 1
                        
                        if i % 10 == 0:
                            logging.info(f"[ETF/STOCK] Progress: {i}/{len(all_symbols)} symbols processed")
                    else:
                        failed_count += 1
                else:
                    failed_count += 1
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                failed_count += 1
                logging.error(f"[ETF/STOCK] ERROR {symbol}: {e}")
        
        logging.info(f"[ETF/STOCK] Completed: {fetched_count} symbols fetched, {failed_count} failed")
        return fetched_count > 0
        
    except Exception as e:
        logging.error(f"[ETF/STOCK] ERROR: Error fetching data for {target_date}: {e}")
        return False

def main():
    """Main function to fetch EOD data for all strategies"""
    print("=" * 80)
    print("EOD Data Fetch - ETF and STOCK Strategies (RS SKIPPED)")
    print("Dates: October 27, 28, 29, 30, 31, 2025")
    print("=" * 80)
    print()
    
    # Initialize fetchers (SKIP RS - only ETF/STOCK)
    rs_fetcher = None
    logging.info("[RS] SKIPPED: RS data fetching is disabled")
    
    etf_stock_fetcher = None
    if IndianMarketDataFetcher:
        try:
            etf_stock_fetcher = IndianMarketDataFetcher()
            logging.info(f"[ETF/STOCK] Fetcher initialized. Database: {etf_stock_fetcher.db_path}")
        except Exception as e:
            logging.error(f"[ETF/STOCK] Failed to initialize fetcher: {e}")
    
    # Get target dates
    target_dates = get_target_dates()
    
    print(f"Fetching data for {len(target_dates)} dates:")
    for d in target_dates:
        print(f"   - {d.strftime('%Y-%m-%d')} ({d.strftime('%A')})")
    print()
    
    # Results tracking (SKIP RS)
    etf_stock_results = {"success": 0, "failed": 0}
    
    # Fetch data for each date
    for i, target_date in enumerate(target_dates, 1):
        print(f"\n{'='*60}")
        print(f"Processing Date {i}/{len(target_dates)}: {target_date.strftime('%Y-%m-%d')} ({target_date.strftime('%A')})")
        print(f"{'='*60}")
        
        # SKIP RS data fetching
        print(f"[RS Strategy] SKIPPED: RS data fetching is disabled")
        
        # Fetch ETF/Stock data
        if etf_stock_fetcher:
            print(f"\n[ETF/STOCK Strategy] Fetching data...")
            etf_stock_success = fetch_etf_stock_data_for_date(etf_stock_fetcher, target_date)
            if etf_stock_success:
                etf_stock_results["success"] += 1
                print(f"[ETF/STOCK] SUCCESS: Data fetched for {target_date.strftime('%Y-%m-%d')}")
            else:
                etf_stock_results["failed"] += 1
                print(f"[ETF/STOCK] FAILED: Could not fetch data for {target_date.strftime('%Y-%m-%d')}")
        else:
            print(f"[ETF/STOCK] SKIPPED: Fetcher not available")
        
        # Wait between dates to avoid rate limiting
        if i < len(target_dates):
            print("\nWaiting 5 seconds before next date...")
            time.sleep(5)
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"\n[RS Strategy]")
    print(f"  SKIPPED: RS data fetching was disabled")
    
    print(f"\n[ETF/STOCK Strategy]")
    print(f"  SUCCESS: {etf_stock_results['success']} dates")
    print(f"  FAILED: {etf_stock_results['failed']} dates")
    if etf_stock_fetcher:
        print(f"  Database: {etf_stock_fetcher.db_path}")
    
    print(f"\nLogs: {os.path.join(logs_dir, 'eod_fetch_oct_27_31_2025.log')}")
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()


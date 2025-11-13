#!/usr/bin/env python3
"""
Script to fetch EOD (End of Day) data for ETF and STOCK strategies
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

# Suppress yfinance warnings
yfinance_logger = logging.getLogger('yfinance')
yfinance_logger.setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

# Suppress the signal generator import warning
logging.getLogger('root').setLevel(logging.ERROR)

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'Schedulers', 'etf_stock_scheduler'))

# Import ETF/Stock fetcher (suppress warnings during import)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from scheduler import IndianMarketDataFetcher
    except ImportError as e:
        print(f"[ERROR] Error importing IndianMarketDataFetcher: {e}")
        IndianMarketDataFetcher = None

# Setup logging
logs_dir = os.path.join(current_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'eod_fetch_etf_stock_oct_27_31_2025.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def get_target_dates():
    """Get the list of target dates"""
    return [
        date(2025, 10, 27),   # Monday
        date(2025, 10, 28),   # Tuesday
        date(2025, 10, 29),   # Wednesday
        date(2025, 10, 30),   # Thursday
        date(2025, 10, 31),   # Friday
    ]

def fetch_data_for_date(fetcher, target_date):
    """Fetch ETF and Stock data for a specific date"""
    if not fetcher:
        logger.warning("ETF/Stock fetcher not available")
        return False
    
    try:
        logger.info(f"[FETCH] Fetching data for {target_date.strftime('%Y-%m-%d')} ({target_date.strftime('%A')})")
        
        # Calculate date range for yfinance (with buffer)
        start_date = target_date - pd.Timedelta(days=5)
        end_date = target_date + pd.Timedelta(days=2)
        
        fetched_count = 0
        failed_count = 0
        
        # Get all symbols (ETF + Stock)
        all_symbols = fetcher.all_symbols
        logger.info(f"[FETCH] Fetching data for {len(all_symbols)} symbols (ETF + Stock)...")
        
        for i, symbol in enumerate(all_symbols, 1):
            try:
                # Fetch data with error suppression
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    old_level = yfinance_logger.level
                    yfinance_logger.setLevel(logging.ERROR)
                    try:
                        ticker = yf.Ticker(symbol)
                        data = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                                            end=end_date.strftime('%Y-%m-%d'), 
                                            raise_errors=False)
                    finally:
                        yfinance_logger.setLevel(old_level)
                
                if not data.empty:
                    # Filter data for target date
                    data.index = pd.to_datetime(data.index)
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
                        
                        # Save to database
                        fetcher.save_to_database(symbol_data)
                        fetched_count += 1
                        
                        if i % 20 == 0:
                            logger.info(f"   Progress: {i}/{len(all_symbols)} symbols ({fetched_count} successful, {failed_count} failed)")
                    else:
                        failed_count += 1
                else:
                    failed_count += 1
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                failed_count += 1
                error_msg = str(e).lower()
                # Only log non-trivial errors
                if 'delisted' not in error_msg and 'not found' not in error_msg and '404' not in error_msg:
                    if i % 50 == 0:  # Log every 50th error to reduce noise
                        logger.warning(f"   Error for {symbol}: {e}")
        
        success_rate = (fetched_count / len(all_symbols)) * 100 if all_symbols else 0
        logger.info(f"[OK] Completed {target_date.strftime('%Y-%m-%d')}: {fetched_count}/{len(all_symbols)} symbols ({success_rate:.1f}%)")
        
        return fetched_count > 0
        
    except Exception as e:
        logger.error(f"[ERROR] Error fetching data for {target_date}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to fetch EOD data for ETF and Stock"""
    try:
        print("=" * 80)
        print("EOD Data Fetch - ETF and STOCK Strategies")
        print("Dates: October 27, 28, 29, 30, 31, 2025")
        print("=" * 80)
        print()
        sys.stdout.flush()
        
        # Initialize fetcher
        fetcher = None
        if IndianMarketDataFetcher:
            try:
                fetcher = IndianMarketDataFetcher()
                logger.info(f"[OK] Fetcher initialized")
                logger.info(f"   Database: {fetcher.db_path}")
                logger.info(f"   ETF symbols: {len(fetcher.etf_symbols)}")
                logger.info(f"   Stock symbols: {len(fetcher.stock_symbols)}")
                logger.info(f"   Total symbols: {len(fetcher.all_symbols)}")
            except Exception as e:
                logger.error(f"[ERROR] Failed to initialize fetcher: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return
        
        if not fetcher:
            logger.error("[ERROR] Cannot proceed without fetcher")
            return
        
        # Get target dates
        target_dates = get_target_dates()
        
        print(f"\nTarget Dates ({len(target_dates)}):")
        for d in target_dates:
            print(f"   - {d.strftime('%Y-%m-%d')} ({d.strftime('%A')})")
        print()
        
        # Results tracking
        results = {"success": 0, "failed": 0}
        
        # Fetch data for each date
        for i, target_date in enumerate(target_dates, 1):
            try:
                print(f"\n{'='*80}")
                print(f"Processing Date {i}/{len(target_dates)}: {target_date.strftime('%Y-%m-%d')} ({target_date.strftime('%A')})")
                print(f"{'='*80}")
                sys.stdout.flush()
                
                success = fetch_data_for_date(fetcher, target_date)
                if success:
                    results["success"] += 1
                    print(f"[SUCCESS] Data fetched for {target_date.strftime('%Y-%m-%d')}")
                    logger.info(f"[SUCCESS] Date {target_date.strftime('%Y-%m-%d')} completed successfully")
                else:
                    results["failed"] += 1
                    print(f"[FAILED] Could not fetch data for {target_date.strftime('%Y-%m-%d')}")
                    logger.error(f"[FAILED] Date {target_date.strftime('%Y-%m-%d')} failed")
                
                # Wait between dates to avoid rate limiting
                if i < len(target_dates):
                    print("\nWaiting 5 seconds before next date...")
                    logger.info(f"Waiting 5 seconds before processing next date...")
                    time.sleep(5)
                
                # Force flush output to ensure we see progress
                sys.stdout.flush()
                
            except KeyboardInterrupt:
                logger.warning("Script interrupted by user")
                print("\n[WARNING] Script interrupted by user")
                break
            except Exception as e:
                logger.error(f"[ERROR] Unexpected error processing {target_date.strftime('%Y-%m-%d')}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                print(f"[ERROR] Unexpected error: {e}")
                results["failed"] += 1
                # Continue with next date instead of stopping
                continue
    
        # Final summary
        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"\n[SUCCESS] {results['success']} dates")
        print(f"[FAILED] {results['failed']} dates")
        print(f"\nDatabase: {fetcher.db_path}")
        print(f"Logs: {os.path.join(logs_dir, 'eod_fetch_etf_stock_oct_27_31_2025.log')}")
        print(f"\n{'='*80}")
        logger.info(f"Script completed: {results['success']} successful, {results['failed']} failed")
        
    except KeyboardInterrupt:
        print("\n[WARNING] Script interrupted by user")
        logger.warning("Script interrupted by user")
    except Exception as e:
        logger.error(f"[ERROR] Fatal error in main: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\n[ERROR] Fatal error: {e}")
        print("Check logs for details")

if __name__ == "__main__":
    main()


# Complete ETF EOD Data Fetcher Script for Indian ETFs with Automatic Scheduling
import sqlite3
import yfinance as yf
import logging
import pandas as pd
from datetime import datetime, date, time as dt_time
import holidays
import time
import sys
import os
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
import threading

# Setup enhanced logging with rotation
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('logs/etf_scheduler.log', maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler(sys.stdout)
    ]
)

class IndianETFDataFetcher:
    def __init__(self, db_path='../unified_etf_data.sqlite'):
        """
        Initialize the Indian ETF data fetcher with database path and configuration
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = os.path.abspath(db_path)
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.etf_symbols = [
            'BANKBEES.NS',
            'CPSEETF.NS',
            'GOLDBEES.NS',
            'INFRABEES.NS',
            'ITBEES.NS',
            'JUNIORBEES.NS',
            'LIQUIDBEES.NS',
            'METALIETF.NS',
            'MIDCAPETF.NS',
            'MODEFENCE.NS',
            'MON100.NS',    
            'NIFTYBEES.NS',
            'OILIETF.NS',
            'PHARMABEES.NS',
            'PSUBNKBEES.NS'
        ]
        
        # Initialize Indian market holidays
        self.india_holidays = holidays.India()
        
        # Market hours (IST)
        self.market_open = dt_time(9, 15)  # 9:15 AM
        self.market_close = dt_time(15, 30)  # 3:30 PM
        
    def is_trading_day(self, check_date=None):
        """
        Check if given date (or today) is a trading day in India
        
        Args:
            check_date (date, optional): Date to check. Defaults to today.
            
        Returns:
            bool: True if trading day, False otherwise
        """
        if check_date is None:
            check_date = date.today()
            
        # Check if it's a weekday (Monday=0, Sunday=6)
        if check_date.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Check if it's an Indian holiday
        if check_date in self.india_holidays:
            return False
        
        return True
    
    def is_after_market_close(self):
        """
        Check if current time is after market close (3:30 PM IST)
        
        Returns:
            bool: True if after market close, False otherwise
        """
        now = datetime.now()
        current_time = now.time()
        return current_time >= self.market_close
    
    def validate_data(self, data):
        """
        Validate fetched data before saving
        
        Args:
            data (pandas.Series): EOD data to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        required_fields = ['Open', 'High', 'Low', 'Close', 'Volume']
        for field in required_fields:
            if field not in data or pd.isna(data[field]):
                return False
        return True
    
    def create_database_table(self):
        """
        Create the ETF data table if it doesn't exist
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS etf_unified (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    open REAL NOT NULL,
                    close REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    adj_close REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(symbol, created_at)
                )
            ''')
            
            conn.commit()
            conn.close()
            logging.info("Database table created/verified successfully")
            
        except sqlite3.Error as e:
            logging.error(f"Database error: {e}")
            raise
    
    def fetch_etf_data(self, symbol, retries=3):
        """
        Fetch EOD data for a single Indian ETF symbol with retry logic
        
        Args:
            symbol (str): ETF ticker symbol with NSE suffix (e.g., BANKBEES.NS)
            retries (int): Number of retry attempts
            
        Returns:
            pandas.Series or None: EOD data or None if failed
        """
        for attempt in range(retries):
            try:
                ticker = yf.Ticker(symbol)
                hist_data = ticker.history(period='2d')
                
                if hist_data.empty:
                    logging.warning(f"No data returned for {symbol} on attempt {attempt + 1}")
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return None
                
                latest_data = hist_data.iloc[-1]
                
                # Validate data before returning
                if not self.validate_data(latest_data):
                    logging.warning(f"Invalid data for {symbol} on attempt {attempt + 1}")
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return None
                
                logging.info(f"Successfully fetched data for {symbol}")
                return latest_data
                
            except Exception as e:
                logging.error(f"Error fetching data for {symbol} on attempt {attempt + 1}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return None
    
    def save_to_database(self, symbol, data, date_str):
        """
        Save ETF data to SQLite database using UPSERT
        
        Args:
            symbol (str): ETF ticker symbol
            data (pandas.Series): EOD data
            date_str (str): Date string in YYYY-MM-DD format
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

                        # Remove .NS suffix from symbol before saving
            clean_symbol = symbol.replace('.NS', '') if symbol.endswith('.NS') else symbol
            
            # Handle different possible column names for adjusted close
            # For ETFs, if no Adj Close is available, use Close price
            adj_close_value = data.get('Adj Close', data.get('Adj_Close', data['Close']))
            
            cursor.execute('''
                INSERT OR REPLACE INTO etf_unified 
                (symbol, open, close, high, low, volume, adj_close, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                clean_symbol,
                float(data['Open']),
                float(data['Close']),
                float(data['High']),
                float(data['Low']),
                int(data['Volume']),
                float(adj_close_value),
                date_str
            ))
            
            conn.commit()
            conn.close()
            
            logging.info(f"Successfully saved {symbol} data to database")
            
        except sqlite3.Error as e:
            logging.error(f"Database error saving {symbol}: {e}")
            raise
        except Exception as e:
            logging.error(f"Error saving {symbol} data: {e}")
            raise
    
    def run_daily_fetch(self):
        """
        Main method to fetch and save EOD data for all Indian ETF symbols
        """
        start_time = datetime.now()
        logging.info("=== Starting daily Indian ETF data fetch ===")
        
        # Check if it's a trading day
        if not self.is_trading_day():
            logging.info("Today is not a trading day in India. Skipping data fetch.")
            return 0, 0
        
        # Check if it's after market close
        if not self.is_after_market_close():
            logging.info("Market is still open. Waiting for market close to fetch EOD data.")
            return 0, 0
        
        self.create_database_table()
        date_str = date.today().strftime('%Y-%m-%d')
        
        successful_fetches = 0
        failed_fetches = 0
        
        for symbol in self.etf_symbols:
            try:
                logging.info(f"Fetching data for {symbol}...")
                eod_data = self.fetch_etf_data(symbol)
                
                if eod_data is not None:
                    self.save_to_database(symbol, eod_data, date_str)
                    successful_fetches += 1
                else:
                    logging.error(f"Failed to fetch data for {symbol}")
                    failed_fetches += 1
                
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logging.error(f"Unexpected error processing {symbol}: {e}")
                failed_fetches += 1
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logging.info("=== Daily fetch completed ===")
        logging.info(f"Successful fetches: {successful_fetches}")
        logging.info(f"Failed fetches: {failed_fetches}")
        logging.info(f"Total duration: {duration.total_seconds():.2f} seconds")
        
        return successful_fetches, failed_fetches
    
class ETFScheduler:
    """
    Automatic scheduler for ETF data fetching
    """
    def __init__(self):
        self.fetcher = IndianETFDataFetcher('../unified_etf_data.sqlite')
        self.scheduler = BlockingScheduler()
        self.setup_scheduler()
    
    def setup_scheduler(self):
        """Setup the scheduler with job and event listeners"""
        # Add job to run daily at 4:00 PM IST (16:00)
        self.scheduler.add_job(
            func=self.fetch_etf_data_job,
            trigger=CronTrigger(hour=16, minute=0),  # 4:00 PM IST
            id='etf_daily_fetch',
            name='Daily ETF Data Fetch at 4 PM',
            max_instances=1,
            replace_existing=True
        )
        
        # Add event listeners for monitoring
        self.scheduler.add_listener(self.job_executed_listener, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self.job_error_listener, EVENT_JOB_ERROR)
        
        logging.info("Scheduler configured to run daily at 4:00 PM IST")
    
    def fetch_etf_data_job(self):
        """Job function to fetch ETF data"""
        try:
            logging.info("=== Scheduled ETF data fetch started ===")
            success_count, fail_count = self.fetcher.run_daily_fetch()
            
            if fail_count > 0:
                logging.warning(f"Scheduled fetch completed with some failures. Success: {success_count}, Failed: {fail_count}")
            else:
                logging.info(f"Scheduled fetch completed successfully. Fetched: {success_count} ETFs")
                
        except Exception as e:
            logging.critical(f"Critical error in scheduled fetch: {e}")
            raise
    
    def job_executed_listener(self, event):
        """Listener for successful job execution"""
        logging.info(f"Job {event.job_id} executed successfully at {event.scheduled_run_time}")
    
    def job_error_listener(self, event):
        """Listener for job execution errors"""
        logging.error(f"Job {event.job_id} failed with exception: {event.exception}")
    
    def start_scheduler(self):
        """Start the scheduler"""
        try:
            logging.info("Starting ETF data scheduler...")
            logging.info("Scheduler will run daily at 4:00 PM IST")
            logging.info("Press Ctrl+C to stop the scheduler")
            self.scheduler.start()
        except KeyboardInterrupt:
            logging.info("Scheduler stopped by user")
            self.scheduler.shutdown()
        except Exception as e:
            logging.critical(f"Error starting scheduler: {e}")
            raise

def main():
    """Main function - can run in manual or scheduled mode"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ETF Data Fetcher')
    parser.add_argument('--mode', choices=['manual', 'scheduled'], default='manual',
                       help='Run mode: manual (run once) or scheduled (continuous)')
    
    args = parser.parse_args()
    
    if args.mode == 'manual':
        # Manual execution (original behavior)
        try:
            fetcher = IndianETFDataFetcher()
            success_count, fail_count = fetcher.run_daily_fetch()
            
            if fail_count > 0:
                logging.warning(f"Some fetches failed. Success: {success_count}, Failed: {fail_count}")
                sys.exit(1)
            else:
                logging.info("All data fetches completed successfully")
                sys.exit(0)
        except Exception as e:
            logging.critical(f"Critical error in main execution: {e}")
            sys.exit(2)
    
    elif args.mode == 'scheduled':
        # Scheduled execution
        scheduler = ETFScheduler()
        scheduler.start_scheduler()

if __name__ == "__main__":
    main()

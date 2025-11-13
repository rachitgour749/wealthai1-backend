# Complete ETF & Stock EOD Data Fetcher Script for Indian Markets with Automatic Scheduling
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

# Add path for signal generators
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'live_market_engine', 'live_market_engine'))

# Import signal generators
try:
    from signal_generator import LiveSignalGenerator
    from stock_signal_generator import LiveStockSignalGenerator
except ImportError as e:
    logging.warning(f"Could not import signal generators: {e}")
    LiveSignalGenerator = None
    LiveStockSignalGenerator = None

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

class IndianMarketDataFetcher:
    def __init__(self, db_path=None):
        """
        Initialize the Indian market data fetcher with database path and configuration
        
        Args:
            db_path (str): Path to SQLite database file
        """
        if db_path is None:
            # Use absolute path to ensure consistency
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(current_dir, '..', 'unified_etf_data.sqlite')
        else:
            self.db_path = os.path.abspath(db_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # ETF symbols (corrected NSE symbols)
        self.etf_symbols = [
            'BANKBEES.NS',
            'GOLDBEES.NS',
            'HNGSNGBEES.NS',  # Corrected: Hang Seng ETF
            'INFRABEES.NS',
            'ITBEES.NS',
            'JUNIORBEES.NS',   # Corrected: Junior Nifty ETF
            'LIQUIDBEES.NS',
            'MON100.NS',
            'NIFTYBEES.NS',
            'PHARMABEES.NS',
            'PSUBNKBEES.NS',
            'SENSEX1.BO'    # Corrected: Sensex ETF
        ]
        
        # Stock symbols (Nifty 50 stocks)
        self.stock_symbols = [
            'ADANIENT.NS',
            'ADANIGREEN.NS',
            'ADANIPORTS.NS',
            'APOLLOHOSP.NS',
            'ASIANPAINT.NS',
            'AXISBANK.NS',
            'BAJAJ-AUTO.NS',
            'BAJAJFINSV.NS',
            'BAJFINANCE.NS',
            'BHARTIARTL.NS',
            'BPCL.NS',
            'BRITANNIA.NS',
            'CIPLA.NS',
            'COALINDIA.NS',
            'DIVISLAB.NS',
            'DRREDDY.NS',
            'EICHERMOT.NS',
            'GRASIM.NS',
            'HCLTECH.NS',
            'HDFCBANK.NS',
            'HDFCLIFE.NS',
            'HEROMOTOCO.NS',
            'HINDALCO.NS',
            'HINDUNILVR.NS',
            'ICICIBANK.NS',
            'INDUSINDBK.NS',
            'ITC.NS',
            'JSWSTEEL.NS',
            'KOTAKBANK.NS',
            'LT.NS',
            'LTIM.NS',
            'M&M.NS',
            'MARUTI.NS',
            'NESTLEIND.NS',
            'NTPC.NS',
            'ONGC.NS',
            'POWERGRID.NS',
            'RELIANCE.NS',
            'SBILIFE.NS',
            'SBIN.NS',
            'SUNPHARMA.NS',
            'TATACONSUM.NS',
            'TATAMOTORS.NS',
            'TATASTEEL.NS',
            'TCS.NS',
            'TECHM.NS',
            'TITAN.NS',
            'ULTRACEMCO.NS',
            'UPL.NS',
            'WIPRO.NS'
        ]
        
        # Combined symbols list
        self.all_symbols = self.etf_symbols + self.stock_symbols
        
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
        Create the unified market data table if it doesn't exist
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
                    asset_type TEXT DEFAULT 'ETF',
                    UNIQUE(symbol, created_at)
                )
            ''')
            
            # Add asset_type column if it doesn't exist (for existing databases)
            try:
                cursor.execute('ALTER TABLE etf_unified ADD COLUMN asset_type TEXT DEFAULT "ETF"')
                conn.commit()
                logging.info("Added asset_type column to existing table")
            except sqlite3.OperationalError:
                # Column already exists, ignore
                pass
            
            conn.commit()
            conn.close()
            logging.info("Database table created/verified successfully")
            
        except sqlite3.Error as e:
            logging.error(f"Database error: {e}")
            raise
    
    def fetch_market_data(self, symbol, retries=3):
        """
        Fetch EOD data for a single Indian market symbol (ETF or Stock) with retry logic
        
        Args:
            symbol (str): Market ticker symbol with NSE suffix (e.g., BANKBEES.NS, RELIANCE.NS)
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
        Save market data to SQLite database using UPSERT
        
        Args:
            symbol (str): Market ticker symbol
            data (pandas.Series): EOD data
            date_str (str): Date string in YYYY-MM-DD format
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Remove .NS suffix from symbol before saving
            clean_symbol = symbol.replace('.NS', '') if symbol.endswith('.NS') else symbol
            
            # Determine asset type
            asset_type = 'ETF' if symbol in self.etf_symbols else 'STOCK'
            
            # Handle different possible column names for adjusted close
            # For ETFs and stocks, if no Adj Close is available, use Close price
            adj_close_value = data.get('Adj Close', data.get('Adj_Close', data['Close']))
            
            cursor.execute('''
                INSERT OR REPLACE INTO etf_unified 
                (symbol, date, open, high, low, close, volume, adj_close, asset_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                clean_symbol,
                date_str,
                float(data['Open']),
                float(data['High']),
                float(data['Low']),
                float(data['Close']),
                int(data['Volume']),
                float(adj_close_value),
                asset_type
            ))
            
            conn.commit()
            conn.close()
            
            logging.info(f"Successfully saved {symbol} ({asset_type}) data to database")
            
        except sqlite3.Error as e:
            logging.error(f"Database error saving {symbol}: {e}")
            raise
        except Exception as e:
            logging.error(f"Error saving {symbol} data: {e}")
            raise
    
    def run_daily_fetch(self):
        """
        Main method to fetch and save EOD data for all Indian market symbols (ETFs and Stocks)
        """
        start_time = datetime.now()
        logging.info("=== Starting daily Indian market data fetch (ETFs + Stocks) ===")
        
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
        etf_count = 0
        stock_count = 0
        
        for symbol in self.all_symbols:
            try:
                asset_type = 'ETF' if symbol in self.etf_symbols else 'STOCK'
                logging.info(f"Fetching data for {symbol} ({asset_type})...")
                eod_data = self.fetch_market_data(symbol)
                
                if eod_data is not None:
                    try:
                        self.save_to_database(symbol, eod_data, date_str)
                        successful_fetches += 1
                        if asset_type == 'ETF':
                            etf_count += 1
                        else:
                            stock_count += 1
                    except Exception as db_error:
                        logging.error(f"Failed to save {symbol} to database: {db_error}")
                        failed_fetches += 1
                else:
                    logging.error(f"Failed to fetch data for {symbol}")
                    failed_fetches += 1
                
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logging.error(f"Unexpected error processing {symbol}: {e}")
                failed_fetches += 1
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logging.info("=== Daily market data fetch completed ===")
        logging.info(f"Successful fetches: {successful_fetches} ({etf_count} ETFs, {stock_count} Stocks)")
        logging.info(f"Failed fetches: {failed_fetches}")
        logging.info(f"Total duration: {duration.total_seconds():.2f} seconds")
        
        return successful_fetches, failed_fetches
    
class MarketDataScheduler:
    """
    Automatic scheduler for market data fetching (ETFs and Stocks)
    """
    def __init__(self):
        self.fetcher = IndianMarketDataFetcher()  # Use default path
        self.scheduler = BlockingScheduler()
        
        # Initialize signal generators
        self.etf_signal_generator = None
        self.stock_signal_generator = None
        
        if LiveSignalGenerator:
            try:
                self.etf_signal_generator = LiveSignalGenerator()
                logging.info("ETF signal generator initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize ETF signal generator: {e}")
        
        if LiveStockSignalGenerator:
            try:
                self.stock_signal_generator = LiveStockSignalGenerator()
                logging.info("Stock signal generator initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize Stock signal generator: {e}")
        
        self.setup_scheduler()
    
    def setup_scheduler(self):
        """Setup the scheduler with job and event listeners"""
        # Add job to run daily at 4:00 PM IST - PRODUCTION TIME
        self.scheduler.add_job(
            func=self.fetch_market_data_job,
            trigger=CronTrigger(hour=16, minute=0, timezone='Asia/Kolkata'),  # 4:00 PM IST - PRODUCTION
            id='market_daily_fetch',
            name='Daily Market Data Fetch at 4:00 PM (ETFs + Stocks)',
            max_instances=1,
            replace_existing=True
        )
        
        # Add job to run every Friday at 4:30 PM IST for ETF signal generation
        self.scheduler.add_job(
            func=self.generate_etf_signals_job,
            trigger=CronTrigger(day_of_week='fri', hour=16, minute=30, timezone='Asia/Kolkata'),
            id='etf_friday_signals',
            name='Friday ETF Signal Generation at 4:30 PM',
            max_instances=1,
            replace_existing=True
        )
        
        # Add job to run every Friday at 4:35 PM IST for Stock signal generation
        self.scheduler.add_job(
            func=self.generate_stock_signals_job,
            trigger=CronTrigger(day_of_week='fri', hour=16, minute=35, timezone='Asia/Kolkata'),
            id='stock_friday_signals',
            name='Friday Stock Signal Generation at 4:35 PM',
            max_instances=1,
            replace_existing=True
        )
        
        # Add job to execute signals every Monday at 9:20 AM IST (after market opens)
        # If Monday is a holiday, will execute on Tuesday at 9:20 AM
        # PRODUCTION SCHEDULE
        self.scheduler.add_job(
            func=self.execute_signals_monday_job_with_holiday_check,
            trigger=CronTrigger(day_of_week='mon', hour=9, minute=20, timezone='Asia/Kolkata'),
            id='monday_signal_execution',
            name='Monday Signal Execution at 9:20 AM IST (Production)',
            max_instances=1,  # Prevents multiple simultaneous executions
            replace_existing=True
        )
        
        # Fallback: Also schedule for Tuesday 9:20 AM (in case Monday was holiday)
        self.scheduler.add_job(
            func=self.execute_signals_tuesday_fallback,
            trigger=CronTrigger(day_of_week='tue', hour=9, minute=20, timezone='Asia/Kolkata'),
            id='tuesday_signal_execution_fallback',
            name='Tuesday Signal Execution Fallback at 9:20 AM IST',
            max_instances=1,
            replace_existing=True
        )
        
        # Add event listeners for monitoring
        self.scheduler.add_listener(self.job_executed_listener, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self.job_error_listener, EVENT_JOB_ERROR)
        
        logging.info("Scheduler configured:")
        logging.info("  - Daily EOD fetch: 4:00 PM IST (ETFs + Stocks)")
        logging.info("  - Friday ETF signals: 4:30 PM IST")
        logging.info("  - Friday Stock signals: 4:35 PM IST")
        logging.info("  - Monday Signal Execution: 9:20 AM IST (PRODUCTION)")
        logging.info("  - Tuesday Fallback Execution: 9:20 AM IST (if Monday is holiday)")
        logging.info("")
        logging.info("🚀 Automatic signal execution for ALL users every Monday at 10:22 AM IST")
    
    def fetch_market_data_job(self):
        """Job function to fetch market data (ETFs and Stocks)"""
        try:
            logging.info("=== Scheduled market data fetch started ===")
            success_count, fail_count = self.fetcher.run_daily_fetch()
            
            if fail_count > 0:
                logging.warning(f"Scheduled fetch completed with some failures. Success: {success_count}, Failed: {fail_count}")
            else:
                logging.info(f"Scheduled fetch completed successfully. Fetched: {success_count} market instruments")
                
        except Exception as e:
            logging.critical(f"Critical error in scheduled fetch: {e}")
            raise
    
    def generate_etf_signals_job(self):
        """Job function to generate signals on Friday for ALL deployments (ETF and Stock)"""
        try:
            logging.info("=== Scheduled Signal Generation Started (Friday) ===")
            logging.info("Processing both ETF and Stock Strategy deployments")
            
            # Get ALL active deployments from deploy_details (both ETF and Stock)
            try:
                conn = sqlite3.connect(self.fetcher.db_path)
                cursor = conn.cursor()
                
                # Get ALL deployments (ETF and Stock)
                cursor.execute('''
                    SELECT run_id, user_email, strategy_type, etf_count, stock_count
                    FROM deploy_details 
                    WHERE status = 'running'
                    ORDER BY strategy_type, created_at DESC
                ''')
                
                all_deployments = cursor.fetchall()  # Get ALL rows
                conn.close()
                
                if not all_deployments:
                    logging.warning("No active deployments found")
                    return
                
                # Separate ETF and Stock deployments
                etf_deployments = [(run_id, user_email, etf_count) 
                                   for run_id, user_email, strategy_type, etf_count, stock_count 
                                   in all_deployments if strategy_type == 'ETF Strategy']
                
                stock_deployments = [(run_id, user_email, stock_count) 
                                     for run_id, user_email, strategy_type, etf_count, stock_count 
                                     in all_deployments if strategy_type == 'Stock Strategy']
                
                logging.info(f"Found {len(etf_deployments)} ETF deployment(s) and {len(stock_deployments)} Stock deployment(s)")
                    
            except Exception as db_error:
                logging.error(f"Error fetching deployments from deploy_details: {db_error}")
                return
            
            # ========== PROCESS ETF STRATEGY DEPLOYMENTS ==========
            if etf_deployments and self.etf_signal_generator:
                logging.info(f"\n{'='*80}")
                logging.info("📈 PROCESSING ETF STRATEGY DEPLOYMENTS")
                logging.info(f"{'='*80}")
                
                etf_strategy_config = {
                    'max_positions': 5,
                    'min_rs_score': 0.4,
                    'min_momentum': 0.01,
                    'min_volume_ratio': 0.8,
                    'sell_threshold': 0.4
                }
                
                etf_total_signals = 0
                etf_successful = 0
                etf_failed = 0
                
                for run_id, user_email, etf_count in etf_deployments:
                    try:
                        logging.info(f"\n{'='*60}")
                        logging.info(f"Processing ETF deployment: {run_id}")
                        logging.info(f"User: {user_email}, ETF Count: {etf_count}")
                        logging.info(f"{'='*60}")
                        
                        result = self.etf_signal_generator.run_weekly_signal_generation(
                            run_id=run_id,
                            strategy_type='SurfTrend',
                            strategy_config=etf_strategy_config
                        )
                        
                        if result['success']:
                            signals_count = result['signals_count']
                            buy_count = result['buy_count']
                            sell_count = result['sell_count']
                            
                            logging.info(f"✅ ETF Deployment {run_id}: {signals_count} signals ({buy_count} BUY, {sell_count} SELL)")
                            etf_total_signals += signals_count
                            etf_successful += 1
                        else:
                            error_msg = result.get('error', 'Unknown error')
                            logging.error(f"❌ ETF Deployment {run_id} failed: {error_msg}")
                            etf_failed += 1
                            
                    except Exception as deploy_error:
                        logging.error(f"❌ Error processing ETF deployment {run_id}: {deploy_error}")
                        etf_failed += 1
                        continue
                
                logging.info(f"\n{'='*60}")
                logging.info(f"📊 ETF Signal Generation Summary:")
                logging.info(f"  Total ETF Deployments: {len(etf_deployments)}")
                logging.info(f"  Successful: {etf_successful}")
                logging.info(f"  Failed: {etf_failed}")
                logging.info(f"  Total Signals Generated: {etf_total_signals}")
                logging.info(f"{'='*60}\n")
            
            # ========== PROCESS STOCK STRATEGY DEPLOYMENTS ==========
            if stock_deployments:
                logging.info(f"\n{'='*80}")
                logging.info("📊 PROCESSING STOCK STRATEGY DEPLOYMENTS")
                logging.info(f"{'='*80}")
                
                # Import Stock signal generator
                try:
                    import sys
                    import os
                    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'live_market_engine', 'live_market_engine'))
                    from stock_signal_generator import LiveStockSignalGenerator
                    
                    stock_signal_generator = LiveStockSignalGenerator(self.fetcher.db_path)
                    
                    stock_strategy_config = {
                        'max_positions': 4,
                        'top_buy_count': 1,
                        'top_sell_count': 3
                    }
                    
                    stock_total_signals = 0
                    stock_successful = 0
                    stock_failed = 0
                    
                    for run_id, user_email, stock_count in stock_deployments:
                        try:
                            logging.info(f"\n{'='*60}")
                            logging.info(f"Processing Stock deployment: {run_id}")
                            logging.info(f"User: {user_email}, Stock Count: {stock_count}")
                            logging.info(f"{'='*60}")
                            
                            result = stock_signal_generator.run_weekly_signal_generation(
                                run_id=run_id,
                                strategy_type='StockSurfTrend',
                                strategy_config=stock_strategy_config
                            )
                            
                            if result['success']:
                                signals_count = result['signals_count']
                                buy_count = result['buy_count']
                                sell_count = result['sell_count']
                                
                                logging.info(f"✅ Stock Deployment {run_id}: {signals_count} signals ({buy_count} BUY, {sell_count} SELL)")
                                stock_total_signals += signals_count
                                stock_successful += 1
                            else:
                                error_msg = result.get('error', 'Unknown error')
                                logging.error(f"❌ Stock Deployment {run_id} failed: {error_msg}")
                                stock_failed += 1
                                
                        except Exception as deploy_error:
                            logging.error(f"❌ Error processing Stock deployment {run_id}: {deploy_error}")
                            stock_failed += 1
                            continue
                    
                    logging.info(f"\n{'='*60}")
                    logging.info(f"📊 Stock Signal Generation Summary:")
                    logging.info(f"  Total Stock Deployments: {len(stock_deployments)}")
                    logging.info(f"  Successful: {stock_successful}")
                    logging.info(f"  Failed: {stock_failed}")
                    logging.info(f"  Total Signals Generated: {stock_total_signals}")
                    logging.info(f"{'='*60}\n")
                    
                except ImportError as import_error:
                    logging.error(f"❌ Could not import Stock signal generator: {import_error}")
                except Exception as stock_error:
                    logging.error(f"❌ Error in Stock signal generation: {stock_error}")
            
            # ========== FINAL SUMMARY ==========
            logging.info(f"\n{'='*80}")
            logging.info("🎯 OVERALL SIGNAL GENERATION SUMMARY")
            logging.info(f"{'='*80}")
            logging.info(f"📈 ETF Deployments: {len(etf_deployments)} total")
            if etf_deployments:
                logging.info(f"   ✅ Successful: {etf_successful}, ❌ Failed: {etf_failed}, 📊 Signals: {etf_total_signals}")
            logging.info(f"📊 Stock Deployments: {len(stock_deployments)} total")
            if stock_deployments:
                logging.info(f"   ✅ Successful: {stock_successful}, ❌ Failed: {stock_failed}, 📊 Signals: {stock_total_signals}")
            logging.info(f"{'='*80}\n")
                
        except Exception as e:
            logging.critical(f"Critical error in ETF signal generation: {e}")
            raise
    
    def execute_signals_monday_job_with_holiday_check(self):
        """Job function to execute signals on Monday morning (with holiday check)"""
        try:
            logging.info("=" * 80)
            logging.info("=== Scheduled Monday Signal Execution (9:20 AM) ===")
            logging.info("=" * 80)
            
            # Check if today is a trading day
            if not self.fetcher.is_trading_day():
                logging.warning("⚠️  Today (Monday) is a market holiday. Execution will run tomorrow (Tuesday) at 9:20 AM.")
                return
            
            logging.info("✅ Today is a trading day. Proceeding with signal execution...")
            logging.info("🚀 Executing signals automatically")
            
            # Import execution service
            try:
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Services'))
                from execution.execution_service import ExecutionService
                
                execution_service = ExecutionService(self.fetcher.db_path)
                
                # Execute ONLY BUY signals from last Friday (both ETF and Stock)
                # This will clear previous Monday's data and store new execution data
                result = execution_service.execute_all_signals(side='BUY', signal_type='auto')
                
                if result['success']:
                    logging.info(f"✅ Signal execution completed successfully")
                    logging.info(f"📊 Total: {result['total_signals']}, Successful: {result['successful']}, Failed: {result['failed']}")
                    logging.info("💾 All execution data stored (previous Monday's data cleared)")
                else:
                    logging.error(f"❌ Signal execution failed: {result.get('message', 'Unknown error')}")
                    
            except ImportError as import_error:
                logging.error(f"❌ Could not import execution service: {import_error}")
            except Exception as exec_error:
                logging.error(f"❌ Error in signal execution: {exec_error}")
                
        except Exception as e:
            logging.critical(f"Critical error in signal execution: {e}")
            raise
    
    def execute_signals_tuesday_fallback(self):
        """Fallback job for Tuesday if Monday was a holiday"""
        try:
            logging.info("=" * 80)
            logging.info("=== Tuesday Fallback Signal Execution (9:20 AM) ===")
            logging.info("=" * 80)
            
            # Check if execution already happened yesterday (Monday)
            # If Monday was a trading day, Tuesday fallback should NOT run
            from datetime import datetime, timedelta
            yesterday = datetime.now().date() - timedelta(days=1)
            
            # Check if yesterday (Monday) was a trading day
            if self.fetcher.is_trading_day(yesterday):
                logging.info("✅ Monday was a trading day. Execution already completed yesterday.")
                logging.info("⏭️  Skipping Tuesday fallback execution.")
                return
            
            logging.info("⚠️  Monday was a holiday. Executing signals today (Tuesday)...")
            
            # Import execution service
            try:
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Services'))
                from execution.execution_service import ExecutionService
                
                execution_service = ExecutionService(self.fetcher.db_path)
                
                # Execute ONLY BUY signals from last Friday
                result = execution_service.execute_all_signals(side='BUY', signal_type='auto')
                
                if result['success']:
                    logging.info(f"✅ Tuesday fallback execution completed successfully")
                    logging.info(f"📊 Total: {result['total_signals']}, Successful: {result['successful']}, Failed: {result['failed']}")
                else:
                    logging.error(f"❌ Tuesday fallback execution failed: {result.get('message', 'Unknown error')}")
                    
            except ImportError as import_error:
                logging.error(f"❌ Could not import execution service: {import_error}")
            except Exception as exec_error:
                logging.error(f"❌ Error in signal execution: {exec_error}")
                
        except Exception as e:
            logging.critical(f"Critical error in Tuesday fallback execution: {e}")
            raise
    
    def generate_stock_signals_job(self):
        """Job function to generate Stock signals on Friday"""
        try:
            logging.info("=== Scheduled Stock signal generation started (Friday) ===")
            
            if not self.stock_signal_generator:
                logging.error("Stock signal generator not available")
                return
            
            # Strategy configuration for Stock signals
            strategy_config = {
                'max_positions': 5,
                'min_rs_score': 0.4,
                'min_momentum': 0.01,
                'min_volume_ratio': 0.8,
                'sell_threshold': 0.4
            }
            
            result = self.stock_signal_generator.run_weekly_signal_generation(
                strategy_type='StockSurfTrend',
                strategy_config=strategy_config
            )
            
            if result['success']:
                logging.info(f"Stock signal generation completed: {result['signals_count']} signals ({result['buy_count']} BUY, {result['sell_count']} SELL)")
            else:
                logging.error(f"Stock signal generation failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logging.critical(f"Critical error in Stock signal generation: {e}")
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
            logging.info("Starting market data scheduler (ETFs + Stocks)...")
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
    
    parser = argparse.ArgumentParser(description='Market Data Fetcher (ETFs + Stocks)')
    parser.add_argument('--mode', choices=['manual', 'scheduled'], default='manual',
                       help='Run mode: manual (run once) or scheduled (continuous)')
    
    args = parser.parse_args()
    
    if args.mode == 'manual':
        # Manual execution (original behavior)
        try:
            fetcher = IndianMarketDataFetcher()
            success_count, fail_count = fetcher.run_daily_fetch()
            
            if fail_count > 0:
                logging.warning(f"Some fetches failed. Success: {success_count}, Failed: {fail_count}")
                sys.exit(1)
            else:
                logging.info("All market data fetches completed successfully")
                sys.exit(0)
        except Exception as e:
            logging.critical(f"Critical error in main execution: {e}")
            sys.exit(2)
    
    elif args.mode == 'scheduled':
        # Scheduled execution
        scheduler = MarketDataScheduler()
        scheduler.start_scheduler()

if __name__ == "__main__":
    main()

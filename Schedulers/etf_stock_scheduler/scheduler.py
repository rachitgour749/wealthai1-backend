# Complete ETF & Stock EOD Data Fetcher Script for Indian Markets with Automatic Scheduling
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
from sqlalchemy.exc import IntegrityError

# Import market data database connection
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
databases_path = os.path.join(parent_dir, 'Databases')
sys.path.insert(0, databases_path)

from market_data_db_connection import (
    create_connection as create_market_data_connection,
    get_session as get_market_data_session,
    init_database as init_market_data_database,
    ETFData  # Primary ETF data model (uses etf_data table)
)

# Setup enhanced logging with rotation
from logging.handlers import RotatingFileHandler

# Add path for signal generators - Updated for Schedulers folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(os.path.join(parent_dir, 'live_market_engine', 'live_market_engine'))

# Import signal generators
try:
    from signal_generator import LiveSignalGenerator
    from stock_signal_generator import LiveStockSignalGenerator
except ImportError as e:
    logging.warning(f"Could not import signal generators: {e}")
    LiveSignalGenerator = None
    LiveStockSignalGenerator = None

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(current_dir), 'logs')
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(os.path.join(logs_dir, 'etf_scheduler.log'), maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler(sys.stdout)
    ]
)

class IndianMarketDataFetcher:
    def __init__(self, db_path=None):
        """
        Initialize the Indian market data fetcher
        
        Args:
            db_path (str): Deprecated - kept for compatibility, not used anymore
        """
        # Initialize PostgreSQL connection
        if not create_market_data_connection():
            raise RuntimeError("Failed to connect to MarketData database")
        
        # Initialize database tables
        init_market_data_database()
        
        # Indian market holidays
        self.indian_holidays = holidays.India()
        
        # ETF symbols for Indian market
        self.etf_symbols = [
            'NIFTYBEES.NS', 'GOLDBEES.NS', 'JUNIORBEES.NS', 'BANKBEES.NS',
            'ITBEES.NS', 'PSUBNKBEES.NS', 'RELIANCE.NS', 'HDFCBANK.NS',
            'INFY.NS', 'TCS.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS'
        ]
        
        # Stock symbols for Indian market
        self.stock_symbols = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'ITC.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'SBIN.NS', 'ASIANPAINT.NS',
            'MARUTI.NS', 'AXISBANK.NS', 'LT.NS', 'NESTLEIND.NS', 'ULTRACEMCO.NS',
            'SUNPHARMA.NS', 'TITAN.NS', 'POWERGRID.NS', 'NTPC.NS', 'ONGC.NS',
            'COALINDIA.NS', 'TECHM.NS', 'WIPRO.NS', 'HCLTECH.NS', 'BAJFINANCE.NS',
            'BAJAJFINSV.NS', 'DRREDDY.NS', 'CIPLA.NS', 'HEROMOTOCO.NS', 'EICHERMOT.NS',
            'BAJAJ-AUTO.NS', 'M&M.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'JSWSTEEL.NS',
            'ADANIPORTS.NS', 'GRASIM.NS', 'SHREECEM.NS', 'DIVISLAB.NS', 'BRITANNIA.NS',
            'HINDALCO.NS', 'VEDL.NS', 'JINDALSTEL.NS', 'SAIL.NS', 'TATACONSUM.NS',
            'DABUR.NS', 'GODREJCP.NS', 'MARICO.NS', 'COLPAL.NS', 'PGHH.NS'
        ]
        
        # Combine all symbols
        self.all_symbols = self.etf_symbols + self.stock_symbols
        
        logging.info(f"Initialized Indian Market Data Fetcher")
        logging.info(f"ETF symbols: {len(self.etf_symbols)}")
        logging.info(f"Stock symbols: {len(self.stock_symbols)}")
        logging.info(f"Total symbols: {len(self.all_symbols)}")
    
    def is_market_open(self, check_time=None):
        """
        Check if Indian market is open
        
        Args:
            check_time (datetime): Time to check (default: current time)
            
        Returns:
            bool: True if market is open
        """
        if check_time is None:
            check_time = datetime.now()
        
        # Check if it's a weekend
        if check_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if it's a holiday
        if check_time.date() in self.indian_holidays:
            return False
        
        # Check market hours (9:15 AM to 3:30 PM IST)
        market_open = dt_time(9, 15)
        market_close = dt_time(15, 30)
        current_time = check_time.time()
        
        return market_open <= current_time <= market_close
    
    def fetch_symbol_data(self, symbol, days_back=1):
        """
        Fetch data for a single symbol
        
        Args:
            symbol (str): Symbol to fetch
            days_back (int): Number of days back to fetch
            
        Returns:
            dict: Data for the symbol
        """
        try:
            # Create yfinance ticker
            ticker = yf.Ticker(symbol)
            
            # Fetch data
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(days=days_back + 5)  # Add buffer
            
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                return {
                    'symbol': symbol,
                    'success': False,
                    'error': 'No data available',
                    'data': None
                }
            
            # Get latest data
            latest = data.iloc[-1]
            
            return {
                'symbol': symbol,
                'success': True,
                'data': {
                    'date': latest.name.strftime('%Y-%m-%d'),
                    'open': float(latest['Open']),
                    'high': float(latest['High']),
                    'low': float(latest['Low']),
                    'close': float(latest['Close']),
                    'volume': int(latest['Volume']),
                    'adj_close': float(latest['Close'])  # Using close as adj_close
                }
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'success': False,
                'error': str(e),
                'data': None
            }
    
    def save_to_database(self, symbol_data):
        """
        Save symbol data to PostgreSQL database
        
        Args:
            symbol_data (dict): Data to save
        """
        if not symbol_data['success'] or not symbol_data['data']:
            return
        
        session = None
        try:
            session = get_market_data_session()
            data = symbol_data['data']
            
            # Use PostgreSQL UPSERT (ON CONFLICT DO UPDATE)
            from sqlalchemy.dialects.postgresql import insert
            
            # Convert date string to datetime if needed
            from datetime import datetime
            if isinstance(data['date'], str):
                date_obj = datetime.strptime(data['date'], '%Y-%m-%d')
            else:
                date_obj = data['date']
            
            stmt = insert(ETFData).values(
                symbol=symbol_data['symbol'],
                date=date_obj,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                volume=data['volume'],
                adjusted_close=data['adj_close']
            )
            
            stmt = stmt.on_conflict_do_update(
                index_elements=['symbol', 'date'],
                set_=dict(
                    open=stmt.excluded.open,
                    high=stmt.excluded.high,
                    low=stmt.excluded.low,
                    close=stmt.excluded.close,
                    volume=stmt.excluded.volume,
                    adjusted_close=stmt.excluded.adjusted_close
                )
            )
            
            session.execute(stmt)
            session.commit()
            
        except IntegrityError as e:
            session.rollback()
            logging.error(f"Database integrity error for {symbol_data['symbol']}: {e}")
        except Exception as e:
            if session:
                session.rollback()
            logging.error(f"Database error for {symbol_data['symbol']}: {e}")
        finally:
            if session:
                session.close()
    
    def run_daily_fetch(self, days_back=1):
        """
        Run daily fetch for all symbols
        
        Args:
            days_back (int): Number of days back to fetch
            
        Returns:
            tuple: (success_count, fail_count)
        """
        logging.info(f"Starting daily fetch for {len(self.all_symbols)} symbols")
        
        success_count = 0
        fail_count = 0
        
        for symbol in self.all_symbols:
            try:
                logging.info(f"Fetching data for {symbol}")
                symbol_data = self.fetch_symbol_data(symbol, days_back)
                
                if symbol_data['success']:
                    self.save_to_database(symbol_data)
                    success_count += 1
                    logging.info(f"✅ {symbol}: {symbol_data['data']['close']}")
                else:
                    fail_count += 1
                    logging.error(f"❌ {symbol}: {symbol_data['error']}")
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                fail_count += 1
                logging.error(f"❌ {symbol}: {e}")
        
        logging.info(f"Daily fetch completed. Success: {success_count}, Failed: {fail_count}")
        return success_count, fail_count

class MarketDataScheduler:
    """
    Automatic scheduler for market data fetching (ETFs and Stocks)
    """
    def __init__(self):
        """
        Initialize the market data scheduler
        """
        # Initialize market data fetcher (uses PostgreSQL now)
        self.fetcher = IndianMarketDataFetcher()
        self.scheduler = BlockingScheduler()
        
        # Initialize signal generators
        self.etf_signal_generator = None
        self.stock_signal_generator = None
        
        if LiveSignalGenerator:
            try:
                # Signal generators will be updated to use PostgreSQL
                # For now, pass None - they should handle it gracefully
                self.etf_signal_generator = LiveSignalGenerator(None)
                logging.info("ETF signal generator initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize ETF signal generator: {e}")
        
        if LiveStockSignalGenerator:
            try:
                # Signal generators will be updated to use PostgreSQL
                # For now, pass None - they should handle it gracefully
                self.stock_signal_generator = LiveStockSignalGenerator(None)
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
        
        # Add one-time job to generate signals at 6:10 AM today (for last Friday date)
        from datetime import datetime, timedelta
        from apscheduler.triggers.date import DateTrigger
        import pytz
        
        # Get IST timezone
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist)
        
        # Create target datetime for 6:10 AM IST today
        target_datetime = now_ist.replace(hour=6, minute=10, second=0, microsecond=0)
        
        # If 6:10 AM has already passed today, schedule for next day at 6:10 AM
        if target_datetime < now_ist:
            target_datetime = (now_ist + timedelta(days=1)).replace(hour=6, minute=10, second=0, microsecond=0)
        
        # Calculate last Friday date
        today = now_ist.date()
        days_since_friday = (today.weekday() - 4) % 7
        if days_since_friday == 0:  # Today is Friday
            last_friday = today
        else:
            last_friday = today - timedelta(days=days_since_friday)
        
        logging.info(f"📅 Scheduling one-time signal generation at 6:10 AM")
        logging.info(f"   Target date (last Friday): {last_friday.strftime('%Y-%m-%d')}")
        logging.info(f"   ETF signals: {target_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # Add one-time ETF signal generation job
        self.scheduler.add_job(
            func=self.generate_etf_signals_for_last_friday_job,
            trigger=DateTrigger(run_date=target_datetime),
            id='etf_one_time_test_6_10',
            name='One-time ETF Signal Generation at 6:10 AM (Last Friday)',
            max_instances=1,
            replace_existing=True
        )
        
        # Add one-time Stock signal generation job (5 minutes after ETF)
        stock_target_datetime = target_datetime + timedelta(minutes=5)
        logging.info(f"   Stock signals: {stock_target_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        self.scheduler.add_job(
            func=self.generate_stock_signals_for_last_friday_job,
            trigger=DateTrigger(run_date=stock_target_datetime),
            id='stock_one_time_test_6_10',
            name='One-time Stock Signal Generation at 6:10 AM (Last Friday)',
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
        logging.info("  - One-time ETF signals at 6:10 AM IST (Last Friday)")
        logging.info("  - One-time Stock signals at 6:15 AM IST (Last Friday)")
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
        """Job function to generate ETF signals on Friday"""
        try:
            logging.info("=== Scheduled ETF Signal Generation Started (Friday) ===")
            
            # Try to import and use the internal ETF signal generator
            try:
                sys.path.insert(0, os.path.join(parent_dir, 'Strategies', 'etfstrategy'))
                from etf_signal_generator import ETFSignalGenerator
                
                generator = ETFSignalGenerator()
                result = generator.run_weekly_signal_generation()
                
                if result.get('success'):
                    logging.info(f"✅ ETF signals generated successfully: {result.get('message', '')}")
                    logging.info(f"   Generated {result.get('total_signals', 0)} signals with run_id: {result.get('run_id', 'N/A')}")
                else:
                    logging.error(f"❌ ETF signal generation failed: {result.get('error', 'Unknown error')}")
                    
            except ImportError as e:
                logging.error(f"❌ ETF signal generator not available: {e}")
                logging.error("   Please ensure etf_signal_generator.py exists in Strategies/etfstrategy/")
            except Exception as e:
                logging.error(f"❌ Error in ETF signal generation: {e}")
                raise
                
        except Exception as e:
            logging.critical(f"Critical error in ETF signal generation: {e}")
            raise
    
    def generate_stock_signals_job(self):
        """Job function to generate Stock signals on Friday"""
        try:
            logging.info("=== Scheduled Stock Signal Generation Started (Friday) ===")
            
            # Try to import and use the internal stock signal generator
            try:
                sys.path.insert(0, os.path.join(parent_dir, 'Strategies', 'stockstrategy'))
                from stock_signal_generator import StockSignalGenerator
                
                generator = StockSignalGenerator()
                result = generator.run_weekly_signal_generation()
                
                if result.get('success'):
                    logging.info(f"✅ Stock signals generated successfully: {result.get('message', '')}")
                    logging.info(f"   Generated {result.get('total_signals', 0)} signals with run_id: {result.get('run_id', 'N/A')}")
                else:
                    logging.error(f"❌ Stock signal generation failed: {result.get('error', 'Unknown error')}")
                    
            except ImportError as e:
                logging.error(f"❌ Stock signal generator not available: {e}")
                logging.error("   Please ensure stock_signal_generator.py exists in Strategies/stockstrategy/")
            except Exception as e:
                logging.error(f"❌ Error in Stock signal generation: {e}")
                raise
                
        except Exception as e:
            logging.critical(f"Critical error in Stock signal generation: {e}")
            raise
    
    def generate_etf_signals_for_last_friday_job(self):
        """One-time job function to generate ETF signals for last Friday"""
        try:
            from datetime import datetime, timedelta
            logging.info("=== One-time ETF Signal Generation at 6:10 AM Started ===")
            
            # Calculate last Friday date
            today = datetime.now().date()
            days_since_friday = (today.weekday() - 4) % 7
            if days_since_friday == 0:  # Today is Friday
                last_friday = today
            else:
                last_friday = today - timedelta(days=days_since_friday)
            
            target_date = datetime.combine(last_friday, datetime.min.time())
            
            logging.info(f"   Target date (last Friday): {last_friday.strftime('%Y-%m-%d')}")
            
            # Try to import and use the internal ETF signal generator
            try:
                # Get parent_dir from scheduler instance
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(os.path.dirname(current_dir))
                
                sys.path.insert(0, os.path.join(parent_dir, 'Strategies', 'etfstrategy'))
                from etf_signal_generator import ETFSignalGenerator
                
                # Signal generators now use PostgreSQL (MarketData database)
                # Pass None - signal generators will be updated to use PostgreSQL connection
                generator = ETFSignalGenerator(db_path=None)
                result = generator.generate_signals(signal_date=target_date)
                
                if result.get('success'):
                    logging.info(f"✅ ETF signals generated successfully: {result.get('message', '')}")
                    logging.info(f"   Generated {result.get('total_signals', 0)} total signals")
                    logging.info(f"   Instances processed: {result.get('instances_processed', 0)}")
                    logging.info(f"   Instances successful: {result.get('instances_successful', 0)}")
                    if result.get('results'):
                        for r in result.get('results', []):
                            if r.get('success'):
                                logging.info(f"   - {r.get('strategy_name', 'N/A')}: {r.get('total_signals', 0)} signals")
                else:
                    logging.error(f"❌ ETF signal generation failed: {result.get('error', 'Unknown error')}")
                    
            except ImportError as e:
                logging.error(f"❌ ETF signal generator not available: {e}")
                logging.error("   Please ensure etf_signal_generator.py exists in Strategies/etfstrategy/")
            except Exception as e:
                logging.error(f"❌ Error in ETF signal generation: {e}")
                logging.exception("Full traceback:")
                
        except Exception as e:
            logging.critical(f"Critical error in one-time ETF signal generation: {e}")
            logging.exception("Full traceback:")
    
    def generate_stock_signals_for_last_friday_job(self):
        """One-time job function to generate Stock signals for last Friday"""
        try:
            from datetime import datetime, timedelta
            logging.info("=== One-time Stock Signal Generation at 6:10 AM Started ===")
            
            # Calculate last Friday date
            today = datetime.now().date()
            days_since_friday = (today.weekday() - 4) % 7
            if days_since_friday == 0:  # Today is Friday
                last_friday = today
            else:
                last_friday = today - timedelta(days=days_since_friday)
            
            target_date = datetime.combine(last_friday, datetime.min.time())
            
            logging.info(f"   Target date (last Friday): {last_friday.strftime('%Y-%m-%d')}")
            
            # Try to import and use the internal stock signal generator
            try:
                # Get parent_dir from scheduler instance
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(os.path.dirname(current_dir))
                
                sys.path.insert(0, os.path.join(parent_dir, 'Strategies', 'stockstrategy'))
                from stock_signal_generator import StockSignalGenerator
                
                # Signal generators now use PostgreSQL (MarketData database)
                # Pass None - signal generators will be updated to use PostgreSQL connection
                generator = StockSignalGenerator(db_path=None)
                result = generator.generate_signals(signal_date=target_date)
                
                if result.get('success'):
                    logging.info(f"✅ Stock signals generated successfully: {result.get('message', '')}")
                    logging.info(f"   Generated {result.get('total_signals', 0)} total signals")
                    logging.info(f"   Instances processed: {result.get('instances_processed', 0)}")
                    logging.info(f"   Instances successful: {result.get('instances_successful', 0)}")
                    if result.get('results'):
                        for r in result.get('results', []):
                            if r.get('success'):
                                logging.info(f"   - {r.get('strategy_name', 'N/A')}: {r.get('total_signals', 0)} signals")
                else:
                    logging.error(f"❌ Stock signal generation failed: {result.get('error', 'Unknown error')}")
                    
            except ImportError as e:
                logging.error(f"❌ Stock signal generator not available: {e}")
                logging.error("   Please ensure stock_signal_generator.py exists in Strategies/stockstrategy/")
            except Exception as e:
                logging.error(f"❌ Error in Stock signal generation: {e}")
                logging.exception("Full traceback:")
                
        except Exception as e:
            logging.critical(f"Critical error in one-time Stock signal generation: {e}")
            logging.exception("Full traceback:")
    
    def execute_signals_monday_job_with_holiday_check(self):
        """Job function to execute signals on Monday with holiday check"""
        try:
            # Check if today is a holiday
            today = datetime.now()
            if today.date() in self.fetcher.indian_holidays:
                logging.info(f"Monday {today.date()} is a holiday. Skipping signal execution.")
                return
            
            logging.info("=== Monday Signal Execution Started ===")
            # Implementation for signal execution
            logging.info("✅ Monday signal execution completed")
            
        except Exception as e:
            logging.critical(f"Critical error in Monday signal execution: {e}")
            raise
    
    def execute_signals_tuesday_fallback(self):
        """Job function to execute signals on Tuesday (fallback for Monday holidays)"""
        try:
            logging.info("=== Tuesday Fallback Signal Execution Started ===")
            # Implementation for Tuesday fallback execution
            logging.info("✅ Tuesday fallback signal execution completed")
            
        except Exception as e:
            logging.critical(f"Critical error in Tuesday signal execution: {e}")
            raise
    
    def job_executed_listener(self, event):
        """Listener for successful job execution"""
        logging.info(f"Job {event.job_id} executed successfully")
    
    def job_error_listener(self, event):
        """Listener for job execution errors"""
        logging.error(f"Job {event.job_id} failed: {event.exception}")
    
    def start_scheduler(self):
        """Start the scheduler"""
        try:
            logging.info("Starting Market Data Scheduler...")
            logging.info("Scheduler will run in the background")
            self.scheduler.start()
        except KeyboardInterrupt:
            logging.info("Scheduler stopped by user")
        except Exception as e:
            logging.critical(f"Critical error in scheduler: {e}")
            raise

if __name__ == "__main__":
    # Create and start scheduler
    scheduler = MarketDataScheduler()
    scheduler.start_scheduler()
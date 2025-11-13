"""
RS Strategy Scheduler
Automatically fetches EOD data and generates live signals for RS strategy
"""

import sqlite3
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

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import RS strategy modules
try:
    # Try relative import first (when used as module)
    from .rs_live_signal_generator import RSLiveSignalGenerator
    from .rs_eod_data_fetcher import RSEODDataFetcher
except ImportError:
    try:
        # Try direct import (when run as script)
        from rs_live_signal_generator import RSLiveSignalGenerator
        from rs_eod_data_fetcher import RSEODDataFetcher
    except ImportError as e:
        print(f"Warning: Could not import RS strategy modules: {e}")
        RSLiveSignalGenerator = None
        RSEODDataFetcher = None

class RSStrategyScheduler:
    """
    Scheduler for RS Strategy EOD data fetching and signal generation
    """
    
    def __init__(self, db_path=None):
        """
        Initialize the RS strategy scheduler
        
        Args:
            db_path (str): Path to SQLite database file
        """
        if db_path is None:
            self.db_path = os.path.join(current_dir, "nifty500_data_with_metadata.sqlite")
        else:
            self.db_path = os.path.abspath(db_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Create logs directory
        logs_dir = os.path.join(os.path.dirname(current_dir), '..', 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging(logs_dir)
        
        # Initialize modules
        self.eod_fetcher = None
        self.signal_generator = None
        self.scheduler = None
        
        # Initialize modules if available
        if RSEODDataFetcher:
            try:
                self.eod_fetcher = RSEODDataFetcher(self.db_path)
                self.logger.info("EOD data fetcher initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize EOD data fetcher: {e}")
        
        if RSLiveSignalGenerator:
            try:
                self.signal_generator = RSLiveSignalGenerator(self.db_path)
                self.logger.info("Live signal generator initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize live signal generator: {e}")
        
        # Indian holidays for market closure detection
        self.indian_holidays = holidays.India()
        
        self.logger.info(f"RS Strategy Scheduler initialized with database: {self.db_path}")
    
    def setup_logging(self, logs_dir):
        """Setup logging for the scheduler"""
        log_file = os.path.join(logs_dir, 'rs_scheduler.log')
        
        self.logger = logging.getLogger('rs_scheduler')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler with rotation
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(logging.INFO)
        
        # Console handler with UTF-8 encoding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        # Set encoding to UTF-8 to handle emoji characters
        if hasattr(console_handler.stream, 'reconfigure'):
            console_handler.stream.reconfigure(encoding='utf-8')
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def is_market_open(self, check_date=None):
        """
        Check if the market is open on the given date
        
        Args:
            check_date (date): Date to check (default: today)
            
        Returns:
            bool: True if market is open, False otherwise
        """
        if check_date is None:
            check_date = date.today()
        
        # Check if it's a weekend
        if check_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if it's a holiday
        if check_date in self.indian_holidays:
            return False
        
        return True
    
    def fetch_eod_data_job(self):
        """Job to fetch EOD data for RS strategy"""
        try:
            self.logger.info("Starting EOD data fetch job...")
            
            if not self.eod_fetcher:
                self.logger.error("EOD data fetcher not available")
                return
            
            # Check if market was open today
            if not self.is_market_open():
                self.logger.info("Market was closed today, skipping EOD data fetch")
                return
            
            # Fetch daily data
            result = self.eod_fetcher.fetch_daily_data(days_back=1)
            
            self.logger.info(f"EOD data fetch completed: {result['successful_fetches']} successful, {result['failed_fetches']} failed")
            
        except Exception as e:
            self.logger.error(f"Error in EOD data fetch job: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def generate_signals_job(self):
        """Job to generate live signals for RS strategy"""
        try:
            self.logger.info("Starting live signal generation job...")
            
            if not self.signal_generator:
                self.logger.error("Live signal generator not available")
                return
            
            # Check if market was open today
            if not self.is_market_open():
                self.logger.info("Market was closed today, skipping signal generation")
                return
            
            # Generate signals using default configuration
            result = self.signal_generator.generate_signals(config_id=1)
            
            self.logger.info(f"Live signal generation completed: {result['total_signals']} signals generated")
            
        except Exception as e:
            self.logger.error(f"Error in signal generation job: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def cleanup_job(self):
        """Job to cleanup old data and signals"""
        try:
            self.logger.info("Starting cleanup job...")
            
            # Cleanup old EOD data (keep 1 year)
            if self.eod_fetcher:
                self.eod_fetcher.cleanup_old_data(days_to_keep=365)
                self.logger.info("EOD data cleanup completed")
            
            # Cleanup old signals (keep 30 days)
            if self.signal_generator:
                self.signal_generator.cleanup_old_signals(days_to_keep=30)
                self.logger.info("Signal cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error in cleanup job: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def job_listener(self, event):
        """Listener for job execution events"""
        if event.exception:
            self.logger.error(f"Job {event.job_id} failed: {event.exception}")
        else:
            self.logger.info(f"Job {event.job_id} completed successfully")
    
    def start_scheduler(self):
        """Start the RS strategy scheduler"""
        try:
            self.logger.info("Starting RS Strategy Scheduler...")
            
            # Create scheduler
            self.scheduler = BlockingScheduler()
            
            # Add job listener
            self.scheduler.add_listener(self.job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
            
            # Schedule EOD data fetch - daily at 6:00 PM IST (after market close)
            self.scheduler.add_job(
                self.fetch_eod_data_job,
                trigger=CronTrigger(hour=18, minute=0, timezone='Asia/Kolkata'),
                id='rs_eod_fetch',
                name='RS Strategy EOD Data Fetch',
                replace_existing=True
            )
            
            # Schedule signal generation - daily at 6:00 PM IST (after EOD data fetch)
            self.scheduler.add_job(
                self.generate_signals_job,
                trigger=CronTrigger(hour=18, minute=0, timezone='Asia/Kolkata'),
                id='rs_signal_generation',
                name='RS Strategy Signal Generation',
                replace_existing=True
            )
            
            # Schedule cleanup - weekly on Sunday at 2:00 AM IST
            self.scheduler.add_job(
                self.cleanup_job,
                trigger=CronTrigger(day_of_week=6, hour=2, minute=0, timezone='Asia/Kolkata'),
                id='rs_cleanup',
                name='RS Strategy Cleanup',
                replace_existing=True
            )
            
            # Log scheduled jobs
            jobs = self.scheduler.get_jobs()
            self.logger.info(f"Scheduled {len(jobs)} jobs:")
            for job in jobs:
                try:
                    # Try to get next run time - different APScheduler versions have different attributes
                    if hasattr(job, 'next_run_time') and job.next_run_time:
                        next_run = job.next_run_time.strftime('%Y-%m-%d %H:%M:%S IST')
                    elif hasattr(job, 'next_run') and job.next_run:
                        next_run = job.next_run.strftime('%Y-%m-%d %H:%M:%S IST')
                    else:
                        next_run = 'Not scheduled'
                    self.logger.info(f"   - {job.name}: Next run at {next_run}")
                except Exception as e:
                    self.logger.info(f"   - {job.name}: Scheduled (next run time unavailable)")
            
            self.logger.info("RS Strategy Scheduler started successfully")
            self.logger.info("EOD data fetch: Daily at 6:00 PM IST")
            self.logger.info("Signal generation: Daily at 6:00 PM IST")
            self.logger.info("Cleanup: Weekly on Sunday at 2:00 AM IST")
            
            # Start the scheduler (BlockingScheduler will block here)
            self.scheduler.start()
            
        except Exception as e:
            self.logger.error(f"Failed to start RS strategy scheduler: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def stop_scheduler(self):
        """Stop the RS strategy scheduler"""
        try:
            if self.scheduler:
                self.scheduler.shutdown(wait=False)
                self.logger.info("RS Strategy Scheduler stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping RS strategy scheduler: {e}")
    
    def run_manual_eod_fetch(self):
        """Manually run EOD data fetch"""
        try:
            self.logger.info("Running manual EOD data fetch...")
            self.fetch_eod_data_job()
        except Exception as e:
            self.logger.error(f"Error in manual EOD fetch: {e}")
    
    def run_manual_signal_generation(self):
        """Manually run signal generation"""
        try:
            self.logger.info("Running manual signal generation...")
            self.generate_signals_job()
        except Exception as e:
            self.logger.error(f"Error in manual signal generation: {e}")
    
    def run_manual_cleanup(self):
        """Manually run cleanup"""
        try:
            self.logger.info("Running manual cleanup...")
            self.cleanup_job()
        except Exception as e:
            self.logger.error(f"Error in manual cleanup: {e}")

def main():
    """Main function to run the RS strategy scheduler"""
    try:
        # Initialize scheduler
        scheduler = RSStrategyScheduler()
        
        # Start scheduler
        scheduler.start_scheduler()
        
    except KeyboardInterrupt:
        print("\n🛑 Scheduler stopped by user")
    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

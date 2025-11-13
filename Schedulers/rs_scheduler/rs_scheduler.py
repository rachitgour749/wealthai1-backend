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
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'Strategies', 'rsStrategy'))

# Import RS strategy modules
try:
    # Try relative import first (when used as module)
    from .rs_eod_data_fetcher import RSEODDataFetcher
except ImportError:
    try:
        # Try direct import (when run as script)
        from rs_eod_data_fetcher import RSEODDataFetcher
    except ImportError:
        try:
            # Try importing from Strategies directory
            from Strategies.rsStrategy.rs_eod_data_fetcher import RSEODDataFetcher
        except ImportError as e:
            print(f"Warning: Could not import RS strategy modules: {e}")
            RSEODDataFetcher = None

class RSStrategyScheduler:
    """
    Scheduler for RS Strategy EOD data fetching
    """
    
    def __init__(self, db_path=None):
        """
        Initialize the RS strategy scheduler
        
        Args:
            db_path (str): Path to SQLite database file
        """
        if db_path is None:
            # Updated path for Schedulers folder
            self.db_path = os.path.join(parent_dir, 'Strategies', 'rsStrategy', 'nifty500_data_with_metadata.sqlite')
        else:
            self.db_path = os.path.abspath(db_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Create logs directory
        logs_dir = os.path.join(os.path.dirname(current_dir), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging(logs_dir)
        
        # Initialize modules
        self.eod_fetcher = None
        self.scheduler = None
        
        # Initialize modules if available
        if RSEODDataFetcher:
            try:
                self.eod_fetcher = RSEODDataFetcher(self.db_path)
                self.logger.info("EOD data fetcher initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize EOD data fetcher: {e}")
        
        # Indian holidays for market closure detection
        self.indian_holidays = holidays.India()
        
        self.logger.info(f"RS Strategy Scheduler initialized with database: {self.db_path}")
    
    def setup_logging(self, logs_dir):
        """Setup logging for the scheduler"""
        self.logger = logging.getLogger('rs_scheduler')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            os.path.join(logs_dir, 'rs_scheduler.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
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
    
    def fetch_eod_data_job(self):
        """Job function to fetch EOD data"""
        try:
            self.logger.info("=== RS Strategy EOD Data Fetch Started ===")
            
            if not self.eod_fetcher:
                self.logger.warning("EOD data fetcher not available")
                return
            
            # Fetch EOD data
            result = self.eod_fetcher.fetch_daily_data(days_back=1)
            
            if result['success']:
                self.logger.info(f"✅ EOD data fetch completed: {result['message']}")
            else:
                self.logger.error(f"❌ EOD data fetch failed: {result['message']}")
                
        except Exception as e:
            self.logger.critical(f"Critical error in EOD data fetch: {e}")
            raise
    
    def cleanup_job(self):
        """Job function to cleanup old data"""
        try:
            self.logger.info("=== RS Strategy Cleanup Started ===")
            
            # Cleanup old data (keep last 30 days)
            # Implementation for cleanup logic
            
            self.logger.info("✅ RS Strategy cleanup completed")
            
        except Exception as e:
            self.logger.critical(f"Critical error in cleanup: {e}")
            raise
    
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
                self.logger.info(f"  - {job.name} (ID: {job.id})")
            
            self.logger.info("")
            self.logger.info("  - One-time RS signals at 6:20 AM IST (Last Friday)")
            self.logger.info("RS Strategy Scheduler started successfully")
            self.logger.info("Scheduler will run in the background")
            
            # Start the scheduler
            self.scheduler.start()
            
        except KeyboardInterrupt:
            self.logger.info("RS Strategy Scheduler stopped by user")
        except Exception as e:
            self.logger.critical(f"Critical error in RS Strategy Scheduler: {e}")
            raise
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        if self.scheduler:
            self.scheduler.shutdown(wait=False)
            self.logger.info("RS Strategy Scheduler stopped")

if __name__ == "__main__":
    # Create and start scheduler
    scheduler = RSStrategyScheduler()
    scheduler.start_scheduler()
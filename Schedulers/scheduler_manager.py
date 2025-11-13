"""
Scheduler Manager for WealthAI Backend
Automatically starts all schedulers when server starts
"""

import os
import sys
import threading
import logging
from datetime import datetime

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import schedulers
try:
    from Schedulers.etf_stock_scheduler.scheduler import MarketDataScheduler
    from Schedulers.rs_scheduler.rs_scheduler import RSStrategyScheduler
except ImportError as e:
    print(f"Warning: Could not import schedulers: {e}")
    MarketDataScheduler = None
    RSStrategyScheduler = None

class SchedulerManager:
    """
    Manages all schedulers for the WealthAI backend
    """
    
    def __init__(self):
        """Initialize the scheduler manager"""
        self.etf_scheduler = None
        self.rs_scheduler = None
        self.etf_thread = None
        self.rs_thread = None
        self.initialized = False
        
        # Setup logging
        self.setup_logging()
        
        self.logger.info("Scheduler Manager initialized")
    
    def setup_logging(self):
        """Setup logging for the scheduler manager"""
        # Create logs directory
        logs_dir = os.path.join(current_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger('scheduler_manager')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(os.path.join(logs_dir, 'scheduler_manager.log'))
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def start_etf_scheduler(self):
        """Start the ETF/Stock scheduler in a separate thread"""
        try:
            if not MarketDataScheduler:
                self.logger.error("ETF/Stock scheduler not available")
                return False
            
            self.logger.info("🕐 Starting ETF/Stock scheduler...")
            self.etf_scheduler = MarketDataScheduler()
            
            # Start scheduler in a separate thread
            self.etf_thread = threading.Thread(target=self.etf_scheduler.start_scheduler, daemon=True)
            self.etf_thread.start()
            
            self.logger.info("✅ ETF/Stock scheduler started successfully")
            self.logger.info("📊 Will fetch data for 62 market instruments (12 ETFs + 50 Stocks)")
            self.logger.info("📅 ETF signals: Friday at 4:30 PM IST")
            self.logger.info("📅 Stock signals: Friday at 4:35 PM IST")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start ETF/Stock scheduler: {e}")
            return False
    
    def start_rs_scheduler(self):
        """Start the RS strategy scheduler in a separate thread"""
        try:
            if not RSStrategyScheduler:
                self.logger.error("RS strategy scheduler not available")
                return False
            
            self.logger.info("🕐 Starting RS strategy scheduler...")
            self.rs_scheduler = RSStrategyScheduler()
            
            # Start scheduler in a separate thread
            self.rs_thread = threading.Thread(target=self.rs_scheduler.start_scheduler, daemon=True)
            self.rs_thread.start()
            
            self.logger.info("✅ RS strategy scheduler started successfully")
            self.logger.info("📊 Will fetch Nifty 500 data and generate signals")
            self.logger.info("📅 EOD data: Daily at 6:00 PM IST")
            self.logger.info("📅 Signal generation: Daily at 6:00 PM IST")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start RS strategy scheduler: {e}")
            return False
    
    def start_all_schedulers(self):
        """Start all schedulers"""
        try:
            self.logger.info("🚀 Starting all schedulers...")
            
            # Start ETF/Stock scheduler
            etf_success = self.start_etf_scheduler()
            
            # Wait a bit before starting RS scheduler
            import time
            time.sleep(2)
            
            # Start RS scheduler
            rs_success = self.start_rs_scheduler()
            
            if etf_success and rs_success:
                self.initialized = True
                self.logger.info("✅ All schedulers started successfully!")
                return True
            else:
                self.logger.error("❌ Some schedulers failed to start")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to start schedulers: {e}")
            return False
    
    def stop_all_schedulers(self):
        """Stop all schedulers"""
        try:
            self.logger.info("🛑 Stopping all schedulers...")
            
            # Stop ETF scheduler
            if self.etf_scheduler:
                try:
                    self.etf_scheduler.scheduler.shutdown(wait=False)
                    self.logger.info("✅ ETF/Stock scheduler stopped")
                except Exception as e:
                    self.logger.error(f"Error stopping ETF scheduler: {e}")
            
            # Stop RS scheduler
            if self.rs_scheduler:
                try:
                    self.rs_scheduler.stop_scheduler()
                    self.logger.info("✅ RS strategy scheduler stopped")
                except Exception as e:
                    self.logger.error(f"Error stopping RS scheduler: {e}")
            
            self.initialized = False
            self.logger.info("✅ All schedulers stopped")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping schedulers: {e}")
    
    def get_status(self):
        """Get the status of all schedulers"""
        status = {
            'initialized': self.initialized,
            'etf_scheduler_running': self.etf_thread and self.etf_thread.is_alive() if self.etf_thread else False,
            'rs_scheduler_running': self.rs_thread and self.rs_thread.is_alive() if self.rs_thread else False,
            'timestamp': datetime.now().isoformat()
        }
        return status

# Global scheduler manager instance
scheduler_manager = SchedulerManager()

def start_schedulers():
    """Start all schedulers - called by server.py"""
    return scheduler_manager.start_all_schedulers()

def stop_schedulers():
    """Stop all schedulers - called by server.py"""
    return scheduler_manager.stop_all_schedulers()

def get_scheduler_status():
    """Get scheduler status - called by server.py"""
    return scheduler_manager.get_status()

if __name__ == "__main__":
    # Test the scheduler manager
    print("Testing Scheduler Manager...")
    
    # Start schedulers
    success = start_schedulers()
    
    if success:
        print("✅ Schedulers started successfully!")
        
        # Wait for a bit
        import time
        time.sleep(5)
        
        # Get status
        status = get_scheduler_status()
        print(f"Status: {status}")
        
        # Stop schedulers
        stop_schedulers()
        print("✅ Schedulers stopped")
    else:
        print("❌ Failed to start schedulers")

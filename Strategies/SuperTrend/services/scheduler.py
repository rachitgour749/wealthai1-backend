"""
APScheduler for daily jobs (optional - can be used for automated EOD runs)
"""
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import requests


class StrategyScheduler:
    """
    Scheduler for automated strategy execution
    """
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        """
        Initialize scheduler
        
        Args:
            api_url: Base URL for API
        """
        self.api_url = api_url
        self.scheduler = BackgroundScheduler()
    
    def eod_job(self):
        """Run end-of-day processing"""
        try:
            print(f"[{datetime.now()}] Running EOD job...")
            response = requests.post(f"{self.api_url}/api/run/eod")
            
            if response.status_code == 200:
                result = response.json()
                print(f"EOD job completed: {result['message']}")
            else:
                print(f"EOD job failed: {response.text}")
        
        except Exception as e:
            print(f"Error in EOD job: {str(e)}")
    
    def start(self, eod_time: str = "18:00"):
        """
        Start scheduler
        
        Args:
            eod_time: Time to run EOD job (HH:MM format)
        """
        # Schedule EOD job
        hour, minute = eod_time.split(":")
        self.scheduler.add_job(
            self.eod_job,
            'cron',
            hour=int(hour),
            minute=int(minute),
            id='eod_job'
        )
        
        self.scheduler.start()
        print(f"Scheduler started. EOD job scheduled for {eod_time} daily.")
    
    def stop(self):
        """Stop scheduler"""
        self.scheduler.shutdown()
        print("Scheduler stopped.")


if __name__ == "__main__":
    # Test scheduler
    scheduler = StrategyScheduler()
    scheduler.start("18:00")
    
    # Keep running
    try:
        import time
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.stop()


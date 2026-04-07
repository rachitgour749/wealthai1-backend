"""
Scheduler Service

Dynamic scheduler that loads and schedules all enabled strategies from configuration.
Supports signal generation jobs and data fetching jobs.
"""

import logging
import importlib
from typing import Dict, Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import pytz

from Services.scheduler.config_utils import scheduler_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SchedulerService:
    """
    Dynamic scheduler service that manages all strategy signal generation
    and data fetching jobs based on configuration.
    """
    
    def __init__(self):
        """Initialize the scheduler"""
        # Get timezone from config
        timezone_str = scheduler_config.get_timezone()
        self.timezone = pytz.timezone(timezone_str)
        
        # Initialize AsyncIO scheduler
        self.scheduler = AsyncIOScheduler(timezone=self.timezone)
        self.jobs = {}
        
        logger.info(f"Scheduler initialized with timezone: {timezone_str}")
    
    def start_scheduler(self):
        """
        Start the scheduler and register system jobs
        """
        try:
            # Start the scheduler
            self.scheduler.start()
            logger.info("✓ Scheduler started successfully")

            # ---------------------------------------------------------
            # Register System Jobs (Daily Report at Market Close)
            # ---------------------------------------------------------
            try:
                from Services.scheduler.jobs.daily_report_job import run_daily_trade_report
                # Schedule for 15:35 (3:35 PM IST) - 5 mins after NSE close
                self.scheduler.add_job(
                    run_daily_trade_report,
                    trigger=CronTrigger(hour=15, minute=35, day_of_week='mon-fri', timezone=self.timezone),
                    id='system_daily_trade_report',
                    name='System: Daily Trade Report (Excel)',
                    replace_existing=True
                )
                logger.info("✓ Registered system job: Daily Trade Report (15:35 IST)")
            except Exception as e:
                logger.error(f"Failed to register system job: {e}")
            
            # Log scheduled jobs
            self._log_scheduled_jobs()
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # Dynamic strategy generation and execution schedulers removed.
    
    def _create_trigger(self, job_config: Dict) -> CronTrigger:
        """
        Create a cron trigger based on job configuration
        
        Args:
            job_config: Job configuration with frequency, time, etc.
        
        Returns:
            CronTrigger object
        """
        frequency = job_config.get('frequency', 'daily')
        time_str = job_config.get('time', '00:00')
        
        # Parse time
        hour, minute = map(int, time_str.split(':'))
        
        if frequency == 'weekly':
            # Convert day name to number (0=Monday, 6=Sunday)
            day_map = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2,
                'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6
            }
            day_of_week = job_config.get('day_of_week', 'monday').lower()
            day_num = day_map.get(day_of_week, 0)
            
            trigger = CronTrigger(
                day_of_week=day_num,
                hour=hour,
                minute=minute,
                timezone=self.timezone
            )
            
        elif frequency == 'daily':
            trigger = CronTrigger(
                hour=hour,
                minute=minute,
                timezone=self.timezone
            )
            
        else:
            # Default to daily
            logger.warning(f"Unknown frequency '{frequency}', defaulting to daily")
            trigger = CronTrigger(
                hour=hour,
                minute=minute,
                timezone=self.timezone
            )
        
        return trigger
    
    def _log_scheduled_jobs(self):
        """Log all scheduled jobs with their next run times"""
        logger.info("\n" + "="*60)
        logger.info("SCHEDULED JOBS")
        logger.info("="*60)
        
        jobs = self.scheduler.get_jobs()
        
        if not jobs:
            logger.info("No jobs scheduled")
        else:
            for job in jobs:
                next_run = job.next_run_time
                logger.info(f"  {job.name}")
                logger.info(f"    ID: {job.id}")
                logger.info(f"    Next run: {next_run.strftime('%Y-%m-%d %H:%M:%S %Z') if next_run else 'N/A'}")
                logger.info("")
        
        logger.info("="*60 + "\n")
    
    def shutdown(self):
        """Shutdown the scheduler gracefully"""
        logger.info("Shutting down scheduler...")
        self.scheduler.shutdown(wait=True)
        logger.info("Scheduler shut down successfully")
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """
        Get status of a specific job
        
        Args:
            job_id: ID of the job
        
        Returns:
            Dictionary with job status or None if not found
        """
        job = self.scheduler.get_job(job_id)
        
        if not job:
            return None
        
        return {
            'id': job.id,
            'name': job.name,
            'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
            'trigger': str(job.trigger)
        }
    
    def trigger_job_now(self, job_id: str):
        """
        Manually trigger a job immediately
        
        Args:
            job_id: ID of the job to trigger
        """
        job = self.scheduler.get_job(job_id)
        
        if not job:
            logger.error(f"Job not found: {job_id}")
            return False
        
        try:
            job.modify(next_run_time=datetime.now(self.timezone))
            logger.info(f"Triggered job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to trigger job '{job_id}': {e}")
            return False


# Global scheduler instance
_scheduler_service = None


def get_scheduler() -> SchedulerService:
    """Get the global scheduler instance"""
    global _scheduler_service
    
    if _scheduler_service is None:
        _scheduler_service = SchedulerService()
    
    return _scheduler_service


def start_scheduler():
    """Start the global scheduler"""
    scheduler = get_scheduler()
    scheduler.start_scheduler()
    return scheduler

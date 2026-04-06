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
        Start the scheduler and register all enabled strategies from config
        """
        try:
            # Get all enabled strategies from config
            enabled_strategies = scheduler_config.get_all_enabled_strategies()
            
            logger.info(f"Found {len(enabled_strategies)} enabled strategies/jobs")
            
            # Register each enabled strategy
            for strategy_name in enabled_strategies:
                try:
                    self.register_strategy_job(strategy_name)
                except Exception as e:
                    logger.error(f"Failed to register job '{strategy_name}': {e}")
                    import traceback
                    traceback.print_exc()
            
            # Start the scheduler
            self.scheduler.start()
            logger.info("✓ Scheduler started successfully")
            
            # Log scheduled jobs
            self._log_scheduled_jobs()
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def register_strategy_job(self, strategy_name: str):
        """
        Dynamically register a strategy job based on configuration
        
        Args:
            strategy_name: Name of the strategy from config
        """
        strategy_config = scheduler_config.get_strategy_config(strategy_name)
        
        if not strategy_config:
            logger.warning(f"No configuration found for strategy: {strategy_name}")
            return
        
        # Determine job type and register accordingly
        if 'signal_generation' in strategy_config:
            self._register_signal_job(strategy_name, strategy_config)
        elif 'data_fetch' in strategy_config:
            self._register_data_fetch_job(strategy_name, strategy_config)
        else:
            logger.warning(f"Unknown job type for strategy: {strategy_name}")
    
    def _register_signal_job(self, strategy_name: str, config: Dict):
        """
        Register a signal generation job
        
        Args:
            strategy_name: Name of the strategy
            config: Strategy configuration dictionary
        """
        sig_gen = config['signal_generation']
        
        try:
            # Dynamically import the generator function
            module_path = config['generator_module']
            function_name = config['generator_function']
            
            logger.info(f"Loading signal generator: {module_path}.{function_name}")
            
            module = importlib.import_module(module_path)
            generator_func = getattr(module, function_name)
            
            # Create cron trigger based on frequency
            trigger = self._create_trigger(sig_gen)
            
            # Add job to scheduler
            job = self.scheduler.add_job(
                generator_func,
                trigger=trigger,
                id=f'signal_{strategy_name}',
                name=f'Signal Generation: {strategy_name}',
                replace_existing=True
            )
            
            self.jobs[f'signal_{strategy_name}'] = job
            logger.info(f"✓ Registered signal job: {strategy_name}")
            
        except Exception as e:
            logger.error(f"Failed to register signal job '{strategy_name}': {e}")
            raise
    
    def _register_data_fetch_job(self, strategy_name: str, config: Dict):
        """
        Register a data fetching job
        
        Args:
            strategy_name: Name of the data fetch job
            config: Job configuration dictionary
        """
        data_fetch = config['data_fetch']
        
        try:
            # Dynamically import the fetch function
            module_path = config['fetch_module']
            function_name = config['fetch_function']
            
            logger.info(f"Loading data fetch function: {module_path}.{function_name}")
            
            module = importlib.import_module(module_path)
            fetch_func = getattr(module, function_name)
            
            # Create cron trigger based on frequency
            trigger = self._create_trigger(data_fetch)
            
            # Add job to scheduler
            job = self.scheduler.add_job(
                fetch_func,
                trigger=trigger,
                id=f'data_{strategy_name}',
                name=f'Data Fetch: {strategy_name}',
                replace_existing=True
            )
            
            self.jobs[f'data_{strategy_name}'] = job
            logger.info(f"✓ Registered data fetch job: {strategy_name}")
            
        except Exception as e:
            logger.error(f"Failed to register data fetch job '{strategy_name}': {e}")
            raise
    
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

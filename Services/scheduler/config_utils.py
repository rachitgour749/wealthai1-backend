"""
Scheduler Configuration Utility

Provides centralized access to scheduler configuration including:
- Trading calendars (NSE India, NYSE/NASDAQ US)
- Strategy schedules and timing
- Execution settings
- Holiday management
"""

import json
import os
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from pathlib import Path


class SchedulerConfig:
    """Centralized scheduler configuration manager"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.load_config()
    
    def load_config(self, config_path: Optional[str] = None):
        """Load scheduler configuration from JSON file"""
        if config_path is None:
            # Default path: config/scheduler_config.json
            # Current file is in Services/scheduler/config_utils.py
            # Need to go up 2 levels to reach project root, then into config/
            base_dir = Path(__file__).parent.parent.parent  # Go up from Services/scheduler to project root
            config_path = base_dir / "config" / "scheduler_config.json"
        
        # Debug: Print resolved path
        print(f"[DEBUG] Scheduler config path resolved to: {config_path}")
        print(f"[DEBUG] Config file exists: {Path(config_path).exists()}")
        
        try:
            with open(config_path, 'r') as f:
                self._config = json.load(f)
            print(f"[DEBUG] Successfully loaded scheduler config")
        except FileNotFoundError as e:
            print(f"[ERROR] Config file not found at: {config_path}")
            print(f"[ERROR] Current file location: {Path(__file__).absolute()}")
            print(f"[ERROR] Base dir: {base_dir}")
            raise
    
    @property
    def config(self) -> Dict:
        """Get full configuration"""
        return self._config
    
    def get_strategy_config(self, strategy_name: str) -> Optional[Dict]:
        """Get configuration for a specific strategy"""
        return self._config.get('strategies', {}).get(strategy_name)
    
    def is_strategy_enabled(self, strategy_name: str) -> bool:
        """Check if a strategy is enabled"""
        strategy = self.get_strategy_config(strategy_name)
        return strategy.get('enabled', False) if strategy else False
    
    def get_nse_holidays(self, year: int = 2026) -> List[str]:
        """Get NSE India holidays for specified year"""
        return self._config.get('trading_calendar', {}).get(f'nse_holidays_{year}', [])
    
    def get_us_holidays(self, year: int = 2026) -> List[str]:
        """Get US market holidays for specified year"""
        return self._config.get('trading_calendar', {}).get(f'us_holidays_{year}', [])
    
    def get_us_early_close(self, year: int = 2026) -> List[str]:
        """Get US market early close dates for specified year"""
        return self._config.get('trading_calendar', {}).get(f'us_early_close_{year}', [])
    
    def is_trading_day(self, check_date: date, market: str = 'NSE') -> bool:
        """
        Check if a given date is a trading day
        
        Args:
            check_date: Date to check
            market: 'NSE' for India or 'US' for US markets
        
        Returns:
            True if it's a trading day, False otherwise
        """
        # Check if weekend
        if check_date.weekday() in self._config.get('trading_calendar', {}).get('weekend_days', [5, 6]):
            return False
        
        # Check if holiday
        date_str = check_date.strftime('%Y-%m-%d')
        year = check_date.year
        
        if market.upper() == 'NSE':
            holidays = self.get_nse_holidays(year)
        else:
            holidays = self.get_us_holidays(year)
        
        return date_str not in holidays
    
    def get_next_trading_day(self, from_date: date, market: str = 'NSE') -> date:
        """Get the next trading day after the given date"""
        from datetime import timedelta
        
        next_day = from_date + timedelta(days=1)
        while not self.is_trading_day(next_day, market):
            next_day += timedelta(days=1)
        
        return next_day
    
    def get_previous_trading_day(self, from_date: date, market: str = 'NSE') -> date:
        """Get the previous trading day before the given date"""
        from datetime import timedelta
        
        prev_day = from_date - timedelta(days=1)
        while not self.is_trading_day(prev_day, market):
            prev_day -= timedelta(days=1)
        
        return prev_day
    
    def get_execution_settings(self) -> Dict:
        """Get execution settings"""
        return self._config.get('execution_settings', {})
    
    def get_logging_config(self) -> Dict:
        """Get logging configuration"""
        return self._config.get('logging', {})
    
    def get_timezone(self) -> str:
        """Get configured timezone"""
        return self._config.get('timezone', 'Asia/Kolkata')
    
    def get_all_enabled_strategies(self) -> List[str]:
        """Get list of all enabled strategy names"""
        strategies = self._config.get('strategies', {})
        return [name for name, config in strategies.items() if config.get('enabled', False)]


# Singleton instance
scheduler_config = SchedulerConfig()

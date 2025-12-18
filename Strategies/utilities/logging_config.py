"""
Centralized Logging Configuration System for Strategies

This module provides a flexible logging system that allows strategies to control
log output through configuration files or dictionaries.

Usage:
    # Initialize with config file
    logger = StrategyLogger('Rotation_ETF', config_path='path/to/logging_config.json')
    
    # Initialize with config dict
    logger = StrategyLogger('RS_Stocks', config_dict={'enabled': True, 'categories': {'debug': False}})
    
    # Use logger
    logger.info("Loading data...")
    logger.debug("Debug information")
    logger.trade("Executed trade: BUY RELIANCE")
"""

import json
import os
from typing import Dict, Optional, Any
from pathlib import Path


class StrategyLogger:
    """
    Centralized logger for strategy modules with configurable log categories.
    
    Supports the following log categories:
    - debug: Detailed debugging information
    - info: General informational messages
    - progress: Progress updates and status messages
    - error: Error messages (always enabled)
    - warning: Warning messages
    - trade: Trade execution logs
    - performance: Performance metrics and statistics
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        'enabled': True,
        'categories': {
            'debug': False,
            'info': True,
            'progress': True,
            'error': True,  # Always enabled
            'trade': True,
            'performance': True
        }
    }
    
    def __init__(self, strategy_name: str, config_path: Optional[str] = None, 
                 config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize the StrategyLogger.
        
        Args:
            strategy_name: Name of the strategy (e.g., 'Rotation_ETF', 'RS_Stocks')
            config_path: Path to JSON configuration file (optional)
            config_dict: Configuration dictionary (optional, takes precedence over config_path)
        """
        self.strategy_name = strategy_name
        self.config = self._load_config(config_path, config_dict)
        
    def _load_config(self, config_path: Optional[str] = None, 
                     config_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load configuration from file or dictionary.
        
        Priority:
        1. config_dict (if provided)
        2. config_path (if provided)
        3. Auto-detect config file in strategy directory
        4. Default configuration
        
        Args:
            config_path: Path to JSON configuration file
            config_dict: Configuration dictionary
            
        Returns:
            Configuration dictionary
        """
        # Priority 1: Use provided config_dict
        if config_dict is not None:
            return self._merge_with_defaults(config_dict)
        
        # Priority 2: Use provided config_path
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                return self._merge_with_defaults(loaded_config)
            except Exception as e:
                print(f"[WARNING] Failed to load config from {config_path}: {e}")
                print(f"[WARNING] Using default configuration for {self.strategy_name}")
        
        # Priority 3: Auto-detect config file in strategy directory
        # Try to find logging_config.json in the strategy's directory or parent directories
        try:
            # Get the directory of the calling module
            import inspect
            frame = inspect.currentframe()
            caller_frame = frame.f_back.f_back  # Go up two frames
            caller_file = caller_frame.f_globals.get('__file__')
            
            if caller_file:
                current_dir = Path(caller_file).parent
                
                # Search current directory and up to 3 parent directories
                for _ in range(4):
                    auto_config_path = current_dir / 'logging_config.json'
                    
                    if auto_config_path.exists():
                        with open(auto_config_path, 'r') as f:
                            loaded_config = json.load(f)
                        return self._merge_with_defaults(loaded_config)
                    
                    # Move to parent directory
                    parent = current_dir.parent
                    if parent == current_dir:  # Reached root
                        break
                    current_dir = parent
        except Exception:
            pass  # Silently fail and use defaults
        
        # Priority 4: Use default configuration
        return self.DEFAULT_CONFIG.copy()
    
    def _merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge provided config with defaults to ensure all categories exist.
        
        Args:
            config: User-provided configuration
            
        Returns:
            Merged configuration with defaults
        """
        merged = self.DEFAULT_CONFIG.copy()
        
        # Update enabled flag
        if 'enabled' in config:
            merged['enabled'] = config['enabled']
        
        # Update categories
        if 'categories' in config:
            merged['categories'].update(config['categories'])
        
        # Ensure error is always enabled
        merged['categories']['error'] = True
        
        return merged
    
    def is_enabled(self, category: str) -> bool:
        """
        Check if a log category is enabled.
        
        Args:
            category: Log category name
            
        Returns:
            True if category is enabled, False otherwise
        """
        # If logging is globally disabled, only allow errors
        if not self.config.get('enabled', True):
            return category == 'error'
        
        # Check if category is enabled
        return self.config.get('categories', {}).get(category, False)
    
    def _log(self, category: str, message: str, prefix: str = ""):
        """
        Internal logging method.
        
        Args:
            category: Log category
            message: Message to log
            prefix: Optional prefix for the message
        """
        # Return early if not enabled - prevents any print() calls including blank lines
        if not self.is_enabled(category):
            return
            
        # Only print if enabled
        if prefix:
            print(f"{prefix} {message}")
        else:
            print(message)
    
    def debug(self, message: str):
        """
        Log debug message.
        
        Args:
            message: Debug message
        """
        self._log('debug', message, prefix="[DEBUG]")
    
    def info(self, message: str):
        """
        Log informational message.
        
        Args:
            message: Info message
        """
        self._log('info', message)
    
    def progress(self, message: str):
        """
        Log progress update.
        
        Args:
            message: Progress message
        """
        self._log('progress', message)
    
    def error(self, message: str):
        """
        Log error message (always enabled).
        
        Args:
            message: Error message
        """
        self._log('error', message, prefix="[ERROR]")
    
    def warning(self, message: str):
        """
        Log warning message.
        
        Args:
            message: Warning message
        """
        self._log('info', message, prefix="[WARNING]")
    
    def trade(self, message: str):
        """
        Log trade execution message.
        
        Args:
            message: Trade message
        """
        self._log('trade', message)
    
    def performance(self, message: str):
        """
        Log performance metric.
        
        Args:
            message: Performance message
        """
        self._log('performance', message)
    
    def update_config(self, config_dict: Dict[str, Any]):
        """
        Update logging configuration dynamically.
        
        Args:
            config_dict: New configuration dictionary
        """
        self.config = self._merge_with_defaults(config_dict)
    
    def set_category(self, category: str, enabled: bool):
        """
        Enable or disable a specific log category.
        
        Args:
            category: Category name
            enabled: True to enable, False to disable
        """
        if category in self.config.get('categories', {}):
            self.config['categories'][category] = enabled
    
    def enable_all(self):
        """Enable all log categories."""
        for category in self.config.get('categories', {}).keys():
            self.config['categories'][category] = True
    
    def disable_all(self):
        """Disable all log categories except errors."""
        for category in self.config.get('categories', {}).keys():
            if category != 'error':
                self.config['categories'][category] = False
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Current configuration dictionary
        """
        return self.config.copy()

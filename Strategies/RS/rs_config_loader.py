"""
RS Strategy Configuration Loader
Centralized configuration management for RS_ETF and RS_Stocks strategies
"""

import json
import os
from typing import Dict, Any
from pathlib import Path


class RSConfig:
    """Load and manage RS strategy configuration"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RSConfig, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._load_config()
    
    def _load_config(self):
        """Load configuration from rs_config.json"""
        try:
            # Get the directory where this file is located
            current_dir = Path(__file__).parent
            config_path = current_dir / 'rs_config.json'
            
            if not config_path.exists():
                print(f"Warning: Configuration file not found at {config_path}")
                print("Using default configuration")
                self._config = self._get_default_config()
                return
            
            with open(config_path, 'r') as f:
                self._config = json.load(f)
            
            print(f"✓ RS Strategy configuration loaded from {config_path}")
            
        except Exception as e:
            print(f"Error loading RS configuration: {e}")
            print("Using default configuration")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file is not found"""
        return {
            "stop_loss_settings": {
                "daily_stop_loss_check": True
            },
            "strategy_defaults": {
                "stop_loss_pct": 15.0,
                "capital_reset_threshold_pct": 25.0,
                "buffer_capital_pct": 10.0,
                "max_positions": 20,
                "transaction_cost_pct": 0.1
            }
        }
    
    def get_daily_stop_loss_check(self) -> bool:
        """
        Get whether to check stop loss daily or weekly
        
        Returns:
            bool: True for daily check, False for weekly (signal day only) check
        """
        return self._config.get('stop_loss_settings', {}).get('daily_stop_loss_check', True)
    
    def get_stop_loss_pct(self) -> float:
        """Get default stop loss percentage"""
        return self._config.get('strategy_defaults', {}).get('stop_loss_pct', 15.0)
    
    def get_capital_reset_threshold_pct(self) -> float:
        """Get capital reset threshold percentage"""
        return self._config.get('strategy_defaults', {}).get('capital_reset_threshold_pct', 25.0)
    
    def get_buffer_capital_pct(self) -> float:
        """Get buffer capital percentage"""
        return self._config.get('strategy_defaults', {}).get('buffer_capital_pct', 10.0)
    
    def get_max_positions(self) -> int:
        """Get default max positions"""
        return self._config.get('strategy_defaults', {}).get('max_positions', 20)
    
    def get_transaction_cost_pct(self) -> float:
        """Get transaction cost percentage"""
        return self._config.get('strategy_defaults', {}).get('transaction_cost_pct', 0.1)
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary"""
        return self._config.copy()
    
    def reload(self):
        """Reload configuration from file"""
        self._config = None
        self._load_config()


# Singleton instance
def get_rs_config() -> RSConfig:
    """Get the RS configuration singleton instance"""
    return RSConfig()


# Convenience function for quick access
def is_daily_stop_loss_enabled() -> bool:
    """Quick check if daily stop loss is enabled"""
    return get_rs_config().get_daily_stop_loss_check()


if __name__ == "__main__":
    # Test the configuration loader
    config = get_rs_config()
    print("\n=== RS Strategy Configuration ===")
    print(f"Daily Stop Loss Check: {config.get_daily_stop_loss_check()}")
    print(f"Stop Loss %: {config.get_stop_loss_pct()}")
    print(f"Capital Reset Threshold %: {config.get_capital_reset_threshold_pct()}")
    print(f"Buffer Capital %: {config.get_buffer_capital_pct()}")
    print(f"Max Positions: {config.get_max_positions()}")
    print(f"Transaction Cost %: {config.get_transaction_cost_pct()}")
    print("\nFull Configuration:")
    import pprint
    pprint.pprint(config.get_all_config())

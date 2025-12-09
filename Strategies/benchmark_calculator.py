"""
Benchmark Calculator Module

Calculates Nifty 50 buy-and-hold benchmark performance metrics
for comparison with trading strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional


def safe_float(value, default=0.0):
    """Convert value to float, handling infinity and NaN values safely."""
    try:
        if value is None or np.isnan(value) or np.isinf(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_divide(numerator, denominator, default=0.0):
    """Safely divide two numbers, handling division by zero."""
    try:
        if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            return default
        result = numerator / denominator
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (TypeError, ZeroDivisionError):
        return default


def safe_power(base, exponent, default=0.0):
    """Safely calculate power, handling edge cases."""
    try:
        if base <= 0:
            return default
        result = base ** exponent
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (TypeError, ValueError, OverflowError):
        return default


class BenchmarkCalculator:
    """Calculate Nifty 50 buy-and-hold benchmark performance"""
    
    def __init__(self, initial_capital: float, index_data: pd.DataFrame, trading_dates: List[datetime]):
        """
        Initialize benchmark calculator
        
        Args:
            initial_capital: Starting capital amount
            index_data: DataFrame with index prices (must have 'adjusted_close' column)
            trading_dates: List of trading dates for the backtest period
        """
        self.initial_capital = initial_capital
        self.index_data = index_data
        self.trading_dates = trading_dates
        self.benchmark_snapshots = []
        
    def calculate_benchmark_values(self) -> List[Dict]:
        """
        Calculate benchmark buy-and-hold values for each trading date
        
        Returns:
            List of dictionaries with date and benchmark_value
        """
        if not self.trading_dates:
            return []
        
        # Get starting index price
        start_date = self.trading_dates[0]
        try:
            start_index_price = self.index_data.loc[start_date]['adjusted_close']
        except KeyError:
            print(f"Warning: No index data for start date {start_date}")
            return []
        
        # Calculate benchmark value for each date
        benchmark_snapshots = []
        for date in self.trading_dates:
            try:
                current_index_price = self.index_data.loc[date]['adjusted_close']
                # Benchmark value = initial_capital * (current_price / start_price)
                benchmark_value = self.initial_capital * (current_index_price / start_index_price)
                
                benchmark_snapshots.append({
                    'date': date,
                    'benchmark_value': benchmark_value,
                    'index_price': current_index_price
                })
            except KeyError:
                # If no data for this date, use previous value
                if benchmark_snapshots:
                    benchmark_snapshots.append({
                        'date': date,
                        'benchmark_value': benchmark_snapshots[-1]['benchmark_value'],
                        'index_price': benchmark_snapshots[-1]['index_price']
                    })
        
        self.benchmark_snapshots = benchmark_snapshots
        return benchmark_snapshots
    
    def calculate_benchmark_metrics(self, risk_free_rate: float = 6.0) -> Dict:
        """
        Calculate comprehensive benchmark performance metrics
        
        Args:
            risk_free_rate: Annual risk-free rate in percentage (default 6%)
            
        Returns:
            Dictionary with benchmark metrics
        """
        if not self.benchmark_snapshots:
            self.calculate_benchmark_values()
        
        if not self.benchmark_snapshots:
            return self._get_empty_metrics()
        
        # Convert to DataFrame for easier calculation
        df = pd.DataFrame(self.benchmark_snapshots)
        df['returns'] = df['benchmark_value'].pct_change()
        
        # Basic metrics
        start_value = df['benchmark_value'].iloc[0]
        end_value = df['benchmark_value'].iloc[-1]
        total_return = safe_float((safe_divide(end_value, start_value, 1.0) - 1) * 100)
        
        # Calculate CAGR
        start_date = df['date'].iloc[0]
        end_date = df['date'].iloc[-1]
        years = (end_date - start_date).days / 365.25
        cagr = safe_float((safe_power(safe_divide(end_value, start_value, 1.0), safe_divide(1.0, years, 1.0)) - 1) * 100)
        
        # Calculate XIRR (simplified as CAGR for buy-and-hold)
        xirr = cagr
        
        # Calculate annualized return
        annualized_return = cagr
        
        # Calculate maximum drawdown
        max_drawdown = self._calculate_max_drawdown(df)
        
        # Calculate Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio(df, risk_free_rate)
        
        # Calculate volatility
        volatility = self._calculate_volatility(df)
        
        # Calculate Calmar ratio
        calmar_ratio = abs(cagr / max_drawdown) if max_drawdown < 0 else 0.0
        
        # Calculate win rate (days with positive returns)
        positive_days = len(df[df['returns'] > 0])
        total_days = len(df[df['returns'].notna()])
        win_rate = safe_float((positive_days / total_days * 100) if total_days > 0 else 0)
        
        return {
            'total_return_pct': safe_float(total_return),
            'cagr_pct': safe_float(cagr),
            'xirr_pct': safe_float(xirr),
            'annualized_return_pct': safe_float(annualized_return),
            'max_drawdown_pct': safe_float(max_drawdown),
            'sharpe_ratio': safe_float(sharpe_ratio),
            'volatility_pct': safe_float(volatility),
            'calmar_ratio': safe_float(calmar_ratio),
            'win_rate_pct': safe_float(win_rate),
            'beta': 1.0,  # Benchmark beta is always 1.0
            'treynor_ratio': safe_float(cagr - risk_free_rate),  # Simplified for benchmark
            'final_value': safe_float(end_value),
            'total_days': len(df)
        }
    
    def get_benchmark_values_array(self) -> List[float]:
        """Get array of benchmark values for graphing"""
        if not self.benchmark_snapshots:
            self.calculate_benchmark_values()
        return [snap['benchmark_value'] for snap in self.benchmark_snapshots]
    
    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown percentage"""
        try:
            cumulative_max = df['benchmark_value'].expanding().max()
            drawdown = (df['benchmark_value'] - cumulative_max) / cumulative_max * 100
            max_drawdown = drawdown.min()
            return safe_float(max_drawdown)
        except Exception:
            return 0.0
    
    def _calculate_sharpe_ratio(self, df: pd.DataFrame, risk_free_rate: float) -> float:
        """Calculate Sharpe ratio"""
        try:
            returns = df['returns'].dropna()
            if len(returns) < 2:
                return 0.0
            
            # Annualize returns and volatility
            mean_return = returns.mean() * 252  # Assuming daily returns
            std_return = returns.std() * np.sqrt(252)
            
            if std_return == 0:
                return 0.0
            
            sharpe = (mean_return * 100 - risk_free_rate) / (std_return * 100)
            return safe_float(sharpe)
        except Exception:
            return 0.0
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate annualized volatility percentage"""
        try:
            returns = df['returns'].dropna()
            if len(returns) < 2:
                return 0.0
            
            volatility = returns.std() * np.sqrt(252) * 100
            return safe_float(volatility)
        except Exception:
            return 0.0
    
    def _get_empty_metrics(self) -> Dict:
        """Return empty metrics dict"""
        return {
            'total_return_pct': 0.0,
            'cagr_pct': 0.0,
            'xirr_pct': 0.0,
            'annualized_return_pct': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0,
            'volatility_pct': 0.0,
            'calmar_ratio': 0.0,
            'win_rate_pct': 0.0,
            'beta': 1.0,
            'treynor_ratio': 0.0,
            'final_value': self.initial_capital,
            'total_days': 0
        }

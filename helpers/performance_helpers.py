"""
Performance Helper Utilities

Utility functions for calculating financial performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


def calculate_performance_metrics(equity_curve: pd.Series, 
                                  benchmark_curve: Optional[pd.Series] = None,
                                  risk_free_rate: float = 0.07) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        equity_curve: Strategy equity curve (Series with dates as index)
        benchmark_curve: Benchmark equity curve (optional, for alpha calculation)
        risk_free_rate: Annual risk-free rate (default: 7% for India)
        
    Returns:
        Dictionary containing:
        - total_return: Total return percentage
        - cagr: Compound Annual Growth Rate
        - volatility: Annualized volatility
        - sharpe_ratio: Risk-adjusted return metric
        - max_drawdown: Maximum peak-to-trough decline
        - win_rate: Percentage of positive return periods
        - benchmark_return: Benchmark return (if provided)
        - alpha: Excess return over benchmark (if provided)
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return {}
    
    # Calculate returns
    returns = equity_curve.pct_change().dropna()
    
    # Total return
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    
    # CAGR (Compound Annual Growth Rate)
    years = len(equity_curve) / 52  # Assuming weekly data
    cagr = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(52) * 100
    
    # Sharpe Ratio
    excess_returns = returns - (risk_free_rate / 52)
    sharpe_ratio = (excess_returns.mean() / returns.std() * np.sqrt(52)) if returns.std() > 0 else 0
    
    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0
    
    metrics = {
        'total_return': round(total_return, 2),
        'cagr': round(cagr, 2),
        'volatility': round(volatility, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'max_drawdown': round(max_drawdown, 2),
        'win_rate': round(win_rate, 2)
    }
    
    # Add benchmark comparison if provided
    if benchmark_curve is not None and not benchmark_curve.empty:
        benchmark_return = (benchmark_curve.iloc[-1] / benchmark_curve.iloc[0] - 1) * 100
        alpha = total_return - benchmark_return
        metrics['benchmark_return'] = round(benchmark_return, 2)
        metrics['alpha'] = round(alpha, 2)
    
    return metrics

"""
ETF Rotation Strategy Handler
Handles backtest execution for ETF Rotation strategy
"""
import sys
import os
from typing import Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
import pandas as pd

# Add Strategies path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Strategies'))

from Handlers.base_handler import BaseStrategyHandler
from APIs.unified_schemas import UnifiedBacktestRequest, UnifiedBacktestResponse
from Strategies.Rotation_ETF.services.backtester import ETFRotationBacktester


class ETFRotationHandler(BaseStrategyHandler):
    """Handler for ETF Rotation strategy"""
    
    def validate_request(self, request: UnifiedBacktestRequest) -> None:
        """Validate ETF Rotation specific parameters"""
        if not request.tickers:
            raise ValueError("ETF_Rotation requires 'tickers' parameter")
        if not request.capital_per_week:
            raise ValueError("ETF_Rotation requires 'capital_per_week' parameter")
        if not request.accumulation_weeks:
            raise ValueError("ETF_Rotation requires 'accumulation_weeks' parameter")
        if request.brokerage_percent is None:
            raise ValueError("ETF_Rotation requires 'brokerage_percent' parameter")
    
    async def run_backtest(self, request: UnifiedBacktestRequest) -> UnifiedBacktestResponse:
        """
        Run ETF Rotation backtest
        
        Args:
            request: Unified backtest request
            
        Returns:
            UnifiedBacktestResponse with results
        """
        try:
            # Validate request
            self.validate_request(request)
            
            # Clean tickers (remove .NS)
            request.tickers = self._clean_tickers(request.tickers)
            
            # Initialize backtester
            backtester = ETFRotationBacktester(db_path="unified_etf_data.sqlite")
            
            # Run backtest
            result = backtester.run_backtest(
                tickers=request.tickers,
                start_date=request.start_date,
                end_date=request.end_date,
                capital_per_week=request.capital_per_week,
                accumulation_weeks=request.accumulation_weeks,
                brokerage_percent=request.brokerage_percent,
                compounding_enabled=request.compounding_enabled or False
            )
            
            if "error" in result:
                return UnifiedBacktestResponse(
                    success=False,
                    strategy_type=request.strategy_type,
                    metrics={},
                    error=result["error"]
                )
            
            # Calculate metrics (Raw values)
            etf_metrics = backtester.calculate_metrics(
                request.capital_per_week,
                request.accumulation_weeks,
                request.risk_free_rate or 8.0
            )
            
            # Helper to parse string values back to float
            def parse_metric(val):
                if isinstance(val, (int, float)): return val
                if isinstance(val, str):
                    clean = val.replace('₹', '').replace(',', '').replace('%', '')
                    try:
                        return float(clean)
                    except ValueError:
                        return val
                return val

            # Standardize Strategy Metrics
            flat_metrics = {
                "total_investment": parse_metric(etf_metrics.get('Total Investment', 0)),
                "final_capital": parse_metric(etf_metrics.get('Final Value', 0)),
                "total_return_pct": parse_metric(etf_metrics.get('Total Return', 0)),
                "cagr": parse_metric(etf_metrics.get('CAGR', 0)),
                "xirr": parse_metric(etf_metrics.get('XIRR', 0)),
                "sharpe_ratio": parse_metric(etf_metrics.get('Sharpe Ratio', 0)),
                "calmar_ratio": parse_metric(etf_metrics.get('Calmar Ratio', 0)),
                "max_drawdown_pct": parse_metric(etf_metrics.get('Max Drawdown', 0)),
                "volatility": parse_metric(etf_metrics.get('Volatility', 0)),
                "total_trades": etf_metrics.get('Total Trades', 0),
                "win_rate_pct": parse_metric(etf_metrics.get('Win Rate', 0))
            }
            
            # Calculate benchmark metrics
            total_investment = request.accumulation_weeks * request.capital_per_week
            benchmark_metrics = backtester.calculate_benchmark_metrics(
                total_investment,
                request.risk_free_rate or 8.0
            )
            
            flat_benchmark_metrics = {
                "total_investment": parse_metric(benchmark_metrics.get('Total Investment', 0)),
                "final_capital": parse_metric(benchmark_metrics.get('Final Value', 0)),
                "total_return_pct": parse_metric(benchmark_metrics.get('Total Return', 0)),
                "cagr": parse_metric(benchmark_metrics.get('CAGR', 0)),
                "xirr": parse_metric(benchmark_metrics.get('XIRR', 0)),
                "sharpe_ratio": parse_metric(benchmark_metrics.get('Sharpe Ratio', 0)),
                "calmar_ratio": parse_metric(benchmark_metrics.get('Calmar Ratio', 0)),
                "max_drawdown_pct": parse_metric(benchmark_metrics.get('Max Drawdown', 0)),
                "volatility": parse_metric(benchmark_metrics.get('Volatility', 0))
            }
            
            # Prepare performance data
            performance_data = {
                "dates": [],
                "etf_strategy": [],
                "cumulative_investment": [],
                "benchmark_buyhold": []
            }
            
            if not backtester.weekly_nav_df.empty:
                performance_data["dates"] = [str(date) for date in backtester.weekly_nav_df['date']]
                performance_data["strategy"] = backtester.weekly_nav_df['nav'].tolist()
                performance_data["cumulative_investment"] = backtester.weekly_nav_df['cumulative_investment'].tolist()
                
                if not backtester.nifty50_df.empty:
                    # Resample benchmark data to match strategy dates
                    strategy_dates = pd.to_datetime(backtester.weekly_nav_df['date'])
                    
                    # Ensure benchmark df index is datetime
                    if not isinstance(backtester.nifty50_df.index, pd.DatetimeIndex):
                        backtester.nifty50_df.index = pd.to_datetime(backtester.nifty50_df['date'])
                    
                    # Reindex/Resample
                    # Using asof to get closest value on or before the date
                    benchmark_aligned = backtester.nifty50_df.reindex(strategy_dates, method='ffill')
                    
                    performance_data["benchmark_buyhold"] = benchmark_aligned['nav'].fillna(method='ffill').tolist()
            
            # Prepare transaction log
            # Use full portfolio log with all details (costs, debug_info, etc.)
            transaction_log = backtester.portfolio_log
            
            # Get cost breakdown
            cost_breakdown = backtester.get_transaction_costs_summary()

            # Combined final metrics
            metrics = {
                **flat_metrics,
                "benchmark_metrics": flat_benchmark_metrics
            }

            # Sanitize all data
            metrics = self._sanitize_data(metrics)
            performance_data = self._sanitize_data(performance_data)
            transaction_log = self._sanitize_data(transaction_log)
            # cost_breakdown values must be sanitized for numpy support
            cost_breakdown = self._sanitize_data(cost_breakdown)
            
            # Cache backtest results
            print(f"[ETF_ROTATION_HANDLER] About to cache results...")
            print(f"[ETF_ROTATION_HANDLER] Backtester has {len(backtester.portfolio_log)} portfolio log entries")
            
            try:
                from APIs.common.cache import cache_backtest_results
                print(f"[ETF_ROTATION_HANDLER] Calling cache_backtest_results...")
                cache_backtest_results("ETF_Rotation", backtester)
                print(f"[ETF_ROTATION_HANDLER] ✅ Cache call completed")
            except Exception as e:
                print(f"[ETF_ROTATION_HANDLER] ❌ ERROR caching: {e}")
                import traceback
                traceback.print_exc()
            
            return UnifiedBacktestResponse(
                success=True,
                strategy_type=request.strategy_type,
                metrics=metrics,
                performance_data=performance_data,
                transaction_log=transaction_log,
                cost_breakdown=cost_breakdown
            )
            
        except Exception as e:
            return UnifiedBacktestResponse(
                success=False,
                strategy_type=request.strategy_type,
                metrics={},
                error=f"ETF Rotation backtest failed: {str(e)}"
            )

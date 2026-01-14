"""
ETF Payout Strategy Handler
Handles backtest execution for ETF Payout strategy
"""
import sys
import os
from sqlalchemy.orm import Session

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Strategies'))

from Handlers.base_handler import BaseStrategyHandler
from APIs.unified_schemas import UnifiedBacktestRequest, UnifiedBacktestResponse
from Strategies.CustomStrategies.Rotation_ETF_Payout.backtester import RotationETFPayoutBacktester


class ETFPayoutHandler(BaseStrategyHandler):
    """Handler for ETF Payout strategy"""
    
    def validate_request(self, request: UnifiedBacktestRequest) -> None:
        """Validate ETF Payout specific parameters"""
        if not request.tickers:
            raise ValueError("ETF_Payout requires 'tickers' parameter")
        if not request.capital_per_week:
            raise ValueError("ETF_Payout requires 'capital_per_week' parameter")
        if not request.accumulation_weeks:
            raise ValueError("ETF_Payout requires 'accumulation_weeks' parameter")
        if request.brokerage_percent is None:
            raise ValueError("ETF_Payout requires 'brokerage_percent' parameter")
    
    async def run_backtest(self, request: UnifiedBacktestRequest) -> UnifiedBacktestResponse:
        """Run ETF Payout backtest"""
        try:
            self.validate_request(request)
            
            # Clean tickers (remove .NS)
            request.tickers = self._clean_tickers(request.tickers)
            
            # Initialize backtester
            backtester = RotationETFPayoutBacktester(db_path="unified_etf_data.sqlite")
            backtester._verbose = True # Enable for debugging
            
            # Set payout specific parameters from request
            if hasattr(request, 'withdraw_amount') and request.withdraw_amount is not None:
                backtester.withdraw_amount = request.withdraw_amount
            if hasattr(request, 'payout_start_week') and request.payout_start_week is not None:
                backtester.payout_start_week = request.payout_start_week
            
            # Run backtest (parent method doesn't accept withdraw_amount/payout_start_week)
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
            
            # Calculate metrics
            metrics = backtester.calculate_metrics(
                request.capital_per_week,
                request.accumulation_weeks,
                request.risk_free_rate or 8.0
            )
            
            # Add payout-specific metrics
            metrics['total_withdrawn'] = backtester.total_withdrawn_amount
            
            # Calculate benchmark metrics
            total_investment = request.accumulation_weeks * request.capital_per_week
            backtester.nifty50_df = backtester.calculate_benchmark_buyhold(
                start_date=request.start_date,
                end_date=request.end_date,
                total_investment=total_investment,
                brokerage_percent=request.brokerage_percent
            )
            
            benchmark_metrics = backtester.calculate_benchmark_metrics(
                total_investment,
                request.risk_free_rate or 8.0
            )
            
            # Prepare performance data
            performance_data = {
                "dates": [],
                "etf_strategy": [],
                "cumulative_investment": [],
                "cumulative_withdrawn": [],
                "benchmark_buyhold": []
            }
            
            if not backtester.weekly_nav_df.empty:
                performance_data["dates"] = [str(date) for date in backtester.weekly_nav_df['date']]
                performance_data["etf_strategy"] = backtester.weekly_nav_df['nav'].tolist()
                performance_data["cumulative_investment"] = backtester.weekly_nav_df['cumulative_investment'].tolist()
                
                # Populate benchmark_buyhold if available
                if not backtester.nifty50_df.empty:
                    performance_data["benchmark_buyhold"] = backtester.nifty50_df['nav'].tolist()
                
                # Populate cumulative_withdrawn from backtester
                cumulative_withdrawn = 0.0
                withdrawal_dict = {str(log['date']): log['cumulative_withdrawn'] for log in backtester.withdrawal_log}
                
                performance_data["cumulative_withdrawn"] = []
                for idx, row in backtester.weekly_nav_df.iterrows():
                    date_str = str(row['date'])
                    if date_str in withdrawal_dict:
                        cumulative_withdrawn = withdrawal_dict[date_str]
                    performance_data["cumulative_withdrawn"].append(cumulative_withdrawn)
            
            # Normalize benchmark metrics
            benchmark_metrics = self._normalize_benchmark_metrics(benchmark_metrics)
            
            # Sanitize all data
            metrics = self._sanitize_data({
                **metrics,
                "benchmark_metrics": benchmark_metrics
            })
            performance_data = self._sanitize_data(performance_data)
            
            # Cache backtest results for transaction-log endpoint
            try:
                from APIs.centralized_backtest import cache_backtest_results
                cache_backtest_results("ETF_Payout", backtester)
            except Exception as e:
                print(f"[ETF_PAYOUT_HANDLER] Warning: Could not cache backtest results: {e}")
            
            return UnifiedBacktestResponse(
                success=True,
                strategy_type=request.strategy_type,
                metrics=metrics,
                performance_data=performance_data
            )
            
        except Exception as e:
            return UnifiedBacktestResponse(
                success=False,
                strategy_type=request.strategy_type,
                metrics={},
                error=f"ETF Payout backtest failed: {str(e)}"
            )

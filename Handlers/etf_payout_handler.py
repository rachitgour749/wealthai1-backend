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
            
            # Initialize backtester
            backtester = RotationETFPayoutBacktester(db_path="unified_etf_data.sqlite")
            
            # Set payout-specific parameters as instance attributes
            backtester.withdraw_amount = request.withdraw_amount or 0.0
            backtester.payout_start_week = request.payout_start_week or 1
            
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
                # Note: cumulative_withdrawn might not be in the dataframe
                if 'cumulative_withdrawn' in backtester.weekly_nav_df.columns:
                    performance_data["cumulative_withdrawn"] = backtester.weekly_nav_df['cumulative_withdrawn'].tolist()
            
            # Sanitize data
            metrics = self._sanitize_data(metrics)
            performance_data = self._sanitize_data(performance_data)
            
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

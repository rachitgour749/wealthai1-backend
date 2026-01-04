"""
Rotation Stocks Strategy Handler
Handles backtest execution for Rotation Stocks strategy
"""
import sys
import os
from sqlalchemy.orm import Session

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Strategies'))

from Handlers.base_handler import BaseStrategyHandler
from APIs.unified_schemas import UnifiedBacktestRequest, UnifiedBacktestResponse
from Strategies.Rotation_Stocks.services.backtester import StockRotationBacktester


class RotationStocksHandler(BaseStrategyHandler):
    """Handler for Rotation Stocks strategy"""
    
    def validate_request(self, request: UnifiedBacktestRequest) -> None:
        """Validate Rotation Stocks specific parameters"""
        if not request.tickers:
            raise ValueError("Rotation_Stocks requires 'tickers' parameter")
        if not request.capital_per_week:
            raise ValueError("Rotation_Stocks requires 'capital_per_week' parameter")
        if not request.accumulation_weeks:
            raise ValueError("Rotation_Stocks requires 'accumulation_weeks' parameter")
        if request.brokerage_percent is None:
            raise ValueError("Rotation_Stocks requires 'brokerage_percent' parameter")
    
    async def run_backtest(self, request: UnifiedBacktestRequest) -> UnifiedBacktestResponse:
        """Run Rotation Stocks backtest"""
        try:
            self.validate_request(request)
            
            # Initialize backtester
            backtester = StockRotationBacktester(db_path="unified_stock_data.sqlite")
            
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
            
            # Calculate metrics
            metrics = backtester.calculate_metrics(
                request.capital_per_week,
                request.accumulation_weeks,
                request.risk_free_rate or 8.0
            )
            
            # Prepare performance data
            performance_data = {
                "dates": [],
                "stock_strategy": [],
                "cumulative_investment": [],
                "benchmark_buyhold": []
            }
            
            if not backtester.weekly_nav_df.empty:
                performance_data["dates"] = [str(date) for date in backtester.weekly_nav_df['date']]
                performance_data["stock_strategy"] = backtester.weekly_nav_df['nav'].tolist()
                performance_data["cumulative_investment"] = backtester.weekly_nav_df['cumulative_investment'].tolist()
            
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
                error=f"Rotation Stocks backtest failed: {str(e)}"
            )

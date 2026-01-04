"""
SuperTrend Strategy Handler
Handles backtest execution for SuperTrend strategy
"""
import sys
import os
from sqlalchemy.orm import Session

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Strategies'))

from Handlers.base_handler import BaseStrategyHandler
from APIs.unified_schemas import UnifiedBacktestRequest, UnifiedBacktestResponse
from Strategies.SuperTrend.backtester_core.backtest_engine import BacktestEngine


class SuperTrendHandler(BaseStrategyHandler):
    """Handler for SuperTrend strategy"""
    
    def validate_request(self, request: UnifiedBacktestRequest) -> None:
        """Validate SuperTrend specific parameters"""
        if not request.initial_capital:
            raise ValueError("SuperTrend requires 'initial_capital' parameter")
    
    async def run_backtest(self, request: UnifiedBacktestRequest) -> UnifiedBacktestResponse:
        """Run SuperTrend backtest"""
        try:
            self.validate_request(request)
            
            # Note: SuperTrend requires stock_data and index_data to be loaded first
            # This is a simplified implementation - you may need to load data from database
            # For now, return an error message indicating this strategy needs special handling
            
            return UnifiedBacktestResponse(
                success=False,
                strategy_type=request.strategy_type,
                metrics={},
                error="SuperTrend strategy requires direct API call to /api/supertrend/backtest due to complex data requirements. Please use the dedicated SuperTrend endpoint."
            )
            
        except Exception as e:
            return UnifiedBacktestResponse(
                success=False,
                strategy_type=request.strategy_type,
                metrics={},
                error=f"SuperTrend backtest failed: {str(e)}"
            )

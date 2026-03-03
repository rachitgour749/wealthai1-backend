"""
SuperTrend Strategy Handler
Handles backtest execution for SuperTrend strategy
"""
import sys
import os
from typing import Dict, Any
from datetime import datetime
import pandas as pd

# Add Strategies path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Strategies'))

from Handlers.base_handler import BaseStrategyHandler
from APIs.unified_schemas import UnifiedBacktestRequest, UnifiedBacktestResponse
from Strategies.SuperTrend.services.backtester import SuperTrendBacktester

class SuperTrendHandler(BaseStrategyHandler):
    """Handler for SuperTrend strategy"""
    
    def validate_request(self, request: UnifiedBacktestRequest) -> None:
        """Validate SuperTrend specific parameters"""
        if not request.tickers:
            raise ValueError("SuperTrend strategy requires 'tickers' parameter")
        if request.initial_capital is None and request.total_capital is None:
            raise ValueError("SuperTrend strategy requires 'initial_capital' or 'total_capital' parameter")
    
    async def run_backtest(self, request: UnifiedBacktestRequest) -> UnifiedBacktestResponse:
        """
        Run SuperTrend backtest
        """
        try:
            # Validate request
            self.validate_request(request)
            
            # Clean tickers
            tickers = self._clean_tickers(request.tickers)
            
            # Derive market from strategy_type as the primary source of truth.
            if request.strategy_type == 'US_SuperTrend_Strategy' or 'US' in request.strategy_type.upper():
                market = 'US'
            else:
                market = 'INDIA'
            asset_type = 'STOCK'

            if hasattr(request, 'parameters') and isinstance(request.parameters, dict):
                market = request.parameters.get('market', market).upper()
                asset_type = request.parameters.get('asset_type', asset_type).upper()

            # Initialize backtester 
            backtester = SuperTrendBacktester(market=market, asset_type=asset_type)
            
            # Update params
            params_to_update = {}
            if hasattr(request, 'parameters') and isinstance(request.parameters, dict):
                params_to_update.update(request.parameters)
            
            # Fallbacks exactly on request object if applicable
            for param in ['atr_multiplier', 'atr_period', 'stop_loss_pct', 'number_of_slots', 'brokerage_percent']:
                if hasattr(request, param) and getattr(request, param) is not None:
                    params_to_update[param] = getattr(request, param)
                    
            # Specifically check 'perweek capital' mapping -> mapped to initial_capital 
            # as total portfolio capital. Note: handled centrally.
            
            if params_to_update:
                backtester.strategy.update_config(params_to_update)

            initial_capital = request.initial_capital or request.total_capital
            
            # Run backtest
            result = backtester.run_backtest(
                tickers=tickers,
                start_date=request.start_date,
                end_date=request.end_date,
                initial_capital=initial_capital,
                risk_free_rate=request.risk_free_rate or 8.0,
                config_params=params_to_update
            )
            
            if "error" in result:
                return UnifiedBacktestResponse(
                    success=False,
                    strategy_type=request.strategy_type,
                    metrics={},
                    error=result["error"]
                )
            
            # Prepare result
            metrics = result.get("metrics", {})
            performance_data = result.get("performance_data", [])
            transaction_log = result.get("transaction_log", [])
            
            # Format performance data for frontend charting
            chart_data = {
                "dates": [str(p['date']) for p in performance_data],
                "strategy": [p['strategy'] for p in performance_data],
                "benchmark_buyhold": [p['benchmark_buyhold'] for p in performance_data],
                "cumulative_investment": [p['cumulative_investment'] for p in performance_data]
            }
            
            # Sanitize data
            metrics = self._sanitize_data(metrics)
            chart_data = self._sanitize_data(chart_data)
            transaction_log = self._sanitize_data(transaction_log)
            
            return UnifiedBacktestResponse(
                success=True,
                strategy_type=request.strategy_type,
                metrics=metrics,
                performance_data=chart_data,
                transaction_log=transaction_log
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return UnifiedBacktestResponse(
                success=False,
                strategy_type=request.strategy_type,
                metrics={},
                error=f"SuperTrend backtest failed: {str(e)}"
            )

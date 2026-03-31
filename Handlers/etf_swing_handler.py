"""
ETF Swing Strategy Handler
Handles backtest execution for ETF Swing strategy
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
from Strategies.ETF_Swing_Strategy.services.backtester import ETFSwingBacktester

class ETFSwingHandler(BaseStrategyHandler):
    """Handler for ETF Swing strategy"""
    
    def validate_request(self, request: UnifiedBacktestRequest) -> None:
        """Validate ETF Swing specific parameters"""
        if not request.tickers:
            raise ValueError("ETF_Swing_Strategy requires 'tickers' parameter")
        if request.initial_capital is None and request.total_capital is None:
            raise ValueError("ETF_Swing_Strategy requires 'initial_capital' or 'total_capital' parameter")
    
    async def run_backtest(self, request: UnifiedBacktestRequest) -> UnifiedBacktestResponse:
        """
        Run ETF Swing backtest
        """
        try:
            # Validate request
            self.validate_request(request)
            
            # Clean tickers
            tickers = self._clean_tickers(request.tickers)
            
            # Initialize backtester
            # config_path can be used to pass customization if needed
            backtester = ETFSwingBacktester()
            
            # Update strategy parameters from request if provided
            if hasattr(request, 'sma_lookback') and request.sma_lookback:
                backtester.strategy.sma_lookback = request.sma_lookback
            if hasattr(request, 'stop_loss_pct') and request.stop_loss_pct:
                backtester.strategy.stop_loss_pct = request.stop_loss_pct
            if hasattr(request, 'profit_threshold_pct') and request.profit_threshold_pct:
                backtester.strategy.profit_threshold_pct = request.profit_threshold_pct
            if hasattr(request, 'number_of_slots') and request.number_of_slots:
                backtester.strategy.num_slots = request.number_of_slots
                # Re-initialize slots if changed
                backtester.strategy.slots = [{"id": i, "status": "FREE", "data": {}} for i in range(backtester.strategy.num_slots)]

            initial_capital = request.initial_capital or request.total_capital
            
            # Run backtest
            result = backtester.run_backtest(
                tickers=tickers,
                start_date=request.start_date,
                end_date=request.end_date,
                initial_capital=initial_capital
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
                "nav": [p['nav'] for p in performance_data],
                "cash": [p['cash'] for p in performance_data]
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
                error=f"ETF Swing backtest failed: {str(e)}"
            )

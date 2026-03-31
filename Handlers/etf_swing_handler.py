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
            
<<<<<<< HEAD
            # Derive market from strategy_type as the primary source of truth.
            # We CANNOT rely on request.market because the Pydantic schema defines it with
            # default="INDIA", so getattr always returns "INDIA" even for US_ETF_Swing_Strategy.
            if request.strategy_type == 'US_ETF_Swing_Strategy':
                market = 'US'
            else:
                market = 'INDIA'
            asset_type = 'ETF'

            # Allow explicit override only from the parameters dict (deliberate frontend field)
            if hasattr(request, 'parameters') and isinstance(request.parameters, dict):
                market = request.parameters.get('market', market).upper()
                asset_type = request.parameters.get('asset_type', asset_type).upper()

            # Initialize backtester with context
            backtester = ETFSwingBacktester(market=market, asset_type=asset_type)
            
            # Update strategy parameters from request if provided
            # Update strategy parameters from request if provided
            params_to_update = {}
            if hasattr(request, 'sma_lookback') and request.sma_lookback:
                params_to_update['sma_lookback'] = request.sma_lookback
            if hasattr(request, 'stop_loss_pct') and request.stop_loss_pct:
                params_to_update['stop_loss_pct'] = request.stop_loss_pct
            if hasattr(request, 'profit_threshold_pct') and request.profit_threshold_pct:
                params_to_update['profit_threshold_pct'] = request.profit_threshold_pct
            if hasattr(request, 'number_of_slots') and request.number_of_slots:
                params_to_update['number_of_slots'] = request.number_of_slots
            if hasattr(request, 'brokerage_percent') and request.brokerage_percent is not None:
                params_to_update['brokerage_percent'] = request.brokerage_percent
            
            if params_to_update:
                backtester.strategy.update_config(params_to_update)
=======
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
>>>>>>> feature/chatai

            initial_capital = request.initial_capital or request.total_capital
            
            # Run backtest
            result = backtester.run_backtest(
                tickers=tickers,
                start_date=request.start_date,
                end_date=request.end_date,
<<<<<<< HEAD
                initial_capital=initial_capital,
                risk_free_rate=request.risk_free_rate or 8.0
=======
                initial_capital=initial_capital
>>>>>>> feature/chatai
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
<<<<<<< HEAD
                "strategy": [p['strategy'] for p in performance_data],
                "benchmark_buyhold": [p['benchmark_buyhold'] for p in performance_data],
                "cumulative_investment": [p['cumulative_investment'] for p in performance_data]
=======
                "nav": [p['nav'] for p in performance_data],
                "cash": [p['cash'] for p in performance_data]
>>>>>>> feature/chatai
            }
            
            # Sanitize data
            metrics = self._sanitize_data(metrics)
            chart_data = self._sanitize_data(chart_data)
            transaction_log = self._sanitize_data(transaction_log)
            
<<<<<<< HEAD
            # Get cost breakdown
            cost_breakdown = backtester.get_transaction_costs_summary()
            cost_breakdown = self._sanitize_data(cost_breakdown)
            
            # Cache backtest results for other endpoints
            try:
                from APIs.common.cache import cache_backtest_results
                cache_backtest_results(request.strategy_type, backtester)
            except Exception as e:
                print(f"Error caching {request.strategy_type}: {e}")

=======
>>>>>>> feature/chatai
            return UnifiedBacktestResponse(
                success=True,
                strategy_type=request.strategy_type,
                metrics=metrics,
                performance_data=chart_data,
<<<<<<< HEAD
                transaction_log=transaction_log,
                cost_breakdown=cost_breakdown
=======
                transaction_log=transaction_log
>>>>>>> feature/chatai
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

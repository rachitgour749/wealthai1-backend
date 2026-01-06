"""
RS ETF Strategy Handler
Handles backtest execution for RS ETF Rotation strategy
"""
import sys
import os
from datetime import datetime
from sqlalchemy.orm import Session

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Strategies'))

from Handlers.base_handler import BaseStrategyHandler
from APIs.unified_schemas import UnifiedBacktestRequest, UnifiedBacktestResponse
from Strategies.RS_ETF.rs_etf_backtester_core import RSETFStrategyBacktester


class RSETFHandler(BaseStrategyHandler):
    """Handler for RS ETF Rotation strategy"""
    
    def validate_request(self, request: UnifiedBacktestRequest) -> None:
        """Validate RS ETF specific parameters"""
        if not request.total_capital:
            raise ValueError("RS_ETF_Rotation requires 'total_capital' parameter")
    
    async def run_backtest(self, request: UnifiedBacktestRequest) -> UnifiedBacktestResponse:
        """Run RS ETF backtest"""
        try:
            self.validate_request(request)
            
            # Create config dict
            config_dict = {
                'main_index': request.main_index or "^NSEI",
                'etf_universe': request.etf_universe or "ALL_ETFS",
                'custom_etfs': request.custom_etfs,
                'max_positions': request.max_positions or 20,
                'position_size_pct': request.position_size_pct,
                'total_capital': request.total_capital,
                'stop_loss_pct': request.stop_loss_pct or 15.0,
                'buffer_capital_pct': request.buffer_capital_pct or 10.0,
                'capital_reset_threshold_pct': request.capital_reset_threshold_pct or 25.0,
                'max_holding_period': request.max_holding_period or 52,
                'transaction_cost_pct': request.transaction_cost_pct or 0.1,
                'min_price': request.min_price or 10.0,
                'min_turnover': request.min_turnover or 1000000.0,
                'lookback_weeks': request.lookback_weeks or 5,
                'lookback_months': request.lookback_months or 20,
                'lookback_quarters': request.lookback_quarters or 60
            }
            
            # Initialize backtester
            backtester = RSETFStrategyBacktester.from_config_dict(self.db, config_dict)
            
            # Convert dates
            start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00')).replace(tzinfo=None)
            end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00')).replace(tzinfo=None)
            
            # Run backtest
            backtester.run_backtest(start_date, end_date)
            results = backtester.calculate_metrics(risk_free_rate=request.risk_free_rate or 8.0)
            
            # Prepare performance data
            portfolio_snapshots = results.get('portfolio_snapshots', [])
            performance_data = {
                "dates": [],
                "rs_strategy": [],
                "cumulative_investment": [],
                "benchmark_buyhold": []
            }
            
            if portfolio_snapshots:
                performance_data["dates"] = [
                    snapshot.get('date') if isinstance(snapshot.get('date'), str) 
                    else snapshot.get('date').isoformat() if hasattr(snapshot.get('date'), 'isoformat')
                    else str(snapshot.get('date'))
                    for snapshot in portfolio_snapshots
                ]
                performance_data["rs_strategy"] = [snapshot.get('total_value', 0) for snapshot in portfolio_snapshots]
                performance_data["cumulative_investment"] = [request.total_capital] * len(portfolio_snapshots)
                performance_data["benchmark_buyhold"] = results.get('benchmark_buyhold', [])
            
            # Prepare metrics
            metrics = {
                'total_return': results.get('total_return_pct', 0),
                'annualized_return_pct': results.get('annualized_return_pct', 0),
                'cagr_pct': results.get('cagr_pct', 0),
                'xirr_pct': results.get('xirr_pct', 0),
                'max_drawdown': results.get('max_drawdown_pct', 0),
                'sharpe_ratio': results.get('sharpe_ratio', 0),
                'beta': results.get('beta', 1.0),
                'treynor_ratio': results.get('treynor_ratio', 0.0),
                'calmar_ratio': results.get('calmar_ratio', 0.0),
                'win_rate_pct': results.get('win_rate_pct', 0),
                'total_trades': results.get('total_trades', 0),
                'final_capital': results.get('final_capital', 0),
                'benchmark_metrics': results.get('benchmark_metrics', {}),
                'alpha_pct': results.get('alpha_pct', 0)
            }
            
            # Sanitize data
            metrics = self._sanitize_data(metrics)
            performance_data = self._sanitize_data(performance_data)
            transaction_log = self._sanitize_data(results.get('trades', []))
            
            # Cache backtest results
            try:
                from APIs.centralized_backtest import cache_backtest_results
                cache_backtest_results("RS_ETF_Rotation", backtester)
            except Exception as e:
                print(f"[RS_ETF_HANDLER] Warning: Could not cache backtest results: {e}")
            
            return UnifiedBacktestResponse(
                success=True,
                strategy_type=request.strategy_type,
                metrics=metrics,
                performance_data=performance_data,
                transaction_log=transaction_log,
                portfolio_snapshots=self._sanitize_data(portfolio_snapshots)
            )
            
        except Exception as e:
            return UnifiedBacktestResponse(
                success=False,
                strategy_type=request.strategy_type,
                metrics={},
                error=f"RS ETF backtest failed: {str(e)}"
            )

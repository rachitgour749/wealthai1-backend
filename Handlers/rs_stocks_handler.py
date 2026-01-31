"""
RS Stocks Strategy Handler
Handles backtest execution for RS Stocks strategy
"""
import sys
import os
from datetime import datetime
from sqlalchemy.orm import Session

# Add Strategies path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Strategies'))

from Handlers.base_handler import BaseStrategyHandler
from APIs.unified_schemas import UnifiedBacktestRequest, UnifiedBacktestResponse
from Strategies.RS_Stocks.rs_backtester_core import RSStrategyBacktester


class RSStocksHandler(BaseStrategyHandler):
    """Handler for RS Stocks strategy"""
    
    def validate_request(self, request: UnifiedBacktestRequest) -> None:
        """Validate RS Stocks specific parameters"""
        if not request.total_capital:
            raise ValueError("RS_Stocks requires 'total_capital' parameter")
    
    async def run_backtest(self, request: UnifiedBacktestRequest) -> UnifiedBacktestResponse:
        """Run RS Stocks backtest"""
        db_created = False
        db_session = self.db
        
        try:
            self.validate_request(request)
            
            # Ensure we have a database session
            if db_session is None:
                from Strategies.RS_Stocks.database import get_db
                db_session = next(get_db())
                db_created = True
            
            # Create config dict
            from Strategies.RS.rs_config_loader import get_rs_config
            rs_global_config = get_rs_config()
            
            # Prepare config
            # Priority: API Parameter (stop_loss_pct or stop_loss) > Global Config Default
            sl_val = request.stop_loss_pct if request.stop_loss_pct is not None else getattr(request, 'stop_loss', None)
            if sl_val is None:
                sl_val = rs_global_config.get_stop_loss_pct()
            
            config_dict = {
                'main_index': self._clean_tickers(request.main_index) or "^NSEI",
                'stock_universe': request.stock_universe or "NIFTY_500",
                'custom_stocks': self._clean_tickers(request.custom_stocks),
                'max_positions': request.max_positions or 20,
                'position_size_pct': request.position_size_pct,
                'total_capital': request.total_capital,
                'stop_loss_pct': sl_val,
                'daily_stop_loss_check': getattr(request, 'daily_stop_loss_check', rs_global_config.get_daily_stop_loss_check()),
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
            backtester = RSStrategyBacktester.from_config_dict(db_session, config_dict)
            
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
                performance_data["strategy"] = [snapshot.get('total_value', 0) for snapshot in portfolio_snapshots]
                performance_data["cumulative_investment"] = [request.total_capital] * len(portfolio_snapshots)
                performance_data["benchmark_buyhold"] = results.get('benchmark_buyhold', [])
            
            # Prepare metrics
            # Prepare metrics
            metrics = {
                'total_investment': request.total_capital,
                'final_capital': results.get('final_capital', 0),
                'total_return_pct': results.get('total_return_pct', 0),
                'cagr': results.get('cagr_pct', 0),
                'xirr': results.get('xirr_pct', 0),
                'max_drawdown_pct': results.get('max_drawdown_pct', 0),
                'sharpe_ratio': results.get('sharpe_ratio', 0),
                'calmar_ratio': results.get('calmar_ratio', 0.0),
                'volatility': results.get('volatility', 0), # Added if available, else 0
                'beta': results.get('beta', 1.0),
                'treynor_ratio': results.get('treynor_ratio', 0.0),
                'win_rate_pct': results.get('win_rate_pct', 0),
                'total_trades': results.get('total_trades', 0),
                'alpha_pct': results.get('alpha_pct', 0),
                
                'annualized_return_pct': results.get('annualized_return_pct', 0), # Keep for backward compatibility if needed, but not part of standard set
                
                'benchmark_metrics': {
                    "total_investment": results.get('benchmark_metrics', {}).get('total_investment', 0),
                    "final_capital": results.get('benchmark_metrics', {}).get('final_capital', 0),
                    "total_return_pct": results.get('benchmark_metrics', {}).get('total_return_pct', 0),
                    "cagr": results.get('benchmark_metrics', {}).get('cagr', 0),
                    "xirr": results.get('benchmark_metrics', {}).get('xirr', 0),
                    "sharpe_ratio": results.get('benchmark_metrics', {}).get('sharpe_ratio', 0),
                    "calmar_ratio": results.get('benchmark_metrics', {}).get('calmar_ratio', 0),
                    "max_drawdown_pct": results.get('benchmark_metrics', {}).get('max_drawdown_pct', 0),
                    "volatility": results.get('benchmark_metrics', {}).get('volatility', 0)
                }
            }
            
            # Sanitize data
            metrics = self._sanitize_data(metrics)
            performance_data = self._sanitize_data(performance_data)
            # Prepare cost breakdown from trade log
            transaction_log = results.get('trades', [])
            
            # Helper to calculate costs for a list of trades
            def calculate_costs_for_trades(trades_list):
                 if not trades_list:
                     return {
                        'transaction_costs': 0.0,
                        'capital_gains_tax': 0.0,
                        'total_costs': 0.0,
                        'total_brokerage': 0.0,
                        'transactions': 0
                     }
                 
                 # Convert to DF for easy summation if needed, or simple sum
                 total_costs = sum(t.get('total_costs', 0) for t in trades_list)
                 capital_gains_tax = sum(t.get('capital_gains_tax', 0) for t in trades_list)
                 brokerage = sum(t.get('brokerage', 0) for t in trades_list)
                 transaction_costs = total_costs - capital_gains_tax # Derived if total includes tax
                 
                 # Logic for transactions count (Unique Dates to match Rotation's 'Active Periods')
                 dates = set(t.get('date') for t in trades_list if t.get('date'))
                 transactions = len(dates)
                 
                 return {
                    'transaction_costs': round(transaction_costs, 2),
                    'capital_gains_tax': round(capital_gains_tax, 2),
                    'total_costs': round(total_costs, 2),
                    'total_brokerage': round(brokerage, 2),
                    'transactions': transactions
                 }

            cost_breakdown = {}
            # Yearly breakdown
            import pandas as pd
            if transaction_log:
                # Convert to DF for easy grouping
                try:
                    df_trades = pd.DataFrame(transaction_log)
                    if 'date' in df_trades.columns:
                       df_trades['year'] = pd.to_datetime(df_trades['date']).dt.year.astype(str)
                       for year, group in df_trades.groupby('year'):
                           cost_breakdown[year] = calculate_costs_for_trades(group.to_dict('records'))
                       
                       # Add Total
                       cost_breakdown['Total'] = calculate_costs_for_trades(transaction_log)
                except ImportError:
                    pass 
                except Exception as e:
                    print(f"Error grouping costs by year: {e}")

            if not cost_breakdown:
                 cost_breakdown['Total'] = calculate_costs_for_trades([])

            # Sanitize data
            metrics = self._sanitize_data(metrics)
            performance_data = self._sanitize_data(performance_data)
            transaction_log = self._sanitize_data(transaction_log)
            
            # Cache backtest results
            try:
                from APIs.common.cache import cache_backtest_results
                cache_backtest_results("RS_Stocks", backtester)
            except Exception as e:
                print(f"[RS_STOCKS_HANDLER] Warning: Could not cache backtest results: {e}")
            
            return UnifiedBacktestResponse(
                success=True,
                strategy_type=request.strategy_type,
                metrics=metrics,
                performance_data=performance_data,
                transaction_log=transaction_log,
                cost_breakdown=self._sanitize_data(cost_breakdown),
                portfolio_snapshots=self._sanitize_data(portfolio_snapshots)
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return UnifiedBacktestResponse(
                success=False, 
                error=str(e), 
                strategy_type="RS_Stocks",
                metrics={}
            )
        finally:
            if db_created and db_session:
                db_session.close()

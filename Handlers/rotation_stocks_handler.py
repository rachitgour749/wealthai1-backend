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
            raise ValueError("Stock_Rotation requires 'tickers' parameter")
        if not request.capital_per_week:
            raise ValueError("Stock_Rotation requires 'capital_per_week' parameter")
        if not request.accumulation_weeks:
            raise ValueError("Stock_Rotation requires 'accumulation_weeks' parameter")
        if request.brokerage_percent is None:
            raise ValueError("Stock_Rotation requires 'brokerage_percent' parameter")
    
    async def run_backtest(self, request: UnifiedBacktestRequest) -> UnifiedBacktestResponse:
        """Run Rotation Stocks backtest"""
        try:
            self.validate_request(request)
            
            # Extract market
            market = getattr(request, 'market', 'INDIA') or 'INDIA'
            if hasattr(request, 'parameters') and isinstance(request.parameters, dict):
                market = request.parameters.get('market', market)
            
            # Clean tickers (remove .NS)
            request.tickers = self._clean_tickers(request.tickers)
            
            # Initialize backtester with correct market context
            backtester = StockRotationBacktester(market=market, db_path="unified_stock_data.sqlite")
            
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
                performance_data["strategy"] = backtester.weekly_nav_df['nav'].tolist()
                performance_data["cumulative_investment"] = backtester.weekly_nav_df['cumulative_investment'].tolist()
            
            # Calculate benchmark metrics and data
            total_investment = request.accumulation_weeks * request.capital_per_week
            
            # Calculate and store benchmark data
            nifty_df = backtester.calculate_benchmark_buyhold(
                request.start_date, 
                request.end_date, 
                total_investment, 
                request.brokerage_percent
            )
            backtester.nifty50_df = nifty_df
            
            # Populate benchmark chart data aligned with dates
            if not backtester.weekly_nav_df.empty and not nifty_df.empty:
                # Create a mapping of date strings to values
                # Ensure date format matches key in performance_data["dates"]
                # weekly_nav_df['date'] are datetime objects, converted to str in performance_data
                
                # Resample or reindex nifty_df to match strategy dates
                strategy_dates = backtester.weekly_nav_df['date'].tolist()
                benchmark_values = []
                
                # Create lookup for O(1) access
                # Explicitly handle string conversion for matching
                nifty_lookup = {d.strftime('%Y-%m-%d'): nav for d, nav in zip(nifty_df['date'], nifty_df['nav'])}
                
                for date_obj in strategy_dates:
                    date_str = date_obj.strftime('%Y-%m-%d')
                    # Find closest previous date if exact match not found (forward fill logic)
                    if date_str in nifty_lookup:
                        benchmark_values.append(nifty_lookup[date_str])
                    else:
                        # Fallback: get nearest available date from nifty_df
                        # But simplest is to reuse last known value
                         benchmark_values.append(benchmark_values[-1] if benchmark_values else total_investment)

                performance_data["benchmark_buyhold"] = benchmark_values

            benchmark_metrics = backtester.calculate_benchmark_metrics(
                total_investment,
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
                "total_investment": parse_metric(metrics.get('Total Investment', 0)),
                "final_capital": parse_metric(metrics.get('Final Value', 0)),
                "total_return_pct": parse_metric(metrics.get('Total Return', 0)),
                "cagr": parse_metric(metrics.get('CAGR', 0)),
                "xirr": parse_metric(metrics.get('XIRR', 0)),
                "sharpe_ratio": parse_metric(metrics.get('Sharpe Ratio', 0)),
                "calmar_ratio": parse_metric(metrics.get('Calmar Ratio', 0)),
                "max_drawdown_pct": parse_metric(metrics.get('Max Drawdown', 0)),
                "volatility": parse_metric(metrics.get('Volatility', 0)),
                "total_trades": metrics.get('Total Trades', 0),
                "win_rate_pct": parse_metric(metrics.get('Win Rate', 0)),
                "beta": parse_metric(metrics.get('Beta', 1.0)),
                "treynor_ratio": parse_metric(metrics.get('Treynor Ratio', 0)),
            }

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

            # Combined final metrics
            metrics = {
                **flat_metrics,
                "benchmark_metrics": flat_benchmark_metrics
            }
            
            # Get transaction log and cost breakdown
            # Use full portfolio log with all details
            transaction_log = backtester.portfolio_log
            cost_breakdown = backtester.get_transaction_costs_summary()
            
            # Sanitize data
            metrics = self._sanitize_data(metrics)
            performance_data = self._sanitize_data(performance_data)
            transaction_log = self._sanitize_data(transaction_log)
            # cost_breakdown contains numpy types from pandas sum(), must be sanitized
            cost_breakdown = self._sanitize_data(cost_breakdown)
            
            # Cache backtest results
            try:
                from APIs.common.cache import cache_backtest_results
                cache_backtest_results("Stock_Rotation", backtester)
            except Exception as e:
                print(f"Warning: Could not cache backtest results: {e}")
            
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
                error=f"Rotation Stocks backtest failed: {str(e)}"
            )

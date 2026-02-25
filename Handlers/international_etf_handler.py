"""
International ETF Strategy Handler
Handles backtest execution for International ETF Rotation strategy
"""
import sys
import os
from sqlalchemy.orm import Session

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Strategies'))

from Handlers.base_handler import BaseStrategyHandler
from APIs.unified_schemas import UnifiedBacktestRequest, UnifiedBacktestResponse
from Strategies.Rotation_International_ETF.services.backtester import InternationalETFRotationBacktester


class InternationalETFHandler(BaseStrategyHandler):
    """Handler for International ETF Rotation strategy"""
    
    def validate_request(self, request: UnifiedBacktestRequest) -> None:
        """Validate International ETF specific parameters"""
        if not request.tickers:
            raise ValueError("International_ETF_Rotation requires 'tickers' parameter")
        if not request.capital_per_week:
            raise ValueError("International_ETF_Rotation requires 'capital_per_week' parameter")
        if not request.accumulation_weeks:
            raise ValueError("International_ETF_Rotation requires 'accumulation_weeks' parameter")
        if request.brokerage_percent is None:
            raise ValueError("International_ETF_Rotation requires 'brokerage_percent' parameter")
    
    async def run_backtest(self, request: UnifiedBacktestRequest) -> UnifiedBacktestResponse:
        """Run International ETF backtest"""
        try:
            self.validate_request(request)
            
            # Clean tickers (remove .NS)
            request.tickers = self._clean_tickers(request.tickers)
            
            # Extract market
            market = getattr(request, 'market', 'US') or 'US'
            if hasattr(request, 'parameters') and isinstance(request.parameters, dict):
                market = request.parameters.get('market', market)
            
            # Initialize backtester
            backtester = InternationalETFRotationBacktester(market=market, db_path="international_etf_data.sqlite")
            
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
            
            # Calculate metrics (Raw values)
            etf_metrics = backtester.calculate_metrics(
                request.capital_per_week,
                request.accumulation_weeks,
                request.risk_free_rate or 8.0
            )
            
            # Helper to parse string values back to float
            def parse_metric(val):
                if isinstance(val, (int, float)): return val
                if isinstance(val, str):
                    clean = val.replace('₹', '').replace('$', '').replace(',', '').replace('%', '')
                    try:
                        return float(clean)
                    except ValueError:
                        return val
                return val

            # Standardize Strategy Metrics
            flat_metrics = {
                "total_investment": parse_metric(etf_metrics.get('Total Investment', 0)),
                "final_capital": parse_metric(etf_metrics.get('Final Value', 0)),
                "total_return_pct": parse_metric(etf_metrics.get('Total Return', 0)),
                "cagr": parse_metric(etf_metrics.get('CAGR', 0)),
                "xirr": parse_metric(etf_metrics.get('XIRR', 0)),
                "sharpe_ratio": parse_metric(etf_metrics.get('Sharpe Ratio', 0)),
                "calmar_ratio": parse_metric(etf_metrics.get('Calmar Ratio', 0)),
                "max_drawdown_pct": parse_metric(etf_metrics.get('Max Drawdown', 0)),
                "volatility": parse_metric(etf_metrics.get('Volatility', 0)),
                "total_trades": etf_metrics.get('Total Trades', 0),
                "win_rate_pct": parse_metric(etf_metrics.get('Win Rate', 0))
            }

            # Calculate benchmark metrics
            total_investment = request.accumulation_weeks * request.capital_per_week
            backtester.sp500_df = backtester.calculate_benchmark_buyhold(
                start_date=request.start_date,
                end_date=request.end_date,
                total_investment=total_investment,
                brokerage_percent=request.brokerage_percent
            )
            
            benchmark_metrics = backtester.calculate_benchmark_metrics(
                total_investment,
                request.risk_free_rate or 8.0
            )

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
            
            # Prepare performance data
            performance_data = {
                "dates": [],
                "etf_strategy": [],
                "cumulative_investment": [],
                "benchmark_buyhold": []
            }
            
            if not backtester.weekly_nav_df.empty:
                performance_data["dates"] = [str(date) for date in backtester.weekly_nav_df['date']]
                performance_data["strategy"] = backtester.weekly_nav_df['nav'].tolist()
                performance_data["cumulative_investment"] = backtester.weekly_nav_df['cumulative_investment'].tolist()
                
                if not backtester.sp500_df.empty:
                    performance_data["benchmark_buyhold"] = backtester.sp500_df['nav'].tolist()
            
            # Prepare transaction log
            # Use full portfolio log with all details
            transaction_log = backtester.portfolio_log
            
            # Get cost breakdown
            cost_breakdown = backtester.get_transaction_costs_summary()
            
            # Combined final metrics
            metrics = {
                **flat_metrics,
                "benchmark_metrics": flat_benchmark_metrics
            }

            # Sanitize all data
            metrics = self._sanitize_data(metrics)
            performance_data = self._sanitize_data(performance_data)
            transaction_log = self._sanitize_data(transaction_log)
            # cost_breakdown must be sanitized for numpy safety
            cost_breakdown = self._sanitize_data(cost_breakdown)
            
            # Cache backtest results
            try:
                from APIs.common.cache import cache_backtest_results
                cache_backtest_results("International_ETF_Rotation", backtester)
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
            import traceback
            traceback.print_exc()
            return UnifiedBacktestResponse(
                success=False,
                strategy_type=request.strategy_type,
                metrics={},
                error=f"International ETF backtest failed: {str(e)}"
            )

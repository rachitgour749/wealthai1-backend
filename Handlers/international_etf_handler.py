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
            
            # Initialize backtester
            backtester = InternationalETFRotationBacktester(db_path="unified_etf_data.sqlite")
            
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
            etf_metrics = backtester.calculate_metrics(
                request.capital_per_week,
                request.accumulation_weeks,
                request.risk_free_rate or 8.0
            )
            
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
            
            # Prepare performance data
            performance_data = {
                "dates": [],
                "etf_strategy": [],
                "cumulative_investment": [],
                "benchmark_buyhold": []
            }
            
            if not backtester.weekly_nav_df.empty:
                performance_data["dates"] = [str(date) for date in backtester.weekly_nav_df['date']]
                performance_data["etf_strategy"] = backtester.weekly_nav_df['nav'].tolist()
                performance_data["cumulative_investment"] = backtester.weekly_nav_df['cumulative_investment'].tolist()
                
                if not backtester.sp500_df.empty:
                    performance_data["benchmark_buyhold"] = backtester.sp500_df['nav'].tolist()
            
            # Prepare transaction log
            transaction_log = []
            for log in backtester.portfolio_log:
                costs = log.get('costs', {})
                transaction_costs = costs.get('total_costs', 0) if costs else 0
                
                transaction_log.append({
                    'week': log.get('week', 0),
                    'date': log.get('execution_date', '').strftime('%Y-%m-%d') if hasattr(log.get('execution_date', ''), 'strftime') else str(log.get('execution_date', '')),
                    'action': log.get('action', 'NONE'),
                    'ticker': log.get('ticker', ''),
                    'units': log.get('units', 0),
                    'price': log.get('price', 0),
                    'amount': log.get('amount', 0),
                    'transaction_costs': transaction_costs,
                    'capital_gains_tax': log.get('capital_gains_tax', 0),
                    'nav': log.get('nav', 0)
                })
            
            # Normalize benchmark metrics
            benchmark_metrics = self._normalize_benchmark_metrics(benchmark_metrics)
            
            # Sanitize all data
            metrics = self._sanitize_data({
                **etf_metrics,
                "benchmark_metrics": benchmark_metrics
            })
            performance_data = self._sanitize_data(performance_data)
            transaction_log = self._sanitize_data(transaction_log)
            
            # Cache backtest results
            try:
                from APIs.centralized_backtest import cache_backtest_results
                cache_backtest_results("International_ETF_Rotation", backtester)
            except Exception as e:
                print(f"Warning: Could not cache backtest results: {e}")
            
            return UnifiedBacktestResponse(
                success=True,
                strategy_type=request.strategy_type,
                metrics=metrics,
                performance_data=performance_data,
                transaction_log=transaction_log
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

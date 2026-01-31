import os
import sys
from fastapi import HTTPException
from APIs.unified_schemas import UnifiedBacktestRequest, UnifiedBacktestResponse
from Strategies.ETF_Buy_on_Dip.services.backtester import ETFBuyOnDipBacktester
from APIs.common.cache import cache_backtest_results
from Handlers.base_handler import BaseStrategyHandler
import traceback

class ETFBuyOnDipHandler(BaseStrategyHandler):
    def __init__(self, db):
        super().__init__(db)
        self.backtester = ETFBuyOnDipBacktester()

    def validate_request(self, request: UnifiedBacktestRequest) -> None:
        """Validate ETF Buy-on-Dip specific parameters"""
        if not request.tickers:
            raise ValueError("ETF_Buy_on_Dip requires 'tickers' parameter")
        if not request.capital_per_month and not request.capital_per_week:
            # Payout/Dip usually uses monthly, but schema might have both
            raise ValueError("ETF_Buy_on_Dip requires 'capital_per_month' parameter")
    async def run_backtest(self, request: UnifiedBacktestRequest) -> UnifiedBacktestResponse:
        try:
            # Validate request
            self.validate_request(request)

            # Clean tickers
            request.tickers = self._clean_tickers(request.tickers)

            # Extract parameters
            # Use capital_per_month explicitly
            capital = request.capital_per_month or request.capital_per_week or 5000.0
            
            if not request.tickers:
                raise HTTPException(status_code=400, detail="Tickers are required for ETF Buy-on-Dip strategy")

            # Run Backtest
            results = self.backtester.run_backtest(
                tickers=request.tickers,
                start_date=request.start_date,
                end_date=request.end_date,
                capital_per_month=capital,
                brokerage_percent=request.brokerage_percent or 0.0
            )
            
            if results.get("error"):
                raise HTTPException(status_code=400, detail=results["error"])

            # Cache results
            try:
                cache_backtest_results("ETF_Buy_on_Dip", self.backtester)
            except Exception as e:
                print(f"[ETF_BUY_ON_DIP_HANDLER] ERROR caching: {e}")

            # Format Response
            metrics = results.get("metrics", {})
            
            # Prepare transaction log and cost breakdown
            transaction_log = results.get("transaction_log", [])
            
            # Helper to calculate costs for a list of trades
            # Helper to calculate costs for a list of trades
            def calculate_costs_for_trades(trades_list):
                 transaction_costs = sum(t.get('transaction_costs', 0) for t in trades_list)
                 capital_gains_tax = 0.0
                 transactions = len(trades_list)
                 
                 return {
                    'transaction_costs': round(transaction_costs, 2),
                    'capital_gains_tax': round(capital_gains_tax, 2),
                    'total_costs': round(transaction_costs + capital_gains_tax, 2),
                    'total_brokerage': 0.0, # Not detailed in simple Buy on Dip
                    'transactions': transactions
                 }

            cost_breakdown = {}
            # Yearly breakdown
            import pandas as pd
            if transaction_log:
                try:
                    df_trades = pd.DataFrame(transaction_log)
                    if 'date' in df_trades.columns:
                       df_trades['year'] = pd.to_datetime(df_trades['date']).dt.year.astype(str)
                       for year, group in df_trades.groupby('year'):
                           cost_breakdown[year] = calculate_costs_for_trades(group.to_dict('records'))
                except Exception as e:
                    print(f"Error grouping costs by year: {e}")

            if not cost_breakdown and transaction_log:
                 cost_breakdown['Total'] = calculate_costs_for_trades(transaction_log)
            
            # Standardize Performance Data
            snapshots = results.get("portfolio_snapshots", [])
            performance_data = {
                "dates": [s['date'] for s in snapshots],
                "strategy": [s['current_value'] for s in snapshots],
                "cumulative_investment": [s['total_invested'] for s in snapshots],
                "benchmark_buyhold": results.get("benchmark_data", [])
            }

            # Sanitize all data for JSON serialization
            sanitized_metrics = self._sanitize_data({
                "total_invested": metrics.get("total_invested", 0),
                "final_value": metrics.get("final_value", 0),
                "absolute_return": metrics.get("absolute_return", 0),
                "absolute_return_pct": metrics.get("absolute_return_pct", 0),
                "xirr": metrics.get("xirr", 0),
                "benchmark_metrics": results.get("benchmark_metrics", {}),
                "cagr": metrics.get("cagr", 0.0),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0.0)
            })
            performance_data = self._sanitize_data(performance_data)
            transaction_log = self._sanitize_data(transaction_log)
            cost_breakdown = self._sanitize_data(cost_breakdown)

            return UnifiedBacktestResponse(
                success=True,
                strategy_type="ETF_Buy_on_Dip",
                metrics=sanitized_metrics,
                performance_data=performance_data,
                transaction_log=transaction_log,
                cost_breakdown=cost_breakdown
            )

        except HTTPException as he:
            raise he
        except Exception as e:
            print(f"Error in ETF Buy-on-Dip handler: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

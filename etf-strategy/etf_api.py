from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import sys
import os
import sqlite3
import json
from datetime import datetime

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import the ETF backtester
from backtester_core import ETFRotationBacktester

# Create ETF router
etf_router = APIRouter(prefix="/api", tags=["ETF Strategy"])

# Pydantic models for request/response
class BacktestRequest(BaseModel):
    tickers: List[str]
    start_date: str
    end_date: str
    capital_per_week: float
    accumulation_weeks: int
    brokerage_percent: float
    compounding_enabled: bool = False
    risk_free_rate: float = 8.0

class ETFMetadata(BaseModel):
    ticker: str
    name: str
    category: str
    expense_ratio: float
    aum: float

class BacktestResult(BaseModel):
    success: bool
    total_investment: float
    final_portfolio_value: float
    total_return: float
    total_brokerage: float
    error: Optional[str] = None

class BacktestResults(BaseModel):
    total_return: Optional[str] = None
    cagr: Optional[str] = None
    sharpe_ratio: Optional[str] = None
    max_drawdown: Optional[str] = None

class SaveETFStrategyRequest(BaseModel):
    strategy_name: str
    strategy_type: str
    user_id: str
    tickers: List[str]
    start_date: str
    end_date: str
    capital_per_week: float
    accumulation_weeks: int
    brokerage_percent: float
    compounding_enabled: bool
    risk_free_rate: float
    use_custom_dates: bool
    backtest_results: BacktestResults
    created_at: str

class SavedETFStrategy(BaseModel):
    id: int
    strategy_name: str
    strategy_type: str
    user_id: str
    tickers: List[str]
    start_date: str
    end_date: str
    capital_per_week: float
    accumulation_weeks: int
    brokerage_percent: float
    compounding_enabled: bool
    risk_free_rate: float
    use_custom_dates: bool
    backtest_results: Dict[str, Any]
    created_at: str

# Global ETF backtester instance
etf_backtester = None

def initialize_etf_backtester(db_path: str = "unified_etf_data.sqlite"):
    """Initialize the ETF backtester"""
    global etf_backtester
    try:
        etf_backtester = ETFRotationBacktester(db_path=db_path)
        print("✅ ETF Backtester initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing ETF Backtester: {e}")
        etf_backtester = None
        return False

def cleanup_etf_backtester():
    """Clean up ETF backtester resources"""
    global etf_backtester
    if etf_backtester:
        etf_backtester.cleanup()
        etf_backtester = None

# ============================================================================
# ETF STRATEGY ROUTES
# ============================================================================

@etf_router.get("/etfs")
async def get_available_etfs():
    """Get list of available ETFs"""
    try:
        if etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        # Load ETF metadata
        metadata = etf_backtester.load_etf_metadata()
        etfs = []
        
        for ticker, data in metadata.items():
            etfs.append({
                "ticker": ticker,
                "name": data.get('name', ticker),
                "category": data.get('category', 'Unknown'),
                "expense_ratio": data.get('expense_ratio', 0.0),
                "aum": data.get('aum', 0.0)
            })
        
        return {"etfs": etfs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading ETFs: {str(e)}")

@etf_router.get("/default")
async def get_default_etf_selection():
    """Get default ETF selection"""
    try:
        if etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        metadata = etf_backtester.load_etf_metadata()
        available_etfs = list(metadata.keys())
        default_selection = etf_backtester.get_default_etf_selection(available_etfs, 5)
        
        return {"default_etfs": default_selection}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting default selection: {str(e)}")

@etf_router.post("/etfs/date-range")
async def calculate_etf_date_range(request: Dict[str, Any]):
    """Calculate common date range for selected ETFs"""
    try:
        if etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        tickers = request.get("tickers", [])
        print(f"Calculating date range for ETF tickers: {tickers}")
        start_date, end_date, years = etf_backtester.calculate_common_date_range(tickers)
        
        if start_date and end_date:
            return {
                "start_date": start_date,
                "end_date": end_date,
                "years": years
            }
        else:
            raise HTTPException(status_code=400, detail="Could not calculate date range. Please try different ETFs.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating date range: {str(e)}")

@etf_router.post("/diagnose")
async def diagnose_etf_data(request: Dict[str, Any]):
    """Diagnose ETF data availability and provide recommendations"""
    try:
        if etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        tickers = request.get("tickers", [])
        diagnosis = etf_backtester.diagnose_etf_data(tickers)
        return diagnosis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error diagnosing ETF data: {str(e)}")

@etf_router.get("/etfs/overview")
async def get_etf_overview():
    """Get ETF overview with descriptions"""
    try:
        if etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        metadata = etf_backtester.load_etf_metadata()
        etf_overview = []
        
        for symbol, meta in metadata.items():
            description = etf_backtester.generate_etf_description(symbol)
            sector = etf_backtester.get_etf_sector_classification(symbol)
            etf_overview.append({
                'symbol': symbol,
                'description': description,
                'sector': sector,
                'start_date': meta['start_date'],
                'end_date': meta['end_date'],
                'years_available': round(meta['years_available'], 1),
                'total_records': meta['total_records']
            })
        
        # Sort by start date
        etf_overview.sort(key=lambda x: x['start_date'])
        return {"etf_overview": etf_overview}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting ETF overview: {str(e)}")

@etf_router.post("/metrics")
async def calculate_etf_metrics(request: BacktestRequest):
    """Calculate performance metrics for ETF rotation strategy"""
    try:
        if etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        print(f"Running ETF backtest with parameters: {request}")
        
        # Run the backtest
        result = etf_backtester.run_backtest(
            tickers=request.tickers,
            start_date=request.start_date,
            end_date=request.end_date,
            capital_per_week=request.capital_per_week,
            accumulation_weeks=request.accumulation_weeks,
            brokerage_percent=request.brokerage_percent,
            compounding_enabled=request.compounding_enabled
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=f"ETF backtest failed: {result['error']}")
        
        # Calculate metrics
        etf_metrics = etf_backtester.calculate_metrics(
            request.capital_per_week,
            request.accumulation_weeks,
            request.risk_free_rate
        )
        
        # Calculate benchmark metrics
        total_investment = request.accumulation_weeks * request.capital_per_week
        benchmark_metrics = etf_backtester.calculate_benchmark_metrics(
            total_investment,
            request.risk_free_rate
        )
        
        # Prepare performance data for charts
        performance_data = {
            "dates": [],
            "etf_strategy": [],
            "cumulative_investment": [],
            "benchmark_buyhold": []
        }
        
        if not etf_backtester.weekly_nav_df.empty:
            performance_data["dates"] = [str(date) for date in etf_backtester.weekly_nav_df['date']]
            performance_data["etf_strategy"] = etf_backtester.weekly_nav_df['nav'].tolist()
            performance_data["cumulative_investment"] = etf_backtester.weekly_nav_df['cumulative_investment'].tolist()
            
            if not etf_backtester.nifty50_df.empty:
                # Align benchmark data with weekly data
                benchmark_dates = [str(date) for date in etf_backtester.nifty50_df['date']]
                benchmark_navs = etf_backtester.nifty50_df['nav'].tolist()
                performance_data["benchmark_buyhold"] = benchmark_navs
        
        return {
            "success": True,
            "etf_metrics": etf_metrics,
            "benchmark_metrics": benchmark_metrics,
            "backtest_result": result,
            "performance_data": performance_data
        }
        
    except Exception as e:
        print(f"Error calculating ETF metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error calculating ETF metrics: {str(e)}")

@etf_router.get("/metrics/table")
async def get_etf_metrics_table():
    """Get formatted metrics comparison table for ETFs"""
    try:
        if etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        # This would need to be called after a backtest is run
        if not hasattr(etf_backtester, 'weekly_nav_df') or etf_backtester.weekly_nav_df is None:
            raise HTTPException(status_code=400, detail="No ETF backtest data available. Run backtest first.")
        
        # Calculate metrics
        etf_metrics = etf_backtester.calculate_metrics(50000, 52, 8.0)  # Default values
        total_investment = 52 * 50000
        benchmark_metrics = etf_backtester.calculate_benchmark_metrics(total_investment, 8.0)
        
        # Create formatted table
        formatted_table = etf_backtester.create_formatted_metrics_table(etf_metrics, benchmark_metrics)
        
        if not formatted_table.empty:
            return {"metrics_table": formatted_table.to_dict('records')}
        else:
            return {"metrics_table": []}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting ETF metrics table: {str(e)}")

@etf_router.get("/transaction-costs/summary")
async def get_etf_transaction_costs_summary():
    """Get transaction costs summary for ETFs"""
    try:
        if etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        if not etf_backtester.transaction_costs_log:
            return {"costs_summary": {
                'Total All Costs': '₹0',
                'Capital Gains Tax': '₹0',
                'Cost as % of Volume': '0.00%',
                'Total Transactions': '0'
            }}
        
        costs_summary = etf_backtester.get_transaction_costs_summary()
        return {"costs_summary": costs_summary}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting ETF transaction costs summary: {str(e)}")

@etf_router.get("/transaction-log")
async def get_etf_transaction_log():
    """Get transaction log from the latest ETF backtest"""
    try:
        if etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        if not hasattr(etf_backtester, 'portfolio_log') or not etf_backtester.portfolio_log:
            return {"transaction_log": [], "trading_summary": {}}
        
        # Convert portfolio log to frontend format
        transaction_log = []
        for log in etf_backtester.portfolio_log:
            # Extract transaction costs from the costs dictionary
            costs = log.get('costs', {})
            transaction_costs = costs.get('total_costs', 0) if costs else 0
            
            # Handle churning transactions specially to show both sell and buy tickers
            if log.get('action') == 'churn':
                sell_transactions = log.get('sell_transactions', [])
                buy_transaction = log.get('buy_transaction', {})
                
                # Extract sell tickers
                sell_tickers = []
                for sell_txn in sell_transactions:
                    if sell_txn.get('ticker'):
                        sell_tickers.append(sell_txn.get('ticker'))
                
                # Extract buy ticker
                buy_ticker = buy_transaction.get('ticker', 'N/A')
                
                # Create combined ticker string
                if sell_tickers and buy_ticker != 'N/A':
                    ticker_display = f"SELL: {', '.join(sell_tickers)} → BUY: {buy_ticker}"
                elif sell_tickers:
                    ticker_display = f"SELL: {', '.join(sell_tickers)}"
                elif buy_ticker != 'N/A':
                    ticker_display = f"BUY: {buy_ticker}"
                else:
                    ticker_display = "N/A"
                
                # Calculate total sell amount and units
                total_sell_amount = sum(txn.get('amount', 0) for txn in sell_transactions)
                total_sell_units = sum(txn.get('units', 0) for txn in sell_transactions)
                
                transaction_log.append({
                    'week': log.get('week', 0),
                    'date': log.get('execution_date', '').strftime('%Y-%m-%d') if hasattr(log.get('execution_date', ''), 'strftime') else str(log.get('execution_date', '')),
                    'action': log.get('action', 'NONE'),
                    'ticker': ticker_display,
                    'sell_tickers': sell_tickers,
                    'buy_ticker': buy_ticker,
                    'units_sold': total_sell_units,
                    'units_bought': log.get('units', 0),
                    'sell_amount': total_sell_amount,
                    'buy_amount': log.get('amount', 0),
                    'price': log.get('price', 0),
                    'amount': log.get('amount', 0),
                    'transaction_costs': transaction_costs,
                    'capital_gains_tax': log.get('capital_gains_tax', 0),
                    'nav': log.get('nav', 0),
                    'churning_details': {
                        'sell_transactions': sell_transactions,
                        'buy_transaction': buy_transaction,
                        'total_raised': log.get('total_raised', 0)
                    }
                })
            else:
                # Handle regular buy/sell transactions
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
        
        # Calculate trading summary
        total_trades = len(transaction_log)
        buy_trades = len([t for t in transaction_log if t['action'] == 'BUY'])
        sell_trades = len([t for t in transaction_log if t['action'] == 'SELL'])
        churn_trades = len([t for t in transaction_log if t['action'] == 'CHURN'])
        
        # Calculate churning statistics
        total_churn_sells = sum(len(t.get('sell_tickers', [])) for t in transaction_log if t['action'] == 'CHURN')
        total_churn_buys = len([t for t in transaction_log if t['action'] == 'CHURN' and t.get('buy_ticker') != 'N/A'])
        
        trading_summary = {
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'churn_trades': churn_trades,
            'churning_statistics': {
                'total_churn_operations': churn_trades,
                'total_sell_transactions_in_churns': total_churn_sells,
                'total_buy_transactions_in_churns': total_churn_buys,
                'average_sells_per_churn': total_churn_sells / churn_trades if churn_trades > 0 else 0
            },
            'no_trade_weeks': getattr(etf_backtester, 'skipped_days', []),
            'trading_frequency': f"{(total_trades / max(1, len(etf_backtester.portfolio_log))) * 100:.1f}%"
        }
        
        return {
            "transaction_log": transaction_log,
            "trading_summary": trading_summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading ETF transaction log: {str(e)}")

@etf_router.get("/transaction-costs")
async def get_etf_transaction_costs():
    """Get transaction costs data from the latest ETF backtest"""
    try:
        if etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        if not hasattr(etf_backtester, 'transaction_costs_log') or not etf_backtester.transaction_costs_log:
            return {"transaction_costs": []}
        
        # Convert transaction costs log to frontend format
        transaction_costs = []
        for cost in etf_backtester.transaction_costs_log:
            transaction_costs.append({
                'date': cost.get('date', '').strftime('%Y-%m-%d') if hasattr(cost.get('date', ''), 'strftime') else str(cost.get('date', '')),
                'cumulative_cost': cost.get('cumulative_costs', 0),
                'weekly_cost': cost.get('weekly_costs', 0),
                'total_costs': cost.get('total_costs', 0)
            })
        
        return {"transaction_costs": transaction_costs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading ETF transaction costs: {str(e)}")

@etf_router.get("/skipped-trades")
async def get_etf_skipped_trades():
    """Get skipped trades information from the latest ETF backtest"""
    try:
        if etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        if not hasattr(etf_backtester, 'skipped_days') or not etf_backtester.skipped_days:
            return {"skipped_trades": []}
        
        # Convert skipped days to frontend format
        skipped_trades = []
        for skip in etf_backtester.skipped_days:
            skipped_trades.append({
                'week': skip.get('week', 0),
                'date': skip.get('date', ''),
                'signal_date': skip.get('signal_date', 'N/A'),
                'reason': skip.get('reason', 'Unknown')
            })
        
        return {"skipped_trades": skipped_trades}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading ETF skipped trades: {str(e)}")

@etf_router.get("/trade-execution-status")
async def get_etf_trade_execution_status():
    """Get real-time trade execution status and statistics for ETFs"""
    try:
        if etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        # Get current backtest statistics
        stats = {
            'total_weeks_processed': getattr(etf_backtester, 'total_weeks', 0),
            'successful_signals': getattr(etf_backtester, 'successful_signals', 0),
            'successful_executions': getattr(etf_backtester, 'successful_executions', 0),
            'portfolio_log_entries': len(getattr(etf_backtester, 'portfolio_log', [])),
            'transaction_costs_entries': len(getattr(etf_backtester, 'transaction_costs_log', [])),
            'skipped_trades_count': len(getattr(etf_backtester, 'skipped_days', [])),
            'current_cash': getattr(etf_backtester, 'current_cash', 0),
            'current_holdings': getattr(etf_backtester, 'current_holdings', {}),
            'last_trade_date': None,
            'last_trade_action': None,
            'last_trade_ticker': None
        }
        
        # Get last trade information
        if etf_backtester.portfolio_log:
            last_trade = etf_backtester.portfolio_log[-1]
            stats['last_trade_date'] = last_trade.get('execution_date', '').strftime('%Y-%m-%d') if hasattr(last_trade.get('execution_date', ''), 'strftime') else str(last_trade.get('execution_date', ''))
            stats['last_trade_action'] = last_trade.get('action', 'NONE')
            stats['last_trade_ticker'] = last_trade.get('ticker', '')
        
        return {"trade_execution_status": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading ETF trade execution status: {str(e)}")

@etf_router.get("/charts/equity-curve")
async def get_etf_equity_curve_chart(show_benchmark: bool = True, show_etf_strategy: bool = True):
    """Get equity curve chart data for ETFs"""
    try:
        if etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        if not hasattr(etf_backtester, 'weekly_nav_df') or etf_backtester.weekly_nav_df is None:
            raise HTTPException(status_code=400, detail="No ETF backtest data available. Run backtest first.")
        
        # Return data for frontend charting
        if not etf_backtester.weekly_nav_df.empty:
            chart_data = {
                "dates": [str(date) for date in etf_backtester.weekly_nav_df['date']],
                "etf_strategy": etf_backtester.weekly_nav_df['nav'].tolist(),
                "cumulative_investment": etf_backtester.weekly_nav_df['cumulative_investment'].tolist(),
                "benchmark_buyhold": []
            }
            
            if not etf_backtester.nifty50_df.empty:
                chart_data["benchmark_buyhold"] = etf_backtester.nifty50_df['nav'].tolist()
            
            return {"chart_data": chart_data}
        else:
            return {"chart_data": {}}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting ETF equity curve chart: {str(e)}")

@etf_router.get("/charts/transaction-costs")
async def get_etf_transaction_costs_chart():
    """Get transaction costs chart data for ETFs"""
    try:
        if etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        if not etf_backtester.transaction_costs_log:
            return {"chart_data": {}}
        
        # Return data for frontend charting
        costs_df = pd.DataFrame(etf_backtester.transaction_costs_log)
        costs_df['date'] = pd.to_datetime(costs_df['date'])
        costs_df = costs_df.sort_values('date')
        costs_df['cumulative_total_costs'] = costs_df['total_impact'].cumsum()
        
        chart_data = {
            "dates": [str(date) for date in costs_df['date']],
            "cumulative_costs": costs_df['cumulative_total_costs'].tolist()
        }
        
        return {"chart_data": chart_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting ETF transaction costs chart: {str(e)}")

@etf_router.post("/cleanup")
async def cleanup_etf_resources():
    """Clean up ETF resources and clear cache"""
    try:
        cleanup_etf_backtester()
        return {"success": True, "message": "ETF resources cleaned up successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning up ETF resources: {str(e)}")

@etf_router.get("/costs/summary")
async def get_etf_costs_summary():
    """Get comprehensive costs summary including transaction costs and capital gains tax for ETFs"""
    try:
        if etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        if not hasattr(etf_backtester, 'portfolio_log') or not etf_backtester.portfolio_log:
            return {
                "total_all_costs": 0,
                "capital_gains_tax": 0,
                "transaction_costs": 0,
                "cost_as_percent_of_volume": 0,
                "total_transactions": 0,
                "total_volume": 0
            }
        
        # Calculate costs from portfolio log - FIXED: Extract transaction costs from costs dictionary
        total_capital_gains_tax = sum(log.get('capital_gains_tax', 0) for log in etf_backtester.portfolio_log)
        
        # Fix: Extract transaction costs from the costs dictionary, not directly from log
        total_transaction_costs = 0
        for log in etf_backtester.portfolio_log:
            costs = log.get('costs', {})
            transaction_cost = costs.get('total_costs', 0) if costs else 0
            total_transaction_costs += transaction_cost
        
        total_all_costs = total_capital_gains_tax + total_transaction_costs
        
        # Calculate total volume (sum of all transaction amounts)
        total_volume = sum(log.get('amount', 0) for log in etf_backtester.portfolio_log)
        
        # Calculate cost as percentage of volume
        cost_as_percent = (total_all_costs / total_volume * 100) if total_volume > 0 else 0
        
        # Count total transactions
        total_transactions = len(etf_backtester.portfolio_log)
        
        return {
            "total_all_costs": round(total_all_costs, 2),
            "capital_gains_tax": round(total_capital_gains_tax, 2),
            "transaction_costs": round(total_transaction_costs, 2),
            "cost_as_percent_of_volume": round(cost_as_percent, 3),
            "total_transactions": total_transactions,
            "total_volume": round(total_volume, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating ETF costs summary: {str(e)}")

@etf_router.get("/costs/analysis")
async def get_etf_costs_analysis():
    """Get detailed costs analysis over time for the ETF chart"""
    try:
        if etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        if not hasattr(etf_backtester, 'portfolio_log') or not etf_backtester.portfolio_log:
            return {"costs_data": []}
        
        # Create cumulative costs data over time
        costs_data = []
        cumulative_transaction_costs = 0
        cumulative_capital_gains_tax = 0
        cumulative_total_costs = 0
        
        # Group by date and calculate cumulative costs
        date_costs = {}
        for log in etf_backtester.portfolio_log:
            date = log.get('execution_date', '').strftime('%Y-%m-%d') if hasattr(log.get('execution_date', ''), 'strftime') else str(log.get('execution_date', ''))
            
            # Fix: Extract transaction costs from the costs dictionary, not directly from log
            costs = log.get('costs', {})
            transaction_cost = costs.get('total_costs', 0) if costs else 0
            capital_gains_tax = log.get('capital_gains_tax', 0)
            
            if date not in date_costs:
                date_costs[date] = {'transaction_costs': 0, 'capital_gains_tax': 0}
            
            date_costs[date]['transaction_costs'] += transaction_cost
            date_costs[date]['capital_gains_tax'] += capital_gains_tax
        
        # Convert to cumulative data
        for date in sorted(date_costs.keys()):
            cumulative_transaction_costs += date_costs[date]['transaction_costs']
            cumulative_capital_gains_tax += date_costs[date]['capital_gains_tax']
            cumulative_total_costs = cumulative_transaction_costs + cumulative_capital_gains_tax
            
            costs_data.append({
                'date': date,
                'cumulative_transaction_costs': round(cumulative_transaction_costs, 2),
                'cumulative_capital_gains_tax': round(cumulative_capital_gains_tax, 2),
                'total_cumulative_costs': round(cumulative_total_costs, 2)
            })
        
        return {"costs_data": costs_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating ETF costs analysis: {str(e)}")

@etf_router.get("/costs/breakdown")
async def get_etf_costs_breakdown():
    """Get detailed breakdown of costs by type and period for ETFs"""
    try:
        if etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        if not hasattr(etf_backtester, 'portfolio_log') or not etf_backtester.portfolio_log:
            return {"breakdown": {}}
        
        # Calculate costs by year
        yearly_costs = {}
        for log in etf_backtester.portfolio_log:
            date = log.get('execution_date')
            if hasattr(date, 'year'):
                year = date.year
            else:
                # Try to extract year from string date
                try:
                    year = int(str(date)[:4])
                except:
                    year = 2023  # fallback
            
            if year not in yearly_costs:
                yearly_costs[year] = {
                    'transaction_costs': 0,
                    'capital_gains_tax': 0,
                    'total_costs': 0,
                    'transactions': 0
                }
            
            # Fix: Extract transaction costs from the costs dictionary, not directly from log
            costs = log.get('costs', {})
            transaction_cost = costs.get('total_costs', 0) if costs else 0
            capital_gains_tax = log.get('capital_gains_tax', 0)
            
            yearly_costs[year]['transaction_costs'] += transaction_cost
            yearly_costs[year]['capital_gains_tax'] += capital_gains_tax
            yearly_costs[year]['total_costs'] += transaction_cost + capital_gains_tax
            yearly_costs[year]['transactions'] += 1
        
        # Round all values
        for year in yearly_costs:
            for key in ['transaction_costs', 'capital_gains_tax', 'total_costs']:
                yearly_costs[year][key] = round(yearly_costs[year][key], 2)
        
        return {"breakdown": yearly_costs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating ETF costs breakdown: {str(e)}")

# ============================================================================
# SAVED STRATEGY DATABASE FUNCTIONS
# ============================================================================

def init_saved_etf_strategies_table(db_path: str = "unified_etf_data.sqlite"):
    """Initialize the etf-savedStrategy table if it doesn't exist"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "etf-savedStrategy" (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                strategy_type TEXT NOT NULL,
                user_id TEXT NOT NULL,
                tickers TEXT NOT NULL,  -- JSON array of tickers
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                capital_per_week REAL NOT NULL,
                accumulation_weeks INTEGER NOT NULL,
                brokerage_percent REAL NOT NULL,
                compounding_enabled BOOLEAN NOT NULL,
                risk_free_rate REAL NOT NULL,
                use_custom_dates BOOLEAN NOT NULL,
                backtest_results TEXT NOT NULL,  -- JSON object
                created_at TEXT NOT NULL,
                created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index on user_id for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_etf_savedStrategy_user_id 
            ON "etf-savedStrategy"(user_id)
        ''')
        
        conn.commit()
        conn.close()
        print("✅ ETF-savedStrategy table initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing ETF-savedStrategy table: {e}")
        return False

# ============================================================================
# SAVED STRATEGY ROUTES
# ============================================================================

@etf_router.post("/save-strategy")
async def save_etf_strategy(request: SaveETFStrategyRequest):
    """Save an ETF strategy to the database with validation"""
    try:
        # Initialize the table if it doesn't exist
        if not init_saved_etf_strategies_table():
            raise HTTPException(status_code=500, detail="Failed to initialize database table")
        
        # Check if strategy already exists using the backtester core validation
        from backtester_core import ETFRotationBacktester
        
        backtester = ETFRotationBacktester()
        validation_result = backtester.check_strategy_exists(
            strategy_name=request.strategy_name,
            user_id=request.user_id,
            tickers=request.tickers,
            start_date=request.start_date,
            end_date=request.end_date,
            capital_per_week=request.capital_per_week,
            accumulation_weeks=request.accumulation_weeks,
            brokerage_percent=request.brokerage_percent,
            compounding_enabled=request.compounding_enabled,
            risk_free_rate=request.risk_free_rate,
            use_custom_dates=request.use_custom_dates
        )
        
        # If strategy exists, return appropriate response
        if validation_result.get("exists", False):
            existing_strategy = validation_result.get("existing_strategy", {})
            return {
                "success": False,
                "message": validation_result.get("message", "Strategy already exists"),
                "existing_strategy": existing_strategy,
                "strategy_exists": True
            }
        
        # If validation failed due to error, return error
        if "error" in validation_result:
            raise HTTPException(status_code=500, detail=validation_result["error"])
        
        # Strategy doesn't exist, proceed with saving
        conn = sqlite3.connect("unified_etf_data.sqlite")
        cursor = conn.cursor()
        
        # Convert tickers list to JSON string
        tickers_json = json.dumps(request.tickers)
        
        # Convert backtest_results to JSON string (handle None values)
        backtest_results_dict = request.backtest_results.dict()
        # Filter out None values
        backtest_results_dict = {k: v for k, v in backtest_results_dict.items() if v is not None}
        backtest_results_json = json.dumps(backtest_results_dict)
        
        # Insert the strategy
        cursor.execute('''
            INSERT INTO "etf-savedStrategy" (
                strategy_name, strategy_type, user_id, tickers, start_date, end_date,
                capital_per_week, accumulation_weeks, brokerage_percent, compounding_enabled,
                risk_free_rate, use_custom_dates, backtest_results, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            request.strategy_name, request.strategy_type, request.user_id, tickers_json,
            request.start_date, request.end_date, request.capital_per_week, request.accumulation_weeks,
            request.brokerage_percent, request.compounding_enabled, request.risk_free_rate,
            request.use_custom_dates, backtest_results_json, request.created_at
        ))
        
        strategy_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": "Strategy saved successfully",
            "strategy_id": strategy_id,
            "strategy_exists": False
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving strategy: {str(e)}")

@etf_router.get("/get-saved-strategies-list/{user_id}")
async def get_saved_etf_strategies(user_id: str):
    """Get all saved ETF strategies for a specific user"""
    try:
        # Initialize the table if it doesn't exist
        if not init_saved_etf_strategies_table():
            raise HTTPException(status_code=500, detail="Failed to initialize database table")
        
        conn = sqlite3.connect("unified_etf_data.sqlite")
        cursor = conn.cursor()
        
        # Get strategies for the user
        cursor.execute('''
            SELECT id, strategy_name, strategy_type, user_id, tickers, start_date, end_date,
                   capital_per_week, accumulation_weeks, brokerage_percent, compounding_enabled,
                   risk_free_rate, use_custom_dates, backtest_results, created_at, created_timestamp
            FROM "etf-savedStrategy" 
            WHERE user_id = ?
            ORDER BY created_timestamp DESC
        ''', (user_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        strategies = []
        for row in rows:
            try:
                # Parse JSON fields
                tickers = json.loads(row[4]) if row[4] else []  # tickers
                backtest_results = json.loads(row[13]) if row[13] else {}  # backtest_results
                
                strategies.append({
                    "id": row[0],
                    "strategy_name": row[1],
                    "strategy_type": row[2],
                    "user_id": row[3],
                    "tickers": tickers,
                    "start_date": row[5],
                    "end_date": row[6],
                    "capital_per_week": row[7],
                    "accumulation_weeks": row[8],
                    "brokerage_percent": row[9],
                    "compounding_enabled": bool(row[10]),
                    "risk_free_rate": row[11],
                    "use_custom_dates": bool(row[12]),
                    "backtest_results": backtest_results,
                    "created_at": row[14],
                    "created_timestamp": row[15]
                })
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse JSON for strategy ID {row[0]}: {e}")
                continue
        
        # Ensure we always return a proper structure
        if not strategies:
            strategies = []
        
        return {"strategies": strategies}
        
    except Exception as e:
        print(f"Error retrieving saved strategies: {str(e)}")
        # Return empty array instead of throwing error to prevent frontend crashes
        return {"strategies": []}

@etf_router.get("/get-saved-strategy/{strategy_id}")
async def get_saved_etf_strategy_by_id(strategy_id: int):
    """Get a specific saved ETF strategy by ID"""
    try:
        # Initialize the table if it doesn't exist
        if not init_saved_etf_strategies_table():
            raise HTTPException(status_code=500, detail="Failed to initialize database table")
        
        conn = sqlite3.connect("unified_etf_data.sqlite")
        cursor = conn.cursor()
        
        # Get the specific strategy
        cursor.execute('''
            SELECT id, strategy_name, strategy_type, user_id, tickers, start_date, end_date,
                   capital_per_week, accumulation_weeks, brokerage_percent, compounding_enabled,
                   risk_free_rate, use_custom_dates, backtest_results, created_at, created_timestamp
            FROM "etf-savedStrategy" 
            WHERE id = ?
        ''', (strategy_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        try:
            # Parse JSON fields
            tickers = json.loads(row[4]) if row[4] else []  # tickers
            backtest_results = json.loads(row[13]) if row[13] else {}  # backtest_results
            
            strategy = {
                "id": row[0],
                "strategy_name": row[1],
                "strategy_type": row[2],
                "user_id": row[3],
                "tickers": tickers,
                "start_date": row[5],
                "end_date": row[6],
                "capital_per_week": row[7],
                "accumulation_weeks": row[8],
                "brokerage_percent": row[9],
                "compounding_enabled": bool(row[10]),
                "risk_free_rate": row[11],
                "use_custom_dates": bool(row[12]),
                "backtest_results": backtest_results,
                "created_at": row[14],
                "created_timestamp": row[15]
            }
            
            return {"strategy": strategy}
            
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"Error parsing strategy data: {str(e)}")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving strategy: {str(e)}")

@etf_router.delete("/delete-saved-strategy/{strategy_id}")
async def delete_saved_etf_strategy(strategy_id: int):
    """Delete a saved ETF strategy by ID"""
    try:
        # Initialize the table if it doesn't exist
        if not init_saved_etf_strategies_table():
            raise HTTPException(status_code=500, detail="Failed to initialize database table")
        
        conn = sqlite3.connect("unified_etf_data.sqlite")
        cursor = conn.cursor()
        
        # Check if strategy exists
        cursor.execute('SELECT id FROM "etf-savedStrategy" WHERE id = ?', (strategy_id,))
        if not cursor.fetchone():
            conn.close()
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Delete the strategy
        cursor.execute('DELETE FROM "etf-savedStrategy" WHERE id = ?', (strategy_id,))
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": "Strategy deleted successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting strategy: {str(e)}")

@etf_router.put("/update-saved-strategy/{strategy_id}")
async def update_saved_etf_strategy(strategy_id: int, request: SaveETFStrategyRequest):
    """Update a saved ETF strategy"""
    try:
        # Initialize the table if it doesn't exist
        if not init_saved_etf_strategies_table():
            raise HTTPException(status_code=500, detail="Failed to initialize database table")
        
        conn = sqlite3.connect("unified_etf_data.sqlite")
        cursor = conn.cursor()
        
        # Check if strategy exists
        cursor.execute('SELECT id FROM "etf-savedStrategy" WHERE id = ?', (strategy_id,))
        if not cursor.fetchone():
            conn.close()
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Convert tickers list to JSON string
        tickers_json = json.dumps(request.tickers)
        
        # Convert backtest_results to JSON string (handle None values)
        backtest_results_dict = request.backtest_results.dict()
        # Filter out None values
        backtest_results_dict = {k: v for k, v in backtest_results_dict.items() if v is not None}
        backtest_results_json = json.dumps(backtest_results_dict)
        
        # Update the strategy
        cursor.execute('''
            UPDATE "etf-savedStrategy" SET
                strategy_name = ?, strategy_type = ?, user_id = ?, tickers = ?, start_date = ?, end_date = ?,
                capital_per_week = ?, accumulation_weeks = ?, brokerage_percent = ?, compounding_enabled = ?,
                risk_free_rate = ?, use_custom_dates = ?, backtest_results = ?, created_at = ?
            WHERE id = ?
        ''', (
            request.strategy_name, request.strategy_type, request.user_id, tickers_json,
            request.start_date, request.end_date, request.capital_per_week, request.accumulation_weeks,
            request.brokerage_percent, request.compounding_enabled, request.risk_free_rate,
            request.use_custom_dates, backtest_results_json, request.created_at, strategy_id
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": "Strategy updated successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating strategy: {str(e)}")

@etf_router.get("/get-saved-strategies-count/{user_id}")
async def get_saved_etf_strategies_count(user_id: str):
    """Get count of saved ETF strategies for a specific user"""
    try:
        # Initialize the table if it doesn't exist
        if not init_saved_etf_strategies_table():
            raise HTTPException(status_code=500, detail="Failed to initialize database table")
        
        conn = sqlite3.connect("unified_etf_data.sqlite")
        cursor = conn.cursor()
        
        # Get count of strategies for the user
        cursor.execute('''
            SELECT COUNT(*) FROM "etf-savedStrategy" WHERE user_id = ?
        ''', (user_id,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return {"count": count}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting strategy count: {str(e)}")

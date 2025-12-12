from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import sys
import os
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the ETF backtester
from ..services.backtester import ETFRotationBacktester
# Signal generator removed - not needed
LiveSignalGenerator = None
from ..etf_schemas import (
    BacktestRequest, ETFMetadata, BacktestResult, BacktestResults,
    SaveETFStrategyRequest, SavedETFStrategy
)

# Create ETF router
etf_router = APIRouter(prefix="/api", tags=["ETF Strategy"])

# Pydantic models for request/response


# Global ETF backtester instance
etf_backtester = None

def initialize_etf_backtester(db_path: str = "unified_etf_data.sqlite"):
    """Initialize the ETF backtester
    
    Args:
        db_path: Deprecated - kept for compatibility. Now uses PostgreSQL for all operations.
    """
    global etf_backtester
    try:
        etf_backtester = ETFRotationBacktester(db_path=db_path)  # db_path ignored, uses PostgreSQL
        print("ETF Backtester initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing ETF Backtester: {e}")
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
        metadata = etf_backtester.load_metadata()
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
        
        metadata = etf_backtester.load_metadata()
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
        if not tickers:
            raise HTTPException(status_code=400, detail="No tickers provided in request")
        
        print(f"Calculating date range for ETF tickers: {tickers}")
        
        # Enable verbose mode for debugging
        etf_backtester.set_verbose(True)
        
        start_date, end_date, years = etf_backtester.calculate_common_date_range(tickers)
        
        if start_date and end_date:
            return {
                "start_date": start_date,
                "end_date": end_date,
                "years": years
            }
        else:
            # Provide more detailed error message
            available_symbols = list(etf_backtester.etf_metadata.keys())[:20] if etf_backtester.etf_metadata else []
            error_msg = f"Could not calculate date range for ETFs: {tickers}. "
            if not etf_backtester.etf_metadata:
                error_msg += "Metadata is empty. "
            elif available_symbols:
                error_msg += f"Available symbols (sample): {available_symbols}. "
            error_msg += "Please check if the ETFs exist in the database."
            raise HTTPException(status_code=400, detail=error_msg)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] Error calculating date range: {e}")
        print(f"[ERROR] Traceback: {error_trace}")
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
        
        metadata = etf_backtester.load_metadata()
        etf_overview = []
        
        for symbol, meta in metadata.items():
            description = etf_backtester.generate_asset_description(symbol)
            sector = etf_backtester.get_asset_sector_classification(symbol)
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
                    'total_brokerage': 0,
                    'transactions': 0
                }
            
            # Fix: Extract transaction costs from the costs dictionary, not directly from log
            costs = log.get('costs', {})
            transaction_cost = costs.get('total_costs', 0) if costs else 0
            brokerage = costs.get('brokerage', 0) if costs else 0
            capital_gains_tax = log.get('capital_gains_tax', 0)
            
            yearly_costs[year]['transaction_costs'] += transaction_cost
            yearly_costs[year]['capital_gains_tax'] += capital_gains_tax
            yearly_costs[year]['total_costs'] += transaction_cost + capital_gains_tax
            yearly_costs[year]['total_brokerage'] += brokerage
            yearly_costs[year]['transactions'] += 1
        
        # Round all values
        for year in yearly_costs:
            for key in ['transaction_costs', 'capital_gains_tax', 'total_costs', 'total_brokerage']:
                yearly_costs[year][key] = round(yearly_costs[year][key], 2)
        
        return {"breakdown": yearly_costs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating ETF costs breakdown: {str(e)}")

# ============================================================================
# SAVED STRATEGY DATABASE FUNCTIONS
# ============================================================================

def init_saved_etf_strategies_table(db_path: str = None):
    """Initialize the etf_saved_strategy table in PostgreSQL (db_path parameter ignored, kept for compatibility)"""
    try:
        # Import database connection and models
        from Databases.app_data_db_connection import create_connection, init_database, get_engine
        
        # Check if connection already exists, if not create it
        try:
            # Try to get engine to check if connection exists
            engine = get_engine()
            # Connection exists, just initialize tables
        except RuntimeError:
            # Connection doesn't exist, create it
            if not create_connection():
                error_msg = "Failed to connect to PostgreSQL database"
                logger.error(error_msg)  # Removed unicode cross
                return False, error_msg
        
        # Initialize all tables (including etf_saved_strategy) using the centralized init_database
        if not init_database():
            error_msg = "Failed to initialize database tables"
            logger.error(error_msg)
            return False, error_msg
            
        logger.info("etf_saved_strategy table initialized successfully in PostgreSQL")
        return True, None
    except Exception as e:
        error_msg = f"Error initializing etf_saved_strategy table: {e}"
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        return False, str(e)

# ============================================================================
# SAVED STRATEGY ROUTES
# ============================================================================

@etf_router.post("/save-strategy")
async def save_etf_strategy(request: SaveETFStrategyRequest):
    """Save an ETF strategy to the database with validation"""
    try:
        # Initialize the table if it doesn't exist
        success, error_msg = init_saved_etf_strategies_table()
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to initialize database table: {error_msg}")
        
        # Check if strategy already exists using the backtester core validation
        from ..services.backtester import ETFRotationBacktester
        
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
        # Import helper functions
        from Databases.strategy_db_helpers import save_etf_strategy as save_strategy_db
        
        # Convert tickers list to JSON string
        tickers_json = json.dumps(request.tickers)
        
        # Convert backtest_results to JSON string (handle None values)
        backtest_results_dict = request.backtest_results.dict()
        # Filter out None values
        backtest_results_dict = {k: v for k, v in backtest_results_dict.items() if v is not None}
        backtest_results_json = json.dumps(backtest_results_dict)
        
        # Prepare strategy data
        strategy_data = {
            'strategy_name': request.strategy_name,
            'strategy_type': request.strategy_type,
            'user_id': request.user_id,
            'tickers': tickers_json,
            'start_date': request.start_date,
            'end_date': request.end_date,
            'capital_per_week': request.capital_per_week,
            'accumulation_weeks': request.accumulation_weeks,
            'brokerage_percent': request.brokerage_percent,
            'compounding_enabled': request.compounding_enabled,
            'risk_free_rate': request.risk_free_rate,
            'use_custom_dates': request.use_custom_dates,
            'backtest_results': backtest_results_json,
            'created_at': request.created_at,
            'status': 'deploy'
        }
        
        # Save to PostgreSQL
        strategy_id = save_strategy_db(strategy_data)
        
        return {
            "success": True,
            "message": "Strategy saved successfully",
            "strategy_id": strategy_id,
            "strategy_exists": False
        }
        
    except Exception as e:
        print(f"ERROR: Detailed error in save strategy: {str(e)}")
        print(f"ERROR: Exception type: {type(e).__name__}")
        import traceback
        print(f"ERROR: Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error saving strategy: {str(e)}")

@etf_router.get("/get-saved-strategies-list/{user_id}")
async def get_saved_etf_strategies(user_id: str):
    """Get all saved ETF strategies for a specific user"""
    try:
        # Initialize the table if it doesn't exist
        if not init_saved_etf_strategies_table():
            raise HTTPException(status_code=500, detail="Failed to initialize database table")
        
        # Import helper functions
        from Databases.strategy_db_helpers import get_etf_strategies_by_user
        
        # Get strategies from PostgreSQL
        strategies_list = get_etf_strategies_by_user(user_id)
        
        # Parse JSON fields and format response
        strategies = []
        for strategy in strategies_list:
            try:
                # Parse JSON fields
                tickers = json.loads(strategy['tickers']) if strategy.get('tickers') else []
                backtest_results = json.loads(strategy['backtest_results']) if strategy.get('backtest_results') else {}
                
                strategies.append({
                    "id": strategy['id'],
                    "strategy_name": strategy['strategy_name'],
                    "strategy_type": strategy['strategy_type'],
                    "user_id": strategy['user_id'],
                    "tickers": tickers,
                    "start_date": strategy['start_date'],
                    "end_date": strategy['end_date'],
                    "capital_per_week": strategy['capital_per_week'],
                    "accumulation_weeks": strategy['accumulation_weeks'],
                    "brokerage_percent": strategy['brokerage_percent'],
                    "compounding_enabled": strategy['compounding_enabled'],
                    "risk_free_rate": strategy['risk_free_rate'],
                    "use_custom_dates": strategy['use_custom_dates'],
                    "backtest_results": backtest_results,
                    "created_at": strategy['created_at'],
                    "created_timestamp": strategy.get('created_timestamp'),
                    "status": strategy.get('status', 'deploy'),
                    "run_id": strategy.get('run_id'),
                    "webhook_url": strategy.get('webhook_url'),
                    "client_information_json": strategy.get('client_information_json'),
                    "last_execution_date": strategy.get('last_execution_date'),
                    "next_execution_date": strategy.get('next_execution_date')
                })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse strategy ID {strategy.get('id')}: {e}")
                continue
        
        # Ensure we always return a proper structure
        if not strategies:
            strategies = []
        
        return {"strategies": strategies}
        
    except Exception as e:
        print(f"Error retrieving saved strategies: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return empty array instead of throwing error to prevent frontend crashes
        return {"strategies": []}

@etf_router.get("/get-saved-strategy/{strategy_id}")
async def get_saved_etf_strategy_by_id(strategy_id: int):
    """Get a specific saved ETF strategy by ID"""
    try:
        # Initialize the table if it doesn't exist
        if not init_saved_etf_strategies_table():
            raise HTTPException(status_code=500, detail="Failed to initialize database table")
        
        # Import helper functions
        from Databases.strategy_db_helpers import get_etf_strategy_by_id
        
        # Get strategy from PostgreSQL
        strategy = get_etf_strategy_by_id(strategy_id)
        
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        try:
            # Parse JSON fields
            tickers = json.loads(strategy['tickers']) if strategy.get('tickers') else []
            backtest_results = json.loads(strategy['backtest_results']) if strategy.get('backtest_results') else {}
            
            strategy_response = {
                "id": strategy['id'],
                "strategy_name": strategy['strategy_name'],
                "strategy_type": strategy['strategy_type'],
                "user_id": strategy['user_id'],
                "tickers": tickers,
                "start_date": strategy['start_date'],
                "end_date": strategy['end_date'],
                "capital_per_week": strategy['capital_per_week'],
                "accumulation_weeks": strategy['accumulation_weeks'],
                "brokerage_percent": strategy['brokerage_percent'],
                "compounding_enabled": strategy['compounding_enabled'],
                "risk_free_rate": strategy['risk_free_rate'],
                "use_custom_dates": strategy['use_custom_dates'],
                "backtest_results": backtest_results,
                "created_at": strategy['created_at'],
                "created_timestamp": strategy['created_timestamp']
            }
            
            return {"strategy": strategy_response}
            
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"Error parsing strategy data: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving strategy: {str(e)}")

@etf_router.delete("/delete-saved-strategy/{strategy_id}")
async def delete_saved_etf_strategy(strategy_id: int):
    """Delete a saved ETF strategy by ID"""
    session = None
    try:
        # Initialize the table if it doesn't exist
        if not init_saved_etf_strategies_table():
            raise HTTPException(status_code=500, detail="Failed to initialize database table")
        
        from Databases.app_data_db_connection import get_session
        from sqlalchemy import text
        
        session = get_session()
        
        # Check if strategy exists
        result = session.execute(text("SELECT id FROM etf_saved_strategy WHERE id = :strategy_id"), {"strategy_id": strategy_id})
        if not result.fetchone():
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Delete the strategy
        session.execute(text("DELETE FROM etf_saved_strategy WHERE id = :strategy_id"), {"strategy_id": strategy_id})
        session.commit()
        
        return {
            "success": True,
            "message": "Strategy deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if session:
            session.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting strategy: {str(e)}")
    finally:
        if session:
            session.close()

@etf_router.put("/update-saved-strategy/{strategy_id}")
async def update_saved_etf_strategy(strategy_id: int, request: SaveETFStrategyRequest):
    """Update a saved ETF strategy"""
    try:
        # Initialize the table if it doesn't exist
        if not init_saved_etf_strategies_table():
            raise HTTPException(status_code=500, detail="Failed to initialize database table")
        
        # Import helper functions
        from Databases.strategy_db_helpers import get_etf_strategy_by_id, save_etf_strategy as save_strategy_db
        
        # Check if strategy exists
        existing_strategy = get_etf_strategy_by_id(strategy_id)
        if not existing_strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Convert tickers list to JSON string
        tickers_json = json.dumps(request.tickers)
        
        # Convert backtest_results to JSON string (handle None values)
        backtest_results_dict = request.backtest_results.dict()
        # Filter out None values
        backtest_results_dict = {k: v for k, v in backtest_results_dict.items() if v is not None}
        backtest_results_json = json.dumps(backtest_results_dict)
        
        # Prepare strategy data for update
        strategy_data = {
            'id': strategy_id,
            'strategy_name': request.strategy_name,
            'strategy_type': request.strategy_type,
            'user_id': request.user_id,
            'tickers': tickers_json,
            'start_date': request.start_date,
            'end_date': request.end_date,
            'capital_per_week': request.capital_per_week,
            'accumulation_weeks': request.accumulation_weeks,
            'brokerage_percent': request.brokerage_percent,
            'compounding_enabled': request.compounding_enabled,
            'risk_free_rate': request.risk_free_rate,
            'use_custom_dates': request.use_custom_dates,
            'backtest_results': backtest_results_json,
            'created_at': request.created_at
        }
        
        # Update in PostgreSQL
        save_strategy_db(strategy_data)
        
        return {
            "success": True,
            "message": "Strategy updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating strategy: {str(e)}")

@etf_router.get("/get-saved-strategies-count/{user_id}")
async def get_saved_etf_strategies_count(user_id: str):
    """Get count of saved ETF strategies for a specific user"""
    session = None
    try:
        # Initialize the table if it doesn't exist
        if not init_saved_etf_strategies_table():
            raise HTTPException(status_code=500, detail="Failed to initialize database table")
        
        from Databases.app_data_db_connection import get_session
        from sqlalchemy import text
        
        session = get_session()
        
        # Get count of strategies for the user
        result = session.execute(text("SELECT COUNT(*) FROM etf_saved_strategy WHERE user_id = :user_id"), {"user_id": user_id})
        count = result.scalar()
        
        return {"count": count}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting strategy count: {str(e)}")
    finally:
        if session:
            session.close()

# ============================================================================
# NEW RS-STYLE ENDPOINTS FOR ETF STRATEGIES
# ============================================================================

@etf_router.get("/get-saved-strategies-table/{user_id}")
async def get_saved_etf_strategies_table(user_id: str):
    """Get saved ETF strategies in table format like RS strategy"""
    try:
        # Initialize the table if it doesn't exist
        if not init_saved_etf_strategies_table():
            raise HTTPException(status_code=500, detail="Failed to initialize database table")
        
        # Import helper functions
        from Databases.strategy_db_helpers import get_etf_strategies_by_user
        
        # Get strategies from PostgreSQL
        strategies_list = get_etf_strategies_by_user(user_id)
        
        # Format response
        strategies = []
        for strategy in strategies_list:
            strategy_response = {
                "id": strategy['id'],
                "strategy_name": strategy['strategy_name'],
                "strategy_type": strategy['strategy_type'],
                "user_id": strategy['user_id'],
                "config_id": strategy.get('config_id'),
                "backtest_id": strategy.get('backtest_id'),
                "start_date": strategy['start_date'],
                "end_date": strategy['end_date'],
                "etf_universe": strategy.get('etf_universe', 'NIFTY500'),
                "backtest_results": json.loads(strategy['backtest_results']) if isinstance(strategy.get('backtest_results'), str) else (strategy.get('backtest_results') or {}),
                "strategy_config": json.loads(strategy['strategy_config']) if isinstance(strategy.get('strategy_config'), str) else (strategy.get('strategy_config') or {}),
                "run_id": strategy.get('run_id'),
                "client_information_json": json.loads(strategy['client_information_json']) if isinstance(strategy.get('client_information_json'), str) else (strategy.get('client_information_json') or {}),
                "webhook_url": strategy.get('webhook_url'),
                "status": strategy.get('status', 'deploy'),
                "created_at": strategy.get('created_at')
            }
            strategies.append(strategy_response)
        
        return {
            "success": True,
            "strategies": strategies
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting ETF strategies table: {str(e)}")

@etf_router.post("/stop-etf-strategy")
async def stop_etf_strategy(request: dict):
    """Stop a running ETF strategy"""
    session = None
    try:
        strategy_id = request.get("strategy_id")
        user_id = request.get("user_id")
        
        if not strategy_id or not user_id:
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        from Databases.app_data_db_connection import get_session
        from sqlalchemy import text
        
        session = get_session()
        
        # Update status to 'stopped'
        result = session.execute(text("""
            UPDATE etf_saved_strategy 
            SET status = 'stopped', updated_at = CURRENT_TIMESTAMP
            WHERE id = :strategy_id AND user_id = :user_id
        """), {"strategy_id": strategy_id, "user_id": user_id})
        
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        session.commit()
        
        return {
            "success": True,
            "message": "ETF strategy stopped successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if session:
            session.rollback()
        raise HTTPException(status_code=500, detail=f"Error stopping ETF strategy: {str(e)}")
    finally:
        if session:
            session.close()

@etf_router.post("/restart-etf-strategy")
async def restart_etf_strategy(request: dict):
    """Restart a stopped ETF strategy"""
    session = None
    try:
        strategy_id = request.get("strategy_id")
        user_id = request.get("user_id")
        
        if not strategy_id or not user_id:
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        from Databases.app_data_db_connection import get_session
        from sqlalchemy import text
        
        session = get_session()
        
        # Update status to 'running'
        result = session.execute(text("""
            UPDATE etf_saved_strategy 
            SET status = 'running', updated_at = CURRENT_TIMESTAMP
            WHERE id = :strategy_id AND user_id = :user_id
        """), {"strategy_id": strategy_id, "user_id": user_id})
        
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        session.commit()
        
        return {
            "success": True,
            "message": "ETF strategy restarted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if session:
            session.rollback()
        raise HTTPException(status_code=500, detail=f"Error restarting ETF strategy: {str(e)}")
    finally:
        if session:
            session.close()

@etf_router.delete("/delete-etf-strategy/{strategy_id}")
async def delete_etf_strategy(strategy_id: int):
    """Delete an ETF strategy"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from sqlalchemy import text
        
        session = get_session()
        
        # Delete the strategy
        result = session.execute(text("DELETE FROM etf_saved_strategy WHERE id = :strategy_id"), {"strategy_id": strategy_id})
        
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        session.commit()
        
        return {
            "success": True,
            "message": "ETF strategy deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if session:
            session.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting ETF strategy: {str(e)}")
    finally:
        if session:
            session.close()

@etf_router.post("/stop-strategy/{strategy_id}")
async def stop_etf_strategy(strategy_id: int):
    """
    Stop a running ETF strategy by changing status to 'stopped'
    
    Database: PostgreSQL (Neon) - app_data_db_connection (ApplicationData database)
    Table: etf_saved_strategy
    """
    session = None
    try:
        # Initialize the table if it doesn't exist
        if not init_saved_etf_strategies_table():
            raise HTTPException(status_code=500, detail="Failed to initialize database table")
        
        # Use PostgreSQL connection
        from Databases.app_data_db_connection import get_session
        from sqlalchemy import text
        
        session = get_session()
        
        # Check if strategy exists and get current status
        result = session.execute(text("""
            SELECT id, status FROM etf_saved_strategy 
            WHERE id = :strategy_id
        """), {"strategy_id": strategy_id})
        
        strategy = result.fetchone()
        
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        current_status = strategy[1]
        if current_status != 'running':
            raise HTTPException(status_code=400, detail=f"Strategy is not running. Current status: {current_status}")
        
        # Update status to 'stopped'
        session.execute(text("""
            UPDATE etf_saved_strategy 
            SET status = 'stopped', updated_at = CURRENT_TIMESTAMP
            WHERE id = :strategy_id
        """), {"strategy_id": strategy_id})
        
        session.commit()
        
        return {
            "success": True,
            "message": "Strategy stopped successfully",
            "strategy_id": strategy_id,
            "new_status": "stopped"
        }
    except HTTPException:
        raise
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Error stopping strategy: {e}")
        raise HTTPException(status_code=500, detail=f"Error stopping strategy: {str(e)}")
    finally:
        if session:
            session.close()

@etf_router.post("/restart-strategy/{strategy_id}")
async def restart_etf_strategy(strategy_id: int):
    """
    Restart a stopped ETF strategy by changing status to 'running'
    
    Database: PostgreSQL (Neon) - app_data_db_connection (ApplicationData database)
    Table: etf_saved_strategy
    """
    session = None
    try:
        # Initialize the table if it doesn't exist
        if not init_saved_etf_strategies_table():
            raise HTTPException(status_code=500, detail="Failed to initialize database table")
        
        # Use PostgreSQL connection
        from Databases.app_data_db_connection import get_session
        from sqlalchemy import text
        
        session = get_session()
        
        # Check if strategy exists and get current status
        result = session.execute(text("""
            SELECT id, status FROM etf_saved_strategy 
            WHERE id = :strategy_id
        """), {"strategy_id": strategy_id})
        
        strategy = result.fetchone()
        
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        current_status = strategy[1]
        if current_status != 'stopped':
            raise HTTPException(status_code=400, detail=f"Strategy is not stopped. Current status: {current_status}")
        
        # Update status to 'running'
        session.execute(text("""
            UPDATE etf_saved_strategy 
            SET status = 'running', updated_at = CURRENT_TIMESTAMP
            WHERE id = :strategy_id
        """), {"strategy_id": strategy_id})
        
        session.commit()
        
        return {
            "success": True,
            "message": "Strategy restarted successfully",
            "strategy_id": strategy_id,
            "new_status": "running"
        }
    except HTTPException:
        raise
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Error restarting strategy: {e}")
        raise HTTPException(status_code=500, detail=f"Error restarting strategy: {str(e)}")
    finally:
        if session:
            session.close()

@etf_router.delete("/delete-strategy-by-run-id/{run_id}")
async def delete_etf_strategy_by_run_id(run_id: str):
    """
    Delete an ETF strategy by run_id
    
    Database: PostgreSQL (Neon) - app_data_db_connection (ApplicationData database)
    Table: etf_saved_strategy
    """
    session = None
    try:
        # Initialize the table if it doesn't exist
        if not init_saved_etf_strategies_table():
            raise HTTPException(status_code=500, detail="Failed to initialize database table")
        
        # Use PostgreSQL connection
        from Databases.app_data_db_connection import get_session
        from sqlalchemy import text
        
        session = get_session()
        
        # Check if strategy exists
        result = session.execute(text("""
            SELECT id FROM etf_saved_strategy 
            WHERE run_id = :run_id
        """), {"run_id": run_id})
        
        strategy = result.fetchone()
        
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Delete the strategy
        result = session.execute(text("""
            DELETE FROM etf_saved_strategy 
            WHERE run_id = :run_id
        """), {"run_id": run_id})
        
        session.commit()
        
        return {
            "success": True,
            "message": "Strategy deleted successfully",
            "run_id": run_id
        }
    except HTTPException:
        raise
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Error deleting strategy by run_id: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting strategy: {str(e)}")
    finally:
        if session:
            session.close()

# ============================================================================
# ETF SIGNAL GENERATION ENDPOINTS
# ============================================================================

@etf_router.post("/etf-signals/generate")
async def generate_etf_signals(request: Dict[str, Any] = None):
    """
    Generate ETF trading signals
    
    If run_id is provided, generates signals for that specific strategy.
    If no run_id is provided, generates signals for ALL running strategies in etf_saved_strategy table.
    
    Request body (optional):
    {
        "run_id": "run_etf_strategy_2025-10-15_1760544030144",  # Optional - if not provided, processes all running
        "strategy_type": "SurfTrend",  # Optional, default: "SurfTrend"
        "strategy_config": {}  # Optional strategy configuration
    }
    
    Returns (single run_id):
    {
        "success": bool,
        "run_id": str,
        "signals_count": int,
        "buy_count": int,
        "sell_count": int,
        "duration_seconds": float,
        "signals": List[Dict],
        "message": str
    }
    
    Returns (batch - all running):
    {
        "success": bool,
        "total_strategies": int,
        "successful": int,
        "failed": int,
        "results": List[Dict],
        "duration_seconds": float
    }
    """
    try:
        if LiveSignalGenerator is None:
            raise HTTPException(
                status_code=500, 
                detail="Signal generator not available. Check imports and dependencies."
            )
        
        # Handle empty request body
        if request is None:
            request = {}
        
        run_id = request.get("run_id") if request else None
        strategy_type = request.get("strategy_type", "SurfTrend") if request else "SurfTrend"
        strategy_config = request.get("strategy_config") if request else None
        
        # Initialize signal generator
        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
            "unified_etf_data.sqlite"
        )
        
        generator = LiveSignalGenerator(db_path=db_path)
        
        # If run_id is provided, generate for that specific strategy
        if run_id:
            result = generator.run_weekly_signal_generation(
                run_id=run_id,
                strategy_type=strategy_type,
                strategy_config=strategy_config
            )
            
            generator.cleanup()
            
            if result.get("success"):
                return {
                    "success": True,
                    "run_id": result["run_id"],
                    "signals_count": result["signals_count"],
                    "buy_count": result["buy_count"],
                    "sell_count": result["sell_count"],
                    "duration_seconds": result.get("duration_seconds", 0),
                    "signals": result.get("signals", []),
                    "message": f"Successfully generated {result['signals_count']} signals"
                }
            else:
                return {
                    "success": False,
                    "run_id": result.get("run_id"),
                    "signals_count": 0,
                    "buy_count": 0,
                    "sell_count": 0,
                    "error": result.get("error", "Unknown error"),
                    "message": result.get("message", "Signal generation failed")
                }
        
        # No run_id provided - generate for all running strategies
        start_time = datetime.now()
        running_strategies = generator.get_all_running_strategies()
        
        if not running_strategies:
            generator.cleanup()
            return {
                "success": True,
                "total_strategies": 0,
                "successful": 0,
                "failed": 0,
                "results": [],
                "duration_seconds": 0,
                "message": "No running strategies found with run_id"
            }
        
        results = []
        successful = 0
        failed = 0
        
        for strategy in running_strategies:
            strategy_run_id = strategy['run_id']
            strategy_type_from_db = strategy.get('strategy_type', strategy_type)
            
            try:
                result = generator.run_weekly_signal_generation(
                    run_id=strategy_run_id,
                    strategy_type=strategy_type_from_db,
                    strategy_config=strategy_config
                )
                
                if result.get("success"):
                    successful += 1
                else:
                    failed += 1
                
                results.append({
                    "run_id": strategy_run_id,
                    "user_id": strategy.get('user_id'),
                    "strategy_name": strategy.get('strategy_name'),
                    "success": result.get("success", False),
                    "signals_count": result.get("signals_count", 0),
                    "buy_count": result.get("buy_count", 0),
                    "sell_count": result.get("sell_count", 0),
                    "error": result.get("error"),
                    "message": result.get("message", "Signal generation completed")
                })
                
            except Exception as e:
                failed += 1
                results.append({
                    "run_id": strategy_run_id,
                    "user_id": strategy.get('user_id'),
                    "strategy_name": strategy.get('strategy_name'),
                    "success": False,
                    "signals_count": 0,
                    "buy_count": 0,
                    "sell_count": 0,
                    "error": str(e),
                    "message": f"Error generating signals: {str(e)}"
                })
        
        generator.cleanup()
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        return {
            "success": True,
            "total_strategies": len(running_strategies),
            "successful": successful,
            "failed": failed,
            "results": results,
            "duration_seconds": total_duration,
            "message": f"Processed {len(running_strategies)} strategies: {successful} successful, {failed} failed"
        }
            
    except Exception as e:
        print(f"Error generating ETF signals: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating ETF signals: {str(e)}")


@etf_router.get("/etf-signals/recent")
async def get_recent_etf_signals(days: int = 7):
    """
    Get recent ETF signals from the database
    
    Query parameters:
    - days: Number of days to look back (default: 7)
    
    Returns:
    {
        "success": bool,
        "signals": List[Dict],
        "count": int
    }
    """
    try:
        if LiveSignalGenerator is None:
            raise HTTPException(
                status_code=500, 
                detail="Signal generator not available. Check imports and dependencies."
            )
        
        # Get database path
        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
            "unified_etf_data.sqlite"
        )
        
        generator = LiveSignalGenerator(db_path=db_path)
        
        # Get recent signals
        signals = generator.get_recent_signals(days=days)
        
        # Cleanup
        generator.cleanup()
        
        return {
            "success": True,
            "signals": signals,
            "count": len(signals)
        }
        
    except Exception as e:
        print(f"Error getting recent ETF signals: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting recent ETF signals: {str(e)}")


@etf_router.get("/etf-signals/run/{run_id}")
async def get_etf_signals_by_run_id(run_id: str):
    """
    Get ETF signals for a specific run_id
    
    Path parameters:
    - run_id: Deployment run ID
    
    Returns:
    {
        "success": bool,
        "run_id": str,
        "signals": List[Dict],
        "count": int,
        "buy_count": int,
        "sell_count": int
    }
    """
    try:
        if LiveSignalGenerator is None:
            raise HTTPException(
                status_code=500, 
                detail="Signal generator not available. Check imports and dependencies."
            )
        
        # Query signals from PostgreSQL
        from Databases.app_data_db_connection import get_session
        from Databases.strategy_models import LiveSignal, LiveRun
        
        session = get_session()
        try:
            # Query signals with join to runs
            signals_query = session.query(LiveSignal, LiveRun).join(
                LiveRun, LiveSignal.run_id == LiveRun.run_id
            ).filter(
                LiveSignal.run_id == run_id
            ).order_by(LiveSignal.created_at.desc())
            
            signals = []
            for signal, run in signals_query.all():
                signal_dict = {
                    'id': signal.id,
                    'run_id': signal.run_id,
                    'user_id': signal.user_id,
                    'strategy_version': signal.strategy_version,
                    'signal_date': signal.signal_date,
                    'signal_symbol': signal.signal_symbol,
                    'etf_name': signal.etf_name,
                    'side': signal.side,
                    'score': signal.score,
                    'reason': signal.reason,
                    'payload_json': signal.payload_json,
                    'created_at': signal.created_at.isoformat() if signal.created_at else None,
                    'status': run.status,
                    'summary_json': run.summary_json
                }
                
                # Parse JSON fields
                if signal_dict.get('payload_json'):
                    try:
                        signal_dict['payload'] = json.loads(signal_dict['payload_json'])
                    except:
                        signal_dict['payload'] = {}
                if signal_dict.get('summary_json'):
                    try:
                        signal_dict['summary'] = json.loads(signal_dict['summary_json'])
                    except:
                        signal_dict['summary'] = {}
                signals.append(signal_dict)
        finally:
            session.close()
        
        buy_count = len([s for s in signals if s.get('side') == 'BUY'])
        sell_count = len([s for s in signals if s.get('side') == 'SELL'])
        
        return {
            "success": True,
            "run_id": run_id,
            "signals": signals,
            "count": len(signals),
            "buy_count": buy_count,
            "sell_count": sell_count
        }
        
    except Exception as e:
        print(f"Error getting ETF signals by run_id: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting ETF signals by run_id: {str(e)}")


@etf_router.get("/etf-signals/runs")
async def get_etf_signal_runs(days: int = 30):
    """
    Get list of signal generation runs within the specified days
    
    Query parameters:
    - days: Number of days to look back (default: 30)
    
    Returns:
    {
        "success": bool,
        "runs": List[Dict],
        "count": int
    }
    """
    try:
        # Query runs from PostgreSQL
        from Databases.app_data_db_connection import get_session
        from Databases.strategy_models import LiveRun, LiveSignal
        from sqlalchemy import func
        
        session = get_session()
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # Query runs with signal counts
            runs_query = session.query(
                LiveRun,
                func.count(LiveSignal.id).label('signal_count'),
                func.sum(func.case((LiveSignal.side == 'BUY', 1), else_=0)).label('buy_count'),
                func.sum(func.case((LiveSignal.side == 'SELL', 1), else_=0)).label('sell_count')
            ).outerjoin(
                LiveSignal, LiveRun.run_id == LiveSignal.run_id
            ).filter(
                LiveRun.signal_date >= cutoff_date
            ).group_by(LiveRun.run_id).order_by(LiveRun.created_at.desc())
            
            runs = []
            for run, signal_count, buy_count, sell_count in runs_query.all():
                run_dict = {
                    'id': run.id,
                    'run_id': run.run_id,
                    'strategy_version': run.strategy_version,
                    'signal_date': run.signal_date,
                    'created_at': run.created_at.isoformat() if run.created_at else None,
                    'status': run.status,
                    'webhook_attempts': run.webhook_attempts,
                    'last_error': run.last_error,
                    'summary_json': run.summary_json,
                    'signal_count': signal_count or 0,
                    'buy_count': int(buy_count) if buy_count else 0,
                    'sell_count': int(sell_count) if sell_count else 0
                }
                
                if run_dict.get('summary_json'):
                    try:
                        run_dict['summary'] = json.loads(run_dict['summary_json'])
                    except:
                        run_dict['summary'] = {}
                runs.append(run_dict)
        finally:
            session.close()
        
        return {
            "success": True,
            "runs": runs,
            "count": len(runs)
        }
        
    except Exception as e:
        print(f"Error getting ETF signal runs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting ETF signal runs: {str(e)}")


@etf_router.get("/etf-signals/latest")
async def get_latest_etf_signals():
    """
    Get the latest ETF signals (most recent run)
    
    Returns:
    {
        "success": bool,
        "run_id": str,
        "signals": List[Dict],
        "count": int,
        "buy_count": int,
        "sell_count": int,
        "signal_date": str
    }
    """
    try:
        # Query from PostgreSQL
        from Databases.app_data_db_connection import get_session
        from Databases.strategy_models import LiveRun, LiveSignal
        
        session = get_session()
        try:
            # Get the most recent run
            run = session.query(LiveRun).order_by(LiveRun.created_at.desc()).first()
            
            if not run:
                return {
                    "success": False,
                    "message": "No signals found",
                    "run_id": None,
                    "signals": [],
                    "count": 0,
                    "buy_count": 0,
                    "sell_count": 0
                }
            
            run_id = run.run_id
            strategy_version = run.strategy_version
            signal_date = run.signal_date
            
            # Get signals for this run
            signals_query = session.query(LiveSignal).filter(
                LiveSignal.run_id == run_id
            ).order_by(LiveSignal.created_at.desc())
            
            signals = []
            for signal in signals_query.all():
                signal_dict = {
                    'id': signal.id,
                    'run_id': signal.run_id,
                    'user_id': signal.user_id,
                    'strategy_version': signal.strategy_version,
                    'signal_date': signal.signal_date,
                    'signal_symbol': signal.signal_symbol,
                    'etf_name': signal.etf_name,
                    'side': signal.side,
                    'score': signal.score,
                    'reason': signal.reason,
                    'payload_json': signal.payload_json,
                    'created_at': signal.created_at.isoformat() if signal.created_at else None
                }
                
                if signal_dict.get('payload_json'):
                    try:
                        signal_dict['payload'] = json.loads(signal_dict['payload_json'])
                    except:
                        signal_dict['payload'] = {}
                signals.append(signal_dict)
        finally:
            session.close()
        
        buy_count = len([s for s in signals if s.get('side') == 'BUY'])
        sell_count = len([s for s in signals if s.get('side') == 'SELL'])
        
        return {
            "success": True,
            "run_id": run_id,
            "strategy_version": strategy_version,
            "signal_date": signal_date,
            "signals": signals,
            "count": len(signals),
            "buy_count": buy_count,
            "sell_count": sell_count
        }
        
    except Exception as e:
        print(f"Error getting latest ETF signals: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting latest ETF signals: {str(e)}")

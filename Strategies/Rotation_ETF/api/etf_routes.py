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
        import math
        
        # Helper function to recursively sanitize NaN/inf values
        def sanitize_data(obj):
            """Recursively convert NaN/inf to 0 in nested structures"""
            if isinstance(obj, dict):
                return {k: sanitize_data(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_data(item) for item in obj]
            elif isinstance(obj, (int, float)):
                if math.isnan(obj) or math.isinf(obj):
                    return 0
                return obj
            else:
                return obj
        
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
        
        # Sanitize all data before returning
        response_data = {
            "success": True,
            "etf_metrics": sanitize_data(etf_metrics),
            "benchmark_metrics": sanitize_data(benchmark_metrics),
            "backtest_result": sanitize_data(result),
            "performance_data": sanitize_data(performance_data)
        }
        
        return response_data
        
    except Exception as e:
        print(f"Error calculating ETF metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error calculating ETF metrics: {str(e)}")

@etf_router.get("/metrics/table")
async def get_etf_metrics_table():
    """Get formatted metrics comparison table for ETFs"""
    try:
        import math
        
        # Helper function to recursively sanitize NaN/inf values
        def sanitize_data(obj):
            """Recursively convert NaN/inf to 0 in nested structures"""
            if isinstance(obj, dict):
                return {k: sanitize_data(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_data(item) for item in obj]
            elif isinstance(obj, (int, float)):
                if math.isnan(obj) or math.isinf(obj):
                    return 0
                return obj
            else:
                return obj
        
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
            table_data = formatted_table.to_dict('records')
            return {"metrics_table": sanitize_data(table_data)}
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
        import math
        
        if etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        if not hasattr(etf_backtester, 'portfolio_log') or not etf_backtester.portfolio_log:
            return {"transaction_log": [], "trading_summary": {}}
        
        # Helper function to recursively sanitize NaN/inf values
        def sanitize_data(obj):
            """Recursively convert NaN/inf to 0 in nested structures"""
            if isinstance(obj, dict):
                return {k: sanitize_data(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_data(item) for item in obj]
            elif isinstance(obj, (int, float)):
                if math.isnan(obj) or math.isinf(obj):
                    return 0
                return obj
            else:
                return obj
        
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
        
        # Sanitize all data to remove NaN values
        transaction_log = sanitize_data(transaction_log)
        
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

@etf_router.get("/debug/portfolio-log")
async def debug_portfolio_log():
    """Debug endpoint to inspect raw portfolio_log data"""
    try:
        import math
        
        if etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized")
        
        if not hasattr(etf_backtester, 'portfolio_log') or not etf_backtester.portfolio_log:
            return {"portfolio_log": [], "count": 0}
        
        # Helper function to handle NaN values
        def safe_float(value):
            if isinstance(value, (int, float)):
                if math.isnan(value) or math.isinf(value):
                    return None
                return value
            return value
        
        # Return simplified view of portfolio_log
        debug_data = []
        for i, log in enumerate(etf_backtester.portfolio_log):
            debug_data.append({
                'index': i,
                'week': log.get('week', 0),
                'execution_date': log.get('execution_date', '').strftime('%Y-%m-%d %A') if hasattr(log.get('execution_date', ''), 'strftime') else str(log.get('execution_date', '')),
                'signal_date': log.get('signal_date', '').strftime('%Y-%m-%d %A') if hasattr(log.get('signal_date', ''), 'strftime') else str(log.get('signal_date', '')),
                'action': log.get('action', 'NONE'),
                'ticker': log.get('ticker', ''),
                'nav': safe_float(log.get('nav', 0))
            })
        
        return {
            "portfolio_log": debug_data,
            "count": len(debug_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in debug endpoint: {str(e)}")

@etf_router.get("/debug/backtest-state")
async def debug_backtest_state():
    """Enhanced debug endpoint to check complete backtester state"""
    try:
        if etf_backtester is None:
            return {
                "error": "Backtester not initialized",
                "backtester_exists": False
            }
        
        # Gather comprehensive state information
        state = {
            "backtester_exists": True,
            "backtester_type": type(etf_backtester).__name__,
            "has_portfolio_log": hasattr(etf_backtester, 'portfolio_log'),
            "portfolio_log_count": len(etf_backtester.portfolio_log) if hasattr(etf_backtester, 'portfolio_log') else 0,
            "has_transaction_costs_log": hasattr(etf_backtester, 'transaction_costs_log'),
            "transaction_costs_count": len(etf_backtester.transaction_costs_log) if hasattr(etf_backtester, 'transaction_costs_log') else 0,
            "has_skipped_days": hasattr(etf_backtester, 'skipped_days'),
            "skipped_days_count": len(etf_backtester.skipped_days) if hasattr(etf_backtester, 'skipped_days') else 0,
            "has_weekly_nav_df": hasattr(etf_backtester, 'weekly_nav_df'),
            "weekly_nav_rows": len(etf_backtester.weekly_nav_df) if hasattr(etf_backtester, 'weekly_nav_df') and etf_backtester.weekly_nav_df is not None else 0,
            "total_weeks": getattr(etf_backtester, 'total_weeks', 0),
            "successful_signals": getattr(etf_backtester, 'successful_signals', 0),
            "successful_executions": getattr(etf_backtester, 'successful_executions', 0),
            "current_cash": getattr(etf_backtester, 'current_cash', 0),
            "current_holdings": getattr(etf_backtester, 'current_holdings', {}),
            "etf_metadata_count": len(etf_backtester.etf_metadata) if hasattr(etf_backtester, 'etf_metadata') else 0
        }
        
        # Add sample of portfolio_log if available
        if state["portfolio_log_count"] > 0:
            sample_logs = []
            for i, log in enumerate(etf_backtester.portfolio_log[:3]):  # First 3 entries
                sample_logs.append({
                    'week': log.get('week', 0),
                    'action': log.get('action', 'NONE'),
                    'ticker': log.get('ticker', ''),
                    'execution_date': str(log.get('execution_date', ''))
                })
            state["portfolio_log_sample"] = sample_logs
        
        return state
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

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

# ============================================================================
# SAVED STRATEGY DATABASE FUNCTIONS
# ============================================================================

def init_saved_etf_strategies_table(db_path: str = None):
    """
    Now a no-op as tables are managed centrally.
    Kept for compatibility with existing imports.
    """
    return True, None

# ============================================================================
# SAVED STRATEGY ROUTES
# ============================================================================

@etf_router.post("/save-strategy")
async def save_etf_strategy(request: SaveETFStrategyRequest):
    """Save an ETF strategy to the unified saved_instances table"""
    session = None
    try:
        from Services.strategy_manager.models import SavedInstance
        from Databases.app_data_db_connection import get_session
        import uuid

        session = get_session()
        
        # Check if strategy with same name and user exists
        existing = session.query(SavedInstance).filter(
            SavedInstance.user_id == request.user_id,
            SavedInstance.strategy_name == request.strategy_name,
            SavedInstance.strategy_type == request.strategy_type
        ).first()

        if existing:
            return {
                "success": False,
                "message": f"Strategy '{request.strategy_name}' already exists for this user.",
                "strategy_exists": True
            }

        # Convert backtest results to dict
        backtest_results_dict = request.backtest_results.dict()
        backtest_results_dict = {k: v for k, v in backtest_results_dict.items() if v is not None}

        # Combine parameters
        strat_params = {
            "capital_per_week": request.capital_per_week,
            "accumulation_weeks": request.accumulation_weeks,
            "brokerage_percent": request.brokerage_percent,
            "compounding_enabled": request.compounding_enabled,
            "risk_free_rate": request.risk_free_rate,
            "backtest_results": backtest_results_dict
        }

        # Generate a run_id if not present
        run_id = f"run_{request.strategy_type.lower()}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

        new_instance = SavedInstance(
            user_id=request.user_id,
            strategy_name=request.strategy_name,
            strategy_type=request.strategy_type,
            tickers=json.dumps(request.tickers),
            start_date=request.start_date,
            end_date=request.end_date,
            strategies_parameters=strat_params,
            use_custom_date=request.use_custom_dates,
            run_id=run_id,
            status='deploy',
            source='internal'
        )

        session.add(new_instance)
        session.commit()
        
        return {
            "success": True,
            "message": "Strategy saved successfully",
            "strategy_id": new_instance.id,
            "strategy_exists": False
        }
        
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Error saving strategy: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error saving strategy: {str(e)}")
    finally:
        if session:
            session.close()

@etf_router.get("/get-saved-strategies-list/{user_id}")
async def get_saved_etf_strategies(user_id: str):
    """Get all saved ETF strategies for a specific user from unified table"""
    session = None
    try:
        from Services.strategy_manager.models import SavedInstance
        from Databases.app_data_db_connection import get_session
        
        session = get_session()
        
        # Query unified table for ETF strategies
        instances = session.query(SavedInstance).filter(
            SavedInstance.user_id == user_id,
            SavedInstance.strategy_type.like('%ETF%')
        ).order_by(SavedInstance.created_at.desc()).all()
        
        strategies = []
        for inst in instances:
            params = inst.strategies_parameters or {}
            
            # Unpack params for compatibility
            strategies.append({
                "id": inst.id,
                "strategy_name": inst.strategy_name,
                "strategy_type": inst.strategy_type,
                "user_id": inst.user_id,
                "tickers": json.loads(inst.tickers) if inst.tickers else [],
                "start_date": inst.start_date,
                "end_date": inst.end_date,
                "capital_per_week": params.get('capital_per_week', 0),
                "accumulation_weeks": params.get('accumulation_weeks', 0),
                "brokerage_percent": params.get('brokerage_percent', 0),
                "compounding_enabled": params.get('compounding_enabled', False),
                "risk_free_rate": params.get('risk_free_rate', 8.0),
                "use_custom_dates": inst.use_custom_date,
                "backtest_results": params.get('backtest_results', {}),
                "created_at": str(inst.created_at),
                "status": inst.status or 'deploy',
                "run_id": inst.run_id,
                "webhook_url": inst.webhook_url,
                "client_information_json": inst.client_info,
                "last_execution_date": str(inst.last_execution_date) if inst.last_execution_date else None,
                "next_execution_date": str(inst.next_execution_date) if inst.next_execution_date else None
            })
        
        return {"strategies": strategies}
        
    except Exception as e:
        logger.error(f"Error retrieving saved strategies: {e}")
        return {"strategies": []}
    finally:
        if session:
            session.close()

@etf_router.get("/get-saved-strategy/{strategy_id}")
async def get_saved_etf_strategy_by_id(strategy_id: int):
    """Get a specific saved ETF strategy by ID from unified table"""
    session = None
    try:
        from Services.strategy_manager.models import SavedInstance
        from Databases.app_data_db_connection import get_session
        
        session = get_session()
        inst = session.query(SavedInstance).filter(SavedInstance.id == strategy_id).first()
        
        if not inst:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        params = inst.strategies_parameters or {}
        
        strategy_response = {
            "id": inst.id,
            "strategy_name": inst.strategy_name,
            "strategy_type": inst.strategy_type,
            "user_id": inst.user_id,
            "tickers": json.loads(inst.tickers) if inst.tickers else [],
            "start_date": inst.start_date,
            "end_date": inst.end_date,
            "capital_per_week": params.get('capital_per_week', 0),
            "accumulation_weeks": params.get('accumulation_weeks', 0),
            "brokerage_percent": params.get('brokerage_percent', 0),
            "compounding_enabled": params.get('compounding_enabled', False),
            "risk_free_rate": params.get('risk_free_rate', 8.0),
            "use_custom_dates": inst.use_custom_date,
            "backtest_results": params.get('backtest_results', {}),
            "created_at": str(inst.created_at)
        }
        
        return {"strategy": strategy_response}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving strategy: {str(e)}")
    finally:
        if session:
            session.close()

@etf_router.get("/get-saved-strategies-count/{user_id}")
async def get_saved_etf_strategies_count(user_id: str):
    """Get count of saved ETF strategies for user from unified table"""
    session = None
    try:
        from Services.strategy_manager.models import SavedInstance
        from Databases.app_data_db_connection import get_session
        
        session = get_session()
        count = session.query(SavedInstance).filter(
            SavedInstance.user_id == user_id,
            SavedInstance.strategy_type.like('%ETF%')
        ).count()
        
        return {"count": count}
    except Exception as e:
        logger.error(f"Error getting strategy count: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting strategy count: {str(e)}")
    finally:
        if session:
            session.close()

@etf_router.get("/get-saved-strategies-table/{user_id}")
async def get_saved_etf_strategies_table(user_id: str):
    """Get saved ETF strategies in table format for unified dashboard"""
    session = None
    try:
        from Services.strategy_manager.models import SavedInstance
        from Databases.app_data_db_connection import get_session
        
        session = get_session()
        instances = session.query(SavedInstance).filter(
            SavedInstance.user_id == user_id,
            SavedInstance.strategy_type.like('%ETF%')
        ).all()
        
        strategies = []
        for inst in instances:
            params = inst.strategies_parameters or {}
            
            strategies.append({
                "id": inst.id,
                "strategy_name": inst.strategy_name,
                "strategy_type": inst.strategy_type,
                "user_id": inst.user_id,
                "start_date": inst.start_date,
                "end_date": inst.end_date,
                "backtest_results": params.get('backtest_results', {}),
                "strategy_config": params, # Everything else is in params
                "run_id": inst.run_id,
                "client_information_json": inst.client_info,
                "webhook_url": inst.webhook_url,
                "status": inst.status or 'deploy',
                "created_at": str(inst.created_at)
            })
        
        return {
            "success": True,
            "strategies": strategies
        }
    except Exception as e:
        logger.error(f"Error getting strategies table: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting ETF strategies table: {str(e)}")
    finally:
        if session:
            session.close()

@etf_router.post("/stop-etf-strategy")
async def stop_etf_strategy_v1(request: dict):
    """Stop a running ETF strategy (V1)"""
    session = None
    try:
        strategy_id = request.get("strategy_id")
        user_id = request.get("user_id")
        
        if not strategy_id or not user_id:
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        from Services.strategy_manager.models import SavedInstance
        from Databases.app_data_db_connection import get_session
        
        session = get_session()
        inst = session.query(SavedInstance).filter(
            SavedInstance.id == strategy_id, 
            SavedInstance.user_id == user_id
        ).first()
        
        if not inst:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        inst.status = 'stopped'
        session.commit()
        
        return {"success": True, "message": "ETF strategy stopped successfully"}
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
async def restart_etf_strategy_v1(request: dict):
    """Restart a stopped ETF strategy (V1)"""
    session = None
    try:
        strategy_id = request.get("strategy_id")
        user_id = request.get("user_id")
        
        if not strategy_id or not user_id:
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        from Services.strategy_manager.models import SavedInstance
        from Databases.app_data_db_connection import get_session
        
        session = get_session()
        inst = session.query(SavedInstance).filter(
            SavedInstance.id == strategy_id, 
            SavedInstance.user_id == user_id
        ).first()
        
        if not inst:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        inst.status = 'running'
        session.commit()
        
        return {"success": True, "message": "ETF strategy restarted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        if session:
            session.rollback()
        raise HTTPException(status_code=500, detail=f"Error restarting ETF strategy: {str(e)}")
    finally:
        if session:
            session.close()

@etf_router.post("/stop-strategy/{strategy_id}")
async def stop_etf_strategy_v2(strategy_id: int):
    """Stop a running ETF strategy (V2)"""
    session = None
    try:
        from Services.strategy_manager.models import SavedInstance
        from Databases.app_data_db_connection import get_session
        
        session = get_session()
        inst = session.query(SavedInstance).filter(SavedInstance.id == strategy_id).first()
        
        if not inst:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        inst.status = 'stopped'
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
        logger.error(f"Error stopping strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error stopping strategy: {str(e)}")
    finally:
        if session:
            session.close()

@etf_router.post("/restart-strategy/{strategy_id}")
async def restart_etf_strategy_v2(strategy_id: int):
    """Restart a stopped ETF strategy (V2)"""
    session = None
    try:
        from Services.strategy_manager.models import SavedInstance
        from Databases.app_data_db_connection import get_session
        
        session = get_session()
        inst = session.query(SavedInstance).filter(SavedInstance.id == strategy_id).first()
        
        if not inst:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        inst.status = 'running'
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
        logger.error(f"Error restarting strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error restarting strategy: {str(e)}")
    finally:
        if session:
            session.close()


# ============================================================================
# ETF SIGNAL GENERATION ENDPOINTS
# ============================================================================

# ============================================================================
# ETF SIGNAL RETRIEVAL ENDPOINTS
# ============================================================================

@etf_router.get("/etf-signals/recent")
async def get_recent_etf_signals(days: int = 7):
    """Get recent ETF signals from the unified TradingSignal table"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from Databases.signal_models import TradingSignal
        from datetime import datetime, timedelta
        
        session = get_session()
        cutoff = datetime.now() - timedelta(days=days)
        
        signals = session.query(TradingSignal).filter(
            TradingSignal.strategy_type.like('%ETF%'),
            TradingSignal.signal_date >= cutoff
        ).order_by(TradingSignal.signal_date.desc()).all()
        
        return {
            "success": True,
            "signals": [s.to_dict() for s in signals],
            "count": len(signals)
        }
    except Exception as e:
        logger.error(f"Error getting recent ETF signals: {e}")
        return {"success": False, "signals": [], "count": 0}
    finally:
        if session:
            session.close()

@etf_router.get("/etf-signals/run/{run_id}")
async def get_etf_signals_by_run_id(run_id: str):
    """Get ETF signals for a specific run_id from unified TradingSignal table"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from Databases.signal_models import TradingSignal
        
        session = get_session()
        signals = session.query(TradingSignal).filter(
            TradingSignal.run_id == run_id
        ).order_by(TradingSignal.signal_date.desc()).all()
        
        return {
            "success": True,
            "run_id": run_id,
            "signals": [s.to_dict() for s in signals],
            "count": len(signals),
            "buy_count": len([s for s in signals if s.order_side == 'BUY']),
            "sell_count": len([s for s in signals if s.order_side == 'SELL'])
        }
    except Exception as e:
        logger.error(f"Error getting signals for run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if session:
            session.close()

@etf_router.get("/etf-signals/runs")
async def get_etf_signal_runs(days: int = 30):
    """Get list of signal generation runs within the specified days by grouping TradingSignal entries"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from Databases.signal_models import TradingSignal
        from sqlalchemy import func
        from datetime import datetime, timedelta
        
        session = get_session()
        cutoff = datetime.now() - timedelta(days=days)
        
        # Group TradingSignal by run_id to simulate 'runs'
        runs_query = session.query(
            TradingSignal.run_id,
            func.min(TradingSignal.signal_date).label('run_date'),
            TradingSignal.strategy_name,
            func.count(TradingSignal.id).label('signal_count'),
            func.sum(func.case((TradingSignal.order_side == 'BUY', 1), else_=0)).label('buy_count'),
            func.sum(func.case((TradingSignal.order_side == 'SELL', 1), else_=0)).label('sell_count')
        ).filter(
            TradingSignal.strategy_type.like('%ETF%'),
            TradingSignal.signal_date >= cutoff
        ).group_by(
            TradingSignal.run_id, 
            TradingSignal.strategy_name
        ).order_by(func.min(TradingSignal.signal_date).desc())
        
        runs = []
        for run_id, run_date, strat_name, count, buys, sells in runs_query.all():
            runs.append({
                'run_id': run_id,
                'signal_date': str(run_date),
                'strategy_name': strat_name,
                'signal_count': count,
                'buy_count': int(buys or 0),
                'sell_count': int(sells or 0),
                'status': 'completed' # Logic assumes if they are in TradingSignal, it's done
            })
            
        return {
            "success": True,
            "runs": runs,
            "count": len(runs)
        }
    except Exception as e:
        logger.error(f"Error getting ETF signal runs: {e}")
        return {"success": False, "runs": [], "count": 0}
    finally:
        if session:
            session.close()

@etf_router.get("/etf-signals/latest")
async def get_latest_etf_signals():
    """Get the latest ETF signals from the most recent run in TradingSignal table"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from Databases.signal_models import TradingSignal
        
        session = get_session()
        
        # Find the latest run_id
        latest_run = session.query(TradingSignal.run_id).filter(
            TradingSignal.strategy_type.like('%ETF%')
        ).order_by(TradingSignal.signal_date.desc()).first()
        
        if not latest_run:
            return {
                "success": False,
                "message": "No signals found",
                "run_id": None,
                "signals": [],
                "count": 0
            }
        
        run_id = latest_run[0]
        signals = session.query(TradingSignal).filter(
            TradingSignal.run_id == run_id
        ).order_by(TradingSignal.id.asc()).all()
        
        return {
            "success": True,
            "run_id": run_id,
            "signal_date": str(signals[0].signal_date) if signals else None,
            "signals": [s.to_dict() for s in signals],
            "count": len(signals),
            "buy_count": len([s for s in signals if s.order_side == 'BUY']),
            "sell_count": len([s for s in signals if s.order_side == 'SELL'])
        }
    except Exception as e:
        logger.error(f"Error getting latest ETF signals: {e}")
        return {"success": False, "message": str(e)}
    finally:
        if session:
            session.close()

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
from ..services.backtester import InternationalETFRotationBacktester
# Signal generator removed - not needed
LiveSignalGenerator = None
from ..etf_schemas import (
    BacktestRequest, ETFMetadata, BacktestResult, BacktestResults,
    SaveETFStrategyRequest, SavedETFStrategy
)

# Create ETF router
international_etf_router = APIRouter(prefix="/api/international-etf", tags=["International ETF Strategy"])

# Pydantic models for request/response


# Global ETF backtester instance
international_etf_backtester = None

def initialize_international_etf_backtester(db_path: str = "unified_etf_data.sqlite"):
    """Initialize the ETF backtester
    
    Args:
        db_path: Deprecated - kept for compatibility. Now uses PostgreSQL for all operations.
    """
    global international_etf_backtester
    try:
        international_etf_backtester = InternationalETFRotationBacktester(db_path=db_path)  # db_path ignored, uses PostgreSQL
        print("ETF Backtester initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing ETF Backtester: {e}")
        international_etf_backtester = None
        return False

def cleanup_international_etf_backtester():
    """Clean up ETF backtester resources"""
    global international_etf_backtester
    if international_etf_backtester:
        international_etf_backtester.cleanup()
        international_etf_backtester = None

# ============================================================================
# ETF STRATEGY ROUTES
# ============================================================================

@international_etf_router.get("/etfs")
async def get_available_etfs():
    """Get list of available ETFs"""
    try:
        if international_etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        # Load ETF metadata
        metadata = international_etf_backtester.load_metadata()
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

@international_etf_router.get("/default")
async def get_default_etf_selection():
    """Get default ETF selection"""
    try:
        if international_etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        metadata = international_etf_backtester.load_metadata()
        available_etfs = list(metadata.keys())
        default_selection = international_etf_backtester.get_default_etf_selection(available_etfs, 5)
        
        return {"default_etfs": default_selection}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting default selection: {str(e)}")

@international_etf_router.post("/etfs/date-range")
async def calculate_etf_date_range(request: Dict[str, Any]):
    """Calculate common date range for selected ETFs"""
    try:
        if international_etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        tickers = request.get("tickers", [])
        if not tickers:
            raise HTTPException(status_code=400, detail="No tickers provided in request")
        
        print(f"Calculating date range for ETF tickers: {tickers}")
        
        # Enable verbose mode for debugging
        international_etf_backtester.set_verbose(True)
        
        start_date, end_date, years = international_etf_backtester.calculate_common_date_range(tickers)
        
        if start_date and end_date:
            return {
                "start_date": start_date,
                "end_date": end_date,
                "years": years
            }
        else:
            # Provide more detailed error message
            available_symbols = list(international_etf_backtester.etf_metadata.keys())[:20] if international_etf_backtester.etf_metadata else []
            error_msg = f"Could not calculate date range for ETFs: {tickers}. "
            if not international_etf_backtester.etf_metadata:
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

@international_etf_router.post("/diagnose")
async def diagnose_etf_data(request: Dict[str, Any]):
    """Diagnose ETF data availability and provide recommendations"""
    try:
        if international_etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        tickers = request.get("tickers", [])
        diagnosis = international_etf_backtester.diagnose_etf_data(tickers)
        return diagnosis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error diagnosing ETF data: {str(e)}")

@international_etf_router.get("/etfs/overview")
async def get_etf_overview():
    """Get ETF overview with descriptions"""
    try:
        if international_etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        metadata = international_etf_backtester.load_metadata()
        etf_overview = []
        
        for symbol, meta in metadata.items():
            description = international_etf_backtester.generate_asset_description(symbol)
            sector = international_etf_backtester.get_asset_sector_classification(symbol)
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

@international_etf_router.post("/metrics")
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
        
        if international_etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        print(f"Running ETF backtest with parameters: {request}")
        
        # Run the backtest
        result = international_etf_backtester.run_backtest(
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
        etf_metrics = international_etf_backtester.calculate_metrics(
            request.capital_per_week,
            request.accumulation_weeks,
            request.risk_free_rate
        )
        
        # Calculate benchmark metrics
        total_investment = request.accumulation_weeks * request.capital_per_week
        benchmark_metrics = international_etf_backtester.calculate_benchmark_metrics(
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
        
        if not international_etf_backtester.weekly_nav_df.empty:
            performance_data["dates"] = [str(date) for date in international_etf_backtester.weekly_nav_df['date']]
            performance_data["etf_strategy"] = international_etf_backtester.weekly_nav_df['nav'].tolist()
            performance_data["cumulative_investment"] = international_etf_backtester.weekly_nav_df['cumulative_investment'].tolist()
            
            if not international_etf_backtester.sp500_df.empty:
                # Align benchmark data with weekly data
                benchmark_dates = [str(date) for date in international_etf_backtester.sp500_df['date']]
                benchmark_navs = international_etf_backtester.sp500_df['nav'].tolist()
                performance_data["benchmark_buyhold"] = benchmark_navs
        
        # Get purchase limit status
        purchase_limit_status = international_etf_backtester.get_purchase_limit_status()
        
        # Sanitize all data before returning
        response_data = {
            "success": True,
            "etf_metrics": sanitize_data(etf_metrics),
            "benchmark_metrics": sanitize_data(benchmark_metrics),
            "backtest_result": sanitize_data(result),
            "performance_data": sanitize_data(performance_data),
            "purchase_limit_status": sanitize_data(purchase_limit_status)
        }
        
        return response_data
        
    except Exception as e:
        print(f"Error calculating ETF metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error calculating ETF metrics: {str(e)}")

@international_etf_router.get("/metrics/table")
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
        
        if international_etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        # This would need to be called after a backtest is run
        if not hasattr(international_etf_backtester, 'weekly_nav_df') or international_etf_backtester.weekly_nav_df is None:
            raise HTTPException(status_code=400, detail="No ETF backtest data available. Run backtest first.")
        
        # Calculate metrics
        etf_metrics = international_etf_backtester.calculate_metrics(50000, 52, 8.0)  # Default values
        total_investment = 52 * 50000
        benchmark_metrics = international_etf_backtester.calculate_benchmark_metrics(total_investment, 8.0)
        
        # Create formatted table
        formatted_table = international_etf_backtester.create_formatted_metrics_table(etf_metrics, benchmark_metrics)
        
        if not formatted_table.empty:
            table_data = formatted_table.to_dict('records')
            return {"metrics_table": sanitize_data(table_data)}
        else:
            return {"metrics_table": []}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting ETF metrics table: {str(e)}")

@international_etf_router.get("/transaction-costs/summary")
async def get_etf_transaction_costs_summary():
    """Get transaction costs summary for ETFs"""
    try:
        if international_etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        if not international_etf_backtester.transaction_costs_log:
            return {"costs_summary": {
                'Total All Costs': '₹0',
                'Capital Gains Tax': '₹0',
                'Cost as % of Volume': '0.00%',
                'Total Transactions': '0'
            }}
        
        costs_summary = international_etf_backtester.get_transaction_costs_summary()
        return {"costs_summary": costs_summary}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting ETF transaction costs summary: {str(e)}")

@international_etf_router.get("/transaction-log")
async def get_etf_transaction_log():
    """Get transaction log from the latest ETF backtest"""
    try:
        import math
        
        if international_etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        if not hasattr(international_etf_backtester, 'portfolio_log') or not international_etf_backtester.portfolio_log:
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
        for log in international_etf_backtester.portfolio_log:
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
            'no_trade_weeks': getattr(international_etf_backtester, 'skipped_days', []),
            'trading_frequency': f"{(total_trades / max(1, len(international_etf_backtester.portfolio_log))) * 100:.1f}%"
        }
        
        return {
            "transaction_log": transaction_log,
            "trading_summary": trading_summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading ETF transaction log: {str(e)}")

@international_etf_router.get("/debug/portfolio-log")
async def debug_portfolio_log():
    """Debug endpoint to inspect raw portfolio_log data"""
    try:
        import math
        
        if international_etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized")
        
        if not hasattr(international_etf_backtester, 'portfolio_log') or not international_etf_backtester.portfolio_log:
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
        for i, log in enumerate(international_etf_backtester.portfolio_log):
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

@international_etf_router.get("/debug/backtest-state")
async def debug_backtest_state():
    """Enhanced debug endpoint to check complete backtester state"""
    try:
        if international_etf_backtester is None:
            return {
                "error": "Backtester not initialized",
                "backtester_exists": False
            }
        
        # Gather comprehensive state information
        state = {
            "backtester_exists": True,
            "backtester_type": type(international_etf_backtester).__name__,
            "has_portfolio_log": hasattr(international_etf_backtester, 'portfolio_log'),
            "portfolio_log_count": len(international_etf_backtester.portfolio_log) if hasattr(international_etf_backtester, 'portfolio_log') else 0,
            "has_transaction_costs_log": hasattr(international_etf_backtester, 'transaction_costs_log'),
            "transaction_costs_count": len(international_etf_backtester.transaction_costs_log) if hasattr(international_etf_backtester, 'transaction_costs_log') else 0,
            "has_skipped_days": hasattr(international_etf_backtester, 'skipped_days'),
            "skipped_days_count": len(international_etf_backtester.skipped_days) if hasattr(international_etf_backtester, 'skipped_days') else 0,
            "has_weekly_nav_df": hasattr(international_etf_backtester, 'weekly_nav_df'),
            "weekly_nav_rows": len(international_etf_backtester.weekly_nav_df) if hasattr(international_etf_backtester, 'weekly_nav_df') and international_etf_backtester.weekly_nav_df is not None else 0,
            "total_weeks": getattr(international_etf_backtester, 'total_weeks', 0),
            "successful_signals": getattr(international_etf_backtester, 'successful_signals', 0),
            "successful_executions": getattr(international_etf_backtester, 'successful_executions', 0),
            "current_cash": getattr(international_etf_backtester, 'current_cash', 0),
            "current_holdings": getattr(international_etf_backtester, 'current_holdings', {}),
            "etf_metadata_count": len(international_etf_backtester.etf_metadata) if hasattr(international_etf_backtester, 'etf_metadata') else 0
        }
        
        # Add sample of portfolio_log if available
        if state["portfolio_log_count"] > 0:
            sample_logs = []
            for i, log in enumerate(international_etf_backtester.portfolio_log[:3]):  # First 3 entries
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

@international_etf_router.get("/transaction-costs")
async def get_etf_transaction_costs():
    """Get transaction costs data from the latest ETF backtest"""
    try:
        if international_etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        if not hasattr(international_etf_backtester, 'transaction_costs_log') or not international_etf_backtester.transaction_costs_log:
            return {"transaction_costs": []}
        
        # Convert transaction costs log to frontend format
        transaction_costs = []
        for cost in international_etf_backtester.transaction_costs_log:
            transaction_costs.append({
                'date': cost.get('date', '').strftime('%Y-%m-%d') if hasattr(cost.get('date', ''), 'strftime') else str(cost.get('date', '')),
                'cumulative_cost': cost.get('cumulative_costs', 0),
                'weekly_cost': cost.get('weekly_costs', 0),
                'total_costs': cost.get('total_costs', 0)
            })
        
        return {"transaction_costs": transaction_costs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading ETF transaction costs: {str(e)}")

@international_etf_router.get("/skipped-trades")
async def get_etf_skipped_trades():
    """Get skipped trades information from the latest ETF backtest"""
    try:
        if international_etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        if not hasattr(international_etf_backtester, 'skipped_days') or not international_etf_backtester.skipped_days:
            return {"skipped_trades": []}
        
        # Convert skipped days to frontend format
        skipped_trades = []
        for skip in international_etf_backtester.skipped_days:
            skipped_trades.append({
                'week': skip.get('week', 0),
                'date': skip.get('date', ''),
                'signal_date': skip.get('signal_date', 'N/A'),
                'reason': skip.get('reason', 'Unknown')
            })
        
        return {"skipped_trades": skipped_trades}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading ETF skipped trades: {str(e)}")

@international_etf_router.get("/trade-execution-status")
async def get_etf_trade_execution_status():
    """Get real-time trade execution status and statistics for ETFs"""
    try:
        if international_etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        # Get current backtest statistics
        stats = {
            'total_weeks_processed': getattr(international_etf_backtester, 'total_weeks', 0),
            'successful_signals': getattr(international_etf_backtester, 'successful_signals', 0),
            'successful_executions': getattr(international_etf_backtester, 'successful_executions', 0),
            'portfolio_log_entries': len(getattr(international_etf_backtester, 'portfolio_log', [])),
            'transaction_costs_entries': len(getattr(international_etf_backtester, 'transaction_costs_log', [])),
            'skipped_trades_count': len(getattr(international_etf_backtester, 'skipped_days', [])),
            'current_cash': getattr(international_etf_backtester, 'current_cash', 0),
            'current_holdings': getattr(international_etf_backtester, 'current_holdings', {}),
            'last_trade_date': None,
            'last_trade_action': None,
            'last_trade_ticker': None
        }
        
        # Get last trade information
        if international_etf_backtester.portfolio_log:
            last_trade = international_etf_backtester.portfolio_log[-1]
            stats['last_trade_date'] = last_trade.get('execution_date', '').strftime('%Y-%m-%d') if hasattr(last_trade.get('execution_date', ''), 'strftime') else str(last_trade.get('execution_date', ''))
            stats['last_trade_action'] = last_trade.get('action', 'NONE')
            stats['last_trade_ticker'] = last_trade.get('ticker', '')
        
        return {"trade_execution_status": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading ETF trade execution status: {str(e)}")

@international_etf_router.get("/charts/equity-curve")
async def get_etf_equity_curve_chart(show_benchmark: bool = True, show_etf_strategy: bool = True):
    """Get equity curve chart data for ETFs"""
    try:
        if international_etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        if not hasattr(international_etf_backtester, 'weekly_nav_df') or international_etf_backtester.weekly_nav_df is None:
            raise HTTPException(status_code=400, detail="No ETF backtest data available. Run backtest first.")
        
        # Return data for frontend charting
        if not international_etf_backtester.weekly_nav_df.empty:
            chart_data = {
                "dates": [str(date) for date in international_etf_backtester.weekly_nav_df['date']],
                "etf_strategy": international_etf_backtester.weekly_nav_df['nav'].tolist(),
                "cumulative_investment": international_etf_backtester.weekly_nav_df['cumulative_investment'].tolist(),
                "benchmark_buyhold": []
            }
            
            if not international_etf_backtester.sp500_df.empty:
                chart_data["benchmark_buyhold"] = international_etf_backtester.sp500_df['nav'].tolist()
            
            return {"chart_data": chart_data}
        else:
            return {"chart_data": {}}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting ETF equity curve chart: {str(e)}")

@international_etf_router.get("/charts/transaction-costs")
async def get_etf_transaction_costs_chart():
    """Get transaction costs chart data for ETFs"""
    try:
        if international_etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        if not international_etf_backtester.transaction_costs_log:
            return {"chart_data": {}}
        
        # Return data for frontend charting
        costs_df = pd.DataFrame(international_etf_backtester.transaction_costs_log)
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

@international_etf_router.post("/cleanup")
async def cleanup_etf_resources():
    """Clean up ETF resources and clear cache"""
    try:
        cleanup_international_etf_backtester()
        return {"success": True, "message": "ETF resources cleaned up successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning up ETF resources: {str(e)}")

@international_etf_router.get("/costs/summary")
async def get_etf_costs_summary():
    """Get comprehensive costs summary including transaction costs and capital gains tax for ETFs"""
    try:
        if international_etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        if not hasattr(international_etf_backtester, 'portfolio_log') or not international_etf_backtester.portfolio_log:
            return {
                "total_all_costs": 0,
                "capital_gains_tax": 0,
                "transaction_costs": 0,
                "cost_as_percent_of_volume": 0,
                "total_transactions": 0,
                "total_volume": 0
            }
        
        # Calculate costs from portfolio log - FIXED: Extract transaction costs from costs dictionary
        total_capital_gains_tax = sum(log.get('capital_gains_tax', 0) for log in international_etf_backtester.portfolio_log)
        
        # Fix: Extract transaction costs from the costs dictionary, not directly from log
        total_transaction_costs = 0
        for log in international_etf_backtester.portfolio_log:
            costs = log.get('costs', {})
            transaction_cost = costs.get('total_costs', 0) if costs else 0
            total_transaction_costs += transaction_cost
        
        total_all_costs = total_capital_gains_tax + total_transaction_costs
        
        # Calculate total volume (sum of all transaction amounts)
        total_volume = sum(log.get('amount', 0) for log in international_etf_backtester.portfolio_log)
        
        # Calculate cost as percentage of volume
        cost_as_percent = (total_all_costs / total_volume * 100) if total_volume > 0 else 0
        
        # Count total transactions
        total_transactions = len(international_etf_backtester.portfolio_log)
        
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

@international_etf_router.get("/costs/analysis")
async def get_etf_costs_analysis():
    """Get detailed costs analysis over time for the ETF chart"""
    try:
        if international_etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        if not hasattr(international_etf_backtester, 'portfolio_log') or not international_etf_backtester.portfolio_log:
            return {"costs_data": []}
        
        # Create cumulative costs data over time
        costs_data = []
        cumulative_transaction_costs = 0
        cumulative_capital_gains_tax = 0
        cumulative_total_costs = 0
        
        # Group by date and calculate cumulative costs
        date_costs = {}
        for log in international_etf_backtester.portfolio_log:
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

@international_etf_router.get("/costs/breakdown")
async def get_etf_costs_breakdown():
    """Get detailed breakdown of costs by type and period for ETFs"""
    try:
        if international_etf_backtester is None:
            raise HTTPException(status_code=500, detail="ETF backtester not initialized. Check database connection.")
        
        if not hasattr(international_etf_backtester, 'portfolio_log') or not international_etf_backtester.portfolio_log:
            return {"breakdown": {}}
        
        # Calculate costs by year
        yearly_costs = {}
        for log in international_etf_backtester.portfolio_log:
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
    """
    Deprecated: No-op for compatibility. 
    Tables are now managed by centralized init_database().
    """
    logger.info("Database initialization handled by centralized service.")
    return True, None

# ============================================================================
# SAVED STRATEGY ROUTES
# ============================================================================

@international_etf_router.post("/save-strategy")
async def save_etf_strategy(request: SaveETFStrategyRequest):
    """Save an International ETF strategy to the unified saved_instances table"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from Services.strategy_manager.models import SavedInstance
        from sqlalchemy import and_
        
        session = get_session()
        
        # Map tickers to a consistent format (string)
        tickers_str = ",".join(request.tickers)
        
        # Map performance metrics for consistent querying/display
        results = request.backtest_results
        
        # Check if strategy exists
        existing = session.query(SavedInstance).filter(
            and_(
                SavedInstance.user_id == request.user_id,
                SavedInstance.strategy_name == request.strategy_name,
                SavedInstance.strategy_type == request.strategy_type
            )
        ).first()

        if existing:
            return {
                "success": False,
                "message": f"Strategy '{request.strategy_name}' already exists",
                "strategy_exists": True
            }

        # Create new strategy instance
        new_strategy = SavedInstance(
            user_id=request.user_id,
            strategy_name=request.strategy_name,
            strategy_type=request.strategy_type,
            tickers=tickers_str,
            benchmark="S&P_500",  # International ETF benchmark
            start_date=request.start_date,
            end_date=request.end_date,
            status=request.status or "deploy",
            # Pack all specific parameters into JSON
            strategies_parameters={
                "capital_per_week": request.capital_per_week,
                "accumulation_weeks": request.accumulation_weeks,
                "brokerage_percent": request.brokerage_percent,
                "compounding_enabled": request.compounding_enabled,
                "risk_free_rate": request.risk_free_rate,
                "use_custom_dates": request.use_custom_dates,
                "tickers": request.tickers
            },
            # Map backtest results to standard fields
            cagr=results.cagr_pct,
            sharpe_ratio=results.sharpe_ratio,
            max_drawdown=results.max_drawdown_pct,
            total_return_pct=results.total_return_pct,
            # Store full results for frontend
            backtest_results_json=results.dict()
        )

        session.add(new_strategy)
        session.commit()
        
        return {
            "success": True,
            "message": "International ETF Strategy saved successfully",
            "strategy_id": new_strategy.id,
            "strategy_exists": False
        }
        
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Error saving International ETF strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if session:
            session.close()

@international_etf_router.get("/get-saved-strategies-list/{user_id}")
async def get_saved_etf_strategies(user_id: str):
    """Get all saved International ETF strategies for a specific user"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from Services.strategy_manager.models import SavedInstance
        
        session = get_session()
        instances = session.query(SavedInstance).filter(
            SavedInstance.user_id == user_id,
            SavedInstance.strategy_type == 'Rotation_International_ETF'
        ).order_by(SavedInstance.created_at.desc()).all()
        
        strategies = []
        for instance in instances:
            strategies.append({
                "id": instance.id,
                "strategy_name": instance.strategy_name,
                "strategy_type": instance.strategy_type,
                "user_id": instance.user_id,
                "tickers": instance.tickers.split(',') if instance.tickers else [],
                "start_date": instance.start_date,
                "end_date": instance.end_date,
                "capital_per_week": instance.strategies_parameters.get('capital_per_week') if instance.strategies_parameters else 0,
                "accumulation_weeks": instance.strategies_parameters.get('accumulation_weeks') if instance.strategies_parameters else 0,
                "brokerage_percent": instance.strategies_parameters.get('brokerage_percent') if instance.strategies_parameters else 0,
                "compounding_enabled": instance.strategies_parameters.get('compounding_enabled') if instance.strategies_parameters else False,
                "risk_free_rate": instance.strategies_parameters.get('risk_free_rate') if instance.strategies_parameters else 0,
                "use_custom_dates": instance.strategies_parameters.get('use_custom_dates') if instance.strategies_parameters else False,
                "backtest_results": instance.backtest_results_json or {},
                "created_at": instance.created_at.strftime('%Y-%m-%d %H:%M:%S') if instance.created_at else None,
                "status": instance.status,
                "run_id": instance.run_id
            })
            
        return {"strategies": strategies}
    except Exception as e:
        logger.error(f"Error retrieving strategies for user {user_id}: {e}")
        return {"strategies": []}
    finally:
        if session:
            session.close()

@international_etf_router.get("/get-saved-strategy/{strategy_id}")
async def get_saved_etf_strategy_by_id(strategy_id: int):
    """Get a specific saved International ETF strategy by ID"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from Services.strategy_manager.models import SavedInstance
        
        session = get_session()
        instance = session.query(SavedInstance).filter(SavedInstance.id == strategy_id).first()
        
        if not instance:
            raise HTTPException(status_code=404, detail="Strategy not found")
            
        strategy_response = {
            "id": instance.id,
            "strategy_name": instance.strategy_name,
            "strategy_type": instance.strategy_type,
            "user_id": instance.user_id,
            "tickers": instance.tickers.split(',') if instance.tickers else [],
            "start_date": instance.start_date,
            "end_date": instance.end_date,
            "capital_per_week": instance.strategies_parameters.get('capital_per_week') if instance.strategies_parameters else 0,
            "accumulation_weeks": instance.strategies_parameters.get('accumulation_weeks') if instance.strategies_parameters else 0,
            "brokerage_percent": instance.strategies_parameters.get('brokerage_percent') if instance.strategies_parameters else 0,
            "compounding_enabled": instance.strategies_parameters.get('compounding_enabled') if instance.strategies_parameters else False,
            "risk_free_rate": instance.strategies_parameters.get('risk_free_rate') if instance.strategies_parameters else 0,
            "use_custom_dates": instance.strategies_parameters.get('use_custom_dates') if instance.strategies_parameters else False,
            "backtest_results": instance.backtest_results_json or {},
            "created_at": instance.created_at.strftime('%Y-%m-%d %H:%M:%S') if instance.created_at else None,
            "status": instance.status,
            "run_id": instance.run_id
        }
        
        return {"strategy": strategy_response}
    except Exception as e:
        logger.error(f"Error retrieving strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if session:
            session.close()





@international_etf_router.get("/get-saved-strategies-count/{user_id}")
async def get_saved_etf_strategies_count(user_id: str):
    """Get count of saved International ETF strategies for a user"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from Services.strategy_manager.models import SavedInstance
        
        session = get_session()
        count = session.query(SavedInstance).filter(
            SavedInstance.user_id == user_id,
            SavedInstance.strategy_type == 'Rotation_International_ETF'
        ).count()
        
        return {"count": count}
    except Exception as e:
        logger.error(f"Error getting strategy count for user {user_id}: {e}")
        return {"count": 0}
    finally:
        if session:
            session.close()

# ============================================================================
# NEW RS-STYLE ENDPOINTS FOR ETF STRATEGIES
# ============================================================================

@international_etf_router.get("/get-saved-strategies-table/{user_id}")
async def get_saved_etf_strategies_table(user_id: str):
    """Get saved International ETF strategies in table format"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from Services.strategy_manager.models import SavedInstance
        
        session = get_session()
        instances = session.query(SavedInstance).filter(
            SavedInstance.user_id == user_id,
            SavedInstance.strategy_type == 'Rotation_International_ETF'
        ).order_by(SavedInstance.created_at.desc()).all()
        
        strategies = []
        for instance in instances:
            strategies.append({
                "id": instance.id,
                "strategy_name": instance.strategy_name,
                "strategy_type": instance.strategy_type,
                "user_id": instance.user_id,
                "tickers": instance.tickers.split(',') if instance.tickers else [],
                "start_date": instance.start_date,
                "end_date": instance.end_date,
                "status": instance.status,
                "backtest_results": instance.backtest_results_json or {},
                "created_at": instance.created_at.strftime('%Y-%m-%d %H:%M:%S') if instance.created_at else None,
                "run_id": instance.run_id
            })
        
        return {
            "success": True,
            "strategies": strategies
        }
    except Exception as e:
        logger.error(f"Error getting strategy table for user {user_id}: {e}")
        return {"success": False, "strategies": [], "error": str(e)}
    finally:
        if session:
            session.close()

@international_etf_router.post("/stop-etf-strategy")
async def stop_etf_strategy_batch(request: dict):
    """Stop an International ETF strategy (batch/body request)"""
    session = None
    try:
        strategy_id = request.get("strategy_id")
        user_id = request.get("user_id")
        
        from Databases.app_data_db_connection import get_session
        from Services.strategy_manager.models import SavedInstance
        
        session = get_session()
        instance = session.query(SavedInstance).filter(
            SavedInstance.id == strategy_id,
            SavedInstance.user_id == user_id
        ).first()
        
        if not instance:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        instance.status = 'stopped'
        session.commit()
        
        return {"success": True, "message": "Strategy stopped successfully"}
    except Exception as e:
        if session: session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if session: session.close()

@international_etf_router.post("/restart-etf-strategy")
async def restart_etf_strategy_batch(request: dict):
    """Restart an International ETF strategy (batch/body request)"""
    session = None
    try:
        strategy_id = request.get("strategy_id")
        user_id = request.get("user_id")
        
        from Databases.app_data_db_connection import get_session
        from Services.strategy_manager.models import SavedInstance
        
        session = get_session()
        instance = session.query(SavedInstance).filter(
            SavedInstance.id == strategy_id,
            SavedInstance.user_id == user_id
        ).first()
        
        if not instance:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        instance.status = 'deploy'
        session.commit()
        
        return {"success": True, "message": "Strategy restarted successfully"}
    except Exception as e:
        if session: session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if session: session.close()



@international_etf_router.post("/stop-strategy/{strategy_id}")
async def stop_etf_strategy(strategy_id: int):
    """Stop a specific International ETF strategy by ID"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from Services.strategy_manager.models import SavedInstance
        
        session = get_session()
        instance = session.query(SavedInstance).filter(SavedInstance.id == strategy_id).first()
        
        if not instance:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        instance.status = 'stopped'
        session.commit()
        
        return {"success": True, "message": "Strategy stopped successfully"}
    except Exception as e:
        if session: session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if session: session.close()

@international_etf_router.post("/restart-strategy/{strategy_id}")
async def restart_etf_strategy(strategy_id: int):
    """Restart a specific International ETF strategy by ID"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from Services.strategy_manager.models import SavedInstance
        
        session = get_session()
        instance = session.query(SavedInstance).filter(SavedInstance.id == strategy_id).first()
        
        if not instance:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        instance.status = 'deploy'
        session.commit()
        
        return {"success": True, "message": "Strategy restarted successfully"}
    except Exception as e:
        if session: session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if session: session.close()



# ============================================================================
# ETF SIGNAL GENERATION ENDPOINTS
# ============================================================================

@international_etf_router.post("/etf-signals/generate")
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


@international_etf_router.get("/etf-signals/recent")
async def get_recent_etf_signals(days: int = 7):
    """Get recent International ETF signals from TradingSignal table"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from Databases.signal_models import TradingSignal
        from sqlalchemy import and_
        
        session = get_session()
        cutoff = datetime.now() - timedelta(days=days)
        
        signals = session.query(TradingSignal).filter(
            and_(
                TradingSignal.strategy_type.ilike('%ETF%'),
                TradingSignal.signal_date >= cutoff.date()
            )
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
        if session: session.close()


@international_etf_router.get("/etf-signals/run/{run_id}")
async def get_etf_signals_by_run_id(run_id: str):
    """Get International ETF signals for a specific run_id"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from Databases.signal_models import TradingSignal
        
        session = get_session()
        signals = session.query(TradingSignal).filter(
            TradingSignal.run_id == run_id
        ).all()
        
        signals_list = [s.to_dict() for s in signals]
        buy_count = len([s for s in signals_list if s.get('order_side') == 'BUY'])
        sell_count = len([s for s in signals_list if s.get('order_side') == 'SELL'])
        
        return {
            "success": True,
            "run_id": run_id,
            "signals": signals_list,
            "count": len(signals_list),
            "buy_count": buy_count,
            "sell_count": sell_count
        }
    except Exception as e:
        logger.error(f"Error getting signals for run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if session: session.close()


@international_etf_router.get("/etf-signals/runs")
async def get_etf_signal_runs(days: int = 30):
    """Get list of International ETF signal runs"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from Databases.signal_models import TradingSignal
        from sqlalchemy import func
        
        session = get_session()
        cutoff = datetime.now() - timedelta(days=days)
        
        # Group by run_id to simulate runs
        runs_query = session.query(
            TradingSignal.run_id,
            func.max(TradingSignal.signal_date).label('signal_date'),
            func.count(TradingSignal.id).label('signal_count'),
            func.sum(func.case((TradingSignal.order_side == 'BUY', 1), else_=0)).label('buy_count'),
            func.sum(func.case((TradingSignal.order_side == 'SELL', 1), else_=0)).label('sell_count')
        ).filter(
            TradingSignal.strategy_type.ilike('%ETF%'),
            TradingSignal.signal_date >= cutoff.date()
        ).group_by(TradingSignal.run_id).order_by(func.max(TradingSignal.signal_date).desc())
        
        runs = []
        for row in runs_query.all():
            runs.append({
                "run_id": row.run_id,
                "signal_date": str(row.signal_date),
                "status": "completed",
                "signal_count": row.signal_count,
                "buy_count": int(row.buy_count) if row.buy_count else 0,
                "sell_count": int(row.sell_count) if row.sell_count else 0
            })
            
        return {"success": True, "runs": runs, "count": len(runs)}
    except Exception as e:
        logger.error(f"Error getting ETF runs: {e}")
        return {"success": False, "runs": [], "count": 0}
    finally:
        if session: session.close()


@international_etf_router.get("/etf-signals/latest")
async def get_latest_etf_signals():
    """Get the latest International ETF signals"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from Databases.signal_models import TradingSignal
        
        session = get_session()
        # Find latest run_id
        latest_run = session.query(TradingSignal.run_id).filter(
            TradingSignal.strategy_type.ilike('%ETF%')
        ).order_by(TradingSignal.signal_date.desc()).first()
        
        if not latest_run:
            return {"success": False, "message": "No signals found", "signals": [], "count": 0}
            
        run_id = latest_run[0]
        signals = session.query(TradingSignal).filter(TradingSignal.run_id == run_id).all()
        signals_list = [s.to_dict() for s in signals]
        
        buy_count = len([s for s in signals_list if s.get('order_side') == 'BUY'])
        sell_count = len([s for s in signals_list if s.get('order_side') == 'SELL'])
        
        return {
            "success": True,
            "run_id": run_id,
            "signals": signals_list,
            "count": len(signals_list),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "signal_date": str(signals[0].signal_date) if signals else None
        }
    except Exception as e:
        logger.error(f"Error getting latest ETF signals: {e}")
        return {"success": False, "message": str(e)}
    finally:
        if session: session.close()

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
import pandas as pd
import sys
import os
import json
import logging
import uuid
import yfinance as yf
from datetime import datetime, timedelta

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import the stock backtester
from ..services.backtester import StockRotationBacktester

# Import the stock signal generator
try:
    from ..services.signal_generator import LiveStockSignalGenerator
except ImportError:
    LiveStockSignalGenerator = None

# Import schemas
from ..stock_schemas import (
    BacktestRequest, StockMetadata, BacktestResult, BacktestResults,
    DeployStockRequest, DeployStockResponse, SaveStockStrategyRequest, SavedStockStrategy
)

# Create stock router
stock_router = APIRouter(prefix="/api/stocks", tags=["Stock Strategy"])

# Configure logging
logger = logging.getLogger(__name__)

# Global stock backtester instance
stock_backtester = None

def initialize_stock_backtester(db_path: str = "unified_etf_data.sqlite"):
    """Initialize the stock backtester
    
    Args:
        db_path: Deprecated - kept for compatibility. Now uses PostgreSQL for all operations.
    """
    global stock_backtester
    try:
        stock_backtester = StockRotationBacktester(db_path=db_path)  # db_path ignored, uses PostgreSQL
        print("Stock Backtester initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing Stock Backtester: {e}")
        stock_backtester = None
        return False

def cleanup_stock_backtester():
    """Clean up stock backtester resources"""
    global stock_backtester
    if stock_backtester:
        stock_backtester.cleanup()
        stock_backtester = None

# ============================================================================
# STOCK STRATEGY ROUTES
# ============================================================================

@stock_router.get("/")
async def get_available_stocks():
    """Get list of available stocks"""
    try:
        if stock_backtester is None:
            raise HTTPException(status_code=500, detail="Stock backtester not initialized. Check database connection.")
        
        # Load Stock metadata
        metadata = stock_backtester.load_metadata()
        stocks = []
        
        for ticker, data in metadata.items():
            stocks.append({
                "ticker": ticker,
                "name": data.get('name', ticker),
                "category": data.get('category', 'Unknown'),
                "expense_ratio": data.get('expense_ratio', 0.0),
                "aum": data.get('aum', 0.0)
            })
        
        return {"stocks": stocks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading stocks: {str(e)}")

@stock_router.get("/default")
async def get_default_stock_selection():
    """Get default stock selection"""
    try:
        if stock_backtester is None:
            raise HTTPException(status_code=500, detail="Stock backtester not initialized. Check database connection.")
        
        metadata = stock_backtester.load_metadata()
        available_stocks = list(metadata.keys())
        default_selection = stock_backtester.get_default_stock_selection(available_stocks, 5)
        
        return {"default_stocks": default_selection}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting default selection: {str(e)}")

@stock_router.post("/date-range")
async def calculate_stock_date_range(request: Dict[str, Any]):
    """Calculate common date range for selected stocks"""
    try:
        if stock_backtester is None:
            raise HTTPException(status_code=500, detail="Stock backtester not initialized. Check database connection.")
        
        tickers = request.get("tickers", [])
        print(f"Calculating date range for stock tickers: {tickers}")
        start_date, end_date, years = stock_backtester.calculate_common_date_range(tickers)
        
        if start_date and end_date:
            return {
                "start_date": start_date,
                "end_date": end_date,
                "years": years
            }
        else:
            raise HTTPException(status_code=400, detail="Could not calculate date range. Please try different stocks.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating date range: {str(e)}")

@stock_router.post("/diagnose")
async def diagnose_stock_data(request: Dict[str, Any]):
    """Diagnose stock data availability and provide recommendations"""
    try:
        if stock_backtester is None:
            raise HTTPException(status_code=500, detail="Stock backtester not initialized. Check database connection.")
        
        tickers = request.get("tickers", [])
        # Note: diagnose_stock_data was not in the original stockbacktester_core.py I read, 
        # but it was called in stock_api.py. I might have missed it or it's missing in the core.
        # Checking stockbacktester_core.py again... it wasn't there.
        # If it's missing, I should probably remove this endpoint or implement it.
        # For now, I'll comment it out or return a dummy response if method doesn't exist.
        if hasattr(stock_backtester, 'diagnose_stock_data'):
            diagnosis = stock_backtester.diagnose_stock_data(tickers)
            return diagnosis
        else:
             return {"status": "Not implemented in backtester"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error diagnosing stock data: {str(e)}")

@stock_router.get("/overview")
async def get_stock_overview():
    """Get stock overview with descriptions"""
    try:
        if stock_backtester is None:
            raise HTTPException(status_code=500, detail="Stock backtester not initialized. Check database connection.")
        
        metadata = stock_backtester.load_metadata()
        stock_overview = []
        
        for symbol, meta in metadata.items():
            description = stock_backtester.generate_asset_description(symbol)
            sector = stock_backtester.get_asset_sector_classification(symbol)
            stock_overview.append({
                'symbol': symbol,
                'description': description,
                'sector': sector,
                'start_date': meta['start_date'],
                'end_date': meta['end_date'],
                'years_available': round(meta['years_available'], 1),
                'total_records': meta['total_records']
            })
        
        # Sort by start date
        stock_overview.sort(key=lambda x: x['start_date'])
        return {"stock_overview": stock_overview}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stock overview: {str(e)}")

@stock_router.post("/metrics")
async def calculate_stock_metrics(request: BacktestRequest):
    """Calculate performance metrics for stock rotation strategy"""
    try:
        if stock_backtester is None:
            raise HTTPException(status_code=500, detail="Stock backtester not initialized. Check database connection.")
        
        print(f"Running stock backtest with parameters: {request}")
        
        # Run the backtest
        result = stock_backtester.run_backtest(
            tickers=request.tickers,
            start_date=request.start_date,
            end_date=request.end_date,
            capital_per_week=request.capital_per_week,
            accumulation_weeks=request.accumulation_weeks,
            brokerage_percent=request.brokerage_percent,
            compounding_enabled=request.compounding_enabled
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=f"Stock backtest failed: {result['error']}")
        
        # Calculate metrics
        stock_metrics = stock_backtester.calculate_metrics(
            request.capital_per_week,
            request.accumulation_weeks,
            request.risk_free_rate
        )
        
        # Calculate benchmark metrics
        total_investment = request.accumulation_weeks * request.capital_per_week
        benchmark_metrics = stock_backtester.calculate_benchmark_metrics(
            total_investment,
            request.risk_free_rate
        )
        
        # Prepare performance data for charts
        performance_data = {
            "dates": [],
            "stock_strategy": [],
            "cumulative_investment": [],
            "benchmark_buyhold": []
        }
        
        if not stock_backtester.weekly_nav_df.empty:
            performance_data["dates"] = [str(date) for date in stock_backtester.weekly_nav_df['date']]
            performance_data["stock_strategy"] = stock_backtester.weekly_nav_df['nav'].tolist()
            performance_data["cumulative_investment"] = stock_backtester.weekly_nav_df['cumulative_investment'].tolist()
            
            if not stock_backtester.nifty50_df.empty:
                # Align benchmark data with weekly data
                benchmark_dates = [str(date) for date in stock_backtester.nifty50_df['date']]
                benchmark_navs = stock_backtester.nifty50_df['nav'].tolist()
                performance_data["benchmark_buyhold"] = benchmark_navs
        
        return {
            "success": True,
            "stock_metrics": stock_metrics,
            "benchmark_metrics": benchmark_metrics,
            "backtest_result": result,
            "performance_data": performance_data
        }
        
    except Exception as e:
        print(f"Error calculating stock metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error calculating stock metrics: {str(e)}")

@stock_router.get("/metrics/table")
async def get_stock_metrics_table():
    """Get formatted metrics comparison table for stocks"""
    try:
        if stock_backtester is None:
            raise HTTPException(status_code=500, detail="Stock backtester not initialized. Check database connection.")
        
        # This would need to be called after a backtest is run
        if not hasattr(stock_backtester, 'weekly_nav_df') or stock_backtester.weekly_nav_df is None:
            raise HTTPException(status_code=400, detail="No stock backtest data available. Run backtest first.")
        
        # Calculate metrics
        stock_metrics = stock_backtester.calculate_metrics(50000, 52, 8.0)  # Default values
        total_investment = 52 * 50000
        benchmark_metrics = stock_backtester.calculate_benchmark_metrics(total_investment, 8.0)
        
        # Create formatted table
        formatted_table = stock_backtester.create_formatted_metrics_table(stock_metrics, benchmark_metrics)
        
        if not formatted_table.empty:
            return {"metrics_table": formatted_table.to_dict('records')}
        else:
            return {"metrics_table": []}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stock metrics table: {str(e)}")

@stock_router.get("/transaction-costs/summary")
async def get_stock_transaction_costs_summary():
    """Get transaction costs summary for stocks"""
    try:
        if stock_backtester is None:
            raise HTTPException(status_code=500, detail="Stock backtester not initialized. Check database connection.")
        
        if not stock_backtester.transaction_costs_log:
            return {"costs_summary": {
                'Total All Costs': '₹0',
                'Capital Gains Tax': '₹0',
                'Cost as % of Volume': '0.00%',
                'Total Transactions': '0'
            }}
        
        costs_summary = stock_backtester.get_transaction_costs_summary()
        return {"costs_summary": costs_summary}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stock transaction costs summary: {str(e)}")

@stock_router.get("/transaction-log")
async def get_stock_transaction_log():
    """Get transaction log from the latest stock backtest"""
    try:
        if stock_backtester is None:
            raise HTTPException(status_code=500, detail="Stock backtester not initialized. Check database connection.")
        
        if not hasattr(stock_backtester, 'portfolio_log') or not stock_backtester.portfolio_log:
            return {"transaction_log": [], "trading_summary": {}}
        
        # Convert portfolio log to frontend format
        transaction_log = []
        for log in stock_backtester.portfolio_log:
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
            'no_trade_weeks': getattr(stock_backtester, 'skipped_days', []),
            'trading_frequency': f"{(total_trades / max(1, len(stock_backtester.portfolio_log))) * 100:.1f}%"
        }
        
        return {
            "transaction_log": transaction_log,
            "trading_summary": trading_summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading stock transaction log: {str(e)}")

@stock_router.get("/transaction-costs")
async def get_stock_transaction_costs():
    """Get transaction costs data from the latest stock backtest"""
    try:
        if stock_backtester is None:
            raise HTTPException(status_code=500, detail="Stock backtester not initialized. Check database connection.")
        
        if not hasattr(stock_backtester, 'transaction_costs_log') or not stock_backtester.transaction_costs_log:
            return {"transaction_costs": []}
        
        # Convert transaction costs log to frontend format
        transaction_costs = []
        for cost in stock_backtester.transaction_costs_log:
            transaction_costs.append({
                'date': cost.get('date', '').strftime('%Y-%m-%d') if hasattr(cost.get('date', ''), 'strftime') else str(cost.get('date', '')),
                'cumulative_cost': cost.get('cumulative_costs', 0),
                'weekly_cost': cost.get('weekly_costs', 0),
                'total_costs': cost.get('total_costs', 0)
            })
        
        return {"transaction_costs": transaction_costs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading stock transaction costs: {str(e)}")

@stock_router.get("/skipped-trades")
async def get_stock_skipped_trades():
    """Get skipped trades information from the latest stock backtest"""
    try:
        if stock_backtester is None:
            raise HTTPException(status_code=500, detail="Stock backtester not initialized. Check database connection.")
        
        if not hasattr(stock_backtester, 'skipped_days') or not stock_backtester.skipped_days:
            return {"skipped_trades": []}
        
        # Convert skipped days to frontend format
        skipped_trades = []
        for skip in stock_backtester.skipped_days:
            skipped_trades.append({
                'week': skip.get('week', 0),
                'date': skip.get('date', ''),
                'signal_date': skip.get('signal_date', 'N/A'),
                'reason': skip.get('reason', 'Unknown')
            })
        
        return {"skipped_trades": skipped_trades}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading stock skipped trades: {str(e)}")

@stock_router.get("/trade-execution-status")
async def get_stock_trade_execution_status():
    """Get real-time trade execution status and statistics for stocks"""
    try:
        if stock_backtester is None:
            raise HTTPException(status_code=500, detail="Stock backtester not initialized. Check database connection.")
        
        # Get current backtest statistics
        stats = {
            'total_weeks_processed': getattr(stock_backtester, 'total_weeks', 0),
            'successful_signals': getattr(stock_backtester, 'successful_signals', 0),
            'successful_executions': getattr(stock_backtester, 'successful_executions', 0),
            'portfolio_log_entries': len(getattr(stock_backtester, 'portfolio_log', [])),
            'transaction_costs_entries': len(getattr(stock_backtester, 'transaction_costs_log', [])),
            'skipped_trades_count': len(getattr(stock_backtester, 'skipped_days', [])),
            'current_cash': getattr(stock_backtester, 'current_cash', 0),
            'current_holdings': getattr(stock_backtester, 'current_holdings', {}),
            'last_trade_date': None,
            'last_trade_action': None,
            'last_trade_ticker': None
        }
        
        # Get last trade information
        if stock_backtester.portfolio_log:
            last_trade = stock_backtester.portfolio_log[-1]
            stats['last_trade_date'] = last_trade.get('execution_date', '').strftime('%Y-%m-%d') if hasattr(last_trade.get('execution_date', ''), 'strftime') else str(last_trade.get('execution_date', ''))
            stats['last_trade_action'] = last_trade.get('action', 'NONE')
            stats['last_trade_ticker'] = last_trade.get('ticker', '')
        
        return {"trade_execution_status": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading stock trade execution status: {str(e)}")

@stock_router.get("/charts/equity-curve")
async def get_stock_equity_curve_chart(show_benchmark: bool = True, show_stock_strategy: bool = True):
    """Get equity curve chart data for stocks"""
    try:
        if stock_backtester is None:
            raise HTTPException(status_code=500, detail="Stock backtester not initialized. Check database connection.")
        
        if not hasattr(stock_backtester, 'weekly_nav_df') or stock_backtester.weekly_nav_df is None:
            raise HTTPException(status_code=400, detail="No stock backtest data available. Run backtest first.")
        
        # Return data for frontend charting
        if not stock_backtester.weekly_nav_df.empty:
            chart_data = {
                "dates": [str(date) for date in stock_backtester.weekly_nav_df['date']],
                "stock_strategy": stock_backtester.weekly_nav_df['nav'].tolist(),
                "cumulative_investment": stock_backtester.weekly_nav_df['cumulative_investment'].tolist(),
                "benchmark_buyhold": []
            }
            
            if not stock_backtester.nifty50_df.empty:
                chart_data["benchmark_buyhold"] = stock_backtester.nifty50_df['nav'].tolist()
            
            return {"chart_data": chart_data}
        else:
            return {"chart_data": {}}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stock equity curve chart: {str(e)}")

@stock_router.get("/charts/transaction-costs")
async def get_stock_transaction_costs_chart():
    """Get transaction costs chart data for stocks"""
    try:
        if stock_backtester is None:
            raise HTTPException(status_code=500, detail="Stock backtester not initialized. Check database connection.")
        
        if not stock_backtester.transaction_costs_log:
            return {"chart_data": {}}
        
        # Return data for frontend charting
        costs_df = pd.DataFrame(stock_backtester.transaction_costs_log)
        costs_df['date'] = pd.to_datetime(costs_df['date'])
        costs_df = costs_df.sort_values('date')
        costs_df['cumulative_total_costs'] = costs_df['total_impact'].cumsum()
        
        chart_data = {
            "dates": [str(date) for date in costs_df['date']],
            "cumulative_costs": costs_df['cumulative_total_costs'].tolist()
        }
        
        return {"chart_data": chart_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stock transaction costs chart: {str(e)}")

@stock_router.post("/cleanup")
async def cleanup_stock_resources():
    """Clean up stock resources and clear cache"""
    try:
        cleanup_stock_backtester()
        return {"success": True, "message": "Stock resources cleaned up successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning up stock resources: {str(e)}")

@stock_router.get("/costs/summary")
async def get_stock_costs_summary():
    """Get comprehensive costs summary including transaction costs and capital gains tax for stocks"""
    try:
        if stock_backtester is None:
            raise HTTPException(status_code=500, detail="Stock backtester not initialized. Check database connection.")
        
        if not hasattr(stock_backtester, 'portfolio_log') or not stock_backtester.portfolio_log:
            return {
                "total_all_costs": 0,
                "capital_gains_tax": 0,
                "transaction_costs": 0,
                "cost_as_percent_of_volume": 0,
                "total_transactions": 0,
                "total_volume": 0
            }
        
        # Calculate costs from portfolio log - FIXED: Extract transaction costs from costs dictionary
        total_capital_gains_tax = sum(log.get('capital_gains_tax', 0) for log in stock_backtester.portfolio_log)
        
        # Fix: Extract transaction costs from the costs dictionary, not directly from log
        total_transaction_costs = 0
        for log in stock_backtester.portfolio_log:
            costs = log.get('costs', {})
            transaction_cost = costs.get('total_costs', 0) if costs else 0
            total_transaction_costs += transaction_cost
        
        total_all_costs = total_capital_gains_tax + total_transaction_costs
        
        # Calculate total volume (sum of all transaction amounts)
        total_volume = sum(log.get('amount', 0) for log in stock_backtester.portfolio_log)
        
        # Calculate cost as percentage of volume
        cost_as_percent = (total_all_costs / total_volume * 100) if total_volume > 0 else 0
        
        # Count total transactions
        total_transactions = len(stock_backtester.portfolio_log)
        
        return {
            "total_all_costs": round(total_all_costs, 2),
            "capital_gains_tax": round(total_capital_gains_tax, 2),
            "transaction_costs": round(total_transaction_costs, 2),
            "cost_as_percent_of_volume": round(cost_as_percent, 3),
            "total_transactions": total_transactions,
            "total_volume": round(total_volume, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating stock costs summary: {str(e)}")

@stock_router.get("/costs/analysis")
async def get_stock_costs_analysis():
    """Get detailed costs analysis over time for the stock chart"""
    try:
        if stock_backtester is None:
            raise HTTPException(status_code=500, detail="Stock backtester not initialized. Check database connection.")
        
        if not hasattr(stock_backtester, 'portfolio_log') or not stock_backtester.portfolio_log:
            return {"costs_data": []}
        
        # Create cumulative costs data over time
        costs_data = []
        cumulative_transaction_costs = 0
        cumulative_capital_gains_tax = 0
        cumulative_total_costs = 0
        
        # Group by date and calculate cumulative costs
        date_costs = {}
        for log in stock_backtester.portfolio_log:
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
        raise HTTPException(status_code=500, detail=f"Error calculating stock costs analysis: {str(e)}")

@stock_router.get("/costs/breakdown")
async def get_stock_costs_breakdown():
    """Get detailed breakdown of costs by type and period for stocks"""
    try:
        if stock_backtester is None:
            raise HTTPException(status_code=500, detail="Stock backtester not initialized. Check database connection.")
        
        if not hasattr(stock_backtester, 'portfolio_log') or not stock_backtester.portfolio_log:
            return {"breakdown": {}}
        
        # Calculate costs by year
        yearly_costs = {}
        for log in stock_backtester.portfolio_log:
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
        raise HTTPException(status_code=500, detail=f"Error calculating stock costs breakdown: {str(e)}")

# ============================================================================
# SAVED STRATEGY DATABASE FUNCTIONS
# ============================================================================

def init_saved_strategies_table(db_path: str = None):
    """Initialize the stock_saved_strategy table in PostgreSQL (db_path parameter ignored, kept for compatibility)"""
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
                try:
                    logger.error(error_msg)
                except NameError:
                    print(error_msg)
                return False, error_msg
        
        # Initialize all tables (including stock_saved_strategy)
        if not init_database():
            error_msg = "Failed to initialize database tables"
            try:
                logger.error(error_msg)
            except NameError:
                print(error_msg)
            return False, error_msg
        
        try:
            logger.info("stock_saved_strategy table initialized successfully in PostgreSQL")
        except NameError:
            print("stock_saved_strategy table initialized successfully in PostgreSQL")
        return True, None
    except Exception as e:
        error_msg = f"Error initializing stock_saved_strategy table: {e}"
        try:
            logger.error(error_msg)
        except NameError:
            print(error_msg)
        import traceback
        traceback.print_exc()
        return False, str(e)

# ============================================================================
# SAVED STRATEGY ROUTES
# ============================================================================

@stock_router.post("/save-strategy")
async def save_stock_strategy(request: SaveStockStrategyRequest):
    """Save a stock strategy to the database with validation"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from Services.strategy_manager.models import SavedInstance
        
        session = get_session()
        
        # Check if strategy already exists using the backtester core validation
        from ..services.backtester import StockRotationBacktester
        backtester = StockRotationBacktester()
        
        if hasattr(backtester, 'check_strategy_exists'):
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
                return {
                    "success": False,
                    "message": validation_result.get("message", "Strategy already exists"),
                    "existing_strategy": validation_result.get("existing_strategy", {}),
                    "strategy_exists": True
                }
            
            # If validation failed due to error, return error
            if "error" in validation_result:
                raise HTTPException(status_code=500, detail=validation_result["error"])
        
        # Prepare strategy parameters as JSON
        strategies_parameters = {
            "tickers": sorted(request.tickers),
            "start_date": request.start_date,
            "end_date": request.end_date,
            "capital_per_week": request.capital_per_week,
            "accumulation_weeks": request.accumulation_weeks,
            "brokerage_percent": request.brokerage_percent,
            "compounding_enabled": request.compounding_enabled,
            "risk_free_rate": request.risk_free_rate,
            "use_custom_dates": request.use_custom_dates
        }
        
        # Create new SavedInstance
        new_strategy = SavedInstance(
            strategy_name=request.strategy_name,
            strategy_type=request.strategy_type or "StockSurfTrend",
            user_id=request.user_id,
            strategies_parameters=strategies_parameters,
            backtest_results=request.backtest_results.dict() if request.backtest_results else {},
            status='deploy',
            created_at=datetime.utcnow()
        )
        
        session.add(new_strategy)
        session.commit()
        
        return {
            "success": True,
            "message": "Strategy saved successfully",
            "strategy_id": new_strategy.id,
            "strategy_exists": False
        }
        
    except Exception as e:
        if session:
            session.rollback()
        raise HTTPException(status_code=500, detail=f"Error saving strategy: {str(e)}")
    finally:
        if session:
            session.close()

@stock_router.get("/get-saved-strategies-list/{user_id}")
async def get_saved_stock_strategies(user_id: str):
    """Get all saved stock strategies for a specific user"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from Services.strategy_manager.models import SavedInstance
        
        session = get_session()
        
        # Query for stock strategies
        strategies_list = session.query(SavedInstance).filter(
            SavedInstance.user_id == user_id,
            SavedInstance.strategy_type.in_(['Rotation_Stocks', 'StockSurfTrend'])
        ).all()
        
        strategies = []
        for strategy in strategies_list:
            params = strategy.strategies_parameters or {}
            strategies.append({
                "id": strategy.id,
                "strategy_name": strategy.strategy_name,
                "strategy_type": strategy.strategy_type,
                "user_id": strategy.user_id,
                "tickers": params.get('tickers', []),
                "start_date": params.get('start_date'),
                "end_date": params.get('end_date'),
                "capital_per_week": params.get('capital_per_week'),
                "accumulation_weeks": params.get('accumulation_weeks'),
                "brokerage_percent": params.get('brokerage_percent'),
                "compounding_enabled": params.get('compounding_enabled'),
                "risk_free_rate": params.get('risk_free_rate'),
                "use_custom_dates": params.get('use_custom_dates'),
                "backtest_results": strategy.backtest_results or {},
                "created_at": strategy.created_at.strftime('%Y-%m-%d %H:%M:%S') if strategy.created_at else None,
                "status": strategy.status
            })
        
        return {"strategies": strategies}
        
    except Exception as e:
        print(f"Error retrieving saved strategies: {str(e)}")
        return {"strategies": []}
    finally:
        if session:
            session.close()

@stock_router.get("/get-saved-strategy/{strategy_id}")
async def get_saved_stock_strategy_by_id(strategy_id: int):
    """Get a specific saved stock strategy by ID"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from Services.strategy_manager.models import SavedInstance
        
        session = get_session()
        strategy = session.query(SavedInstance).filter(SavedInstance.id == strategy_id).first()
        
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        params = strategy.strategies_parameters or {}
        strategy_response = {
            "id": strategy.id,
            "strategy_name": strategy.strategy_name,
            "strategy_type": strategy.strategy_type,
            "user_id": strategy.user_id,
            "tickers": params.get('tickers', []),
            "start_date": params.get('start_date'),
            "end_date": params.get('end_date'),
            "capital_per_week": params.get('capital_per_week'),
            "accumulation_weeks": params.get('accumulation_weeks'),
            "brokerage_percent": params.get('brokerage_percent'),
            "compounding_enabled": params.get('compounding_enabled'),
            "risk_free_rate": params.get('risk_free_rate'),
            "use_custom_dates": params.get('use_custom_dates'),
            "backtest_results": strategy.backtest_results or {},
            "created_at": strategy.created_at.strftime('%Y-%m-%d %H:%M:%S') if strategy.created_at else None,
            "status": strategy.status
        }
        
        return {"strategy": strategy_response}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving strategy: {str(e)}")
    finally:
        if session:
            session.close()



@stock_router.get("/get-saved-strategies-count/{user_id}")
async def get_saved_stock_strategies_count(user_id: str):
    """Get count of saved stock strategies for a specific user"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from Services.strategy_manager.models import SavedInstance
        
        session = get_session()
        count = session.query(SavedInstance).filter(
            SavedInstance.user_id == user_id,
            SavedInstance.strategy_type.in_(['Rotation_Stocks', 'StockSurfTrend'])
        ).count()
        
        return {"count": count}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting strategy count: {str(e)}")
    finally:
        if session:
            session.close()

# ============================================================================
# NEW RS-STYLE ENDPOINTS FOR STOCK STRATEGIES
# ============================================================================

@stock_router.get("/get-saved-strategies-table/{user_id}")
async def get_saved_stock_strategies_table(user_id: str):
    """Get saved stock strategies in table format like RS strategy"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from Services.strategy_manager.models import SavedInstance
        
        session = get_session()
        
        # Query for stock strategies
        strategies_list = session.query(SavedInstance).filter(
            SavedInstance.user_id == user_id,
            SavedInstance.strategy_type.in_(['Rotation_Stocks', 'StockSurfTrend'])
        ).all()
        
        strategies = []
        for strategy in strategies_list:
            params = strategy.strategies_parameters or {}
            strategy_response = {
                "id": strategy.id,
                "strategy_name": strategy.strategy_name,
                "strategy_type": strategy.strategy_type,
                "user_id": strategy.user_id,
                "config_id": strategy.id, # Using id as config_id
                "start_date": params.get('start_date'),
                "end_date": params.get('end_date'),
                "stock_universe": "CUSTOM", # Defaulting to CUSTOM
                "backtest_results": strategy.backtest_results or {},
                "status": strategy.status or 'deploy',
                "created_at": strategy.created_at.strftime('%Y-%m-%d %H:%M:%S') if strategy.created_at else None
            }
            strategies.append(strategy_response)
        
        return {
            "success": True,
            "strategies": strategies
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stock strategies table: {str(e)}")
    finally:
        if session:
            session.close()

@stock_router.post("/stop-stock-strategy")
async def stop_stock_strategy(request: dict):
    """Stop a running stock strategy"""
    session = None
    try:
        strategy_id = request.get("strategy_id")
        user_id = request.get("user_id")
        
        if not strategy_id or not user_id:
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        from Databases.app_data_db_connection import get_session
        from Services.strategy_manager.models import SavedInstance
        
        session = get_session()
        strategy = session.query(SavedInstance).filter(
            SavedInstance.id == strategy_id, 
            SavedInstance.user_id == user_id
        ).first()
        
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        strategy.status = 'stopped'
        session.commit()
        
        return {
            "success": True,
            "message": "Stock strategy stopped successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if session:
            session.rollback()
        raise HTTPException(status_code=500, detail=f"Error stopping stock strategy: {str(e)}")
    finally:
        if session:
            session.close()

@stock_router.post("/restart-stock-strategy")
async def restart_stock_strategy(request: dict):
    """Restart a stopped stock strategy"""
    session = None
    try:
        strategy_id = request.get("strategy_id")
        user_id = request.get("user_id")
        
        if not strategy_id or not user_id:
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        from Databases.app_data_db_connection import get_session
        from Services.strategy_manager.models import SavedInstance
        
        session = get_session()
        strategy = session.query(SavedInstance).filter(
            SavedInstance.id == strategy_id, 
            SavedInstance.user_id == user_id
        ).first()
        
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        strategy.status = 'deploy' # Standardized on 'deploy' instead of 'running' for consistency
        session.commit()
        
        return {
            "success": True,
            "message": "Stock strategy restarted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if session:
            session.rollback()
        raise HTTPException(status_code=500, detail=f"Error restarting stock strategy: {str(e)}")
    finally:
        if session:
            session.close()


# ============================================================================
# STOCK SIGNAL GENERATION ENDPOINTS
# ============================================================================

@stock_router.post("/stock-signals/generate")
async def generate_stock_signals(request: Dict[str, Any] = None):
    """
    Generate Stock trading signals
    """
    try:
        if LiveStockSignalGenerator is None:
            raise HTTPException(
                status_code=500, 
                detail="Signal generator not available. Check imports and dependencies."
            )
        
        # Handle empty request body
        if request is None:
            request = {}
        
        run_id = request.get("run_id") if request else None
        strategy_type = request.get("strategy_type", "StockSurfTrend") if request else "StockSurfTrend"
        strategy_config = request.get("strategy_config") if request else None
        
        # Initialize signal generator with the specified database path
        # Use absolute path to the database as specified
        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
            "unified_etf_data.sqlite"
        )
        db_path = os.path.abspath(db_path)
        
        generator = LiveStockSignalGenerator(db_path=db_path)
        
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
        
        # No run_id provided - generate for all running strategies (status='deploy')
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
                "message": "No running strategies found with status='deploy'"
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
                    "message": f"Signal generation failed: {str(e)}"
                })
        
        generator.cleanup()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return {
            "success": True,
            "total_strategies": len(running_strategies),
            "successful": successful,
            "failed": failed,
            "results": results,
            "duration_seconds": duration,
            "message": f"Processed {len(running_strategies)} strategies: {successful} successful, {failed} failed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating stock signals: {str(e)}")

@stock_router.get("/stock-signals/recent")
async def get_recent_stock_signals(days: int = 30):
    """Get recent stock signals from the TradingSignal table"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from Databases.signal_models import TradingSignal
        
        session = get_session()
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter for stock strategies
        signals = session.query(TradingSignal).filter(
            TradingSignal.strategy_type.ilike('%stock%'),
            TradingSignal.signal_date >= cutoff_date
        ).order_by(TradingSignal.signal_date.desc(), TradingSignal.created_at.desc()).all()
        
        return {"signals": [s.to_dict() if hasattr(s, 'to_dict') else {c.name: getattr(s, c.name) for c in s.__table__.columns} for s in signals]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving recent signals: {str(e)}")
    finally:
        if session:
            session.close()

@stock_router.get("/stock-signals/run/{run_id}")
async def get_stock_signals_by_run(run_id: str):
    """Get stock signals for a specific run ID"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from Databases.signal_models import TradingSignal
        
        session = get_session()
        signals = session.query(TradingSignal).filter(TradingSignal.run_id == run_id).all()
        
        return {"run_id": run_id, "signals": [s.to_dict() if hasattr(s, 'to_dict') else {c.name: getattr(s, c.name) for c in s.__table__.columns} for s in signals]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving signals for run {run_id}: {str(e)}")
    finally:
        if session:
            session.close()

@stock_router.get("/stock-signals/runs")
async def get_stock_signal_runs(limit: int = 50):
    """Get a list of recent stock signal logic runs"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from Databases.signal_models import TradingSignal
        from sqlalchemy import func
        
        session = get_session()
        
        # Group by run_id to get summary of runs
        runs_query = session.query(
            TradingSignal.run_id,
            TradingSignal.strategy_type,
            func.min(TradingSignal.signal_date).label('signal_date'),
            func.max(TradingSignal.created_at).label('created_at'),
            func.count(TradingSignal.id).label('signal_count')
        ).filter(
            TradingSignal.strategy_type.ilike('%stock%')
        ).group_by(
            TradingSignal.run_id, 
            TradingSignal.strategy_type
        ).order_by(
            func.max(TradingSignal.created_at).desc()
        ).limit(limit)
        
        runs = []
        for run in runs_query:
            runs.append({
                "run_id": run.run_id,
                "strategy_type": run.strategy_type,
                "signal_date": run.signal_date.strftime('%Y-%m-%d') if run.signal_date else None,
                "created_at": run.created_at.strftime('%Y-%m-%d %H:%M:%S') if run.created_at else None,
                "signal_count": run.signal_count
            })
            
        return {"runs": runs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving signal runs: {str(e)}")
    finally:
        if session:
            session.close()

@stock_router.get("/stock-signals/latest")
async def get_latest_stock_signals():
    """Get the most recent run's stock signals"""
    session = None
    try:
        from Databases.app_data_db_connection import get_session
        from Databases.signal_models import TradingSignal
        
        session = get_session()
        
        # Find the most recent run_id
        latest_run = session.query(TradingSignal.run_id).filter(
            TradingSignal.strategy_type.ilike('%stock%')
        ).order_by(TradingSignal.created_at.desc()).first()
        
        if not latest_run:
            return {"run_id": None, "signals": [], "message": "No signals found"}
            
        run_id = latest_run[0]
        signals = session.query(TradingSignal).filter(TradingSignal.run_id == run_id).all()
        
        return {
            "run_id": run_id, 
            "signals": [s.to_dict() if hasattr(s, 'to_dict') else {c.name: getattr(s, c.name) for c in s.__table__.columns} for s in signals]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving latest signals: {str(e)}")
    finally:
        if session:
            session.close()

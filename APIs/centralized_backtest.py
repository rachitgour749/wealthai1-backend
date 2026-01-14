"""
Centralized Backtest API
Single endpoint to handle all strategy types
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
import sys
import os

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from APIs.unified_schemas import UnifiedBacktestRequest, UnifiedBacktestResponse
from Handlers.etf_rotation_handler import ETFRotationHandler
from Handlers.rs_etf_handler import RSETFHandler
from Handlers.rs_stocks_handler import RSStocksHandler
from Handlers.international_etf_handler import InternationalETFHandler
from Handlers.rotation_stocks_handler import RotationStocksHandler
from Handlers.etf_payout_handler import ETFPayoutHandler
from Handlers.supertrend_handler import SuperTrendHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
centralized_router = APIRouter(prefix="/api", tags=["Centralized Backtest API"])

# In-memory cache for latest backtest results per strategy type
# This allows transaction-log and other endpoints to access the most recent backtest data
_backtest_results_cache: Dict[str, Dict[str, Any]] = {}

def cache_backtest_results(strategy_type: str, backtester_instance: Any) -> None:
    """Cache backtest results for later retrieval"""
    print(f"[CACHE] cache_backtest_results called for {strategy_type}")
    print(f"[CACHE] Backtester instance type: {type(backtester_instance)}")
    print(f"[CACHE] Has portfolio_log: {hasattr(backtester_instance, 'portfolio_log')}")
    
    try:
        portfolio_log = getattr(backtester_instance, 'portfolio_log', [])
        print(f"[CACHE] portfolio_log length: {len(portfolio_log) if isinstance(portfolio_log, list) else 'NOT A LIST'}")
        
        # Calculate cost breakdown from portfolio_log
        cost_breakdown = {}
        cost_summary = {}
        cost_analysis = {}
        
        # Calculate costs from portfolio_log - grouped by year
        if portfolio_log and len(portfolio_log) > 0:
            yearly_breakdown = {}
            
            # Debug: Print first log entry to see structure
            if len(portfolio_log) > 0:
                first_log = portfolio_log[0]
                print(f"[CACHE] DEBUG - First portfolio_log entry keys: {list(first_log.keys())}")
                print(f"[CACHE] DEBUG - First portfolio_log entry: {first_log}")
            
            for log in portfolio_log:
                # Get the year from execution_date
                execution_date = log.get('execution_date')
                if hasattr(execution_date, 'year'):
                    year = str(execution_date.year)
                else:
                    # Try to parse string date
                    try:
                        from datetime import datetime
                        if isinstance(execution_date, str):
                            year = str(datetime.strptime(execution_date, '%Y-%m-%d').year)
                        else:
                            year = "Unknown"
                    except:
                        year = "Unknown"
                
                # Initialize year if not exists
                if year not in yearly_breakdown:
                    yearly_breakdown[year] = {
                        'transaction_costs': 0,
                        'capital_gains_tax': 0,
                        'total_costs': 0,
                        'total_brokerage': 0.0,
                        'transactions': 0
                    }
                
                # Extract costs - handle multiple formats
                transaction_cost = 0
                capital_gains_tax = 0
                
                # Try to get from 'costs' dictionary first
                costs = log.get('costs', {})
                if costs and isinstance(costs, dict):
                    transaction_cost = costs.get('total_costs', 0)
                
                # If still 0, try direct fields
                if transaction_cost == 0:
                    transaction_cost = log.get('transaction_costs', 0) or log.get('transaction_cost', 0) or log.get('brokerage', 0)
                
                # Get capital gains tax
                capital_gains_tax = log.get('capital_gains_tax', 0)
                
                # Add to yearly totals
                yearly_breakdown[year]['transaction_costs'] += transaction_cost
                yearly_breakdown[year]['capital_gains_tax'] += capital_gains_tax
                yearly_breakdown[year]['total_costs'] += (transaction_cost + capital_gains_tax)
                yearly_breakdown[year]['transactions'] += 1
            
            # Round all values
            for year in yearly_breakdown:
                yearly_breakdown[year]['transaction_costs'] = round(yearly_breakdown[year]['transaction_costs'], 2)
                yearly_breakdown[year]['capital_gains_tax'] = round(yearly_breakdown[year]['capital_gains_tax'], 2)
                yearly_breakdown[year]['total_costs'] = round(yearly_breakdown[year]['total_costs'], 2)
            
            cost_breakdown = {'breakdown': yearly_breakdown}
            print(f"[CACHE] Calculated yearly cost_breakdown for years: {list(yearly_breakdown.keys())}")
        
        # Try to get additional cost data from backtester methods
        try:
            if hasattr(backtester_instance, 'get_cost_summary'):
                cost_summary = backtester_instance.get_cost_summary()
                print(f"[CACHE] Got cost_summary from backtester")
        except Exception as e:
            print(f"[CACHE] Could not get cost_summary: {e}")
        
        try:
            if hasattr(backtester_instance, 'get_cost_analysis'):
                cost_analysis = backtester_instance.get_cost_analysis()
                print(f"[CACHE] Got cost_analysis from backtester")
        except Exception as e:
            print(f"[CACHE] Could not get cost_analysis: {e}")
        
        _backtest_results_cache[strategy_type] = {
            'portfolio_log': portfolio_log.copy() if isinstance(portfolio_log, list) else [],
            'weekly_nav_df': backtester_instance.weekly_nav_df.copy() if hasattr(backtester_instance, 'weekly_nav_df') and hasattr(backtester_instance.weekly_nav_df, 'copy') else None,
            'trading_summary': getattr(backtester_instance, 'trading_summary', {}).copy() if isinstance(getattr(backtester_instance, 'trading_summary', {}), dict) else {},
            'withdrawal_log': getattr(backtester_instance, 'withdrawal_log', []).copy() if isinstance(getattr(backtester_instance, 'withdrawal_log', []), list) else [],
            'total_withdrawn_amount': getattr(backtester_instance, 'total_withdrawn_amount', 0),
            'cost_breakdown': cost_breakdown,
            'cost_summary': cost_summary,
            'cost_analysis': cost_analysis
        }
        
        cached_count = len(_backtest_results_cache[strategy_type]['portfolio_log'])
        print(f"[CACHE] ✅ Successfully cached {cached_count} transactions for {strategy_type}")
        logger.info(f"✅ Cached backtest results for {strategy_type}: {cached_count} transactions")
    except Exception as e:
        print(f"[CACHE] ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"❌ Failed to cache backtest results for {strategy_type}: {e}")

def get_cached_backtest_results(strategy_type: str) -> Dict[str, Any]:
    """Retrieve cached backtest results"""
    print(f"[CACHE] get_cached_backtest_results called for {strategy_type}")
    print(f"[CACHE] Cache keys: {list(_backtest_results_cache.keys())}")
    
    result = _backtest_results_cache.get(strategy_type, {})
    
    if result:
        print(f"[CACHE] ✅ Found cached data with {len(result.get('portfolio_log', []))} transactions")
    else:
        print(f"[CACHE] ❌ No cached data found for {strategy_type}")
    
    return result


@centralized_router.post("/run_backtest", response_model=UnifiedBacktestResponse)
async def run_backtest(request: UnifiedBacktestRequest) -> UnifiedBacktestResponse:
    """
    Centralized backtest endpoint supporting all strategy types.
    
    **Supported Strategy Types:**
    - `ETF_Rotation`: Weekly SIP-based ETF rotation strategy
    - `RS_ETF_Rotation`: Relative Strength ETF strategy with lump sum capital
    - `International_ETF_Rotation`: Weekly SIP for international ETFs
    - `Rotation_Stocks`: Weekly SIP-based stock rotation strategy
    - `ETF_Payout`: ETF rotation with periodic withdrawals
    - `SuperTrend`: Technical indicator-based trading strategy
    
    **Request Parameters:**
    The required parameters vary by strategy type. See the schema for details.
    
    **Example Request (ETF Rotation):**
    ```json
    {
        "strategy_type": "ETF_Rotation",
        "start_date": "2020-01-01",
        "end_date": "2023-12-31",
        "tickers": ["NIFTYBEES.NS", "BANKBEES.NS", "GOLDBEES.NS"],
        "capital_per_week": 50000,
        "accumulation_weeks": 52,
        "brokerage_percent": 0.1,
        "compounding_enabled": false,
        "risk_free_rate": 8.0
    }
    ```
    
    **Example Request (RS ETF):**
    ```json
    {
        "strategy_type": "RS_ETF_Rotation",
        "start_date": "2020-01-01",
        "end_date": "2023-12-31",
        "total_capital": 1000000,
        "etf_universe": "ALL_ETFS",
        "max_positions": 20,
        "risk_free_rate": 8.0
    }
    ```
    
    **Returns:**
    - Unified response with metrics, performance data, and transaction logs
    """
    
    try:
        logger.info(f"Received backtest request for strategy: {request.strategy_type}")
        
        # Route to appropriate strategy handler
        handler = None
        
        if request.strategy_type == "ETF_Rotation":
            handler = ETFRotationHandler(None)
            
        elif request.strategy_type == "RS_ETF_Rotation":
            handler = RSETFHandler(None)
            
        elif request.strategy_type == "RS_Stocks":
            handler = RSStocksHandler(None)
            
        elif request.strategy_type == "International_ETF_Rotation":
            handler = InternationalETFHandler(None)
            
        elif request.strategy_type == "Rotation_Stocks":
            handler = RotationStocksHandler(None)
            
        elif request.strategy_type == "ETF_Payout":
            handler = ETFPayoutHandler(None)
            
        elif request.strategy_type == "SuperTrend":
            handler = SuperTrendHandler(None)
            
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy_type: {request.strategy_type}"
            )
        
        # Execute backtest
        logger.info(f"Executing backtest for {request.strategy_type}")
        response = await handler.run_backtest(request)
        
        if response.success:
            logger.info(f"Backtest completed successfully for {request.strategy_type}")
        else:
            logger.error(f"Backtest failed for {request.strategy_type}: {response.error}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in centralized backtest: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@centralized_router.get("/health")
async def health_check():
    """Health check endpoint for centralized API"""
    return {
        "status": "healthy",
        "service": "Centralized Backtest API",
        "supported_strategies": [
            "ETF_Rotation",
            "RS_ETF_Rotation",
            "RS_Stocks",
            "International_ETF_Rotation",
            "Rotation_Stocks",
            "ETF_Payout",
            "SuperTrend"
        ]
    }


@centralized_router.get("/strategies")
async def list_strategies():
    """
    List all supported strategy types with their required parameters
    """
    return {
        "strategies": [
            {
                "type": "ETF_Rotation",
                "description": "Weekly SIP-based ETF rotation strategy",
                "required_params": [
                    "tickers", "start_date", "end_date",
                    "capital_per_week", "accumulation_weeks", "brokerage_percent"
                ],
                "optional_params": ["compounding_enabled", "risk_free_rate"]
            },
            {
                "type": "RS_ETF_Rotation",
                "description": "Relative Strength ETF strategy with lump sum capital",
                "required_params": [
                    "start_date", "end_date", "total_capital"
                ],
                "optional_params": [
                    "etf_universe", "custom_etfs", "max_positions",
                    "lookback_weeks", "lookback_months", "lookback_quarters",
                    "risk_free_rate"
                ]
            },
            {
                "type": "RS_Stocks",
                "description": "Relative Strength stock strategy with lump sum capital",
                "required_params": [
                    "start_date", "end_date", "total_capital"
                ],
                "optional_params": [
                    "stock_universe", "custom_stocks", "max_positions",
                    "lookback_weeks", "lookback_months", "lookback_quarters",
                    "risk_free_rate"
                ]
            },
            {
                "type": "International_ETF_Rotation",
                "description": "Weekly SIP for international ETFs",
                "required_params": [
                    "tickers", "start_date", "end_date",
                    "capital_per_week", "accumulation_weeks", "brokerage_percent"
                ],
                "optional_params": ["compounding_enabled", "risk_free_rate"]
            },
            {
                "type": "Rotation_Stocks",
                "description": "Weekly SIP-based stock rotation strategy",
                "required_params": [
                    "tickers", "start_date", "end_date",
                    "capital_per_week", "accumulation_weeks", "brokerage_percent"
                ],
                "optional_params": ["compounding_enabled", "risk_free_rate"]
            },
            {
                "type": "ETF_Payout",
                "description": "ETF rotation with periodic withdrawals",
                "required_params": [
                    "tickers", "start_date", "end_date",
                    "capital_per_week", "accumulation_weeks", "brokerage_percent"
                ],
                "optional_params": [
                    "compounding_enabled", "withdraw_amount",
                    "payout_start_week", "risk_free_rate"
                ]
            },
            {
                "type": "SuperTrend",
                "description": "Technical indicator-based trading strategy",
                "required_params": [
                    "start_date", "end_date", "initial_capital"
                ],
                "optional_params": [
                    "brokerage_pct", "buffer_pct", "ema_short", "ema_long",
                    "supertrend_period", "supertrend_stop_pct",
                    "max_holdings", "symbols"
                ]
            }
        ]
    }

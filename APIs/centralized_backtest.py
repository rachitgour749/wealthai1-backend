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
from Handlers.international_etf_handler import InternationalETFHandler
from Handlers.rotation_stocks_handler import RotationStocksHandler
from Handlers.etf_payout_handler import ETFPayoutHandler
from Handlers.supertrend_handler import SuperTrendHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
centralized_router = APIRouter(prefix="/api", tags=["Centralized Backtest API"])


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

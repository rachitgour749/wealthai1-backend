"""
Backtest Service
Handles backtest execution logic for all strategy types
"""
import logging
from fastapi import HTTPException
from typing import Dict, Any

from APIs.unified_schemas import UnifiedBacktestRequest, UnifiedBacktestResponse
from APIs.common.cache import cache_backtest_results

# Import Handlers
from Handlers.etf_rotation_handler import ETFRotationHandler
from Handlers.rs_etf_handler import RSETFHandler
from Handlers.rs_stocks_handler import RSStocksHandler
from Handlers.international_etf_handler import InternationalETFHandler
from Handlers.rotation_stocks_handler import RotationStocksHandler
from Handlers.etf_payout_handler import ETFPayoutHandler
from Handlers.supertrend_handler import SuperTrendHandler
from Handlers.etf_buy_on_dip_handler import ETFBuyOnDipHandler

logger = logging.getLogger(__name__)

async def execute_backtest(request: UnifiedBacktestRequest) -> UnifiedBacktestResponse:
    """
    Execute backtest for the specified strategy type
    """
    try:
        logger.info(f"Received backtest request for strategy: {request.strategy_type}")
        
        # Route to appropriate strategy handler
        handler = None
        
        if request.strategy_type == "ETF_Rotation":
            handler = ETFRotationHandler(None)
            
        elif request.strategy_type == "RS_ETF_Rotation":
            # REJECT call if strategy is disabled, but for now we keep the handler import
            # If the user wants to hard-disable route access, they can do so in routes.py
            # or we can check here.
            handler = RSETFHandler(None)
            
        elif request.strategy_type == "RS_Stocks":
            handler = RSStocksHandler(None)
            
        elif request.strategy_type == "International_ETF_Rotation":
            handler = InternationalETFHandler(None)
            
        elif request.strategy_type == "Stock_Rotation":
            # Verify if this should run (user asked to remove Stock APIs)
            # But the service logic typically remains, just the route access is cut.
            # However, if we removed the stock_backtester initialization, this might fail 
            # if the handler relies on it. 
            # RotationStocksHandler usually initializes its own backtester or uses global.
            handler = RotationStocksHandler(None)
            
        elif request.strategy_type == "ETF_Payout":
            handler = ETFPayoutHandler(None)
            
        elif request.strategy_type == "SuperTrend":
            handler = SuperTrendHandler(None)
            
        elif request.strategy_type == "ETF_Buy_on_Dip":
            handler = ETFBuyOnDipHandler(None)
            
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
        logger.error(f"Unexpected error in backtest execution: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

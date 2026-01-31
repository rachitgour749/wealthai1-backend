"""
Unified API Routes
Consolidates all API endpoints into a single router, delegating logic to services.
"""
from fastapi import APIRouter, HTTPException, Depends, Query, Request
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional

from Databases.app_data_db_connection import get_db
from APIs.unified_schemas import (
    UnifiedBacktestRequest, UnifiedBacktestResponse, 
    DateRangeRequest
)
from Services.strategy_manager.schemas import (
    SaveStrategyRequest, DeployStrategyRequest, DeleteStrategyClientRequest,
    StrategyResponse, StrategyInstanceSchema
)

# Import Services
from APIs.services import backtest_service
from APIs.services import strategy_service
from APIs.services import management_service

# Define Router
api_router = APIRouter(prefix="/api")

# ============================================================================
# Backtest Endpoints
# ============================================================================

@api_router.post("/run_backtest", response_model=UnifiedBacktestResponse, tags=["Centralized APIs"])
async def run_backtest(request: UnifiedBacktestRequest) -> UnifiedBacktestResponse:
    """
    Centralized backtest endpoint supporting all strategy types.
    Delegates to backtest_service.
    """
    return await backtest_service.execute_backtest(request)

@api_router.get("/health", tags=["Centralized APIs"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Unified WealthAI API",
        "supported_strategies": [
            "ETF_Rotation",
            "RS_ETF_Rotation",
            "RS_Stocks",
            "International_ETF_Rotation",
            "Stock_Rotation",
            "ETF_Payout",
            "SuperTrend"
        ]
    }

@api_router.get("/strategies", tags=["Centralized APIs"])
async def list_strategies():
    """List all supported strategy types"""
    # This static data could also be moved to a service if it grows
    return {
        "strategies": [
            {
                "type": "ETF_Rotation",
                "description": "Weekly SIP-based ETF rotation strategy",
                "required_params": ["tickers", "start_date", "end_date", "capital_per_week", "accumulation_weeks", "brokerage_percent"],
                "optional_params": ["compounding_enabled", "risk_free_rate"]
            },
            {
                "type": "RS_ETF_Rotation",
                "description": "Relative Strength ETF strategy with lump sum capital",
                "required_params": ["start_date", "end_date", "total_capital"],
                "optional_params": ["etf_universe", "custom_etfs", "max_positions", "lookback_weeks", "risk_free_rate"]
            },
            {
                "type": "RS_Stocks",
                "description": "Relative Strength stock strategy with lump sum capital",
                "required_params": ["start_date", "end_date", "total_capital"],
                "optional_params": ["stock_universe", "custom_stocks", "max_positions", "lookback_weeks", "risk_free_rate"]
            },
            {
                "type": "International_ETF_Rotation",
                "description": "Weekly SIP for international ETFs",
                "required_params": ["tickers", "start_date", "end_date", "capital_per_week", "accumulation_weeks", "brokerage_percent"],
                "optional_params": ["compounding_enabled", "risk_free_rate"]
            },
            {
                "type": "Stock_Rotation",
                "description": "Weekly SIP-based stock rotation strategy",
                "required_params": ["tickers", "start_date", "end_date", "capital_per_week", "accumulation_weeks", "brokerage_percent"],
                "optional_params": ["compounding_enabled", "risk_free_rate"]
            },
            {
                "type": "ETF_Payout",
                "description": "ETF rotation with periodic withdrawals",
                "required_params": ["tickers", "start_date", "end_date", "capital_per_week", "accumulation_weeks", "brokerage_percent"],
                "optional_params": ["compounding_enabled", "withdraw_amount", "payout_start_week", "risk_free_rate"]
            },
            {
                "type": "SuperTrend",
                "description": "Technical indicator-based trading strategy",
                "required_params": ["start_date", "end_date", "initial_capital"],
                "optional_params": ["brokerage_pct", "buffer_pct", "ema_short", "ema_long", "supertrend_period", "supertrend_stop_pct", "max_holdings", "symbols"]
            },
            {
                "type": "ETF_Buy_on_Dip",
                "description": "Monthly Buy-on-Dip strategy using 52-week high",
                "required_params": ["tickers", "start_date", "end_date", "capital_per_month"],
                "optional_params": ["brokerage_percent"]
            }
        ]
    }

# ============================================================================
# Strategy Information Endpoints
# ============================================================================

@api_router.get("/strategy/assets", tags=["Centralized APIs"])
async def get_assets(strategy_type: str = Query(..., description="Strategy type")):
    """Get available assets for the specified strategy type"""
    return await strategy_service.get_assets(strategy_type)

@api_router.get("/strategy/assets/overview", tags=["Centralized APIs"])
async def get_asset_overview(strategy_type: str = Query(..., description="Strategy type")):
    """Get detailed asset overview for the specified strategy type"""
    return await strategy_service.get_asset_overview(strategy_type)

@api_router.post("/strategy/date-range", tags=["Centralized APIs"])
async def calculate_date_range(request: DateRangeRequest):
    """Calculate date range for the specified strategy parameters"""
    return await strategy_service.calculate_date_range(request)

@api_router.get("/strategy/defaults", tags=["Centralized APIs"])
async def get_strategy_defaults(strategy_type: str = Query(..., description="Strategy type")):
    """Get default parameters for a strategy"""
    # This is a simple pass-through or static return, can stay here or move to service
    # Implementing minimal logic here for now as it wasn't complicated in original
    defaults = {
        "ETF_Rotation": {
            "tickers": ["NIFTYBEES.NS", "JUNIORBEES.NS", "GOLDBEES.NS"],
            "capital_per_week": 2500,
            "accumulation_weeks": 52,
            "brokerage_percent": 0.0,
            "compounding_enabled": False
        },
        "International_ETF_Rotation": {
            "tickers": ["MOQTY50.NS", "MON100.NS", "HNGSNGBEES.NS"],
            "capital_per_week": 10000,
            "accumulation_weeks": 52,
            "brokerage_percent": 0.0,
            "compounding_enabled": False
        },
        "RS_ETF_Rotation": {
            "total_capital": 1000000,
            "max_positions": 3,
            "etf_universe": "ALL_ETFS"
        },
        "RS_Stocks": {
            "total_capital": 1000000,
            "max_positions": 5,
            "stock_universe": "NIFTY_500"
        },
        "ETF_Buy_on_Dip": {
            "tickers": ["NIFTYBEES", "JUNIORBEES", "GOLDBEES", "LIQUIDBEES"],
            "capital_per_month": 5000,
            "brokerage_percent": 0.0
        }
    }
    return defaults.get(strategy_type, {})

@api_router.get("/strategy/transaction-log", tags=["Centralized APIs"])
async def get_transaction_log(strategy_type: str = Query(..., description="Strategy type")):
    """Get transaction log from the latest backtest result"""
    return await strategy_service.get_cached_transaction_log(strategy_type)

@api_router.get("/strategy/metrics", tags=["Centralized APIs"])
async def get_strategy_metrics(strategy_type: str = Query(..., description="Strategy type")):
    """Get calculated metrics from the latest backtest result"""
    # This reuses the transaction log service which returns the full result object including metrics
    return await strategy_service.get_cached_transaction_log(strategy_type)


# ============================================================================
# Strategy Management Endpoints
# ============================================================================

@api_router.post("/save_strategies", response_model=StrategyResponse, tags=["Centralized APIs"])
async def save_strategy(request: SaveStrategyRequest, db: Session = Depends(get_db)):
    """Save a new strategy configuration"""
    return management_service.save_strategy_logic(request, db)

@api_router.post("/deploy_strategy", response_model=StrategyResponse, tags=["Centralized APIs"])
async def deploy_strategy(request: DeployStrategyRequest, db: Session = Depends(get_db)):
    """Deploy a strategy, moving it to 'running' status"""
    return management_service.deploy_strategy_logic(request, db)

@api_router.post("/stop_strategy", response_model=StrategyResponse, tags=["Centralized APIs"])
async def stop_strategy(run_id: str = Query(..., description="Run ID of the strategy to stop"), db: Session = Depends(get_db)):
    """Stop a running strategy"""
    return management_service.stop_strategy_logic(run_id, db)

@api_router.post("/restart_strategy", response_model=StrategyResponse, tags=["Centralized APIs"])
async def restart_strategy(run_id: str = Query(..., description="Run ID of the strategy to restart"), db: Session = Depends(get_db)):
    """Restart a stopped strategy"""
    return management_service.restart_strategy_logic(run_id, db)

@api_router.delete("/delete_strategy", response_model=StrategyResponse, tags=["Centralized APIs"])
async def delete_strategy(run_id: str = Query(..., description="Run ID of the strategy to delete"), db: Session = Depends(get_db)):
    """Delete a strategy instance"""
    return management_service.delete_strategy_logic(run_id, db)

@api_router.post("/delete_strategy_client", response_model=StrategyResponse, tags=["Centralized APIs"])
async def delete_strategy_client(request: DeleteStrategyClientRequest, db: Session = Depends(get_db)):
    """Remove specific clients from the client_info JSONB column of a strategy"""
    return management_service.delete_strategy_client_logic(request, db)

@api_router.get("/get_instances", response_model=List[StrategyInstanceSchema], tags=["Centralized APIs"])
async def get_instances(
    user_id: str = Query(..., description="User ID to filter by"),
    strategy_type: str = Query(..., description="Strategy type to filter by"),
    db: Session = Depends(get_db)
):
    """Fetch all strategy instances for a specific user and strategy type"""
    return management_service.get_instances_logic(user_id, strategy_type, db)

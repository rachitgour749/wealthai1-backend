"""
Centralized Strategy Management APIs
Handles saving, deploying, stopping, restarting, and deleting strategy instances.
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy import text
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from Databases.app_data_db_connection import get_db
from Services.strategy_manager.models import SavedInstance
from Services.strategy_manager.schemas import (
    SaveStrategyRequest, 
    DeployStrategyRequest, 
    DeleteStrategyClientRequest,
    StrategyResponse,
    StrategyInstanceSchema
)
from Services.strategy_manager.utils import generate_run_id, get_next_trading_day

# Configure logging
logger = logging.getLogger(__name__)

# Create router
strategy_mgmt_router = APIRouter(tags=["strategy-management"])

@strategy_mgmt_router.post("/save_strategies", response_model=StrategyResponse)
async def save_strategy(request: SaveStrategyRequest, db: Session = Depends(get_db)):
    """
    Save a new strategy configuration
    """
    try:
        # Generate unique run_id
        run_id = generate_run_id(request.strategy_type)
        
        # Handle tickers (could be a list or a string)
        tickers_str = request.tickers
        if isinstance(request.tickers, list):
            tickers_str = ",".join(request.tickers)
        
        # Prepare strategies_parameters and ensure strategy_name consistency
        params = request.strategies_parameters or {}
        strat_name = request.strategy_name
        
        # If strategy_name is not at root but is in params, move it to root
        if not strat_name and "strategy_name" in params:
            strat_name = params["strategy_name"]
        
        # Remove strategy_name from params to avoid duplication in JSONB column
        # if you want it strictly separate.
        if "strategy_name" in params:
            params_copy = params.copy()
            params_copy.pop("strategy_name")
            params = params_copy

        # Create new instance
        new_instance = SavedInstance(
            user_id=request.user_id,
            strategy_name=strat_name,
            strategy_type=request.strategy_type,
            tickers=tickers_str,
            start_date=request.start_date,
            end_date=request.end_date,
            strategies_parameters=params,
            use_custom_date=request.use_custom_date,
            run_id=run_id,
            status="not deploy" # Default status
        )
        
        db.add(new_instance)
        db.commit()
        db.refresh(new_instance)
        
        logger.info(f"Saved new strategy instance: {run_id} for user {request.user_id}")
        
        return StrategyResponse(
            success=True,
            message="Strategy configuration saved successfully",
            run_id=run_id,
            data={"id": new_instance.id}
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving strategy: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save strategy: {str(e)}")

@strategy_mgmt_router.post("/deploy_strategy", response_model=StrategyResponse)
async def deploy_strategy(request: DeployStrategyRequest, db: Session = Depends(get_db)):
    """
    Deploy a strategy, moving it to 'running' status
    """
    try:
        # Find strategy by run_id
        strategy = db.query(SavedInstance).filter(SavedInstance.run_id == request.run_id).first()
        
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy with run_id {request.run_id} not found")
        
        # Update fields from payload
        strategy.client_info = request.client_info
        strategy.webhook_url = request.webhook_url
        strategy.reference_capital = request.reference_capital
        strategy.email_notification = request.email_notification
        strategy.telegram_notification = request.telegram_notification
        strategy.user_code = request.user_code
        
        # Set next execution date
        strategy.next_execution_date = get_next_trading_day()
        
        # Change status to running
        strategy.status = "running"
        # Note: updated_at is automatically handled by SQLAlchemy's onupdate=func.now()
        
        db.commit()
        
        logger.info(f"Deployed strategy: {request.run_id}, next execution: {strategy.next_execution_date}")
        
        return StrategyResponse(
            success=True,
            message=f"Strategy {request.run_id} deployed and is now running",
            run_id=request.run_id,
            data={"next_execution_date": strategy.next_execution_date.isoformat()}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deploying strategy: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to deploy strategy: {str(e)}")

@strategy_mgmt_router.post("/stop_strategy", response_model=StrategyResponse)
async def stop_strategy(run_id: str = Query(..., description="Run ID of the strategy to stop"), db: Session = Depends(get_db)):
    """
    Stop a running strategy
    """
    try:
        strategy = db.query(SavedInstance).filter(SavedInstance.run_id == run_id).first()
        
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy with run_id {run_id} not found")
        
        strategy.status = "stop"
        # Note: updated_at is automatically handled by SQLAlchemy's onupdate=func.now()
        
        db.commit()
        
        logger.info(f"Stopped strategy: {run_id}")
        
        return StrategyResponse(
            success=True,
            message=f"Strategy {run_id} has been stopped",
            run_id=run_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error stopping strategy: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop strategy: {str(e)}")

@strategy_mgmt_router.post("/restart_strategy", response_model=StrategyResponse)
async def restart_strategy(run_id: str = Query(..., description="Run ID of the strategy to restart"), db: Session = Depends(get_db)):
    """
    Restart a stopped strategy
    """
    try:
        strategy = db.query(SavedInstance).filter(SavedInstance.run_id == run_id).first()
        
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy with run_id {run_id} not found")
        
        strategy.status = "running"
        # Note: updated_at is automatically handled by SQLAlchemy's onupdate=func.now()
        
        db.commit()
        
        logger.info(f"Restarted strategy: {run_id}")
        
        return StrategyResponse(
            success=True,
            message=f"Strategy {run_id} has been restarted and is now running",
            run_id=run_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error restarting strategy: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to restart strategy: {str(e)}")

@strategy_mgmt_router.delete("/delete_strategy", response_model=StrategyResponse)
async def delete_strategy(run_id: str = Query(..., description="Run ID of the strategy to delete"), db: Session = Depends(get_db)):
    """
    Delete a strategy instance
    """
    try:
        strategy = db.query(SavedInstance).filter(SavedInstance.run_id == run_id).first()
        
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy with run_id {run_id} not found")
        
        db.delete(strategy)
        db.commit()
        
        logger.info(f"Deleted strategy: {run_id}")
        
        return StrategyResponse(
            success=True,
            message=f"Strategy {run_id} has been deleted",
            run_id=run_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting strategy: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete strategy: {str(e)}")

@strategy_mgmt_router.post("/delete_strategy_client", response_model=StrategyResponse)
async def delete_strategy_client(request: DeleteStrategyClientRequest, db: Session = Depends(get_db)):
    """
    Remove specific clients from the client_info JSONB column of a strategy
    """
    try:
        # 1. Find strategy by run_id
        strategy = db.query(SavedInstance).filter(SavedInstance.run_id == request.run_id).first()
        
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy with run_id {request.run_id} not found")
        
        # 2. Check if client_info exists
        if not strategy.client_info:
            return StrategyResponse(
                success=True,
                message=f"Strategy {request.run_id} has no client information to delete from",
                run_id=request.run_id
            )
        
        # 3. Remove specified clients
        current_clients = strategy.client_info.copy() # Work on a copy
        removed_clients = []
        
        for client_id in request.clients:
            if client_id in current_clients:
                current_clients.pop(client_id)
                removed_clients.append(client_id)
        
        if not removed_clients:
            return StrategyResponse(
                success=True,
                message="No matching clients found in strategy profile",
                run_id=request.run_id,
                data={"removed_count": 0}
            )
            
        # 4. Update and Save
        strategy.client_info = current_clients
        flag_modified(strategy, "client_info") # Ensure SQLAlchemy tracks the change
        strategy.updated_at = datetime.now() # Manually trigger for clarity or let model handle
        
        db.commit()
        
        logger.info(f"Removed clients {removed_clients} from strategy {request.run_id}")
        
        return StrategyResponse(
            success=True,
            message=f"Successfully removed {len(removed_clients)} clients from strategy {request.run_id}",
            run_id=request.run_id,
            data={"removed_clients": removed_clients, "remaining_count": len(current_clients)}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error removing strategy clients: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove strategy clients: {str(e)}")

@strategy_mgmt_router.get("/get_instances", response_model=List[StrategyInstanceSchema])
async def get_instances(
    user_id: str = Query(..., description="User ID to filter by"),
    strategy_type: str = Query(..., description="Strategy type to filter by"),
    db: Session = Depends(get_db)
):
    """
    Fetch all strategy instances for a specific user and strategy type
    """
    try:
        instances = db.query(SavedInstance).filter(
            SavedInstance.user_id == user_id,
            SavedInstance.strategy_type == strategy_type
        ).order_by(SavedInstance.created_at.desc()).all()
        
        return instances
        
    except Exception as e:
        logger.error(f"Error fetching instances for {user_id}/{strategy_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch instances: {str(e)}")

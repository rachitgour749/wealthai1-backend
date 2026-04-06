"""
Business logic for Strategy Management.
Handles saving, deploying, stopping, restarting, and deleting strategy instances.
"""
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from fastapi import HTTPException
from typing import List, Dict, Any, Optional
import logging
import importlib
from datetime import datetime

from Services.strategy_manager.models import SavedInstance
from Services.strategy_manager.schemas import (
    SaveStrategyRequest, 
    DeployStrategyRequest, 
    DeleteStrategyClientRequest,
    StrategyResponse,
    StrategyInstanceSchema
)
from Services.strategy_manager.utils import generate_run_id, get_next_trading_day
from Services.scheduler.executors.signal_executor import SignalExecutor

logger = logging.getLogger(__name__)

def save_strategy_logic(request: SaveStrategyRequest, db: Session) -> StrategyResponse:
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

def deploy_strategy_logic(request: DeployStrategyRequest, db: Session) -> StrategyResponse:
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
        
        # Set next execution date using the correct market calendar for this strategy
        strategy.next_execution_date = get_next_trading_day(strategy_type=strategy.strategy_type)
        
        # Change status to running
        strategy.status = "running"
        
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

def stop_strategy_logic(run_id: str, db: Session) -> StrategyResponse:
    """
    Stop a running strategy
    """
    try:
        strategy = db.query(SavedInstance).filter(SavedInstance.run_id == run_id).first()
        
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy with run_id {run_id} not found")
        
        strategy.status = "stop"
        
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

def restart_strategy_logic(run_id: str, db: Session) -> StrategyResponse:
    """
    Restart a stopped strategy
    """
    try:
        strategy = db.query(SavedInstance).filter(SavedInstance.run_id == run_id).first()
        
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy with run_id {run_id} not found")
        
        strategy.status = "running"
        
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

def delete_strategy_logic(run_id: str, db: Session) -> StrategyResponse:
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

def delete_strategy_client_logic(request: DeleteStrategyClientRequest, db: Session) -> StrategyResponse:
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
        strategy.updated_at = datetime.now() 
        
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

def get_instances_logic(user_id: str, strategy_type: str, db: Session) -> List[SavedInstance]:
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

def generate_signals_logic(strategy_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Manually trigger signal generation for one or all supported strategy types.
    """
    # Import generators only when needed to avoid circular dependencies
    from Services.scheduler.generators.etf_rotation_generator import generate_etf_rotation_signals
    from Services.scheduler.generators.rotation_stocks_generator import generate_stock_rotation_signals
    from Services.scheduler.generators.international_etf_generator import generate_international_etf_signals
    from Services.scheduler.generators.etf_swing_generator import generate_etf_swing_signals
    from Services.scheduler.generators.supertrend_generator import generate_supertrend_signals
    from Services.scheduler.generators.rs_etf_strategy_generator import generate_rs_etf_signals
    from Services.scheduler.generators.etf_payout_generator import generate_etf_payout_signals
    
    try:
        results = {}
        
        # Mapping of user-facing strategy types to their generator functions
        GENERATORS_MAP = {
            'Rotation_ETF_Payout': generate_etf_payout_signals,
            'ETF_Swing_Strategy': generate_etf_swing_signals,
            'Rotation_ETF': generate_etf_rotation_signals,
            'Rotation_International_ETF': generate_international_etf_signals,
            'RS_ETF': generate_rs_etf_signals,
            'SuperTrend': generate_supertrend_signals
        }
        
        # 1. Determine which strategies to trigger
        strategies_to_trigger = []
        if strategy_type:
            if strategy_type in GENERATORS_MAP:
                strategies_to_trigger.append(strategy_type)
            else:
                available = list(GENERATORS_MAP.keys())
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported strategy type: {strategy_type}. Supported: {available}"
                )
        else:
            # Trigger all supported strategies
            strategies_to_trigger = list(GENERATORS_MAP.keys())
            
        logger.info(f"Triggering signal generation for: {strategies_to_trigger}")
        
        # 2. Execute triggers
        for name in strategies_to_trigger:
            func = GENERATORS_MAP[name]
            logger.info(f"Running generator for: {name}...")
            try:
                # Most generators take no arguments or an optional signal_date
                func()
                results[name] = "Triggered"
            except Exception as gen_err:
                logger.error(f"Failed to trigger {name}: {gen_err}")
                results[name] = f"Error: {str(gen_err)}"
        
        return {
            "success": True,
            "message": "Signal generation process completed",
            "results": results,
            "triggered_strategies": strategies_to_trigger
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering signal generation: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def execute_signals_logic() -> Dict[str, Any]:
    """
    Manually trigger signal execution for all pending signals
    """
    try:
        logger.info("Manually triggering signal execution")
        # Note: SignalExecutor handles its own DB session internally
        executor = SignalExecutor()
        executor.execute_signals()
        
        return {
            "success": True,
            "message": "Signal execution process completed. Check logs for details."
        }
    except Exception as e:
        logger.error(f"Error triggering signal execution: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

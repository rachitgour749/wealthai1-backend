import logging
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime

from Services.strategy_manager.models import SavedInstance
from Services.strategy_manager.utils import generate_run_id, get_next_trading_day

logger = logging.getLogger(__name__)

def create_webhook_strategy(
    user_id: str,
    strategy_type: str,
    strategy_name: str,
    reference_capital: float,
    client_info: Dict[str, float],
    webhook: Optional[str],
    db: Session
) -> Dict[str, Any]:
    """
    Create a new External Strategy configuration (Webhook Strategy).
    Sets status='running' and source='other'.
    """
    try:
        # 1. Generate Run ID
        run_id = generate_run_id(strategy_type)
        
        # 2. Determine execution dates
        # Set next execution to the next trading day (or today if market open logic allows, keeping it simple for now)
        next_exe = get_next_trading_day()
        
        # 3. Create SavedInstance
        new_instance = SavedInstance(
            user_id=user_id,
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            # Core Config
            reference_capital=reference_capital,
            client_info=client_info,
            webhook_url=webhook, # Using 'webhook' from payload as the source identifier
            
            # Application Logic
            run_id=run_id,
            status='running', # Default running
            source='other',   # Default for external
            
            # Dates
            next_execution_date=next_exe,
            last_execution_date=None,
            
            # Defaults for unused fields in this context
            tickers=None,
            start_date=None,
            end_date=None,
            strategies_parameters={},
            use_custom_date=False,
            email_notification=False,
            telegram_notification=False,
            user_code=None,
            rem_exe_count=0
        )
        
        db.add(new_instance)
        db.commit()
        db.refresh(new_instance)
        
        logger.info(f"Created External Strategy: {run_id} for user {user_id}")
        
        return {
            "success": True,
            "message": "External strategy created successfully",
            "run_id": run_id,
            "status": "running",
            "next_execution_date": next_exe.isoformat()
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating webhook strategy: {e}")
        raise e

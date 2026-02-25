import logging
import uuid
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime

from Services.strategy_manager.models import SavedInstance
from Services.strategy_manager.utils import generate_run_id, get_next_trading_day
from Databases.webhook_models import WebhookIndividual

logger = logging.getLogger(__name__)

def create_webhook_strategy(
    user_id: str,
    strategy_type: str,
    strategy_name: str,
    reference_capital: float,
    client_info: Dict[str, float],
    webhook: Optional[str],
    webhook_type: str = "individual",
    db: Session = None
) -> Dict[str, Any]:
    """
    Create a new External Strategy configuration (Webhook Strategy).
    Sets status='running' and source='other'.
    Generates a unique webhook_key for security.
    """
    try:
        # 1. Generate Run ID
        run_id = generate_run_id(strategy_type)
        
        # 2. Generate Webhook Key
        webhook_key = str(uuid.uuid4())
        
        # 3. Determine execution dates
        next_exe = get_next_trading_day()
        
        # 4. Create SavedInstance
        new_instance = SavedInstance(
            user_id=user_id,
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            # Core Config
            reference_capital=reference_capital,
            client_info=client_info,
            webhook_url=webhook,
            webhook_key=webhook_key,
            
            # Application Logic
            run_id=run_id,
            status='running',
            source='other',
            
            # Dates
            next_execution_date=next_exe,
            last_execution_date=None,
            
            # Defaults
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
        
        # 5. Also store the webhook key in the webhook_keys table
        key_record = WebhookIndividual(
            user_email=user_id,
            run_id=run_id,
            strategy_name=strategy_name,
            webhook_key=webhook_key,
            webhook_type=webhook_type,
            is_active=True
        )
        db.add(key_record)
        
        db.commit()
        db.refresh(new_instance)
        
        logger.info(f"Created External Strategy: {run_id} for user {user_id} with webhook key")
        
        return {
            "success": True,
            "message": "External strategy created successfully",
            "run_id": run_id,
            "status": "running",
            "next_execution_date": next_exe.isoformat(),
            "webhook_key": webhook_key,
            "usage": {
                "header_name": "X-Webhook-Secret",
                "header_value": webhook_key,
                "note": "Add this as a custom header in your TradingView/signal alert. Keep it secret!"
            }
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating webhook strategy: {e}")
        raise e

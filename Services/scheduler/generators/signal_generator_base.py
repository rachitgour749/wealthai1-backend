"""
Signal Generator Base

Common helper functions for all strategy signal generators.
Provides utilities for fetching active instances, saving signals, and reusing backtester logic.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from Databases.app_data_db_connection import get_session
from Databases.signal_models import TradingSignal
from Services.strategy_manager.models import SavedInstance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _sanitize_client_json(client_json: Any) -> Any:
    """
    Sanitize client_json values (convert currency strings to float).
    Example: {"ID": "₹50,000.00"} -> {"ID": 50000.0}
    """
    if not isinstance(client_json, dict):
        return client_json
        
    sanitized = {}
    for key, value in client_json.items():
        if isinstance(value, str):
            # Check if it looks like a number/currency
            # Remove currency symbols and commas
            clean_val = value.replace('₹', '').replace('$', '').replace(',', '').strip()
            try:
                # Try converting to float
                sanitized[key] = float(clean_val)
            except ValueError:
                # If not a number, keep original
                sanitized[key] = value
        else:
            sanitized[key] = value
            
    return sanitized


def fetch_active_instances(strategy_type_val: str) -> List[Dict[str, Any]]:
    """
    Fetch all active instances for a specific strategy type from saved_instances table.
    
    Args:
        strategy_type_val: Type of the strategy (e.g., 'ETF Strategy', 'Stock Strategy')
    
    Returns:
        List of dictionaries containing instance data
    """
    session = get_session()
    
    try:
        # Query saved_instances where strategy_type matches and status is 'running' or 'deploy' (flexible)
        # Note: Filtering by TYPE allows users to name their strategies anything (e.g. "My Safe ETF")
        instances = session.query(SavedInstance).filter(
            SavedInstance.strategy_type == strategy_type_val,
            SavedInstance.status.in_(['running', 'deploy'])
        ).all()
        
        logger.info(f"Found {len(instances)} active instances for type: {strategy_type_val}")
        
        # Convert to dictionaries
        result = []
        for instance in instances:
            result.append({
                'id': instance.id,
                'user_id': instance.user_id,
                'user_code': instance.user_code,
                'strategy_name': instance.strategy_name,
                'strategy_type': instance.strategy_type,
                'run_id': instance.run_id,
                'tickers': instance.tickers,  # JSON field
                'strategies_parameters': instance.strategies_parameters,  # JSON field
                'client_json': _sanitize_client_json(instance.client_info),  # JSON field (mapped to client_json)
                'webhook_url': instance.webhook_url,
                'created_at': instance.created_at
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching active instances for {strategy_name}: {e}")
        import traceback
        traceback.print_exc()
        return []
        
    finally:
        session.close()


def save_signals_to_db(signals: List[TradingSignal]) -> bool:
    """
    Bulk insert trading signals to database
    
    Args:
        signals: List of TradingSignal objects
    
    Returns:
        True if successful, False otherwise
    """
    if not signals:
        logger.warning("No signals to save")
        return True
    
    session = get_session()
    
    try:
        # Create table if not exists (as requested by user)
        # Note: SQLAlchemy usually handles this with create_all, but we can try to be safe
        # In this context, we assume the table creation is handled by the app initialization
        # or we rely on the ORM to fail if table is missing.
        
        # Bulk insert
        session.bulk_save_objects(signals)
        session.commit()
        
        logger.info(f"✓ Saved {len(signals)} signals to database")
        return True
        
    except Exception as e:
        logger.error(f"Error saving signals to database: {e}")
        import traceback
        traceback.print_exc()
        session.rollback()
        return False
        
    finally:
        session.close()


def create_signal(
    user_id: str,
    run_id: str,
    user_code: str,
    strategy_name: str,
    strategy_type: str,
    order_side: str,
    symbol_name: str,
    client_json: Dict[str, Any],
    webhook_url: Optional[str],
    signal_date: datetime,
    score: float,
    price: float,
    high_52: float,
    low_52: float,
    executed_at: Optional[datetime] = None,
    execution_status: str = 'pending'
) -> TradingSignal:
    """
    Create a TradingSignal object with the unified schema
    """
    
    # Create signal object
    signal = TradingSignal(
        user_id=user_id,
        run_id=run_id,
        user_code=user_code,
        strategy_name=strategy_name,
        strategy_type=strategy_type,
        order_side=order_side,
        symbol_name=symbol_name,
        client_json=client_json,
        webhook_url=webhook_url,
        signal_date=signal_date,
        score=score,
        price=price,
        high_52=high_52,
        low_52=low_52,
        created_at=datetime.utcnow(),
        executed_at=executed_at,
        execution_status=execution_status
    )
    
    return signal


def get_pending_signals(strategy_name: Optional[str] = None, user_id: Optional[str] = None) -> List[TradingSignal]:
    """
    Get pending signals from database
    """
    session = get_session()
    
    try:
        query = session.query(TradingSignal).filter(
            TradingSignal.execution_status == 'pending'
        )
        
        if strategy_name:
            query = query.filter(TradingSignal.strategy_name == strategy_name)
        
        if user_id:
            query = query.filter(TradingSignal.user_id == user_id)
        
        signals = query.all()
        
        logger.info(f"Found {len(signals)} pending signals")
        return signals
        
    except Exception as e:
        logger.error(f"Error fetching pending signals: {e}")
        return []
        
    finally:
        session.close()


def mark_signal_executed(signal_id: int, execution_result: Dict[str, Any]) -> bool:
    """
    Mark a signal as executed
    
    Args:
        signal_id: Signal ID (primary key)
        execution_result: Result of execution
    """
    session = get_session()
    
    try:
        signal = session.query(TradingSignal).filter(
            TradingSignal.id == signal_id
        ).first()
        
        if not signal:
            logger.warning(f"Signal not found: {signal_id}")
            return False
        
        signal.execution_status = 'executed'
        signal.execution_result = execution_result
        signal.executed_at = datetime.utcnow()
        signal.updated_at = datetime.utcnow()
        
        session.commit()
        logger.info(f"✓ Marked signal as executed: {signal_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error marking signal as executed: {e}")
        session.rollback()
        return False
        
    finally:
        session.close()


def expire_old_signals(days: int = 7) -> int:
    """
    Expire signals older than specified days
    """
    session = get_session()
    
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Update pending signals older than cutoff date
        result = session.query(TradingSignal).filter(
            TradingSignal.execution_status == 'pending',
            TradingSignal.signal_date < cutoff_date
        ).update({
            'execution_status': 'expired',
            'updated_at': datetime.utcnow()
        })
        
        session.commit()
        
        if result > 0:
            logger.info(f"✓ Expired {result} old signals")
        
        return result
        
    except Exception as e:
        logger.error(f"Error expiring old signals: {e}")
        session.rollback()
        return 0
        
    finally:
        session.close()

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


def fetch_active_instances(strategy_name: str) -> List[Dict[str, Any]]:
    """
    Fetch all active instances for a specific strategy
    
    Args:
        strategy_name: Name of the strategy (e.g., 'ETF_Rotation', 'Rotation_Stocks')
    
    Returns:
        List of dictionaries containing instance data
    """
    session = get_session()
    
    try:
        # Query saved_instances where strategy_name matches and status is 'running'
        instances = session.query(SavedInstance).filter(
            SavedInstance.strategy_name == strategy_name,
            SavedInstance.status == 'running'
        ).all()
        
        logger.info(f"Found {len(instances)} active instances for {strategy_name}")
        
        # Convert to dictionaries
        result = []
        for instance in instances:
            result.append({
                'id': instance.id,
                'user_id': instance.user_id,
                'user_code': instance.user_code,
                'strategy_name': instance.strategy_name,
                'tickers': instance.tickers,  # JSON field
                'strategies_parameters': instance.strategies_parameters,  # JSON field
                'client_info': instance.client_info,  # JSON field
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
    strategy_name: str,
    user_id: str,
    user_code: str,
    instance_id: int,
    signal_type: str,
    symbol: str,
    price: float,
    signal_date: datetime,
    strategy_metadata: Dict[str, Any],
    client_info: Dict[str, Any],
    webhook_url: Optional[str] = None,
    quantity: Optional[int] = None,
    score: Optional[float] = None,
    execution_date: Optional[datetime] = None,
    expiry_days: int = 7
) -> TradingSignal:
    """
    Create a TradingSignal object
    
    Args:
        strategy_name: Name of the strategy
        user_id: User ID
        user_code: User code
        instance_id: Instance ID from saved_instances
        signal_type: BUY, SELL, or HOLD
        symbol: Trading symbol
        price: Current price
        signal_date: When signal was generated
        strategy_metadata: Strategy-specific metadata (52w high/low, RS scores, etc.)
        client_info: Client information for order placement
        webhook_url: Optional webhook URL
        quantity: Optional quantity
        score: Optional strategy score
        execution_date: Optional execution date
        expiry_days: Days until signal expires (default 7)
    
    Returns:
        TradingSignal object
    """
    # Generate unique signal ID
    signal_id = f"{strategy_name}_{user_id}_{symbol}_{signal_date.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # Calculate expiry date if not provided
    if not execution_date:
        execution_date = signal_date + timedelta(days=1)
    
    expiry_date = signal_date + timedelta(days=expiry_days)
    
    # Create signal object
    signal = TradingSignal(
        signal_id=signal_id,
        strategy_name=strategy_name,
        user_id=user_id,
        user_code=user_code,
        instance_id=instance_id,
        signal_type=signal_type,
        symbol=symbol,
        quantity=quantity,
        price=price,
        score=score,
        strategy_metadata=strategy_metadata,
        client_info=client_info,
        webhook_url=webhook_url,
        signal_date=signal_date,
        execution_date=execution_date,
        expiry_date=expiry_date,
        status='pending'
    )
    
    return signal


def get_pending_signals(strategy_name: Optional[str] = None, user_id: Optional[str] = None) -> List[TradingSignal]:
    """
    Get pending signals from database
    
    Args:
        strategy_name: Optional filter by strategy name
        user_id: Optional filter by user ID
    
    Returns:
        List of pending TradingSignal objects
    """
    session = get_session()
    
    try:
        query = session.query(TradingSignal).filter(
            TradingSignal.status == 'pending'
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


def mark_signal_executed(signal_id: str, execution_result: Dict[str, Any]) -> bool:
    """
    Mark a signal as executed
    
    Args:
        signal_id: Signal ID
        execution_result: Result of execution
    
    Returns:
        True if successful, False otherwise
    """
    session = get_session()
    
    try:
        signal = session.query(TradingSignal).filter(
            TradingSignal.signal_id == signal_id
        ).first()
        
        if not signal:
            logger.warning(f"Signal not found: {signal_id}")
            return False
        
        signal.status = 'executed'
        signal.execution_result = execution_result
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
    
    Args:
        days: Number of days after which to expire signals
    
    Returns:
        Number of signals expired
    """
    session = get_session()
    
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Update pending signals older than cutoff date
        result = session.query(TradingSignal).filter(
            TradingSignal.status == 'pending',
            TradingSignal.signal_date < cutoff_date
        ).update({
            'status': 'expired',
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

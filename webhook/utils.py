import re
import json
import logging
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import sqlite3
from fastapi import HTTPException

from webhook.models import StrategyCreate, StrategyUpdate, StrategyResponse, ErrorResponse, SuccessResponse

logger = logging.getLogger(__name__)

def validate_strategy_data(strategy_data: Dict[str, Any]) -> bool:
    """
    Validate strategy data
    
    Args:
        strategy_data: Strategy configuration data
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if not isinstance(strategy_data, dict):
            return False
        
        # Check for required fields based on strategy type
        if 'strategy_type' not in strategy_data:
            return False
        
        strategy_type = strategy_data.get('strategy_type', '').lower()
        
        if strategy_type in ['stock', 'stock_rotation']:
            required_fields = ['tickers', 'rebalance_frequency', 'lookback_period']
        elif strategy_type in ['etf', 'etf_rotation']:
            required_fields = ['etfs', 'rebalance_frequency', 'lookback_period']
        else:
            return False
        
        for field in required_fields:
            if field not in strategy_data:
                return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating strategy data: {e}")
        return False

def generate_json_data(client_ids: List[str], capitals: List[float], strategy_type: str) -> Dict[str, Any]:
    """
    Generate JSON data for trading orders based on client IDs and capitals
    
    Args:
        client_ids: List of client IDs
        capitals: List of capital amounts
        strategy_type: Type of strategy
        
    Returns:
        Generated JSON data
    """
    try:
        if len(client_ids) != len(capitals):
            raise ValueError("Number of client IDs must match number of capital amounts")
        
        orders = []
        for i, (client_id, capital) in enumerate(zip(client_ids, capitals)):
            order = {
                "order_id": f"order_{i+1}_{int(datetime.now().timestamp())}",
                "client_id": client_id,
                "capital": capital,
                "strategy_type": strategy_type,
                "timestamp": datetime.now().isoformat(),
                "status": "pending"
            }
            orders.append(order)
        
        return {
            "orders": orders,
            "total_orders": len(orders),
            "total_capital": sum(capitals),
            "generated_at": datetime.now().isoformat(),
            "strategy_type": strategy_type
        }
    except Exception as e:
        logger.error(f"Error generating JSON data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate JSON data: {str(e)}")

def send_webhook_notification(webhook_url: str, data: Dict[str, Any], timeout: int = 30) -> bool:
    """
    Send webhook notification
    
    Args:
        webhook_url: Webhook URL
        data: Data to send
        timeout: Request timeout in seconds
        
    Returns:
        True if successful, False otherwise
    """
    try:
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'StrategyManagement-Webhook/1.0'
        }
        
        response = requests.post(
            webhook_url,
            json=data,
            headers=headers,
            timeout=timeout
        )
        
        response.raise_for_status()
        logger.info(f"Webhook notification sent successfully to {webhook_url}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send webhook notification to {webhook_url}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending webhook notification: {e}")
        return False

def log_strategy_operation(operation: str, strategy_id: Optional[int], user_email: Optional[str], details: Dict[str, Any]):
    """
    Log strategy operation
    
    Args:
        operation: Type of operation
        strategy_id: Strategy ID
        user_email: User email
        details: Additional details
    """
    try:
        log_data = {
            "operation": operation,
            "strategy_id": strategy_id,
            "user_email": user_email,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        
        logger.info(f"Strategy operation: {json.dumps(log_data)}")
        
    except Exception as e:
        logger.error(f"Error logging strategy operation: {e}")

def create_error_response(error: str, detail: Optional[str] = None) -> ErrorResponse:
    """
    Create standardized error response
    
    Args:
        error: Error message
        detail: Additional error details
        
    Returns:
        ErrorResponse object
    """
    return ErrorResponse(
        error=error,
        detail=detail,
        timestamp=datetime.now()
    )

def create_success_response(message: str, data: Optional[Dict[str, Any]] = None) -> SuccessResponse:
    """
    Create standardized success response
    
    Args:
        message: Success message
        data: Additional data
        
    Returns:
        SuccessResponse object
    """
    return SuccessResponse(
        message=message,
        data=data,
        timestamp=datetime.now()
    )

def sanitize_input(input_string: str) -> str:
    """
    Sanitize input string
    
    Args:
        input_string: Input string to sanitize
        
    Returns:
        Sanitized string
    """
    if not input_string:
        return ""
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', input_string)
    
    # Trim whitespace
    sanitized = sanitized.strip()
    
    return sanitized

def validate_webhook_url(url: str) -> bool:
    """
    Validate webhook URL
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Basic URL validation
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        return bool(url_pattern.match(url))
    except Exception:
        return False

def get_database_connection(database_path: str) -> sqlite3.Connection:
    """
    Get database connection
    
    Args:
        database_path: Path to database file
        
    Returns:
        Database connection
    """
    try:
        conn = sqlite3.connect(database_path)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

def cleanup_old_json_data(database_path: str, retention_days: int = 30):
    """
    Clean up old JSON data
    
    Args:
        database_path: Path to database file
        retention_days: Number of days to retain data
    """
    try:
        conn = get_database_connection(database_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        cursor.execute(
            "DELETE FROM saved_json WHERE created_at < ?",
            (cutoff_date.isoformat(),)
        )
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old JSON records")
            
    except Exception as e:
        logger.error(f"Error cleaning up old JSON data: {e}")

def format_strategy_response(strategy_row: sqlite3.Row) -> StrategyResponse:
    """
    Format strategy row to StrategyResponse
    
    Args:
        strategy_row: Database row
        
    Returns:
        StrategyResponse object
    """
    try:
        strategy_data = json.loads(strategy_row['strategy_data']) if strategy_row['strategy_data'] else {}
        
        return StrategyResponse(
            id=strategy_row['id'],
            strategy_name=strategy_row['strategy_name'],
            user_email=strategy_row['user_email'],
            webhook=strategy_row['webhook'],
            strategy_data=strategy_data,
            is_active=bool(strategy_row['is_active']),
            created_at=datetime.fromisoformat(strategy_row['created_at']),
            updated_at=datetime.fromisoformat(strategy_row['updated_at'])
        )
    except Exception as e:
        logger.error(f"Error formatting strategy response: {e}")
        raise HTTPException(status_code=500, detail="Error formatting strategy data")

"""Utility functions for portfolio service"""
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


def get_strategy_by_run_id(run_id: str, db: Session) -> Optional[Dict]:
    """
    Get strategy details from run_id
    
    Searches across all strategy tables:
    - etf_saved_strategy
    - stock_saved_strategy
    - rs_etf_instance
    - rs_stock_instance
    """
    
    tables = [
        ('etf_saved_strategy', 'ETF_Rotation'),
        ('stock_saved_strategy', 'Stock_Rotation'),
        ('rs_etf_instance', 'RS_ETF'),
        ('rs_stock_instance', 'RS_Stocks')
    ]
    
    for table_name, default_type in tables:
        try:
            query = text(f"""
                SELECT id, user_id, strategy_name, strategy_type, status 
                FROM {table_name} 
                WHERE run_id = :run_id
            """)
            
            result = db.execute(query, {"run_id": run_id}).fetchone()
            
            if result:
                return {
                    'id': result[0],
                    'user_id': result[1],
                    'strategy_name': result[2],
                    'strategy_type': result[3] or default_type,
                    'status': result[4] or 'deploy'
                }
        except Exception as e:
            logger.warning(f"Error querying {table_name}: {e}")
            continue
    
    return None


def calculate_brokerage(quantity: int, price: float, side: str) -> float:
    """
    Calculate brokerage based on trade value
    
    Customize based on your broker's rules
    Example: 0.03% or ₹20, whichever is lower
    """
    trade_value = quantity * price
    
    # Example calculation
    brokerage = min(trade_value * 0.0003, 20.0)
    
    return round(brokerage, 2)


def calculate_taxes(quantity: int, price: float, side: str) -> float:
    """
    Calculate taxes (STT, GST, etc.)
    
    Customize based on Indian tax rules
    Example: 0.1% for equity delivery
    """
    trade_value = quantity * price
    
    # Example calculation
    # STT: 0.1% on buy/sell
    stt = trade_value * 0.001
    
    return round(stt, 2)

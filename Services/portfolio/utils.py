"""Utility functions for portfolio service"""
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import Optional, Dict
import logging
import json

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


def get_client_allocations(run_id: str, db: Session) -> Dict[str, float]:
    """
    Get allocated capital for each client in a strategy run
    Returns: Dict[client_code, capital_amount]
    """
    try:
        # 1. Check saved_instances first (new standard)
        query_client_info = text("""
            SELECT client_info
            FROM saved_instances
            WHERE run_id = :run_id
        """)
        
        client_info_result = db.execute(query_client_info, {"run_id": run_id}).fetchone()
        client_info = None
        
        if client_info_result and client_info_result[0]:
            client_info = client_info_result[0]
        else:
            # 2. Fallback to legacy tables
            legacy_tables = [
                'etf_saved_strategy',
                'stock_saved_strategy',
                'rs_etf_instance',
                'rs_stock_instance'
            ]
            
            for table in legacy_tables:
                try:
                    query_legacy = text(f"""
                        SELECT client_information_json
                        FROM {table}
                        WHERE run_id = :run_id
                    """)
                    legacy_result = db.execute(query_legacy, {"run_id": run_id}).fetchone()
                    
                    if legacy_result and legacy_result[0]:
                        client_info = legacy_result[0]
                        break
                except Exception as e:
                    logger.warning(f"Error checking {table} for client info: {e}")
                    continue
        
        allocations = {}
        
        if client_info:
            # Parse if string
            if isinstance(client_info, str):
                try:
                    client_info = json.loads(client_info)
                except:
                    logger.warning(f"Could not parse client_info JSON for run_id={run_id}")
                    return {}
            
            # Process values
            for client_code, capital_str in client_info.items():
                try:
                    # Clean currency string: remove ₹, $, commas, and spaces
                    cleaned_capital = str(capital_str).replace('₹', '').replace('$', '').replace(',', '').strip()
                    allocations[client_code] = float(cleaned_capital)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert capital to float for client {client_code}: {e}")
                    continue
                    
        return allocations
        
    except Exception as e:
        logger.error(f"Error fetching client allocations for {run_id}: {e}")
        return {}


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


from datetime import date, datetime
from typing import List, Tuple


def calculate_cagr(initial_value: float, current_value: float, start_date: date, end_date: date) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR)
    
    Formula: CAGR = [(Ending Value / Beginning Value)^(1 / Years)] - 1
    
    Args:
        initial_value: Starting investment amount
        current_value: Current portfolio value (AUM)
        start_date: Date of first investment
        end_date: Current date
    
    Returns:
        CAGR as percentage (e.g., 15.5 for 15.5%)
    """
    if initial_value <= 0 or current_value <= 0:
        return 0.0
    
    # Calculate time period in years
    days_elapsed = (end_date - start_date).days
    if days_elapsed <= 0:
        return 0.0
    
    years = days_elapsed / 365.25  # Account for leap years
    
    if years < 0.01:  # Less than ~4 days, too short for meaningful CAGR
        return 0.0
    
    try:
        # Check if the ratio is too extreme (would cause overflow)
        ratio = current_value / initial_value
        
        # Cap extreme ratios to prevent overflow
        # If ratio > 1000, it means 100,000% return which is unrealistic for CAGR
        if ratio > 1000:
            logger.warning(f"CAGR: Extreme ratio detected ({ratio:.2f}), capping calculation")
            return 0.0
        
        # CAGR formula
        cagr = (pow(ratio, 1 / years) - 1) * 100
        
        # Cap CAGR at reasonable bounds (-100% to 1000%)
        if cagr > 1000:
            logger.warning(f"CAGR exceeds 1000%, returning 0: {cagr}")
            return 0.0
        if cagr < -100:
            return -100.0
            
        return round(cagr, 2)
    except (ValueError, ZeroDivisionError, OverflowError) as e:
        logger.error(f"CAGR calculation error: {e}")
        return 0.0


def calculate_xirr(cash_flows: List[float], dates: List[date], guess: float = 0.1) -> float:
    """
    Calculate Extended Internal Rate of Return (XIRR) using Newton-Raphson method
    
    Args:
        cash_flows: List of cash flows (negative for investments, positive for returns)
        dates: List of corresponding dates
        guess: Initial guess for XIRR (default 0.1 = 10%)
    
    Returns:
        XIRR as percentage (e.g., 15.5 for 15.5%)
    """
    if len(cash_flows) != len(dates) or len(cash_flows) < 2:
        return 0.0
    
    # Validate: sum should be positive (net gain) or close to zero
    if sum(cash_flows) < -abs(cash_flows[0]) * 0.1:  # Allow 10% tolerance
        return 0.0
    
    # Convert dates to years from start date
    start_date = min(dates)
    years = [(d - start_date).days / 365.25 for d in dates]
    
    # Newton-Raphson iteration
    rate = guess
    max_iterations = 100
    tolerance = 1e-6
    
    for iteration in range(max_iterations):
        # Calculate NPV (Net Present Value)
        npv = sum(cf / pow(1 + rate, yr) for cf, yr in zip(cash_flows, years))
        
        # Calculate derivative of NPV
        npv_derivative = sum(-yr * cf / pow(1 + rate, yr + 1) for cf, yr in zip(cash_flows, years))
        
        # Check for convergence
        if abs(npv) < tolerance:
            return round(rate * 100, 2)
        
        # Avoid division by zero
        if abs(npv_derivative) < 1e-10:
            break
        
        # Newton-Raphson update
        new_rate = rate - npv / npv_derivative
        
        # Prevent extreme values (rate between -99% and 1000%)
        new_rate = max(-0.99, min(new_rate, 10.0))
        
        # Check if rate change is small enough
        if abs(new_rate - rate) < tolerance:
            return round(new_rate * 100, 2)
        
        rate = new_rate
    
    # If didn't converge, try different starting points
    for fallback_guess in [0.5, -0.5, 0.01, 0.2]:
        if fallback_guess == guess:
            continue
        
        try:
            result = calculate_xirr(cash_flows, dates, fallback_guess)
            if result != 0.0:
                return result
        except:
            continue
    
    return 0.0


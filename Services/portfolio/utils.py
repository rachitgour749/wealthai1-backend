"""Utility functions for portfolio service"""
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import Optional, Dict
import logging
import json

# Configure logging
logger = logging.getLogger('Services.portfolio.utils')
logger.setLevel(logging.DEBUG)

# Add console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def get_strategy_by_run_id(run_id: str, db: Session) -> Optional[Dict]:
    """
    Get strategy details from run_id
    
    Queries the saved_instances table only.
    """
    
    try:
        query = text("""
            SELECT id, user_id, strategy_name, strategy_type, status 
            FROM saved_instances 
            WHERE run_id = :run_id
        """)
        
        result = db.execute(query, {"run_id": run_id}).fetchone()
        
        if result:
            return {
                'id': result[0],
                'user_id': result[1],
                'strategy_name': result[2],
                'strategy_type': result[3],
                'status': result[4] or 'deploy'
            }
            
        return None
        
    except Exception as e:
        logger.warning(f"Error querying saved_instances for run_id {run_id}: {e}")
        return None


def get_client_allocations(run_id: str, db: Session) -> Dict[str, float]:
    """
    Get allocated capital for each client in a strategy run
    Returns: Dict[client_code, capital_amount]
    """
    try:
        # Query saved_instances table only
        query_client_info = text("""
            SELECT client_info
            FROM saved_instances
            WHERE run_id = :run_id
        """)
        
        client_info_result = db.execute(query_client_info, {"run_id": run_id}).fetchone()
        
        if not client_info_result or not client_info_result[0]:
            logger.warning(f"No client_info found in saved_instances for run_id={run_id}")
            return {}
        
        client_info = client_info_result[0]
        allocations = {}
        
        # Parse if string
        if isinstance(client_info, str):
            try:
                client_info = json.loads(client_info)
            except Exception as e:
                logger.warning(f"Could not parse client_info JSON for run_id={run_id}: {e}")
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
        db.rollback()
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
    # Round inputs to 2 decimals
    initial_value = round(float(initial_value), 2)
    current_value = round(float(current_value), 2)
    
    logger.debug("="*80)
    logger.debug("CAGR CALCULATION START")
    logger.debug(f"  Initial Value: ₹{initial_value:,.2f}")
    logger.debug(f"  Current Value: ₹{current_value:,.2f}")
    logger.debug(f"  Start Date: {start_date}")
    logger.debug(f"  End Date: {end_date}")
    
    if initial_value <= 0 or current_value <= 0:
        logger.warning(f"  Invalid values - Initial: {initial_value}, Current: {current_value}")
        logger.debug("  CAGR Result: 0.0 (invalid values)")
        logger.debug("="*80)
        return 0.0
    
    # Calculate time period in years
    days_elapsed = (end_date - start_date).days
    logger.debug(f"  Days Elapsed: {days_elapsed}")
    
    if days_elapsed <= 0:
        logger.warning(f"  Invalid time period - Days: {days_elapsed}")
        logger.debug("  CAGR Result: 0.0 (invalid time period)")
        logger.debug("="*80)
        return 0.0
    
    years = round(days_elapsed / 365.25, 4)  # Account for leap years
    logger.debug(f"  Years: {years:.4f}")
    
    # Minimum 1 week (7 days) required for CAGR calculation
    if days_elapsed < 7:
        logger.warning(f"  Time period too short - Days: {days_elapsed} (minimum 7 required)")
        logger.debug("  CAGR Result: 0.0 (< 1 week)")
        logger.debug("="*80)
        return 0.0
    
    try:
        # Check if the ratio is too extreme (would cause overflow)
        ratio = round(current_value / initial_value, 4)
        logger.debug(f"  Ratio (Current/Initial): {ratio:.4f}")
        
        # CAGR formula
        exponent = round(1 / years, 4)
        logger.debug(f"  Exponent (1/years): {exponent:.4f}")
        
        base_result = round(pow(ratio, exponent), 4)
        logger.debug(f"  Base Result (ratio^exponent): {base_result:.4f}")
        
        cagr = round((base_result - 1) * 100, 2)
        logger.debug(f"  Raw CAGR: {cagr:.2f}%")
        
        # Cap CAGR at reasonable bounds to prevent overflow (-100% to 10,000%)
        # High values are expected for short-term strategies
        if cagr > 10000:
            logger.warning(f"  CAGR exceeds 10,000% ({cagr:.2f}%), capping at 10,000%")
            logger.debug("  CAGR Result: 10000.00% (capped)")
            logger.debug("="*80)
            return 10000.0
        if cagr < -100:
            logger.warning(f"  CAGR below -100% ({cagr:.2f}%), capping at -100%")
            logger.debug("  CAGR Result: -100.00% (capped)")
            logger.debug("="*80)
            return -100.0
        
        final_cagr = round(cagr, 2)
        logger.debug(f"  CAGR Result: {final_cagr:.2f}%")
        logger.debug("="*80)
        return final_cagr
    except (ValueError, ZeroDivisionError, OverflowError) as e:
        logger.error(f"  CAGR calculation error: {e}")
        logger.debug("  CAGR Result: 0.00 (error)")
        logger.debug("="*80)
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


"""Utility functions for centralized strategy management system"""
from datetime import datetime, date, timedelta
import random
import string
import holidays
import pytz
from typing import Optional

def generate_run_id(strategy_type: str) -> str:
    """
    Generate a unique run_id for a strategy
    Format: {suffix}_{DDMMYY}_{uniquecode}
    """
    # Define suffixes
    suffix_map = {
        "ETF_Rotation": "ETF",
        "RS_ETF_Rotation": "RS_ETF",
        "ETF_Payout": "ETF_Payout",
        "RS_ETF_Rotation": "RS_ETF",
        "ETF_Payout": "ETF_Payout",
        "International_ETF_Rotation": "ETF_US",
        "External_Strategy": "EXT"
    }
    
    suffix = suffix_map.get(strategy_type, "STRAT")
    
    # Current date in DDMMYY format
    date_str = datetime.now().strftime("%d%m%y")
    
    # Unique 2-digit code (random lowercase letters/digits for variety, or just digits)
    unique_code = ''.join(random.choices(string.digits, k=2))
    
    # Return formatted run_id
    # Note: For production, you might want to check the DB for uniqueness, 
    # but with 2 digits it's 100 possibilities per day per strategy type.
    return f"{suffix}_{date_str}_{unique_code}"

def get_next_trading_day() -> datetime:
    """
    Get the first trading day of the current week.
    If today is Sunday, returns Monday (if not a holiday).
    If Monday is a holiday, return Tuesday, etc.
    Returns timezone-aware datetime in IST (Asia/Kolkata)
    """
    # Using Indian holidays as the system seems focused on NSE
    ind_holidays = holidays.India()
    
    # Get IST timezone
    ist = pytz.timezone('Asia/Kolkata')
    
    now = datetime.now(ist)
    # Find the start of the week (Monday)
    # weekday() returns 0 for Monday, 6 for Sunday
    days_to_monday = now.weekday()
    monday = now - timedelta(days=days_to_monday)
    
    # If it's already past Monday of this week, we might want the NEXT Monday?
    # The requirement says "suppose today is sunday then first trading day of a week is monday"
    # This implies we want the first trading day of the *upcoming* or *current* week.
    
    # If today is Sat or Sun, we look at upcoming Monday.
    # If today is Mon-Fri, maybe we look at today or next Mon?
    # Usually "first trading day of week" means Monday of that week.
    
    target_day = monday
    
    # If Monday is in the past, move to next Monday? 
    # Actually, the user says "suppose today is sunday", so let's assume if it's weekend, we look ahead.
    if now.weekday() >= 5: # Sat or Sun
        target_day = now + timedelta(days=(7 - now.weekday()) % 7)
    
    # Iterate to find the first non-holiday, non-weekend day
    # Start checking from target_day (which should be a Monday)
    current_check = target_day
    while True:
        # 5 is Saturday, 6 is Sunday
        if current_check.weekday() < 5 and current_check.date() not in ind_holidays:
            # Set to 9:00 AM for execution and return timezone-aware datetime
            return current_check.replace(hour=9, minute=0, second=0, microsecond=0)
        current_check += timedelta(days=1)
        
        # Safety break
        if (current_check - target_day).days > 14:
            return target_day.replace(hour=9, minute=0, second=0, microsecond=0)

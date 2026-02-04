import datetime

def is_market_open():
    """
    Checks if the Indian stock market (NSE/BSE) is currently open.
    Market hours: 09:15 to 15:30 IST, Monday to Friday.
    """
    # Indian Standard Time is UTC + 5:30
    ist_offset = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    now_ist = datetime.datetime.now(ist_offset)
    
    # Check if it's a weekend (Monday=0, Sunday=6)
    if now_ist.weekday() >= 5:
        return False
        
    # Current time in minutes from midnight for easy comparison
    current_minutes = now_ist.hour * 60 + now_ist.minute
    
    start_minutes = 9 * 60 + 15   # 09:15
    end_minutes = 15 * 60 + 30    # 15:30
    
    return start_minutes <= current_minutes <= end_minutes

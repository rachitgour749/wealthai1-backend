import random
import logging
from Services.portfolio.price_service import PriceService

logger = logging.getLogger(__name__)

# Master Credentials for IP Validation
IP_USERNAME = "anjanr"
IP_PASSWORD = "vRDunhrENZR"
IP_PORT = "50100"

def generate_random_mac():
    """Generates a random MAC address string for audit headers."""
    return ":".join(["{:02x}".format(random.randint(0, 255)) for _ in range(6)])

def calculate_limit_price(symbol, exchange, side, buffer_pct=0.005):
    """
    Fetches LTP and calculates a buffered limit price.
    Returns (price, error_message)
    """
    try:
        ltp = PriceService.get_current_price(symbol, exchange)
        if not ltp or ltp <= 0:
            return None, f"Could not fetch current price for {symbol} on {exchange}"
        
        # Apply buffer (0.5% default)
        if side.upper() == "BUY":
            price = round(ltp * (1 + buffer_pct), 2)
        else:
            price = round(ltp * (1 - buffer_pct), 2)
            
        logger.info(f"Calculated limit price for {symbol}: LTP={ltp}, Side={side}, Limit={price}")
        return price, None
    except Exception as e:
        logger.error(f"Error calculating limit price: {e}")
        return None, str(e)

def validate_user_ip_creds(user_record):
    """
    Validates if user's static IP credentials match the master backend values.
    Returns (is_valid, error_message)
    """
    if not user_record:
        return False, "User broker session record not found"
    
    if not user_record.static_ip:
        return False, "Static IP not configured for this account. SEBI mandate requires whitelisted static IP."
    
    if (user_record.static_ip_username != IP_USERNAME or 
        user_record.static_ip_password != IP_PASSWORD or 
        user_record.static_ip_port != IP_PORT):
        return False, "Invalid Static IP credentials (Username/Password/Port mismatch)."
    
    return True, None

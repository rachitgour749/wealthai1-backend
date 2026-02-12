"""
MOSWAL Broker Mapping Functions
Maps common order parameters to MOSWAL-specific API fields
"""

def map_exchange(exchange):
    """
    Maps frontend exchange names to MOSWAL exchange codes.
    
    MOSWAL Exchange Codes:
    - NSE: "NSE"
    - BSE: "BSE"
    - NSEFO: "NSEFO" (NSE Futures & Options)
    - MCX: "MCX"
    - MCXFO: "MCX" (MCX Futures & Options)
    """
    if not exchange:
        return "NSE"  # Default to NSE
        
    exchange = str(exchange).upper().strip()
    
    # Normalize exchange codes
    mapping = {
        "NSECM": "NSE",
        "NSE": "NSE",
        "BSECM": "BSE",
        "BSE": "BSE",
        "NFO": "NSEFO",
        "NSEFO": "NSEFO",
        "MCX": "MCX",
        "MCXFO": "MCX",
        "CDS": "CDS"
    }
    return mapping.get(exchange, "NSE")


def map_validity(validity):
    """
    Maps frontend validity to MOSWAL validity codes.
    
    MOSWAL Validity:
    - DAY: "DAY"
    - IOC: "IOC"
    """
    if not validity:
        return "DAY"  # Default to DAY
        
    validity = str(validity).upper().strip()
    mapping = {
        "DAY": "DAY",
        "IOC": "IOC"
    }
    return mapping.get(validity, "DAY")


def map_order_side(order_side):
    """
    Maps frontend order_side (BUY/SELL) to MOSWAL transaction type.
    
    MOSWAL Transaction Type:
    - BUY: "BUY"
    - SELL: "SELL"
    """
    if not order_side:
        return "BUY"
    
    order_side = str(order_side).upper().strip()
    mapping = {
        "BUY": "BUY",
        "SELL": "SELL"
    }
    return mapping.get(order_side, "BUY")


def map_product_type(product_type, exchange=None):
    """
    Maps frontend product_type to MOSWAL product codes.
    
    MOSWAL Product Type:
    - DELIVERY: "DELIVERY" (for NSE/BSE equity)
    - NORMAL: "NORMAL" (for F&O segments)
    - INTRADAY: "INTRADAY"
    - MARGIN: "MARGIN"
    - CO: "CO" (Cover Order)
    - BO: "BO" (Bracket Order)
    
    Note: DELIVERY is converted to NORMAL for F&O segments (NSEFO, MCX)
    """
    if not product_type:
        return "DELIVERY"
    
    product_type = str(product_type).upper().strip()
    
    mapping = {
        "DELIVERY": "DELIVERY",
        "CNC": "DELIVERY",
        "INTRADAY": "INTRADAY",
        "MIS": "INTRADAY",  # Map MIS to INTRADAY
        "MARGIN": "MARGIN",
        "NRML": "NORMAL",   # Map NRML to NORMAL
        "NORMAL": "NORMAL",
        "CO": "CO",
        "BO": "BO"
    }
    
    result = mapping.get(product_type, "DELIVERY")
    
    # Convert DELIVERY to NORMAL for F&O segments
    if result == "DELIVERY" and exchange and exchange.upper() in ["NSEFO", "MCX", "MCXFO"]:
        result = "NORMAL"
    
    return result


def map_order_type(order_type):
    """
    Maps frontend order_type to MOSWAL order_type.
    
    MOSWAL Order Type:
    - MARKET: "MARKET"
    - LIMIT: "LIMIT"
    - STOPLOSS: "STOPLOSS"
    - STOPLOSSLIMIT: "STOPLOSSLIMIT"
    """
    if not order_type:
        return "MARKET"
    
    order_type = str(order_type).upper().strip()
    mapping = {
        "MARKET": "MARKET",
        "LIMIT": "LIMIT",
        "SL": "STOPLOSS",
        "SL-M": "STOPLOSS",
        "SLM": "STOPLOSS",
        "STOPLOSS": "STOPLOSS",
        "SL-L": "STOPLOSSLIMIT",
        "STOPLOSSLIMIT": "STOPLOSSLIMIT"
    }
    return mapping.get(order_type, "MARKET")

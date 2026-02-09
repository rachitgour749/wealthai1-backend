"""
DHAN Broker Mapping Functions
Maps common order parameters to DHAN-specific API fields
"""

def map_exchange(exchange):
    """
    Maps frontend exchange names to DHAN exchange segment IDs.
    
    DHAN Exchange Segment IDs:
    - NSE: 0
    - BSE: 1
    - NFO: 2
    - MCX: 3
    - CDS: 4
    """
    if not exchange:
        return 0  # Default to NSE
        
    exchange = str(exchange).upper().strip()
    mapping = {
        "NSE": 0,
        "BSE": 1,
        "NFO": 2,  # NSE Futures & Options
        "MCX": 3,
        "CDS": 4,  # Currency Derivatives
        "BCD": 4   # Assuming BCD maps to CDS
    }
    return mapping.get(exchange, 0)


def map_validity(validity):
    """
    Maps frontend validity to DHAN validity codes.
    
    DHAN Validity:
    - DAY: 0
    - IOC: 1
    """
    if not validity:
        return 0  # Default to DAY
        
    validity = str(validity).upper().strip()
    mapping = {
        "DAY": 0,
        "IOC": 1
    }
    return mapping.get(validity, 0)


def map_order_side(order_side):
    """
    Maps frontend order_side (BUY/SELL) to DHAN transaction type.
    
    DHAN Transaction Type:
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


def map_product_type(product_type):
    """
    Maps frontend product_type to DHAN product codes.
    
    DHAN Product Type:
    - CNC (Cash and Carry): "CNC"
    - INTRADAY: "INTRADAY"
    - MARGIN: "MARGIN"
    - CO (Cover Order): "CO"
    - BO (Bracket Order): "BO"
    """
    if not product_type:
        return "CNC"
    
    product_type = str(product_type).upper().strip()
    
    mapping = {
        "DELIVERY": "CNC",
        "CNC": "CNC",
        "INTRADAY": "INTRADAY",
        "MIS": "INTRADAY",  # Map MIS to INTRADAY
        "MARGIN": "MARGIN",
        "NRML": "MARGIN",   # Map NRML to MARGIN
        "CO": "CO",
        "BO": "BO"
    }
    return mapping.get(product_type, "CNC")


def map_order_type(order_type):
    """
    Maps frontend order_type to DHAN order_type.
    
    DHAN Order Type:
    - MARKET: "MARKET"
    - LIMIT: "LIMIT"
    - STOP_LOSS: "STOP_LOSS"
    - STOP_LOSS_MARKET: "STOP_LOSS_MARKET"
    """
    if not order_type:
        return "MARKET"
    
    order_type = str(order_type).upper().strip()
    mapping = {
        "MARKET": "MARKET",
        "LIMIT": "LIMIT",
        "SL": "STOP_LOSS",
        "SL-M": "STOP_LOSS_MARKET",
        "SLM": "STOP_LOSS_MARKET"
    }
    return mapping.get(order_type, "MARKET")

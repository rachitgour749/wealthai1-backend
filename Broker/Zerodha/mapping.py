def map_exchange(exchange):
    """
    Maps frontend exchange names to Zerodha compatible exchange names.
    """
    if not exchange:
        return None
        
    exchange = str(exchange).upper().strip()
    mapping = {
        "NSE": "NSE",
        "BSE": "BSE",
        "NFO": "NFO",
        "MCX": "MCX",
        "CDS": "CDS",
        "BCD": "BCD"
    }
    return mapping.get(exchange, exchange)

def map_validity(validity):
    """
    Maps frontend validity to Zerodha validity.
    """
    if not validity:
        return "DAY"
        
    validity = str(validity).upper().strip()
    mapping = {
        "DAY": "DAY",
        "IOC": "IOC",
        "TTL": "TTL"
    }
    return mapping.get(validity, "DAY")

def map_order_side(order_side):
    """
    Maps frontend order_side (BUY/SELL) to Zerodha transaction_type.
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
    Maps frontend product_type to Zerodha product codes.
    Examples:
    - DELIVERY -> CNC (Cash and Carry for equity delivery)
    - INTRADAY -> MIS (Margin Intraday Square-off)
    - MARGIN -> NRML (Normal for futures and options)
    """
    if not product_type:
        return "CNC"
    
    product_type = str(product_type).upper().strip()
    mapping = {
        "DELIVERY": "CNC",
        "CNC": "CNC",
        "INTRADAY": "MIS",
        "MIS": "MIS",
        "MARGIN": "NRML",
        "NRML": "NRML",
        "CO": "CO",  # Cover Order
        "BO": "BO"   # Bracket Order
    }
    return mapping.get(product_type, "CNC")

def map_order_type(order_type):
    """
    Maps frontend order_type to Zerodha order_type.
    Examples:
    - MARKET -> MARKET
    - LIMIT -> LIMIT
    - SL -> SL (Stop Loss Limit)
    - SL-M -> SL-M (Stop Loss Market)
    """
    if not order_type:
        return "MARKET"
    
    order_type = str(order_type).upper().strip()
    mapping = {
        "MARKET": "MARKET",
        "LIMIT": "LIMIT",
        "SL": "SL",
        "SL-M": "SL-M",
        "SLM": "SL-M"
    }
    return mapping.get(order_type, "MARKET")

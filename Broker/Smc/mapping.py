
"""
SMC ACE (ANT Platform) Mapping Functions
Maps common frontend parameters to SMC ACE specific codes.
"""

def map_exchange(exchange: str) -> str:
    """Map frontend exchange to SMC ACE exchange code."""
    mapping = {
        "NSE": "NSE",
        "BSE": "BSE",
        "NFO": "NFO",
        "MCX": "MCX",
        "CDS": "CDS"
    }
    return mapping.get(exchange.upper(), "NSE")

def map_order_side(order_side: str) -> str:
    """Map frontend order side to SMC ACE action."""
    mapping = {
        "BUY": "BUY",
        "SELL": "SELL"
    }
    return mapping.get(order_side.upper(), "BUY")

def map_product_type(product_type: str) -> str:
    """Map frontend product type to SMC ACE product code."""
    mapping = {
        "DELIVERY": "DELIVERY",
        "CNC": "DELIVERY",
        "INTRADAY": "INTRADAY",
        "MIS": "INTRADAY",
        "MARGIN": "MARGIN",
        "NRML": "MARGIN",
        "BO": "BO"
    }
    return mapping.get(product_type.upper(), "DELIVERY")

def map_order_type(order_type: str) -> str:
    """Map frontend order type to SMC ACE order type code."""
    mapping = {
        "MARKET": "MARKET",
        "LIMIT": "LIMIT",
        "SL": "SL",
        "SL-M": "SL-M"
    }
    return mapping.get(order_type.upper(), "MARKET")

def map_validity(validity: str) -> str:
    """Map frontend validity to SMC ACE validity code."""
    mapping = {
        "DAY": "DAY",
        "IOC": "IOC"
    }
    return mapping.get(validity.upper(), "DAY")

"""
ICICI Broker Mapping Functions
Maps common order parameters to ICICI Breeze API fields
"""

def map_exchange(exchange):
    """
    Maps frontend exchange names to ICICI exchange codes.
    
    ICICI Exchange Codes:
    - NSE: "nse"
    - BSE: "bse"
    - NFO: "nfo"
    """
    if not exchange:
        return "nse"
        
    exchange = str(exchange).lower().strip()
    
    mapping = {
        "nse": "nse",
        "nsecm": "nse",
        "bse": "bse",
        "bsecm": "bse",
        "nfo": "nfo",
        "nsefo": "nfo"
    }
    return mapping.get(exchange, "nse")


def map_order_side(order_side):
    """
    Maps frontend order_side (BUY/SELL) to ICICI transaction type.
    
    ICICI Actions:
    - BUY: "buy"
    - SELL: "sell"
    """
    if not order_side:
        return "buy"
    
    order_side = str(order_side).lower().strip()
    mapping = {
        "buy": "buy",
        "sell": "sell"
    }
    return mapping.get(order_side, "buy")


def map_product_type(product_type, exchange=None):
    """
    Maps frontend product_type to ICICI product codes.
    
    ICICI Product Types:
    - cash: "cash" (Delivery)
    - margin: "margin" (Intraday for Equity)
    - nse_equity_iotp: "mtf" (Margin Token Funding)
    - futures: "futures"
    - options: "options"
    """
    if not product_type:
        return "cash"
    
    product_type = str(product_type).upper().strip()
    exchange = str(exchange).upper().strip() if exchange else ""
    
    # Logic for ICICI product mapping
    if exchange in ["NFO", "NSEFO"]:
        if "OPTION" in product_type:
            return "options"
        else:
            return "futures"
    
    mapping = {
        "DELIVERY": "cash",
        "CNC": "cash",
        "INTRADAY": "margin",
        "MIS": "margin",
        "MARGIN": "margin",
        "MTF": "mtf"
    }
    
    return mapping.get(product_type, "cash")


def map_order_type(order_type):
    """
    Maps frontend order_type to ICICI order_type.
    
    ICICI Order Types:
    - market: "market"
    - limit: "limit"
    """
    if not order_type:
        return "market"
    
    order_type = str(order_type).lower().strip()
    mapping = {
        "market": "market",
        "limit": "limit",
        "sl": "limit", # ICICI uses limit for SL-L
        "sl-m": "market" # ICICI uses market for SL-M
    }
    return mapping.get(order_type, "market")

def map_exchange(exchange):
    """
    Maps frontend exchange names to Kotak compatible exchange segment codes.
    es: Exchange segment code. "nse_cm", "bse_cm", "nse_fo", "bse_fo", "cde_fo", "mcx_fo"
    """
    if not exchange:
        return "nse_cm" # Default
        
    exchange = str(exchange).upper().strip()
    mapping = {
        "NSE": "nse_cm",
        "BSE": "bse_cm",
        "NFO": "nse_fo", # NSE Futures & Options
        "BFO": "bse_fo", # BSE Futures & Options
        "MCX": "mcx_fo",
        "CDS": "cde_fo", # Currency Derivatives
        "BCD": "bse_fo"  # Assuming BCD maps to BSE FO or similar if needed
    }
    return mapping.get(exchange, "nse_cm")

def map_validity(validity):
    """
    Maps frontend validity to Kotak validity (rt).
    Values: "DAY", "IOC"
    """
    if not validity:
        return "DAY"
        
    validity = str(validity).upper().strip()
    mapping = {
        "DAY": "DAY",
        "IOC": "IOC"
    }
    return mapping.get(validity, "DAY")

def map_order_side(order_side):
    """
    Maps frontend order_side (BUY/SELL) to Kotak transaction type (tt).
    Values: "B", "S"
    """
    if not order_side:
        return "B"
    
    order_side = str(order_side).upper().strip()
    mapping = {
        "BUY": "B",
        "SELL": "S"
    }
    return mapping.get(order_side, "B")

def map_product_type(product_type):
    """
    Maps frontend product_type to Kotak product codes (pc).
    Values: "NRML", "CNC", "MIS", "CO", "BO", "MTF"
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
        "CO": "CO",
        "BO": "BO",
        "MTF": "MTF"
    }
    return mapping.get(product_type, "CNC")

def map_order_type(order_type):
    """
    Maps frontend order_type to Kotak order_type (pt).
    Values: "L", "MKT", "SL", "SL-M"
    """
    if not order_type:
        return "MKT"
    
    order_type = str(order_type).upper().strip()
    mapping = {
        "MARKET": "MKT",
        "LIMIT": "L",
        "SL": "SL",
        "SL-M": "SL-M",
        "SLM": "SL-M"
    }
    return mapping.get(order_type, "MKT")

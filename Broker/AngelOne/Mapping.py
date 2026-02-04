"""
AngelOne Parameter Mapping Utilities

Maps frontend parameters to AngelOne API compatible formats.
"""

def map_exchange(exchange):
    """
    Maps frontend exchange names to AngelOne compatible exchange names.
    
    Args:
        exchange: Exchange name from frontend (NSE, BSE, NFO, etc.)
    
    Returns:
        AngelOne compatible exchange name
    """
    if not exchange:
        return None
        
    exchange = str(exchange).upper().strip()
    
    # AngelOne exchange mapping
    mapping = {
        "NSE": "NSE",
        "NSECM": "NSE",
        "BSE": "BSE",
        "BSECM": "BSE",
        "NFO": "NFO",
        "NSEFO": "NFO",
        "MCX": "MCX",
        "CDS": "CDS",
        "BCD": "BCD"
    }
    
    return mapping.get(exchange, exchange)


def map_product_type(product_type, exchange_segment=None):
    """
    Maps frontend product_type to AngelOne product codes.
    
    Args:
        product_type: Product type from frontend (DELIVERY, INTRADAY, MARGIN)
        exchange_segment: Exchange segment (optional, for context)
    
    Returns:
        AngelOne compatible product type
    
    Examples:
        - DELIVERY -> DELIVERY
        - INTRADAY -> INTRADAY
        - MARGIN -> MARGIN
        - CNC -> DELIVERY
        - MIS -> INTRADAY
        - NRML -> MARGIN
    """
    if not product_type:
        return "DELIVERY"
    
    product_type = str(product_type).upper().strip()
    
    # AngelOne product type mapping
    mapping = {
        "DELIVERY": "DELIVERY",
        "CNC": "DELIVERY",
        "INTRADAY": "INTRADAY",
        "MIS": "INTRADAY",
        "MARGIN": "MARGIN",
        "NRML": "MARGIN",
        "BO": "BO",  # Bracket Order
        "CO": "CO"   # Cover Order
    }
    
    return mapping.get(product_type, "DELIVERY")


def map_order_type(order_type):
    """
    Maps frontend order_type to AngelOne order_type.
    
    Args:
        order_type: Order type from frontend
    
    Returns:
        AngelOne compatible order type
    
    Examples:
        - MARKET -> MARKET
        - LIMIT -> LIMIT
        - SL -> STOPLOSS_LIMIT
        - SL-M -> STOPLOSS_MARKET
        - STOPLOSS_LIMIT -> STOPLOSS_LIMIT
        - STOPLOSS_MARKET -> STOPLOSS_MARKET
    """
    if not order_type:
        return "MARKET"
    
    order_type = str(order_type).upper().strip()
    
    # AngelOne order type mapping
    mapping = {
        "MARKET": "MARKET",
        "LIMIT": "LIMIT",
        "SL": "STOPLOSS_LIMIT",
        "SL-M": "STOPLOSS_MARKET",
        "SLM": "STOPLOSS_MARKET",
        "STOPLOSS_LIMIT": "STOPLOSS_LIMIT",
        "STOPLOSS_MARKET": "STOPLOSS_MARKET"
    }
    
    return mapping.get(order_type, "MARKET")


def map_order_side(order_side):
    """
    Maps frontend order_side (BUY/SELL) to AngelOne transaction_type.
    
    Args:
        order_side: Order side from frontend (BUY/SELL)
    
    Returns:
        AngelOne compatible transaction type
    """
    if not order_side:
        return "BUY"
    
    order_side = str(order_side).upper().strip()
    
    mapping = {
        "BUY": "BUY",
        "SELL": "SELL"
    }
    
    return mapping.get(order_side, "BUY")


def map_variety(order_type):
    """
    Maps order_type to AngelOne variety.
    
    Args:
        order_type: Order type (MARKET, LIMIT, STOPLOSS_LIMIT, etc.)
    
    Returns:
        AngelOne variety (NORMAL, STOPLOSS, BO, CO)
    
    Examples:
        - MARKET -> NORMAL
        - LIMIT -> NORMAL
        - STOPLOSS_LIMIT -> STOPLOSS
        - STOPLOSS_MARKET -> STOPLOSS
    """
    if not order_type:
        return "NORMAL"
    
    order_type = str(order_type).upper().strip()
    
    # Determine variety based on order type
    if "STOPLOSS" in order_type or "SL" in order_type:
        return "STOPLOSS"
    elif order_type == "BO":
        return "BO"
    elif order_type == "CO":
        return "CO"
    else:
        return "NORMAL"


def format_symbol_for_cash(symbol, exchange):
    """
    Format symbol for cash segment trading.
    
    For NSE cash segment stocks, AngelOne requires '-EQ' suffix.
    For ETFs, use the symbol as-is (no suffix).
    
    Args:
        symbol: Trading symbol (e.g., 'RELIANCE', 'PHARMABEES')
        exchange: Mapped exchange (e.g., 'NSE')
    
    Returns:
        Formatted symbol (e.g., 'RELIANCE-EQ' for stocks, 'PHARMABEES' for ETFs)
    """
    if not symbol:
        return symbol
    
    symbol = str(symbol).strip()
    exchange = str(exchange).upper().strip() if exchange else ""
    
    # ETFs typically have 'BEES', 'ETF', or 'GOLD' in their name
    # They don't need the -EQ suffix
    etf_indicators = ['BEES', 'ETF', 'GOLD', 'SILVER', 'LIQUID', 'MON100', 'CPSE', 'SETF']
    
    # Check if symbol is an ETF
    is_etf = any(indicator in symbol.upper() for indicator in etf_indicators)
    
    # Add -EQ suffix for stocks only on NSE/BSE cash segments
    if exchange in ["NSE", "NSECM", "BSE", "BSECM"] and not is_etf:
        if not symbol.endswith('-EQ'):
            symbol = f"{symbol}-EQ"
    
    return symbol

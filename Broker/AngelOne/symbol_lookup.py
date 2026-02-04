"""
AngelOne Symbol Token Lookup

Maps trading symbols to their exchange instrument IDs (tokens) for AngelOne API.
This allows automatic token resolution without requiring users to provide tokens manually.
"""

# NSE Cash Segment - Common Stocks
NSE_CASH_TOKENS = {
    # Banks
    "YESBANK": "11915",
    "SBIN": "3045",
    "HDFCBANK": "1333",
    "ICICIBANK": "4963",
    "AXISBANK": "5900",
    "KOTAKBANK": "1922",
    "INDUSINDBK": "5258",
    "BANDHANBNK": "579",
    "FEDERALBNK": "1023",
    "IDFCFIRSTB": "11184",
    "PNB": "10666",
    "BANKBARODA": "4668",
    
    # IT
    "TCS": "11536",
    "INFY": "1594",
    "WIPRO": "3787",
    "HCLTECH": "7229",
    "TECHM": "13538",
    "LTI": "17818",
    "COFORGE": "3901",
    "MPHASIS": "4503",
    "PERSISTENT": "14413",
    
    # Auto
    "TATAMOTORS": "3456",
    "M&M": "10999",
    "MARUTI": "10999",
    "BAJAJ-AUTO": "16669",
    "HEROMOTOCO": "1348",
    "EICHERMOT": "910",
    "TVSMOTOR": "8479",
    "ASHOKLEY": "212",
    
    # Pharma
    "SUNPHARMA": "3351",
    "DRREDDY": "881",
    "CIPLA": "694",
    "DIVISLAB": "10940",
    "BIOCON": "11373",
    "AUROPHARMA": "275",
    "LUPIN": "10440",
    "TORNTPHARM": "3518",
    
    # Energy & Oil
    "RELIANCE": "2885",
    "ONGC": "2475",
    "BPCL": "526",
    "IOC": "1624",
    "GAIL": "4717",
    "COALINDIA": "20374",
    "NTPC": "11630",
    "POWERGRID": "14977",
    
    # FMCG
    "HINDUNILVR": "1394",
    "ITC": "1660",
    "NESTLEIND": "17963",
    "BRITANNIA": "547",
    "DABUR": "772",
    "MARICO": "4067",
    "GODREJCP": "10099",
    
    # Metals
    "TATASTEEL": "3499",
    "HINDALCO": "1363",
    "JSWSTEEL": "11723",
    "VEDL": "3063",
    "SAIL": "2963",
    "NMDC": "15332",
    "NATIONALUM": "6364",
    
    # Telecom
    "BHARTIARTL": "10604",
    "IDEA": "7929",
    
    # Cement
    "ULTRACEMCO": "11532",
    "GRASIM": "1232",
    "SHREECEM": "3103",
    "AMBUJACEM": "1270",
    "ACC": "22",
    
    # Infra & Construction
    "LT": "11483",
    "ADANIPORTS": "15083",
    "ADANIENT": "25",
    
    # Others
    "ASIANPAINT": "236",
    "TITAN": "3506",
    "BAJFINANCE": "317",
    "BAJAJFINSV": "16675",
    "HDFC": "1330",
    "HDFCLIFE": "467",
    "SBILIFE": "21808",
}

# NSE ETF Segment
NSE_ETF_TOKENS = {
    # Nifty Index ETFs
    "NIFTYBEES": "15068",
    "JUNIORBEES": "16669",
    "BANKBEES": "15083",
    "PSUBNKBEES": "26017",
    "PVTBNKBEES": "26016",
    
    # Sector ETFs
    "INFRABEES": "26013",
    "PHARMABEES": "26014",
    "ITBEES": "26011",
    "AUTOBEES": "26009",
    "FMCGBEES": "26010",
    "CONSUMBEES": "26012",
    "METALBEES": "26015",
    "PSUBEES": "26018",
    
    # Other Index ETFs
    "GOLDBEES": "1660",
    "LIQUIDBEES": "4717",
    "SILVER": "3045",
    "CPSEETF": "26019",
    "SETFNIF50": "14366",
    "SETFNN50": "14367",
    
    # International ETFs (if traded on NSE)
    "MON100": "26020",
    "HNGSNGBEES": "26021",
}

# NSE F&O Segment
NSE_FO_TOKENS = {
    # Add F&O tokens if needed
}

# BSE Cash Segment
BSE_CASH_TOKENS = {
    # Add BSE tokens if needed
}


def get_symbol_token(symbol: str, exchange: str = "NSE") -> str:
    """
    Get the exchange instrument ID (token) for a given symbol.
    
    Args:
        symbol: Trading symbol (e.g., 'YESBANK', 'RELIANCE', 'PHARMABEES')
        exchange: Exchange name (NSE, BSE, NFO, etc.)
    
    Returns:
        Token ID as string, or None if not found
    
    Example:
        >>> get_symbol_token('YESBANK', 'NSE')
        '11915'
        >>> get_symbol_token('PHARMABEES', 'NSE')
        '26014'
    """
    # Normalize symbol (remove .NS, .BO suffixes and convert to uppercase)
    symbol = symbol.upper().replace('.NS', '').replace('.BO', '').strip()
    
    # Select appropriate token map based on exchange
    if exchange.upper() == "NSE":
        # Check ETFs first (they often have 'BEES' or 'ETF' in name)
        token = NSE_ETF_TOKENS.get(symbol)
        if token:
            return token
        # Then check stocks
        return NSE_CASH_TOKENS.get(symbol)
    elif exchange.upper() == "BSE":
        return BSE_CASH_TOKENS.get(symbol)
    elif exchange.upper() in ["NFO", "MCX", "CDS"]:
        return NSE_FO_TOKENS.get(symbol)
    
    return None


def is_symbol_supported(symbol: str, exchange: str = "NSE") -> bool:
    """
    Check if a symbol is supported in the token mapping.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
    
    Returns:
        True if symbol is supported, False otherwise
    """
    return get_symbol_token(symbol, exchange) is not None


def get_all_supported_symbols(exchange: str = "NSE") -> list:
    """
    Get list of all supported symbols for an exchange.
    
    Args:
        exchange: Exchange name
    
    Returns:
        List of supported symbol names (includes both stocks and ETFs for NSE)
    """
    if exchange.upper() == "NSE":
        # Combine ETFs and stocks
        return list(NSE_ETF_TOKENS.keys()) + list(NSE_CASH_TOKENS.keys())
    elif exchange.upper() == "BSE":
        return list(BSE_CASH_TOKENS.keys())
    elif exchange.upper() in ["NFO", "MCX", "CDS"]:
        return list(NSE_FO_TOKENS.keys())
    
    return []

"""
# AngelOne Symbol Token Lookup
#
# Maps trading symbols to their exchange instrument IDs (tokens) for AngelOne API.
# This allows automatic token resolution without requiring users to provide tokens manually.
# """
import json
import os
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Cache for loaded scrip master
ANGEL_SCRIP_MASTER: Dict[str, Dict] = {} 
SCIP_MASTER_PATH = os.path.join(os.path.dirname(__file__), 'resources', 'OpenAPIScripMaster.json')

def load_scrip_master() -> None:
    """Load Scrip Master JSON into memory if not already loaded."""
    global ANGEL_SCRIP_MASTER
    if ANGEL_SCRIP_MASTER:
        return

    try:
        if os.path.exists(SCIP_MASTER_PATH):
            logger.info("Loading AngelOne Scrip Master...")
            with open(SCIP_MASTER_PATH, 'r') as f:
                data = json.load(f)
                
            # Build optimized dictionary: (symbol, exch_seg) -> token
            for item in data:
                symbol = item.get('symbol')
                exch_seg = item.get('exch_seg')
                token = item.get('token')
                
                if symbol and exch_seg and token:
                    # Determine normalized exchange name
                    # AngelOne uses 'NSE', 'BSE', 'NFO', 'MCX', 'CDS' as exch_seg
                    # But keys should overlap with what we passed, or we normalize in lookup
                    
                    # Store with upper case symbol and exchange
                    key = f"{symbol.upper()}:{exch_seg.upper()}"
                    ANGEL_SCRIP_MASTER[key] = token
            
            logger.info(f"Loaded {len(ANGEL_SCRIP_MASTER)} instruments from Scrip Master.")
        else:
            logger.warning(f"AngelOne Scrip Master not found at {SCIP_MASTER_PATH}")
            
    except Exception as e:
        logger.error(f"Failed to load AngelOne Scrip Master: {e}") 


# NSE Cash Segment - Common Stocks
# NSE Cash Segment - Common Stocks
# Hardcoded list removed to rely on OpenAPIScripMaster.json for accuracy
NSE_CASH_TOKENS = {}

# NSE ETF Segment
# NSE ETF Segment
NSE_ETF_TOKENS = {}

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
    # Select appropriate token map based on exchange
    token = None
    if exchange.upper() == "NSE":
        # Check ETFs first (they often have 'BEES' or 'ETF' in name)
        token = NSE_ETF_TOKENS.get(symbol)
        if not token:
            # Then check stocks
            token = NSE_CASH_TOKENS.get(symbol)
    elif exchange.upper() == "BSE":
        token = BSE_CASH_TOKENS.get(symbol)
    elif exchange.upper() in ["NFO", "MCX", "CDS"]:
        token = NSE_FO_TOKENS.get(symbol)

    # Return if found in hardcoded lists
    if token:
        return token
        
    # Standardize Exchange Name for AngelOne lookup
    # Input 'exchange' might be 'NSECM' or 'NSE', handle accordingly
    # Ideally should use map_exchange from Mapping.py but avoiding circular import
    normalized_exchange = exchange.upper()
    if normalized_exchange in ['NSECM', 'NSE-EQ']:
        normalized_exchange = 'NSE'
    elif normalized_exchange in ['BSECM', 'BSE-EQ']:
        normalized_exchange = 'BSE'
    elif normalized_exchange in ['FO', 'NFO']:
        normalized_exchange = 'NFO'
        
    # Fallback: Check cached Scrip Master
    load_scrip_master()
    
    # Try direct lookup first
    key = f"{symbol}:{normalized_exchange}"
    if key in ANGEL_SCRIP_MASTER:
        return ANGEL_SCRIP_MASTER[key]
        
    # If not found and exchange is NSE/BSE, try adding -EQ
    if normalized_exchange in ['NSE', 'BSE'] and not symbol.endswith('-EQ'):
        key_eq = f"{symbol}-EQ:{normalized_exchange}"
        if key_eq in ANGEL_SCRIP_MASTER:
            return ANGEL_SCRIP_MASTER[key_eq]
            
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
    load_scrip_master()
    if exchange.upper() == "NSE":
        # Return all symbols for NSE
        return [k.split(':')[0] for k in ANGEL_SCRIP_MASTER.keys() if ':NSE' in k]
    
    return []

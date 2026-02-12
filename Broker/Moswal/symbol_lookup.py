"""
MOSWAL Symbol Lookup Helper
Provides symbol name to token conversion for MOSWAL broker using AngelOne ScripMaster
"""
import logging
import json
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Path to ScripMaster file
SCRIP_MASTER_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "AngelOne", "resources", "OpenAPIScripMaster.json"
)

# Cache for loaded scrip data
_scrip_cache = None


def _load_scrip_master():
    """Load and cache the ScripMaster data"""
    global _scrip_cache
    
    if _scrip_cache is not None:
        return _scrip_cache
    
    try:
        abs_path = os.path.abspath(SCRIP_MASTER_PATH)
        logger.info(f"Loading ScripMaster from: {abs_path}")
        
        with open(abs_path, 'r') as f:
            data = json.load(f)
        
        # Create lookup dictionaries for fast access
        _scrip_cache = {
            'by_symbol': {},
            'by_name': {}
        }
        
        for item in data:
            symbol = item.get('symbol', '').upper()
            name = item.get('name', '').upper()
            token = item.get('token')
            exch_seg = item.get('exch_seg', '').upper()
            
            # Store by symbol (e.g., "ITBEES-EQ" -> token)
            if symbol and token:
                key = f"{symbol}_{exch_seg}"
                _scrip_cache['by_symbol'][key] = token
                _scrip_cache['by_symbol'][symbol] = token  # Also store without exchange
            
            # Store by name (e.g., "ITBEES" -> token)
            if name and token:
                key = f"{name}_{exch_seg}"
                _scrip_cache['by_name'][key] = token
                _scrip_cache['by_name'][name] = token  # Also store without exchange
        
        logger.info(f"Loaded {len(data)} symbols from ScripMaster")
        return _scrip_cache
        
    except Exception as e:
        logger.error(f"Failed to load ScripMaster: {e}")
        return None


def get_moswal_token(symbol: str, exchange: str = "NSE") -> str:
    """
    Convert symbol name to MOSWAL symboltoken using ScripMaster data.
    
    Args:
        symbol: Trading symbol (e.g., "ITBEES", "ITBEES-EQ", "SBIN")
        exchange: Exchange name (NSE, BSE, etc.)
    
    Returns:
        str: Numeric symboltoken for MOSWAL
        
    Raises:
        ValueError: If symbol cannot be converted
    """
    # If already numeric, return as-is
    if isinstance(symbol, (int, float)):
        return str(int(symbol))
    
    symbol_str = str(symbol).strip()
    
    # If already numeric string, return it
    if symbol_str.isdigit():
        return symbol_str
    
    # Load ScripMaster
    scrip_data = _load_scrip_master()
    if not scrip_data:
        raise ValueError(
            f"ScripMaster data not available. Cannot convert symbol '{symbol}'. "
            f"Please provide the numeric symboltoken directly."
        )
    
    # Clean symbol name (remove -EQ, -BE suffixes)
    clean_symbol = re.sub(r'-(EQ|BE|BZ|BL|BT|GC|IL|IT|TS)$', '', symbol_str.upper())
    exchange_upper = exchange.upper()
    
    # --- EXCHANGE-SPECIFIC LOOKUP (Recommended) ---
    
    # 1. Try Name + Exchange (e.g., "ITBEES_NSE")
    lookup_key = f"{clean_symbol}_{exchange_upper}"
    if lookup_key in scrip_data['by_name']:
        token = scrip_data['by_name'][lookup_key]
        logger.info(f"Converted symbol '{symbol}' to token '{token}' (by name+exchange)")
        return token
        
    # 2. Try Symbol + Exchange (e.g., "ITBEES-EQ_NSE")
    lookup_key = f"{symbol_str.upper()}_{exchange_upper}"
    if lookup_key in scrip_data['by_symbol']:
        token = scrip_data['by_symbol'][lookup_key]
        logger.info(f"Converted symbol '{symbol}' to token '{token}' (by symbol+exchange)")
        return token

    # 3. Try Cleaned Symbol + Exchange (e.g., "ITBEES_NSE")
    lookup_key = f"{clean_symbol}_{exchange_upper}"
    if lookup_key in scrip_data['by_symbol']:
        token = scrip_data['by_symbol'][lookup_key]
        logger.info(f"Converted symbol '{symbol}' to token '{token}' (by cleaned symbol+exchange)")
        return token

    # --- FALLBACK LOOKUP (Try without exchange if no specific match found) ---
    
    # 4. Try by name
    if clean_symbol in scrip_data['by_name']:
        token = scrip_data['by_name'][clean_symbol]
        logger.info(f"Converted symbol '{symbol}' to token '{token}' (by name fallback)")
        return token
        
    # 5. Try by symbol
    if clean_symbol in scrip_data['by_symbol']:
        token = scrip_data['by_symbol'][clean_symbol]
        logger.info(f"Converted symbol '{symbol}' to token '{token}' (by symbol fallback)")
        return token
        
    if symbol_str.upper() in scrip_data['by_symbol']:
        token = scrip_data['by_symbol'][symbol_str.upper()]
        logger.info(f"Converted symbol '{symbol}' to token '{token}' (by original symbol fallback)")
        return token
    
    # If not found, raise error
    raise ValueError(
        f"Symbol '{symbol}' not found in ScripMaster for exchange '{exchange}'. "
        f"Please verify the symbol name or provide the numeric symboltoken directly."
    )


def is_numeric_token(value: any) -> bool:
    """Check if value is a numeric token"""
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False

"""
DHAN Broker Mapping Functions
Maps common order parameters to DHAN-specific API fields and provides Security ID lookup.
"""
import os
import csv
import logging

logger = logging.getLogger(__name__)

# Global cache for scrip master
_scrip_master_cache = {}
_scrip_master_loaded = False

def _load_scrip_master():
    """Load the scrip master CSV into memory cache."""
    global _scrip_master_loaded, _scrip_master_cache
    if _scrip_master_loaded:
        return True
        
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, 'resources', 'api-scrip-master.csv')
    
    if not os.path.exists(csv_path):
        logger.error(f"Dhan scrip master not found at {csv_path}. Please run update_dhan_scrip.py.")
        return False
        
    try:
        logger.info(f"Loading Dhan scrip master from {csv_path}...")
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Skip header
            header = f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                
                exchange = row[0].upper()
                segment = row[1].upper()
                security_id = row[2]
                symbol = row[5].upper()
                
                # Store by exchange and symbol
                # For Equity (E), we also handle common aliases
                _scrip_master_cache[(exchange, symbol)] = security_id
                
                # If segment is Equity, also store without -EQ suffix if present
                if segment == 'E' and '-' in symbol:
                    base_symbol = symbol.split('-')[0]
                    _scrip_master_cache[(exchange, base_symbol)] = security_id
            
        _scrip_master_loaded = True
        logger.info(f"Loaded {len(_scrip_master_cache)} instruments from Dhan scrip master.")
        return True
    except Exception as e:
        logger.error(f"Error loading Dhan scrip master: {e}")
        return False


def get_security_id(symbol, exchange="NSE"):
    """
    Look up the Dhan Security ID for a given symbol and exchange.
    """
    if not symbol:
        return ""
        
    if not _scrip_master_loaded:
        _load_scrip_master()
        
    symbol = str(symbol).upper().strip()
    exchange = str(exchange).upper().strip()
    
    # Map common frontend exchange names to CSV names
    exch_map = {
        "NSE": "NSE",
        "BSE": "BSE",
        "NFO": "NSE",  # F&O is under NSE in Dhan scrip master
        "MCX": "MCX",
        "CDS": "NSE"   # Currency is often under NSE
    }
    
    csv_exchange = exch_map.get(exchange, exchange)
    
    # Try direct lookup
    security_id = _scrip_master_cache.get((csv_exchange, symbol))
    
    # If not found and it's Equity, try stripping -EQ or adding it
    if not security_id:
        if '-EQ' in symbol:
            security_id = _scrip_master_cache.get((csv_exchange, symbol.replace('-EQ', '')))
        else:
            # Some entries might have -EQ in the CSV even if rare
            security_id = _scrip_master_cache.get((csv_exchange, f"{symbol}-EQ"))
            
    if not security_id:
        logger.warning(f"SecurityId not found for {symbol} on {exchange} (CSV Exchange: {csv_exchange})")
        return symbol  # Fallback to symbol if not found
        
    return security_id


def map_exchange(exchange):
    """
    Maps frontend exchange names to DHAN exchange segment strings.
    
    DHAN Exchange Segment Strings:
    - NSE_EQ: National Stock Exchange Equity Cash
    - NSE_FNO: National Stock Exchange Futures & Options
    - NSE_CURRENCY: National Stock Exchange Currency
    - BSE_EQ: Bombay Stock Exchange Equity Cash
    - MCX_COMM: Multi Commodity Exchange Commodity
    - BSE_CURRENCY: Bombay Stock Exchange Currency
    - BSE_FNO: Bombay Stock Exchange Futures & Options
    """
    if not exchange:
        return "NSE_EQ"
        
    exchange = str(exchange).upper().strip()
    mapping = {
        "NSE": "NSE_EQ",
        "BSE": "BSE_EQ",
        "NFO": "NSE_FNO",
        "MCX": "MCX_COMM",
        "CDS": "NSE_CURRENCY",
        "BCD": "BSE_CURRENCY"
    }
    return mapping.get(exchange, "NSE_EQ")


def map_validity(validity):
    """
    Maps frontend validity to DHAN validity strings.
    
    DHAN Validity:
    - DAY: The order is valid for the entire trading day.
    - IOC: Immediate Or Cancel.
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

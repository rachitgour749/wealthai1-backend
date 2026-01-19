"""Yahoo Finance price fetching service"""
import yfinance as yf
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PriceService:
    """Fetch prices from Yahoo Finance"""
    
    # Symbol mapping for Indian stocks (add .NS suffix)
    SYMBOL_SUFFIX_MAP = {
        'NSE': '.NS',
        'BSE': '.BO'
    }
    
    @staticmethod
    def _format_symbol(symbol: str, exchange: str = 'NSE') -> str:
        """
        Format symbol for Yahoo Finance
        
        Examples:
        - NIFTYBEES (NSE) → NIFTYBEES.NS
        - GOLDBEES (NSE) → GOLDBEES.NS
        """
        # If it's an index (starts with ^) or already has a suffix, return as is
        if symbol.startswith('^') or symbol.endswith(('.NS', '.BO')):
            return symbol
        
        suffix = PriceService.SYMBOL_SUFFIX_MAP.get(exchange, '.NS')
        return f"{symbol}{suffix}"
    
    @staticmethod
    def get_price_on_date(
        symbol: str,
        trade_date: date,
        exchange: str = 'NSE',
        price_type: str = 'open'
    ) -> float:
        """
        Get price for a symbol on a specific date
        
        Args:
            symbol: Stock/ETF symbol (e.g., 'NIFTYBEES')
            trade_date: Date to fetch price for
            exchange: Exchange (NSE/BSE)
            price_type: 'open', 'close', 'high', 'low'
        
        Returns:
            Price as float
        """
        try:
            # Format symbol for Yahoo Finance
            yf_symbol = PriceService._format_symbol(symbol, exchange)
            
            # Fetch data for the date (with buffer for holidays)
            start_date = trade_date - timedelta(days=7)
            end_date = trade_date + timedelta(days=1)
            
            logger.info(f"Fetching {price_type} price for {yf_symbol} on {trade_date}")
            
            # Download data
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                logger.warning(f"No data found for {yf_symbol} on {trade_date}")
                return 0.0
            
            # Get closest date
            hist.index = hist.index.date
            
            if trade_date in hist.index:
                price = hist.loc[trade_date, price_type.capitalize()]
            else:
                # Get nearest date
                nearest_date = min(hist.index, key=lambda d: abs(d - trade_date))
                price = hist.loc[nearest_date, price_type.capitalize()]
                logger.info(f"Using price from {nearest_date} (requested {trade_date})")
            
            return float(price)
            
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return 0.0
    
    @staticmethod
    def get_latest_prices(
        symbols: List[str],
        exchange: str = 'NSE'
    ) -> Dict[str, float]:
        """
        Get latest prices for multiple symbols
        
        Args:
            symbols: List of symbols
            exchange: Exchange (NSE/BSE)
        
        Returns:
            Dict mapping symbol to price
        """
        prices = {}
        
        for symbol in symbols:
            try:
                yf_symbol = PriceService._format_symbol(symbol, exchange)
                ticker = yf.Ticker(yf_symbol)
                
                # Get latest price
                hist = ticker.history(period='1d')
                
                if not hist.empty:
                    prices[symbol] = float(hist['Close'].iloc[-1])
                else:
                    logger.warning(f"No recent data for {symbol}")
                    prices[symbol] = 0.0
                    
            except Exception as e:
                logger.error(f"Error fetching price for {symbol}: {e}")
                prices[symbol] = 0.0
        
        return prices
    
    @staticmethod
    def get_current_price(symbol: str, exchange: str = 'NSE') -> float:
        """Get current/latest price for a symbol"""
        prices = PriceService.get_latest_prices([symbol], exchange)
        return prices.get(symbol, 0.0)

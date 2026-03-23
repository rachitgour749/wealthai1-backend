import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Type, Tuple
from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from Databases.app_data_db_connection import (
    get_session, 
    ETFMarket, 
    StockMarket, 
    USETFMarket,
    Nifty50IndexMarket,
    SP500IndexMarket,
    USStockMarket
)

logger = logging.getLogger(__name__)

class MarketDataService:
    """
    Unified service for fetching market data across different markets and asset types.
    """
    
    # Model Mapping
    MARKET_MODELS = {
        ('INDIA', 'ETF'): ETFMarket,
        ('INDIA', 'STOCK'): StockMarket,
        ('INDIA', 'INDEX'): Nifty50IndexMarket,
        ('US', 'ETF'): USETFMarket,
        ('US', 'STOCK'): USStockMarket,
        ('US', 'INDEX'): SP500IndexMarket,
    }

    @classmethod
    def get_model(cls, market: str, asset_type: str) -> Type:
        """Get the SQLAlchemy model for a given market and asset type."""
        key = (market.upper(), asset_type.upper())
        model = cls.MARKET_MODELS.get(key)
        if not model:
            raise ValueError(f"No market model found for {market} {asset_type}")
        return model

    @classmethod
    def fetch_close_prices(
        cls, 
        tickers: List[str], 
        market: str, 
        asset_type: str, 
        start_date: datetime, 
        end_date: datetime,
        db: Optional[Session] = None
    ) -> pd.DataFrame:
        """
        Fetch close prices for multiple tickers and return as a pivoted DataFrame.
        """
        db_provided = db is not None
        if not db_provided:
            db = get_session()
            
        try:
            model = cls.get_model(market, asset_type)
            
            # Fetch data
            query = db.query(model.date, model.symbol, model.close).filter(
                model.symbol.in_(tickers),
                model.date >= start_date,
                model.date <= end_date
            )
            
            data = query.all()
            
            if not data:
                logger.warning(f"No {market} {asset_type} data found for {tickers} between {start_date} and {end_date}")
                return pd.DataFrame()
                
            df = pd.DataFrame(data, columns=['date', 'symbol', 'close'])
            df['date'] = pd.to_datetime(df['date'])
            
            # Pivot to standard format (Index: Date, Columns: Tickers, Values: Close)
            pivot_df = df.pivot(index='date', columns='symbol', values='close')
            pivot_df.sort_index(inplace=True)
            
            # Professional handling: forward fill missing values (common in multi-ticker datasets)
            pivot_df.ffill(inplace=True)
            
            return pivot_df
            
        except Exception as e:
            logger.error(f"Error fetching market data for {market} {asset_type}: {e}")
            return pd.DataFrame()
        finally:
            if not db_provided:
                db.close()

    @classmethod
    def get_latest_price(cls, symbol: str, market: str, asset_type: str) -> float:
        """Fetch the most recent close price for a single symbol."""
        db = get_session()
        try:
            model = cls.get_model(market, asset_type)
            latest = db.query(model.close).filter(
                model.symbol == symbol
            ).order_by(model.date.desc()).first()
            
            return float(latest[0]) if latest else 0.0
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return 0.0
        finally:
            db.close()

    @classmethod
    def fetch_ohlcv(
        cls,
        ticker: str,
        market: str,
        asset_type: str,
        start_date,
        end_date,
        db: Optional[Session] = None
    ) -> pd.DataFrame:
        """
        Fetch full OHLCV data for a single ticker.
        Returns a DataFrame indexed by date with columns: open, high, low, close, volume.
        Falls back to using close for open/high/low if those columns don't exist on the model.
        """
        db_provided = db is not None
        if not db_provided:
            db = get_session()

        try:
            model = cls.get_model(market, asset_type)

            # Build query — include open/high/low/volume if available on the model
            has_ohlv = all(hasattr(model, c) for c in ['open', 'high', 'low', 'volume'])

            if has_ohlv:
                query = db.query(
                    model.date, model.open, model.high, model.low, model.close, model.volume
                ).filter(
                    model.symbol == ticker,
                    model.date >= start_date,
                    model.date <= end_date
                ).order_by(model.date)
                rows = query.all()
                if not rows:
                    return pd.DataFrame()
                df = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            else:
                query = db.query(model.date, model.close).filter(
                    model.symbol == ticker,
                    model.date >= start_date,
                    model.date <= end_date
                ).order_by(model.date)
                rows = query.all()
                if not rows:
                    return pd.DataFrame()
                df = pd.DataFrame(rows, columns=['date', 'close'])
                for col in ['open', 'high', 'low']:
                    df[col] = df['close']
                df['volume'] = 0

            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV for {ticker} ({market} {asset_type}): {e}")
            return pd.DataFrame()
        finally:
            if not db_provided:
                db.close()

    @classmethod
    def calculate_date_range(
        cls, 
        tickers: List[str], 
        market: str, 
        asset_type: str,
        buffer_days: int = 371  # ~53 weeks (252 trading days)
    ) -> Tuple[Optional[str], Optional[str], float]:
        """
        Calculate common date range for multiple tickers with a buffer for indicators.
        Returns (start_date_str, end_date_str, years_available).
        """
        # Normalize symbols (remove .NS suffix if present)
        tickers = [t.replace(".NS", "").strip() for t in tickers if t and isinstance(t, str)]
            
        db = get_session()
        try:
            model = cls.get_model(market, asset_type)
            
            # Query MIN and MAX dates for each ticker
            from sqlalchemy import text
            placeholders = ','.join([f':t{i}' for i in range(len(tickers))])
            query = text(f"""
                SELECT 
                    symbol,
                    MIN(date) as start_date,
                    MAX(date) as end_date
                FROM {model.__tablename__}
                WHERE symbol IN ({placeholders})
                GROUP BY symbol
            """)
            
            params = {f't{i}': t for i, t in enumerate(tickers)}
            result = db.execute(query, params)
            rows = result.fetchall()
            
            if not rows:
                return None, None, 0.0
            
            # Find latest start and earliest end (common range)
            latest_start = max(pd.to_datetime(row[1]) for row in rows)
            earliest_end = min(pd.to_datetime(row[2]) for row in rows)
            
            # Apply buffer logic
            strategy_start = latest_start + timedelta(days=buffer_days)
            
            if strategy_start >= earliest_end:
                return None, None, 0.0
                
            years = (earliest_end - strategy_start).days / 365.25
            
            return strategy_start.strftime('%Y-%m-%d'), earliest_end.strftime('%Y-%m-%d'), years
            
        except Exception as e:
            logger.error(f"Error calculating date range: {e}")
            return None, None, 0.0
        finally:
            db.close()

    @classmethod
    def generate_asset_description(cls, symbol: str, asset_type: str = "ETF") -> str:
        """Generate intelligent asset descriptions based on symbol names"""
        etf_mappings = {
            'NIFTYBEES': 'Nifty 50 ETF - Broad Market',
            'BANKBEES': 'Banking Sector ETF',
            'JUNIORBEES': 'Nifty Next 50 ETF - Mid Cap',
            'ITBEES': 'Information Technology ETF',
            'PHARMABEES': 'Pharmaceutical Sector ETF',
            'INFRABEES': 'Infrastructure Sector ETF',
            'GOLDBEES': 'Gold Commodity ETF',
            'METALIETF': 'Metal & Mining Sector ETF',
            'OILIETF': 'Oil & Gas Sector ETF',
            'LIQUIDBEES': 'Liquid Fund ETF - Money Market',
            'CPSEETF': 'CPSE (Central PSE) ETF',
            'PSUBNKBEES': 'PSU Banking ETF',
            'MON100': 'NASDAQ 100 ETF - US Tech',
            'MODEFENCE': 'Defence Sector ETF',
            'MIDCAPETF': 'Mid Cap ETF',
        }

        if symbol in etf_mappings and asset_type.upper() == "ETF":
            return etf_mappings[symbol]

        symbol_lower = symbol.lower()
        suffix = " ETF" if asset_type.upper() == "ETF" else ""

        if 'pharma' in symbol_lower:
            return f'Pharmaceutical Sector{suffix}'
        elif 'bank' in symbol_lower:
            return f'Banking Sector{suffix}'
        elif 'it' in symbol_lower or 'tech' in symbol_lower:
            return f'Technology Sector{suffix}'
        elif 'gold' in symbol_lower:
            return f'Gold Commodity{suffix}'
        elif 'oil' in symbol_lower or 'energy' in symbol_lower:
            return f'Oil & Gas Sector{suffix}'
        elif 'metal' in symbol_lower:
            return f'Metal & Mining Sector{suffix}'
        elif 'infra' in symbol_lower:
            return f'Infrastructure Sector{suffix}'
        elif 'defence' in symbol_lower or 'defense' in symbol_lower:
            return f'Defence Sector{suffix}'
        elif 'midcap' in symbol_lower or 'mid' in symbol_lower:
            return f'Mid Cap{suffix}'
        elif 'smallcap' in symbol_lower or 'small' in symbol_lower:
            return f'Small Cap{suffix}'
        elif 'liquid' in symbol_lower:
            return f'Liquid Fund{suffix}'
        elif 'nifty' in symbol_lower:
            return f'Nifty Index{suffix}'
        elif 'sensex' in symbol_lower:
            return f'Sensex Index{suffix}'
        elif 'psu' in symbol_lower:
            return f'PSU Sector{suffix}'
        elif 'cpse' in symbol_lower:
            return f'CPSE{suffix}'
        elif any(geo in symbol_lower for geo in ['us', 'usa', 'nasdaq', 'sp500', 'dow']):
            return f'International{suffix}'
        elif 'fmcg' in symbol_lower:
            return f'FMCG Sector{suffix}'
        elif 'auto' in symbol_lower:
            return f'Automotive Sector{suffix}'
        else:
            return f'{symbol.replace(".NS", "").replace("_", " ")}{suffix}'

    @classmethod
    def get_asset_sector_classification(cls, symbol: str, asset_type: str = "ETF") -> str:
        """Classify asset into sector categories"""
        clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
        symbol_lower = symbol.lower()
        
        sector_map = {
            'HDFCBANK': 'Financials', 'ICICIBANK': 'Financials', 'SBIN': 'Financials', 'AXISBANK': 'Financials',
            'KOTAKBANK': 'Financials', 'BAJFINANCE': 'Financials', 'BAJAJFINSV': 'Financials', 'HDFCLIFE': 'Financials',
            'SBILIFE': 'Financials', 'INDUSINDBK': 'Financials', 'BANKBARODA': 'Financials', 'PNB': 'Financials',
            'JIOFIN': 'Financials', 'PFC': 'Financials', 'RECLTD': 'Financials', 'SHRIRAMFIN': 'Financials',
            'CHOLAFIN': 'Financials', 'MUTHOOTFIN': 'Financials', 'PSUBANK': 'Financials', 'BANKBEES': 'Financials',
            'BANKETF': 'Financials', 'PVTBANK': 'Financials', 'HDFCPVTBAN': 'Financials', 'HDFCPSUBK': 'Financials',
            'TCS': 'Technology', 'INFY': 'Technology', 'HCLTECH': 'Technology', 'WIPRO': 'Technology',
            'TECHM': 'Technology', 'LTIM': 'Technology', 'PERSISTENT': 'Technology', 'COFORGE': 'Technology',
            'MPHASIS': 'Technology', 'LTTS': 'Technology', 'KPITTECH': 'Technology', 'TATAELXSI': 'Technology',
            'ITBEES': 'Technology', 'ITETF': 'Technology', 'HDFCNIFIT': 'Technology', 'MAHKTECH': 'Technology',
            'RELIANCE': 'Energy', 'ONGC': 'Energy', 'BPCL': 'Energy', 'POWERGRID': 'Utilities',
            'NTPC': 'Utilities', 'ADANIGREEN': 'Utilities', 'TATAPOWER': 'Utilities', 'IOC': 'Energy',
            'GAIL': 'Energy', 'HPCL': 'Energy', 'COALINDIA': 'Energy', 'MOENERGY': 'Energy',
            'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'NESTLEIND': 'FMCG', 'BRITANNIA': 'FMCG', 'COLPAL': 'FMCG',
            'TITAN': 'Consumer Discretionary', 'ASIANPAINTS': 'Materials', 'MARICO': 'FMCG',
            'DABUR': 'FMCG', 'GODREJCP': 'FMCG', 'TATACONSUM': 'FMCG', 'VARUN': 'FMCG',
            'FMCGIETF': 'FMCG', 'CONSUMBEES': 'FMCG', 'ICICIFMCG': 'FMCG', 'CONSUMER': 'Consumption',
            'TATAMOTORS': 'Automotive', 'MARUTI': 'Automotive', 'M&M': 'Automotive', 'BAJAJ-AUTO': 'Automotive',
            'EICHERMOT': 'Automotive', 'HEROMOTOCO': 'Automotive', 'TVSMOTOR': 'Automotive', 'BHARATFORG': 'Automotive',
            'AUTOBEES': 'Automotive', 'AUTOIETF': 'Automotive', 'ICICIAUTO': 'Automotive',
            'SUNPHARMA': 'Healthcare', 'DRREDDY': 'Healthcare', 'CIPLA': 'Healthcare', 'DIVISLAB': 'Healthcare',
            'APOLLOHOSP': 'Healthcare', 'LUPIN': 'Healthcare', 'TORNTPHARM': 'Healthcare', 'MANKIND': 'Healthcare',
            'PHARMABEES': 'Healthcare', 'HEALTHY': 'Healthcare', 'HDFCHEALTH': 'Healthcare', 'ICICIPHARM': 'Healthcare',
            'TATASTEEL': 'Materials', 'JSWSTEEL': 'Materials', 'HINDALCO': 'Materials', 'VEDANTA': 'Materials', 'VEDL': 'Materials',
            'ULTRACEMCO': 'Materials', 'AMBUJACEM': 'Materials', 'PIDILITIND': 'Materials', 'JUBLFOOD': 'Consumer Services',
            'SAIL': 'Materials', 'JINDALSTEL': 'Materials', 'SHREECEM': 'Materials', 'GRASIM': 'Materials',
            'METALIETF': 'Materials', 'METAL': 'Materials', 'HDFCSILVER': 'Commodities', 'SILVERBEES': 'Commodities',
            'LT': 'Infrastructure', 'SIEMENS': 'Capital Goods', 'ABB': 'Capital Goods', 'HAL': 'Defence',
            'BEL': 'Defence', 'ADANIENT': 'Services', 'ADANIPORTS': 'Infrastructure',
            'INFRABEES': 'Infrastructure', 'INFRAIETF': 'Infrastructure', 'ICICIINFRA': 'Infrastructure',
            'BHARTIARTL': 'Telecommunication', 'VODAFONE': 'Telecommunication',
            'NIFTYBEES': 'Broad Market', 'JUNIORBEES': 'Mid Cap', 'MIDCAPETF': 'Mid Cap', 
            'ELM250': 'Broad Market', 'EQUAL200': 'Broad Market', 'EMULTIMQ': 'Multi-Factor',
            'MON100': 'Technology (US)', 'MAFANG': 'Technology (US)', 'GOLDBEES': 'Commodities',
            'LIQUIDBEES': 'Cash/Liquid', 'CPSEETF': 'PSU', 'PSUBNKBEES': 'PSU Banking'
        }
        
        if clean_symbol in sector_map: return sector_map[clean_symbol]
        if symbol in sector_map: return sector_map[symbol]

        if symbol in ['NIFTYBEES', 'JUNIORBEES', 'MIDCAPETF', 'ELM250', 'EQUAL200', 'EMULTIMQ']: return 'Broad Market'
        elif 'bank' in symbol_lower or 'psu' in symbol_lower: return 'Financial'
        elif 'it' in symbol_lower or 'tech' in symbol_lower or 'mon100' in symbol_lower: return 'Technology'
        elif 'pharma' in symbol_lower or 'health' in symbol_lower: return 'Healthcare'
        elif 'gold' in symbol_lower or 'silver' in symbol_lower: return 'Commodity'
        elif 'oil' in symbol_lower or 'energy' in symbol_lower: return 'Energy'
        elif 'metal' in symbol_lower: return 'Materials'
        elif 'infra' in symbol_lower: return 'Infrastructure'
        elif 'defence' in symbol_lower or 'defense' in symbol_lower: return 'Defence'
        elif 'liquid' in symbol_lower: return 'Cash/Liquid'
        elif 'cpse' in symbol_lower: return 'PSU'
        elif 'fmcg' in symbol_lower: return 'FMCG'
        elif 'auto' in symbol_lower: return 'Automotive'
        elif 'realty' in symbol_lower or 'real' in symbol_lower: return 'Real Estate'
        elif 'consumer' in symbol_lower: return 'Consumption'
        else: return 'Equity ETF' if asset_type.upper() == 'ETF' else 'Equity Stock'

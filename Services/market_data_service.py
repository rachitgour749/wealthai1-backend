import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Type
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

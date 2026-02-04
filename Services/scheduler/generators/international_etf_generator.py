"""
International ETF Signal Generator

Generates trading signals for International ETF (US) strategy by reusing proven logic
from the backtester's compute_52_week_high_low() function.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional

from Services.scheduler.generators.signal_generator_base import (
    fetch_active_instances,
    save_signals_to_db,
    create_signal,
    expire_old_signals
)
from Services.scheduler.config_utils import scheduler_config
from Strategies.Rotation_International_ETF.services.backtester import InternationalETFRotationBacktester
from Databases.app_data_db_connection import get_session
from Databases.app_data_db_connection import USETFMarket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_international_etf_signals(signal_date: Optional[datetime] = None):
    """
    Generate trading signals for International ETF strategy
    
    Uses US ETF market data
    """
    logger.info("\n" + "="*60)
    logger.info("INTERNATIONAL ETF SIGNAL GENERATION")
    logger.info("="*60)
    
    if signal_date is None:
        signal_date = datetime.now()
    
    # Check if it's a US trading day
    if not scheduler_config.is_trading_day(signal_date.date(), 'US'):
        logger.warning(f"{signal_date.date()} is not a trading day (US). Skipping signal generation.")
        return
    
    logger.info(f"Signal Date: {signal_date.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        instances = fetch_active_instances('Rotation_International_ETF')
        
        if not instances:
            logger.info("No active International ETF instances found")
            return
        
        backtester = InternationalETFRotationBacktester()
        all_signals = []
        
        for instance in instances:
            try:
                signals = _generate_signals_for_instance(
                    instance, signal_date, backtester
                )
                all_signals.extend(signals)
                
            except Exception as e:
                logger.error(f"Error generating signals for instance {instance['id']}: {e}")
                import traceback
                traceback.print_exc()
        
        if all_signals:
            success = save_signals_to_db(all_signals)
            if success:
                logger.info(f"✓ Generated and saved {len(all_signals)} International ETF signals")
        else:
            logger.info("No signals generated")
        
        expire_old_signals(days=7)
        logger.info("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Error in International ETF signal generation: {e}")
        import traceback
        traceback.print_exc()


def _generate_signals_for_instance(
    instance: dict,
    signal_date: datetime,
    backtester: InternationalETFRotationBacktester
) -> List:
    """Generate signals for a single instance"""
    logger.info(f"\nProcessing instance {instance['id']} for user {instance['user_id']}")
    
    tickers = instance.get('tickers', [])
    if not tickers:
        logger.warning(f"No tickers found for instance {instance['id']}")
        return []
    
    logger.info(f"  Tickers: {tickers}")
    
    close_df = _fetch_market_data(tickers, signal_date)
    
    if close_df.empty:
        logger.warning(f"No market data found for instance {instance['id']}")
        return []
    
    high_low_data = backtester.compute_52_week_high_low(close_df, signal_date)
    
    if high_low_data.empty:
        logger.warning(f"Insufficient data for 52-week calculation (instance {instance['id']})")
        return []
    
    sorted_etfs = high_low_data.sort_values('distance_from_low', ascending=True)
    top_etf = sorted_etfs.iloc[0]
    
    logger.info(f"  Selected ETF: {top_etf['symbol']}")
    logger.info(f"    Current Price: ${top_etf['current_price']:.2f}")
    logger.info(f"    52W High: ${top_etf['52w_high']:.2f}")
    logger.info(f"    52W Low: ${top_etf['52w_low']:.2f}")
    logger.info(f"    Distance from Low: {top_etf['distance_from_low']:.2f}%")
    
    metadata = {
        '52w_high': float(top_etf['52w_high']),
        '52w_low': float(top_etf['52w_low']),
        'current_price': float(top_etf['current_price']),
        'distance_from_low': float(top_etf['distance_from_low']),
        'distance_from_high': float(top_etf['distance_from_high']),
        'rank': 1,
        'total_etfs_analyzed': len(high_low_data)
    }
    
    signal = create_signal(
        strategy_name='Rotation_International_ETF',
        user_id=instance['user_id'],
        user_code=instance['user_code'],
        instance_id=instance['id'],
        signal_type='BUY',
        symbol=top_etf['symbol'],
        price=float(top_etf['current_price']),
        signal_date=signal_date,
        strategy_metadata=metadata,
        client_info=instance.get('client_info', {}),
        webhook_url=instance.get('webhook_url'),
        score=float(top_etf['distance_from_low']),
        execution_date=signal_date + timedelta(days=1)
    )
    
    return [signal]


def _fetch_market_data(tickers: List[str], signal_date: datetime) -> pd.DataFrame:
    """Fetch US ETF market data from database"""
    session = get_session()
    
    try:
        lookback_start = signal_date - timedelta(days=400)
        
        data = session.query(USETFMarket).filter(
            USETFMarket.symbol.in_(tickers),
            USETFMarket.date >= lookback_start,
            USETFMarket.date <= signal_date
        ).all()
        
        if not data:
            logger.warning(f"No market data found for tickers: {tickers}")
            return pd.DataFrame()
        
        rows = []
        for record in data:
            rows.append({
                'date': record.date,
                'symbol': record.symbol,
                'close': record.close
            })
        
        df = pd.DataFrame(rows)
        close_df = df.pivot(index='date', columns='symbol', values='close')
        close_df = close_df.sort_index()
        
        logger.info(f"  Fetched {len(close_df)} days of data for {len(close_df.columns)} ETFs")
        
        return close_df
        
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
        
    finally:
        session.close()


if __name__ == "__main__":
    generate_international_etf_signals()

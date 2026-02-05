"""
Stock Rotation Signal Generator

Generates trading signals for Stock Rotation strategy by reusing proven logic
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
from Strategies.Rotation_Stocks.services.backtester import StockRotationBacktester
from Databases.app_data_db_connection import get_session
from Databases.app_data_db_connection import StockMarket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_stock_rotation_signals(signal_date: Optional[datetime] = None):
    """
    Generate trading signals for Stock Rotation strategy
    
    Similar to ETF Rotation but uses stock_market table
    """
    logger.info("\n" + "="*60)
    logger.info("STOCK ROTATION SIGNAL GENERATION")
    logger.info("="*60)
    
    if signal_date is None:
        signal_date = datetime.now()
    
    # Check if it's a trading day
    if not scheduler_config.is_trading_day(signal_date.date(), 'NSE'):
        logger.warning(f"{signal_date.date()} is not a trading day (NSE). Skipping signal generation.")
        return
    
    logger.info(f"Signal Date: {signal_date.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        instances = fetch_active_instances('Rotation_Stocks')
        
        if not instances:
            logger.info("No active Stock Rotation instances found")
            return
        
        backtester = StockRotationBacktester()
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
                logger.info(f"✓ Generated and saved {len(all_signals)} Stock Rotation signals")
        else:
            logger.info("No signals generated")
        
        expire_old_signals(days=7)
        logger.info("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Error in Stock Rotation signal generation: {e}")
        import traceback
        traceback.print_exc()


def _generate_signals_for_instance(
    instance: dict,
    signal_date: datetime,
    backtester: StockRotationBacktester
) -> List:
    """Generate signals for a single instance"""
    logger.info(f"\nProcessing instance {instance['id']} for user {instance['user_id']}")
    
    tickers = instance.get('tickers', [])
    
    # Handle string format (comma-separated)
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(',') if t.strip()]
        
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
    
    sorted_stocks = high_low_data.sort_values('distance_from_low', ascending=True)
    top_stock = sorted_stocks.iloc[0]
    
    logger.info(f"  Selected Stock: {top_stock['symbol']}")
    logger.info(f"    Current Price: ₹{top_stock['current_price']:.2f}")
    logger.info(f"    52W High: ₹{top_stock['52w_high']:.2f}")
    logger.info(f"    52W Low: ₹{top_stock['52w_low']:.2f}")
    logger.info(f"    Distance from Low: {top_stock['distance_from_low']:.2f}%")
    
    metadata = {
        '52w_high': float(top_stock['52w_high']),
        '52w_low': float(top_stock['52w_low']),
        'current_price': float(top_stock['current_price']),
        'distance_from_low': float(top_stock['distance_from_low']),
        'distance_from_high': float(top_stock['distance_from_high']),
        'rank': 1,
        'total_stocks_analyzed': len(high_low_data)
    }
    
    signal = create_signal(
        user_id=instance['user_id'],
        run_id=instance.get('run_id'), # Now available from fetch_active_instances
        user_code=instance['user_code'],
        strategy_name='Rotation_Stocks',
        strategy_type=instance.get('strategy_type', 'Stock Strategy'),
        order_side='BUY',
        symbol_name=top_stock['symbol'],
        client_json=instance.get('client_json', {}), # mapped in base
        webhook_url=instance.get('webhook_url'),
        signal_date=signal_date,
        score=float(top_stock['distance_from_low']),
        price=float(top_stock['current_price']),
        high_52=float(top_stock['52w_high']),
        low_52=float(top_stock['52w_low']),
        executed_at=None, # Pending execution
        execution_status='pending'
    )
    
    return [signal]


def _fetch_market_data(tickers: List[str], signal_date: datetime) -> pd.DataFrame:
    """Fetch stock market data from database"""
    session = get_session()
    
    try:
        lookback_start = signal_date - timedelta(days=400)
        
        # Query Stock market data
        data = session.query(StockMarket).filter(
            StockMarket.symbol.in_(tickers),
            StockMarket.date >= lookback_start,
            StockMarket.date <= signal_date
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
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])

        close_df = df.pivot(index='date', columns='symbol', values='close')
        close_df = close_df.sort_index()
        
        logger.info(f"  Fetched {len(close_df)} days of data for {len(close_df.columns)} stocks")
        
        return close_df
        
    except Exception as e:
        logger.error(f"Failed to fetch market data.")
        logger.error(f"  Tickers ({len(tickers)}): {tickers}")
        logger.error(f"  Date Range: {lookback_start.date()} to {signal_date.date()}")
        logger.error(f"  Error: {str(e)}")
        import traceback
        logger.error(f"  Traceback: {traceback.format_exc()}")
        return pd.DataFrame()
        
    finally:
        session.close()


if __name__ == "__main__":
    generate_stock_rotation_signals()

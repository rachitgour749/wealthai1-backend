"""
ETF Rotation Signal Generator

Generates trading signals for ETF Rotation strategy by reusing proven logic
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
from Strategies.Rotation_ETF.services.backtester import ETFRotationBacktester
from Databases.app_data_db_connection import get_session
from Databases.app_data_db_connection import ETFMarket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_etf_rotation_signals(signal_date: Optional[datetime] = None):
    """
    Generate trading signals for ETF Rotation strategy
    
    This function:
    1. Fetches all active ETF Rotation instances
    2. For each instance, fetches market data
    3. Reuses compute_52_week_high_low() from backtester
    4. Selects ETF with smallest distance from 52-week low
    5. Creates and saves BUY signals
    
    Args:
        signal_date: Optional signal generation date (defaults to today)
    """
    logger.info("\n" + "="*60)
    logger.info("ETF ROTATION SIGNAL GENERATION")
    logger.info("="*60)
    
    # Use current date if not provided
    if signal_date is None:
        signal_date = datetime.now()
    
    # Check if it's a trading day
    if not scheduler_config.is_trading_day(signal_date.date(), 'NSE'):
        logger.warning(f"{signal_date.date()} is not a trading day (NSE). Skipping signal generation.")
        return
    
    logger.info(f"Signal Date: {signal_date.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Fetch active instances
        instances = fetch_active_instances('ETF_Rotation')
        
        if not instances:
            logger.info("No active ETF Rotation instances found")
            return
        
        # Step 2: Initialize backtester (for reusing logic)
        backtester = ETFRotationBacktester()
        
        # Step 3: Generate signals for each instance
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
        
        # Step 4: Save all signals to database
        if all_signals:
            success = save_signals_to_db(all_signals)
            if success:
                logger.info(f"✓ Generated and saved {len(all_signals)} ETF Rotation signals")
        else:
            logger.info("No signals generated")
        
        # Step 5: Expire old signals
        expired_count = expire_old_signals(days=7)
        
        logger.info("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Error in ETF Rotation signal generation: {e}")
        import traceback
        traceback.print_exc()


def _generate_signals_for_instance(
    instance: dict,
    signal_date: datetime,
    backtester: ETFRotationBacktester
) -> List:
    """
    Generate signals for a single instance
    
    Args:
        instance: Instance data dictionary
        signal_date: Signal generation date
        backtester: ETFRotationBacktester instance
    
    Returns:
        List of TradingSignal objects
    """
    logger.info(f"\nProcessing instance {instance['id']} for user {instance['user_id']}")
    
    # Get tickers from instance
    tickers = instance.get('tickers', [])
    
    # Handle string format (comma-separated)
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(',') if t.strip()]
        
    if not tickers:
        logger.warning(f"No tickers found for instance {instance['id']}")
        return []
    
    logger.info(f"  Tickers: {tickers}")
    
    # Fetch market data for tickers
    close_df = _fetch_market_data(tickers, signal_date)
    
    if close_df.empty:
        logger.warning(f"No market data found for instance {instance['id']}")
        return []
    
    # Reuse backtester's compute_52_week_high_low logic
    high_low_data = backtester.compute_52_week_high_low(close_df, signal_date)
    
    if high_low_data.empty:
        logger.warning(f"Insufficient data for 52-week calculation (instance {instance['id']})")
        return []
    
    # Sort by distance_from_low (ascending) - select ETF closest to 52-week low
    sorted_etfs = high_low_data.sort_values('distance_from_low', ascending=True)
    
    # Select top ETF
    top_etf = sorted_etfs.iloc[0]
    
    logger.info(f"  Selected ETF: {top_etf['symbol']}")
    logger.info(f"    Current Price: ₹{top_etf['current_price']:.2f}")
    logger.info(f"    52W High: ₹{top_etf['52w_high']:.2f}")
    logger.info(f"    52W Low: ₹{top_etf['52w_low']:.2f}")
    logger.info(f"    Distance from Low: {top_etf['distance_from_low']:.2f}%")
    logger.info(f"    Distance from High: {top_etf['distance_from_high']:.2f}%")
    
    # Create signal metadata
    metadata = {
        '52w_high': float(top_etf['52w_high']),
        '52w_low': float(top_etf['52w_low']),
        'current_price': float(top_etf['current_price']),
        'distance_from_low': float(top_etf['distance_from_low']),
        'distance_from_high': float(top_etf['distance_from_high']),
        'rank': 1,
        'total_etfs_analyzed': len(high_low_data)
    }
    
    # Create trading signal
    # Create trading signal
    signal = create_signal(
        user_id=instance['user_id'],
        run_id=instance.get('run_id'),
        user_code=instance['user_code'],
        strategy_name='ETF_Rotation',
        strategy_type=instance.get('strategy_type', 'ETF Strategy'),
        order_side='BUY',
        symbol_name=top_etf['symbol'],
        client_json=instance.get('client_json', {}),
        webhook_url=instance.get('webhook_url'),
        signal_date=signal_date,
        score=float(top_etf['distance_from_low']),
        price=float(top_etf['current_price']),
        high_52=float(top_etf['52w_high']),
        low_52=float(top_etf['52w_low']),
        executed_at=None,
        execution_status='pending'
    )
    
    return [signal]


def _fetch_market_data(tickers: List[str], signal_date: datetime) -> pd.DataFrame:
    """
    Fetch market data for tickers from database
    
    Args:
        tickers: List of ticker symbols
        signal_date: Signal date
    
    Returns:
        DataFrame with close prices (index=date, columns=symbols)
    """
    session = get_session()
    
    try:
        # Calculate lookback period (need 252+ trading days for 52-week calculation)
        lookback_start = signal_date - timedelta(days=400)  # ~1.5 years to ensure 252 trading days
        
        # Query ETF market data
        data = session.query(ETFMarket).filter(
            ETFMarket.symbol.in_(tickers),
            ETFMarket.date >= lookback_start,
            ETFMarket.date <= signal_date
        ).all()
        
        if not data:
            logger.warning(f"No market data found for tickers: {tickers}")
            return pd.DataFrame()
        
        # Convert to DataFrame
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
        
        # Pivot to get close prices (index=date, columns=symbols)
        close_df = df.pivot(index='date', columns='symbol', values='close')
        close_df = close_df.sort_index()
        
        logger.info(f"  Fetched {len(close_df)} days of data for {len(close_df.columns)} ETFs")
        
        return close_df
        
    except Exception as e:
        logger.error(f"Failed to fetch market data.")
        logger.error(f"  Tickers ({len(tickers)}): {tickers}")
        logger.error(f"  Date Range: {lookback_start.date()} to {signal_date.date()}")
        logger.error(f"  Error: {str(e)}")
        # Only print traceback for unexpected errors, or keep it but cleaner
        import traceback
        logger.error(f"  Traceback: {traceback.format_exc()}")
        return pd.DataFrame()
        
    finally:
        session.close()


# For testing
if __name__ == "__main__":
    generate_etf_rotation_signals()

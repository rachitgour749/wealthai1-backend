"""
Rotation ETF Payout Signal Generator

Extends standard ETF Rotation with systematic withdrawal feature.
Reuses the backtester's 52-week high-low logic for signal generation.
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
from Strategies.CustomStrategies.Rotation_ETF_Payout.backtester import RotationETFPayoutBacktester
from Databases.app_data_db_connection import get_session
from Databases.app_data_db_connection import ETFMarket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_etf_payout_signals(signal_date: Optional[datetime] = None):
    """
    Generate trading signals for Rotation ETF Payout strategy
    
    Args:
        signal_date: Optional signal generation date (defaults to today)
    """
    logger.info("\n" + "="*60)
    logger.info("ROTATION ETF PAYOUT SIGNAL GENERATION")
    logger.info("="*60)
    
    if signal_date is None:
        signal_date = datetime.now()
        
    # Check if it's a trading day
    if not scheduler_config.is_trading_day(signal_date.date(), 'NSE'):
        logger.warning(f"Skipping: Not a trading day (NSE)")
        return
        
    try:
        # Step 1: Fetch active instances
        instances = fetch_active_instances('Rotation_ETF_Payout')
        
        if not instances:
            logger.info("No active Rotation ETF Payout instances found")
            return
            
        # Step 2: Initialize backtester
        backtester = RotationETFPayoutBacktester()
        
        # Step 3: Generate signals
        all_signals = []
        for instance in instances:
            try:
                signals = _generate_signals_for_instance(instance, signal_date, backtester)
                all_signals.extend(signals)
            except Exception as e:
                logger.error(f"Error for instance {instance['id']}: {e}")
                
        # Step 4: Save signals
        if all_signals:
            save_signals_to_db(all_signals)
            logger.info(f"✓ Generated {len(all_signals)} Rotation ETF Payout signals")
        else:
            logger.info("No signals generated")
            
        expire_old_signals(days=7)
        
    except Exception as e:
        logger.error(f"Error in Rotation ETF Payout signal generation: {e}")


def _generate_signals_for_instance(instance: dict, signal_date: datetime, backtester: RotationETFPayoutBacktester) -> List:
    """Generate signals for a single instance using 52-week high-low logic"""
    logger.info(f"Processing instance {instance['id']} for user {instance['user_id']}")
    
    tickers = instance.get('tickers', [])
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(',') if t.strip()]
        
    if not tickers:
        return []
        
    # Fetch data (1.5 years for 52-week calc)
    session = get_session()
    try:
        start_date = signal_date - timedelta(days=450)
        data = session.query(ETFMarket).filter(
            ETFMarket.symbol.in_(tickers),
            ETFMarket.date >= start_date,
            ETFMarket.date <= signal_date
        ).all()
        
        if not data:
            return []
            
        rows = [{'date': r.date, 'symbol': r.symbol, 'close': r.close} for r in data]
        df = pd.DataFrame(rows)
        pivot_df = df.pivot(index='date', columns='symbol', values='close').sort_index()
        
        # Reuse backtester logic
        high_low_data = backtester.compute_52_week_high_low(pivot_df, signal_date)
        
        if high_low_data.empty:
            return []
            
        # Select ETF closest to 52-week low (ascending)
        sorted_etfs = high_low_data.sort_values('distance_from_low', ascending=True)
        top_etf = sorted_etfs.iloc[0]
        
        logger.info(f"  Selected: {top_etf['symbol']} ({top_etf['distance_from_low']:.2f}% from low)")
        
        # Create BUY signal
        signal = create_signal(
            user_id=instance['user_id'],
            run_id=instance.get('run_id'),
            user_code=instance['user_code'],
            strategy_name='Rotation_ETF_Payout',
            strategy_type=instance.get('strategy_type', 'Rotation_ETF_Payout'),
            order_side='BUY',
            symbol_name=top_etf['symbol'],
            client_json=instance.get('client_json', {}),
            webhook_url=instance.get('webhook_url'),
            signal_date=signal_date,
            score=float(top_etf['distance_from_low']),
            price=float(top_etf['current_price']),
            high_52=float(top_etf['52w_high']),
            low_52=float(top_etf['52w_low']),
            execution_status='pending'
        )
        return [signal]
        
    finally:
        session.close()

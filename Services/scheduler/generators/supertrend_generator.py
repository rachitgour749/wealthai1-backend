"""
SuperTrend Signal Generator

Generates trading signals for SuperTrend strategy (both India and US markets)
by calculating weekly SuperTrend indicators and checking for breakouts.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from Services.scheduler.generators.signal_generator_base import (
    fetch_active_instances,
    save_signals_to_db,
    create_signal,
    expire_old_signals
)
from Services.scheduler.config_utils import scheduler_config
from Strategies.SuperTrend.strategy import SuperTrendStrategy
from Databases.app_data_db_connection import get_session
from Databases.app_data_db_connection import StockMarket, ETFMarket, USETFMarket, USStockMarket, Nifty50IndexMarket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_supertrend_signals(signal_date: Optional[datetime] = None):
    """
    Generate trading signals for SuperTrend strategy
    
    Args:
        signal_date: Optional signal generation date (defaults to today)
    """
    logger.info("\n" + "="*60)
    logger.info("SUPERTREND SIGNAL GENERATION (INDIA & US)")
    logger.info("="*60)
    
    if signal_date is None:
        signal_date = datetime.now()
    
    # Check if it's a trading day (NSE for India, US for US)
    # We will check market-specific holidays inside the instance loop
    
    try:
        # Step 1: Fetch active instances
        # We check for 'SuperTrend' as the primary identifier
        instances = fetch_active_instances('SuperTrend')
        
        if not instances:
            logger.info("No active SuperTrend instances found")
            return
        
        all_signals = []
        
        for instance in instances:
            try:
                # Use strategy_type from instance or default to SuperTrend
                market = instance.get('strategies_parameters', {}).get('market', 'INDIA').upper()
                
                # Check trading day for the specific market
                market_key = 'NSE' if market == 'INDIA' else 'US'
                if not scheduler_config.is_trading_day(signal_date.date(), market_key):
                    logger.warning(f"Skipping {instance['id']} ({market}): Not a trading day for {market_key}")
                    continue
                
                signals = _generate_signals_for_instance(instance, signal_date)
                all_signals.extend(signals)
                
            except Exception as e:
                logger.error(f"Error generating SuperTrend signals for instance {instance['id']}: {e}")
                import traceback
                traceback.print_exc()
        
        # Step 2: Save all signals to database
        if all_signals:
            success = save_signals_to_db(all_signals)
            if success:
                logger.info(f"✓ Generated and saved {len(all_signals)} SuperTrend signals")
        else:
            logger.info("No signals generated")
            
        # Step 3: Expire old signals
        expire_old_signals(days=7)
        
        logger.info("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Error in SuperTrend signal generation: {e}")
        import traceback
        traceback.print_exc()


def _generate_signals_for_instance(instance: dict, signal_date: datetime) -> List:
    """
    Generate signals for a single SuperTrend instance
    """
    logger.info(f"\nProcessing instance {instance['id']} for user {instance['user_id']}")
    
    params = instance.get('strategies_parameters', {})
    market = params.get('market', 'INDIA').upper()
    asset_type = params.get('asset_type', 'STOCK').upper()
    
    # Initialize strategy class for logic reuse
    strategy = SuperTrendStrategy(market=market, asset_type=asset_type)
    strategy.update_config(params)
    
    # Get tickers from instance
    tickers = instance.get('tickers', [])
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(',') if t.strip()]
        
    if not tickers:
        logger.warning(f"No tickers found for instance {instance['id']}")
        return []
        
    logger.info(f"  Market: {market} | Asset: {asset_type} | Tickers: {tickers}")
    
    # Fetch market data for tickers
    all_signals = []
    
    for symbol in tickers:
        try:
            # Fetch daily OHLCV data
            df = _fetch_historical_data(symbol, market, asset_type, signal_date)
            
            if df.empty or len(df) < strategy.atr_period * 5: # Need enough for weekly calc
                logger.warning(f"  Insufficient data for {symbol}")
                continue
                
            # Calculate indicators (Weekly SuperTrend merged back to daily)
            df_with_indicators = strategy.calculate_indicators(df)
            
            # Evaluate signals for the current signal_date
            # Note: evaluate_signals checks if today is the "last day of week"
            # However, for manual trigger, we might want to check the absolute latest state
            # but we'll follow the strategy's end-of-week rule as requested.
            signal_info = strategy.evaluate_signals(symbol, df_with_indicators, signal_date)
            
            if signal_info.get('eligible'):
                # Create BUY signal
                signal = create_signal(
                    user_id=instance['user_id'],
                    run_id=instance.get('run_id'),
                    user_code=instance['user_code'],
                    strategy_name='SuperTrend',
                    strategy_type=instance.get('strategy_type', 'SuperTrend'),
                    order_side='BUY',
                    symbol_name=symbol,
                    client_json=instance.get('client_json', {}),
                    webhook_url=instance.get('webhook_url'),
                    signal_date=signal_date,
                    score=signal_info.get('supertrend', 0.0),
                    price=signal_info.get('close', 0.0),
                    high_52=0.0, # Not used in ST
                    low_52=0.0,
                    execution_status='pending'
                )
                all_signals.append(signal)
            
            # TODO: Sell/Exit signals if necessary
            # The manual trigger usually focuses on NEW signals, 
            # but if strategy requires exit signals (Breakdown), we should handle it.
            # process_exits in ST strategy compares current status in 'slots'.
            # Since this is a stateless generator, we might need to check active positions?
            # For now, we focus on generating entries as per requirement.
            
        except Exception as e:
            logger.error(f"  Error processing {symbol}: {e}")
            
    return all_signals


def _fetch_historical_data(symbol: str, market: str, asset_type: str, signal_date: datetime) -> pd.DataFrame:
    """Fetch daily OHLCV data from the appropriate database table"""
    session = get_session()
    
    try:
        # Calculate lookback period (need enough for weekly SuperTrend calculation)
        lookback_start = signal_date - timedelta(days=600)
        
        # Select table based on market and asset type
        model = None
        if market == 'INDIA':
            if asset_type == 'STOCK': model = StockMarket
            elif asset_type == 'ETF': model = ETFMarket
            elif asset_type == 'INDEX': model = Nifty50IndexMarket
        else: # US
            if asset_type == 'ETF': model = USETFMarket
            elif asset_type == 'STOCK': model = USStockMarket
            
        if not model:
            logger.error(f"  No database model found for {market} / {asset_type}")
            return pd.DataFrame()
            
        # Query data
        data = session.query(model).filter(
            model.symbol == symbol,
            model.date >= lookback_start,
            model.date <= signal_date
        ).order_by(model.date).all()
        
        if not data:
            return pd.DataFrame()
            
        # Convert to DataFrame
        rows = []
        for r in data:
            rows.append({
                'date': r.date,
                'open': r.open,
                'high': r.high,
                'low': r.low,
                'close': r.close,
                'volume': r.volume
            })
            
        df = pd.DataFrame(rows)
        df.set_index('date', inplace=True)
        return df
        
    finally:
        session.close()


if __name__ == "__main__":
    generate_supertrend_signals()

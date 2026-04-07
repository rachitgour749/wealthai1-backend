"""
RS ETF Strategy Signal Generator

Generates trading signals for Relative Strength (RS) ETF strategy.
Calculates RS scores based on weighted returns across multiple periods
and ranks ETFs to identify top performers.
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
from Databases.app_data_db_connection import get_session
from Databases.app_data_db_connection import ETFMarket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_rs_etf_signals(signal_date: Optional[datetime] = None):
    """
    Generate trading signals for RS ETF strategy
    """
    logger.info("\n" + "="*60)
    logger.info("RS ETF SIGNAL GENERATION")
    logger.info("="*60)
    
    if signal_date is None:
        signal_date = datetime.now()
        
    try:
        # Step 1: Fetch active instances
        instances = fetch_active_instances('RS_ETF_Rotation')
        
        if not instances:
            logger.info("No active RS ETF instances found")
            return
            
        all_signals = []
        
        for instance in instances:
            try:
                signals = _generate_signals_for_instance(instance, signal_date)
                all_signals.extend(signals)
            except Exception as e:
                logger.error(f"Error generating RS ETF signals for instance {instance['id']}: {e}")
                
        # Step 2: Save signals
        if all_signals:
            save_signals_to_db(all_signals)
            logger.info(f"✓ Generated {len(all_signals)} RS ETF signals")
        else:
            logger.info("No signals generated")
            
        # Step 3: Expire old
        expire_old_signals(days=7)
        
    except Exception as e:
        logger.error(f"Error in RS ETF signal generation: {e}")


def _generate_signals_for_instance(instance: dict, signal_date: datetime) -> List:
    """Generate signals for a single RS ETF instance"""
    logger.info(f"Processing RS ETF instance {instance['id']} for user {instance['user_id']}")
    
    params = instance.get('strategies_parameters', {})
    lookback_weeks = int(params.get('lookback_weeks', 5))
    lookback_months = int(params.get('lookback_months', 20))
    lookback_quarters = int(params.get('lookback_quarters', 60))
    max_positions = int(params.get('max_positions', 5))
    
    # Get universe
    tickers = instance.get('tickers', [])
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(',') if t.strip()]
        
    if not tickers:
        return []
        
    # Fetch data
    # We need enough data for the longest lookback (60 weeks ~ 420 days)
    lookback_days = max(lookback_weeks * 7, lookback_months * 30, lookback_quarters * 90) + 30
    df = _fetch_rs_market_data(tickers, signal_date, lookback_days)
    
    if df.empty:
        return []
        
    # Calculate RS Scores
    # Logic: Weighted average of returns over different periods
    scores = {}
    current_prices = {}
    
    for symbol in tickers:
        if symbol not in df.columns:
            continue
            
        data = df[symbol].dropna()
        if len(data) < lookback_weeks * 5: # Basic check for enough data
            continue
            
        current_price = data.iloc[-1]
        current_prices[symbol] = current_price
        
        # Period Returns
        def get_return(weeks):
            days = weeks * 5 # Approx trading days
            if len(data) > days:
                prev_price = data.iloc[-days-1]
                return (current_price - prev_price) / prev_price
            return 0.0
            
        ret_w = get_return(lookback_weeks)
        ret_m = get_return(lookback_months)
        ret_q = get_return(lookback_quarters)
        
        # Weighting: 40% Weeks, 30% Months, 30% Quarters (Simplified from RS strategy)
        final_score = (ret_w * 0.4) + (ret_m * 0.3) + (ret_q * 0.3)
        scores[symbol] = final_score
        
    if not scores:
        return []
        
    # Rank and select top performers
    sorted_symbols = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    top_symbols = sorted_symbols[:max_positions]
    
    logger.info(f"  Top {len(top_symbols)} RS ETFs: {top_symbols}")
    
    signals = []
    for symbol in top_symbols:
        signal = create_signal(
            user_id=instance['user_id'],
            run_id=instance.get('run_id'),
            user_code=instance['user_code'],
            strategy_name='RS_ETF',
            strategy_type=instance.get('strategy_type', 'RS_ETF_Rotation'),
            order_side='BUY',
            symbol_name=symbol,
            client_json=instance.get('client_json', {}),
            webhook_url=instance.get('webhook_url'),
            signal_date=signal_date,
            score=float(scores[symbol]),
            price=float(current_prices[symbol]),
            high_52=0.0,
            low_52=0.0,
            execution_status='pending'
        )
        signals.append(signal)
        
    return signals


def _fetch_rs_market_data(tickers: List[str], signal_date: datetime, lookback_days: int) -> pd.DataFrame:
    """Fetch return data for RS calculation"""
    session = get_session()
    try:
        start_date = signal_date - timedelta(days=lookback_days)
        data = session.query(ETFMarket).filter(
            ETFMarket.symbol.in_(tickers),
            ETFMarket.date >= start_date,
            ETFMarket.date <= signal_date
        ).order_by(ETFMarket.date).all()
        
        if not data:
            return pd.DataFrame()
            
        rows = []
        for r in data:
            rows.append({'date': r.date, 'symbol': r.symbol, 'close': r.close})
            
        df = pd.DataFrame(rows)
        if df.empty: return df
        
        pivot_df = df.pivot(index='date', columns='symbol', values='close')
        return pivot_df
    finally:
        session.close()

"""
ETF Swing Strategy Signal Generator

Generates daily trading signals for the ETF Swing Strategy.
Reconstructs portfolio state from executed signals in the database and applies strategy logic.
"""

import logging
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy import desc

from Services.scheduler.generators.signal_generator_base import (
    fetch_active_instances,
    save_signals_to_db,
    create_signal
)
from Services.scheduler.config_utils import scheduler_config
from Strategies.ETF_Swing_Strategy.strategy import ETFSwingStrategy
from Databases.app_data_db_connection import get_session, ETFMarket
from Databases.signal_models import TradingSignal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_etf_swing_signals(signal_date: Optional[datetime] = None):
    """
    Generate trading signals for ETF Swing Strategy
    
    1. Reconstructs portfolio state from DB (TradingSignals)
    2. Fetches market data
    3. Runs strategy logic (Exits then Entries)
    4. Saves new signals to DB
    """
    logger.info("\n" + "="*60)
    logger.info("ETF SWING STRATEGY SIGNAL GENERATION")
    logger.info("="*60)
    
    if signal_date is None:
        signal_date = datetime.now()
        
    # Check if it's a trading day (skip check if forced, but good practice)
    # Note: For Swing strategy, we might want to run even on holidays if we have data, 
    # but strictly speaking only trading days matter.
    if not scheduler_config.is_trading_day(signal_date.date(), 'NSE'):
        logger.warning(f"{signal_date.date()} is not a trading day (NSE). Skipping.")
        return

    logger.info(f"Signal Date: {signal_date.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. Fetch active instances
        instances = fetch_active_instances('ETF_Swing_Strategy')
        if not instances:
            logger.info("No active ETF Swing Strategy instances found.")
            return

        all_signals = []
        
        for instance in instances:
            try:
                signals = _process_instance(instance, signal_date)
                all_signals.extend(signals)
            except Exception as e:
                logger.error(f"Error processing instance {instance['id']}: {e}")
                import traceback
                traceback.print_exc()

        # 2. Save signals
        if all_signals:
            save_signals_to_db(all_signals)
        else:
            logger.info("No new signals generated.")

    except Exception as e:
        logger.error(f"Critical error in ETF Swing Generator: {e}")
        import traceback
        traceback.print_exc()


def _process_instance(instance: Dict, signal_date: datetime) -> List[TradingSignal]:
    """Process a single strategy instance"""
    user_id = instance['user_id']
    strategy_name = instance['strategy_name']
    
    logger.info(f"Processing {strategy_name} for User {user_id}")
    
    # Initialize Strategy
    strategy = ETFSwingStrategy()
    
    # Configure Strategy from Instance
    params = instance.get('strategies_parameters', {})
    if not isinstance(params, dict):
        # Handle case where it might be a JSON string
        try:
             params = json.loads(params) if params else {}
        except:
             params = {}

    strategy.sma_lookback = int(params.get('sma_lookback', 50))
    strategy.stop_loss_pct = float(params.get('stop_loss_pct', 5.0))
    strategy.profit_threshold_pct = float(params.get('profit_threshold_pct', 10.0))
    strategy.num_slots = int(params.get('number_of_slots', 5))
    
    initial_capital = float(params.get('initial_capital', 100000.0)) # Default 1L if missing
    strategy.initialize_portfolio(initial_capital)
    
    # Reconstruct Portfolio State from DB Signals
    _reconstruct_portfolio_state(strategy, user_id, strategy_name)
    
    # Fetch Market Data
    # 1. For Holdings (Current Price & SMA)
    # 2. For Universe (Potential Entries)
    # Universe is defined in 'tickers' list in instance
    universe_tickers = instance.get('tickers', [])
    if isinstance(universe_tickers, str):
        universe_tickers = [t.strip() for t in universe_tickers.split(',')]
    
    # Add held tickers to universe if not present (to ensure we get data for exits)
    held_tickers = [s['data']['symbol'] for s in strategy.slots if s['status'] == 'OCCUPIED']
    all_tickers = list(set(universe_tickers + held_tickers))
    
    if not all_tickers:
        logger.warning("No tickers defined for instance.")
        return []

    market_data_df = _fetch_market_data(all_tickers, signal_date)
    
    if market_data_df.empty:
        logger.warning("No market data available.")
        return []

    generated_signals = []
    
    # --- LOGIC: Exits ---
    # We use 'Process Exits' to check for Stop Loss / Profit
    # Creating a dummy 'prices' dict for the strategy method
    current_prices = {}
    for ticker in all_tickers:
        if ticker in market_data_df.columns:
            # Get latest available price (Close of signal date)
            # In live scenario, we run this After Market Close, so 'Close' is the execution ref for next day? 
            # OR we use Close to decide, and execution is Next Day Open.
            # Strategy logic expects 'prices' map.
            try:
                # Get the last row's close
                price = float(market_data_df[ticker].iloc[-1])
                current_prices[ticker] = price
            except:
                pass
                
    # Update current SMAs for holdings (needed for Profit Exit logic)
    for slot in strategy.slots:
        if slot['status'] == 'OCCUPIED':
            sym = slot['data']['symbol']
            if sym in market_data_df.columns:
                series = market_data_df[sym]
                if len(series) >= strategy.sma_lookback:
                    current_sma = series.rolling(window=strategy.sma_lookback).mean().iloc[-1]
                    strategy.update_holding_sma(sym, current_sma)

    # Note on Date: Strategy uses `current_date` for logs. 
    # Logic: usage of `process_exits`.
    # `process_exits` normally checks against T+1 Open in backtest.
    # In Live: We are at T (Evening). We check against T Close (approx). 
    # If SL hit based on T Close, we signal SELL for T+1 Open.
    
    exits = strategy.process_exits(current_prices, signal_date)
    
    for exit_tx in exits:
        # Create SELL Signal
        # Prepare client_json with signal metadata
        client_json = instance.get('client_json', {}).copy()
        client_json['qty'] = exit_tx['qty']
        client_json['reason'] = exit_tx['reason']

        sig = create_signal(
            user_id=user_id,
            run_id=instance.get('run_id'),
            user_code=instance.get('user_code', ''),
            strategy_name=strategy_name,
            strategy_type='ETF_Swing_Strategy',
            order_side='SELL',
            symbol_name=exit_tx['symbol'],
            client_json=client_json,
            webhook_url=instance.get('webhook_url'),
            signal_date=signal_date,
            score=0.0, # N/A for exit
            price=float(exit_tx['price']), # Estimated Price (Close of T)
            high_52=0.0,
            low_52=0.0,
            execution_status='pending'
        )
        generated_signals.append(sig)
        logger.info(f"Generated SELL signal for {exit_tx['symbol']}")

    # --- LOGIC: Entries ---
    # Evaluate signals for all universe tickers
    eligible_etfs = []
    
    for ticker in universe_tickers:
        if ticker in market_data_df.columns:
            df_ticker = pd.DataFrame({'close': market_data_df[ticker]})
            # Calculate SMA & Distance (Need full dataframe structure for strategy method?)
            # Strategy.evaluate_signals expects DataFrame with 'close' and indices.
            # And it calculates SMA internally in `evaluate_signals`? 
            # No, `evaluate_signals` assumes `df` has 'sma' and 'distance_pct' columns OR calculates them?
            # Looking at `strategy.py`: `evaluate_signals` logs stuff but `calculate_indicators` does the math.
            
            # Helper to clear and prep DF
            full_df = market_data_df[[ticker]].rename(columns={ticker: 'close'})
            full_df = strategy.calculate_indicators(full_df) # Adds 'sma', 'distance_pct'
            
            signal = strategy.evaluate_signals(ticker, full_df, signal_date)
            
            if signal.get('eligible'):
                 eligible_etfs.append(signal)

    # Process Entries
    # Note: `process_entries` handles slot allocation and checking `PENDING_FREE`
    entries = strategy.process_entries(eligible_etfs, signal_date)
    
    for entry_tx in entries:
        # Create BUY Signal
        # Prepare client_json with signal metadata
        client_json = instance.get('client_json', {}).copy()
        client_json['qty'] = entry_tx['qty']

        sig = create_signal(
            user_id=user_id,
            run_id=instance.get('run_id'),
            user_code=instance.get('user_code', ''),
            strategy_name=strategy_name,
            strategy_type='ETF_Swing_Strategy',
            order_side='BUY',
            symbol_name=entry_tx['symbol'],
            client_json=client_json,
            webhook_url=instance.get('webhook_url'),
            signal_date=signal_date,
            score=0.0, # Could use distance?
            price=float(entry_tx['price']),
            high_52=0.0,
            low_52=0.0,
            execution_status='pending'
        )
        generated_signals.append(sig)
        logger.info(f"Generated BUY signal for {entry_tx['symbol']}")

    # Finalize (Live execution doesn't really need finalize loop because we simply save signals, 
    # but the strategy object state is transient here anyway.
    # The 'PENDING_FREE' status was useful for `process_entries` to skip slots freed *this run*.)
    
    return generated_signals


def _reconstruct_portfolio_state(strategy: ETFSwingStrategy, user_id: str, strategy_name: str):
    """
    Rebuild strategy.slots and cash based on Executed Signals from DB.
    """
    session = get_session()
    try:
        # Fetch ALL executed signals for this user/strategy sorted by time
        signals = session.query(TradingSignal).filter(
            TradingSignal.user_id == user_id,
            TradingSignal.strategy_name == strategy_name,
            TradingSignal.execution_status == 'executed'
        ).order_by(TradingSignal.executed_at).all()
        
        if not signals:
            return

        # Replay History
        # This is a simplified replay. In prod, you might snapshot state.
        # Here we trust the signal history linear reconstruction.
        
        for sig in signals:
            if sig.order_side == 'BUY':
                # Allocate to a FREE slot
                # Find first free slot
                free_slots = [s for s in strategy.slots if s['status'] == 'FREE']
                if free_slots:
                    slot = free_slots[0]
                    slot['status'] = 'OCCUPIED'
                    # We need details: Entry Price, Date, Qty
                    # Qty might be in execution_result or estimated
                    qty = 0
                    if sig.execution_result and 'qty' in sig.execution_result:
                         qty = int(sig.execution_result['qty'])
                    elif sig.client_json and 'qty' in sig.client_json:
                         qty = int(sig.client_json['qty'])
                    else:
                         # Fallback estimate
                         qty = int(strategy.slot_capital // sig.price)
                    
                    slot['data'] = {
                        "symbol": sig.symbol_name,
                        "qty": qty,
                        "entry_price": sig.price,
                        "entry_date": sig.executed_at, # Using executed time
                        "sma": 0, # Historical SMA not strictly needed for Exit check unless we store it?
                        # Actually Exit check needs 'current_sma'.
                    }
                    strategy.available_cash -= (sig.price * qty)

            elif sig.order_side == 'SELL':
                # Find the slot holding this symbol
                found_slot = None
                for s in strategy.slots:
                    if s['status'] == 'OCCUPIED' and s['data']['symbol'] == sig.symbol_name:
                        found_slot = s
                        break
                
                if found_slot:
                    # Free it
                    found_slot['status'] = 'FREE'
                    found_slot['data'] = {}
                    # Add cash (approx)
                    # We don't have exact qty from sell signal unless we look at execution_result
                    # But for 'Replay' state to be mostly correct about 'Occupancy', this is enough.
                    # Cash tracking is secondary for 'Signal Generation' unless we want precise position sizing.
                    # For now, we assume standard slot sizing.
                    strategy._recalculate_slot_capital()
        
        logger.info(f"Reconstructed State: {sum(1 for s in strategy.slots if s['status'] == 'OCCUPIED')} slots occupied.")
                    
    except Exception as e:
        logger.error(f"Error reconstructing portfolio: {e}")
    finally:
        session.close()


def _fetch_market_data(tickers: List[str], signal_date: datetime) -> pd.DataFrame:
    """Fetch close prices for tickers"""
    session = get_session()
    try:
        # Need enough lookback for SMA 50 + buffer
        start_date = signal_date - timedelta(days=150)
        
        data = session.query(ETFMarket.date, ETFMarket.symbol, ETFMarket.close).filter(
            ETFMarket.symbol.in_(tickers),
            ETFMarket.date >= start_date,
            ETFMarket.date <= signal_date
        ).all()
        
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data, columns=['date', 'symbol', 'close'])
        df['date'] = pd.to_datetime(df['date'])
        
        # Pivot
        pivot_df = df.pivot(index='date', columns='symbol', values='close')
        pivot_df.sort_index(inplace=True)
        
        # Forward fill missing data?
        pivot_df.fillna(method='ffill', inplace=True)
        
        return pivot_df
        
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return pd.DataFrame()
    finally:
        session.close()

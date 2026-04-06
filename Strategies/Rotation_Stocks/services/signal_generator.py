import sys
import os
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
import pytz
import json

# Add Databases path for market data connection
current_dir = os.path.dirname(os.path.abspath(__file__))
# Adjust path to reach Databases from Strategies/Rotation_Stocks/services
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
databases_path = os.path.join(project_root, 'Databases')
if databases_path not in sys.path:
    sys.path.insert(0, databases_path)

try:
    from .backtester import StockRotationBacktester
except ImportError:
    StockRotationBacktester = None

class LiveStockSignalGenerator:
    """
    Live Stock Signal Generator
    Uses stock_data table for stock data
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the stock signal generator
        """
        # Initialize backtester (uses PostgreSQL for market data)
        self.backtester = StockRotationBacktester(None) if StockRotationBacktester else None
        self.user_id = None  # Will be set from deployment
        self.setup_logging()
        self.create_tables()
    
    def setup_logging(self):
        """Setup logging for stock signal generation"""
        # Ensure logs directory exists
        log_dir = os.path.join(project_root, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'live_stock_signals.log'), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Set console handler to handle Unicode properly
        console_handler = None
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                console_handler = handler
                break
        
        if console_handler:
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
    
    def create_tables(self):
        """Initialize live_stock_signals and live_stock_runs tables in PostgreSQL"""
        try:
            from Databases.app_data_db_connection import create_connection, init_database
            
            # Ensure connection is established
            if not create_connection():
                self.logger.error("Failed to connect to PostgreSQL database")
                raise RuntimeError("Failed to connect to PostgreSQL database")
            
            # Initialize all tables (including live_stock_signals and live_stock_runs)
            if not init_database():
                self.logger.error("Failed to initialize database tables")
                raise RuntimeError("Failed to initialize database tables")
            
            self.logger.info("Live stock signals database tables initialized successfully in PostgreSQL")
            
        except Exception as e:
            self.logger.error(f"Error initializing live stock signals tables: {e}")
            raise
    
    def get_symbols_from_deployment(self, run_id: str) -> List[str]:
        """
        Get stock symbols from stock_saved_strategy table for a specific run_id (PostgreSQL)
        """
        try:
            from Databases.app_data_db_connection import get_session
            from Databases.strategy_models import StockSavedStrategy
            
            session = get_session()
            try:
                # Query stock_saved_strategy table from PostgreSQL
                strategy = session.query(StockSavedStrategy).filter(
                    StockSavedStrategy.run_id == run_id
                ).first()
                
                if not strategy:
                    raise ValueError(f"No Stock Strategy deployment found for run_id: {run_id}")
                
                # Parse tickers from JSON
                try:
                    symbols = json.loads(strategy.tickers) if strategy.tickers else []
                except:
                    symbols = []
                
                if not symbols or len(symbols) == 0:
                    raise ValueError(f"No stock symbols found for run_id: {run_id}")
                
                # Store user_id for later use
                self.user_id = strategy.user_id
                
                self.logger.info(f"Found {len(symbols)} stock symbols from stock_saved_strategy for run_id {run_id}: {symbols}")
                return symbols
            finally:
                session.close()
            
        except Exception as e:
            self.logger.error(f"Error getting symbols from deployment: {str(e)}")
            raise
    
    def get_stock_data_for_signals(self, days_back: int = 365) -> List[str]:
        """
        Get stock symbols for signal generation from stock_market table
        Note: stock_market table should only contain stocks (ETFs are in etf_market table)
        """
        from app_data_db_connection import get_session as get_app_data_session
        from sqlalchemy import text
        
        session = None
        try:
            session = get_app_data_session()
            
            # Calculate the most recent Friday (same logic as ETF signal generator)
            today = datetime.now()
            days_since_friday = (today.weekday() - 4) % 7  # 4 is Friday (0 is Monday)
            last_friday = today - timedelta(days=days_since_friday)
            test_date = last_friday
            
            end_date = test_date.strftime('%Y-%m-%d')
            start_date = (test_date - timedelta(days=days_back)).strftime('%Y-%m-%d')
            self.logger.info(f"Using last Friday's date: {end_date} for stock signal generation")
            
            # Get all symbols from stock_market table
            query = text("SELECT DISTINCT symbol FROM stock_market ORDER BY symbol")
            result = session.execute(query)
            stock_symbols = [row[0] for row in result.fetchall()]
            
            # Filter symbols that have data in the specified date range
            valid_stock_symbols = []
            for symbol in stock_symbols:
                query = text("""
                    SELECT COUNT(*) FROM stock_market 
                    WHERE symbol = :symbol AND date >= :start_date AND date <= :end_date
                """)
                result = session.execute(query, {"symbol": symbol, "start_date": start_date, "end_date": end_date})
                count = result.fetchone()[0]
                if count >= 100:  # At least 100 trading days for testing
                    valid_stock_symbols.append(symbol)
            
            self.logger.info(f"Found {len(valid_stock_symbols)} stock symbols for signal generation")
            return valid_stock_symbols
            
        except Exception as e:
            self.logger.error(f"Error getting stock symbols: {e}")
            raise
        finally:
            if session:
                session.close()
    
    def load_stock_data_from_database(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load stock data from PostgreSQL database"""
        from app_data_db_connection import get_session as get_app_data_session
        from sqlalchemy import text
        
        session = None
        try:
            session = get_market_data_session()
            
            query = text("""
                SELECT date, open, high, low, close, volume, adj_close
                FROM stock_market
                WHERE symbol = :symbol AND date >= CAST(:start_date AS DATE) AND date <= CAST(:end_date AS DATE)
                ORDER BY date
            """)
            
            # Use session.execute for proper parameter binding
            result = session.execute(query, {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date
            })
            rows = result.fetchall()
            columns = result.keys()
            df = pd.DataFrame(rows, columns=columns)
            
            if df.empty:
                return pd.DataFrame()
            
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df['symbol'] = symbol
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()
        finally:
            if session:
                session.close()
    
    def calculate_stock_metrics(self, symbols: List[str], days_back: int = 365) -> pd.DataFrame:
        """Calculate metrics for all stock symbols"""
        try:
            # Calculate the most recent Friday (same logic as ETF signal generator)
            today = datetime.now()
            days_since_friday = (today.weekday() - 4) % 7  # 4 is Friday
            last_friday = today - timedelta(days=days_since_friday)
            test_date = last_friday
            
            end_date = test_date.strftime('%Y-%m-%d')
            start_date = (test_date - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            all_metrics = []
            
            for symbol in symbols:
                try:
                    df = self.load_stock_data_from_database(symbol, start_date, end_date)
                    
                    if df.empty or len(df) < 100:
                        self.logger.warning(f"Insufficient data for {symbol}: {len(df)} days")
                        continue
                    
                    # Get price series
                    close_prices = df['close']
                    volume = df['volume']
                    
                    # Get latest date and price
                    latest_date = close_prices.index[-1]
                    current_price = close_prices.iloc[-1]
                    
                    # Calculate 52-week high and low from CLOSE prices (use available data, min 100 days)
                    window_size = min(len(close_prices), 252)  # Use available data up to 252 days
                    high_52w = close_prices.rolling(window=window_size).max().iloc[-1]
                    low_52w = close_prices.rolling(window=window_size).min().iloc[-1]
                    
                    # Calculate relative strength (RS) score
                    rs_score = (current_price - low_52w) / (high_52w - low_52w) if high_52w != low_52w else 0
                    
                    # Calculate momentum indicators
                    momentum_20d = (close_prices.iloc[-1] - close_prices.iloc[-20]) / close_prices.iloc[-20] if len(close_prices) >= 20 else 0
                    avg_volume = volume.rolling(window=20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
                    volume_ratio = volume.iloc[-1] / avg_volume if avg_volume > 0 else 1
                    
                    metrics = {
                        'symbol': symbol,
                        'date': latest_date,
                        'price': current_price,
                        'high_52w': high_52w,
                        'low_52w': low_52w,
                        'rs_score': rs_score,
                        'momentum_20d': momentum_20d,
                        'volume_ratio': volume_ratio,
                        'volume': volume.iloc[-1]
                    }
                    
                    all_metrics.append(metrics)
                    
                except Exception as e:
                    self.logger.error(f"Error calculating metrics for {symbol}: {e}")
                    continue
            
            if not all_metrics:
                self.logger.warning("No valid stock metrics calculated")
                return pd.DataFrame()
            
            metrics_df = pd.DataFrame(all_metrics)
            self.logger.info(f"Calculated metrics for {len(metrics_df)} stocks")
            
            return metrics_df
            
        except Exception as e:
            self.logger.error(f"Error calculating stock metrics: {e}")
            raise
    
    def print_detailed_momentum_calculation(self, metrics_df: pd.DataFrame):
        """Print detailed momentum calculation for stocks"""
        try:
            if metrics_df.empty:
                print("\n⚠️  No stock data available for momentum calculation")
                return
            
            # Calculate distance percentages
            metrics_df['distance_from_low_pct'] = ((metrics_df['price'] - metrics_df['low_52w']) / metrics_df['low_52w']) * 100
            metrics_df['distance_from_high_pct'] = ((metrics_df['high_52w'] - metrics_df['price']) / metrics_df['high_52w']) * 100
            
            # Filter valid stocks (with proper 52-week data)
            valid_stocks = metrics_df.dropna(subset=['high_52w', 'low_52w', 'price'])
            
            print(f"\n📊 Stock Momentum Calculation for {metrics_df['date'].iloc[0].strftime('%Y-%m-%d')}:")
            print(f"   Available historical data: {len(metrics_df)} stocks")
            print(f"   Required for calculation: 100+ days")
            print(f"   ✅ Using available trading days for momentum calculation")
            print(f"   📈 52-week max closes calculated for {len(valid_stocks)} stocks")
            print(f"   📉 52-week min closes calculated for {len(valid_stocks)} stocks")
            print(f"   💰 Current prices available for {len(valid_stocks)} stocks")
            
            # Display detailed metrics for each stock (limit to top 10 for readability)
            display_stocks = valid_stocks.head(10)
            for _, row in display_stocks.iterrows():
                print(f"   ✅ {row['symbol']}: Max_Close=₹{row['high_52w']:.2f}, Min_Close=₹{row['low_52w']:.2f}, Current=₹{row['price']:.2f}")
                print(f"       Distance from Min Close: {row['distance_from_low_pct']:.2f}%, Distance from Max Close: {row['distance_from_high_pct']:.2f}%")
            
            if len(valid_stocks) > 10:
                print(f"   ... and {len(valid_stocks) - 10} more stocks")
            
            print(f"   🎯 Final momentum result: {len(valid_stocks)} valid stocks with proper 52-week data")
            
            # Rank stocks by distance from 52-week min close (closest first)
            ranked_from_low = valid_stocks.sort_values('distance_from_low_pct', ascending=True)
            
            print(f"   📊 Stock Rankings from 52-week MIN CLOSE (closest to min close first):")
            for i, (_, row) in enumerate(ranked_from_low.head(10).iterrows(), 1):
                print(f"      {i}. {row['symbol']}: {row['distance_from_low_pct']:.2f}% from min close")
            
            # Rank stocks by distance from 52-week max close (closest first)
            ranked_from_high = valid_stocks.sort_values('distance_from_high_pct', ascending=True)
            
            print(f"   📈 Stock Rankings from 52-week MAX CLOSE (closest to max close first):")
            for i, (_, row) in enumerate(ranked_from_high.head(10).iterrows(), 1):
                print(f"      {i}. {row['symbol']}: {row['distance_from_high_pct']:.2f}% from max close")
            
        except Exception as e:
            self.logger.error(f"Error in detailed momentum calculation: {e}")
    
    def print_signal_details(self, signals: List[Dict[str, Any]]):
        """Print stock signal details in the requested format"""
        try:
            if not signals:
                print("\n⚠️  No stock signals generated for this period")
                return
            
            buy_signals = [s for s in signals if s['side'] == 'BUY']
            sell_signals = [s for s in signals if s['side'] == 'SELL']
            
            print(f"\n📊 Final Stock Results:")
            
            if buy_signals:
                print(f"\nBUY -> Buy stock details according to the calculation:")
                for i, signal in enumerate(buy_signals, 1):
                    symbol = signal['symbol']
                    score = signal['score']
                    reason = signal['reason']
                    price = signal['payload']['price']
                    distance_from_low = signal['payload']['distance_from_low_pct']
                    distance_from_high = signal['payload']['distance_from_high_pct']
                    rank = signal['payload']['rank_from_low']
                    
                    print(f"   {i}. {symbol}")
                    print(f"      Price: ₹{price:.2f}")
                    print(f"      Score: {score:.2f}")
                    print(f"      Distance from 52-week min close: {distance_from_low:.2f}%")
                    print(f"      Distance from 52-week max close: {distance_from_high:.2f}%")
                    print(f"      Rank from min close: #{rank}")
                    print(f"      Reason: {reason}")
            
            if sell_signals:
                print(f"\nSELL -> Sell stock details according to the calculation:")
                for i, signal in enumerate(sell_signals, 1):
                    symbol = signal['symbol']
                    score = signal['score']
                    reason = signal['reason']
                    price = signal['payload']['price']
                    distance_from_low = signal['payload']['distance_from_low_pct']
                    distance_from_high = signal['payload']['distance_from_high_pct']
                    rank = signal['payload']['rank_from_high']
                    
                    print(f"   {i}. {symbol}")
                    print(f"      Price: ₹{price:.2f}")
                    print(f"      Score: {score:.2f}")
                    print(f"      Distance from 52-week min close: {distance_from_low:.2f}%")
                    print(f"      Distance from 52-week max close: {distance_from_high:.2f}%")
                    print(f"      Rank from max close: #{rank}")
                    print(f"      Reason: {reason}")
                    
        except Exception as e:
            self.logger.error(f"Error in stock signal details: {e}")
    
    def generate_signals(self, metrics_df: pd.DataFrame, strategy_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Generate BUY/SELL signals for stocks based on simple ranking logic:
        BUY: Closest to 52-week min close (top 1 candidate)
        SELL: Closest to 52-week max close (top 3 candidates)
        """
        try:
            signals = []
            
            # Calculate distance percentages for ranking
            metrics_df['distance_from_low_pct'] = ((metrics_df['price'] - metrics_df['low_52w']) / metrics_df['low_52w']) * 100
            metrics_df['distance_from_high_pct'] = ((metrics_df['high_52w'] - metrics_df['price']) / metrics_df['high_52w']) * 100
            
            # BUY Logic: Closest to 52-week min close (top 1 candidate only)
            buy_candidates = metrics_df.sort_values('distance_from_low_pct', ascending=True)
            top_buy_count = 1  # Take only top 1 closest to min close
            
            for i, (_, row) in enumerate(buy_candidates.head(top_buy_count).iterrows()):
                signal = {
                    'symbol': row['symbol'],
                    'side': 'BUY',
                    'score': round(100 - row['distance_from_low_pct'], 2),  # Higher score = closer to min close
                    'reason': f"Rank {i+1} closest to 52-week min close: {row['distance_from_low_pct']:.2f}% from min close",
                    'payload': {
                        'price': row['price'],
                        'high_52w': row['high_52w'],
                        'low_52w': row['low_52w'],
                        'distance_from_low_pct': row['distance_from_low_pct'],
                        'distance_from_high_pct': row['distance_from_high_pct'],
                        'rank_from_low': i + 1
                    }
                }
                signals.append(signal)
            
            # SELL Logic: Closest to 52-week max close (top 1-3 candidates)
            sell_candidates = metrics_df.sort_values('distance_from_high_pct', ascending=True)
            top_sell_count = min(3, len(sell_candidates))  # Take top 3 closest to max close
            
            for i, (_, row) in enumerate(sell_candidates.head(top_sell_count).iterrows()):
                signal = {
                    'symbol': row['symbol'],
                    'side': 'SELL',
                    'score': round(100 - row['distance_from_high_pct'], 2),  # Higher score = closer to max close
                    'reason': f"Rank {i+1} closest to 52-week max close: {row['distance_from_high_pct']:.2f}% from max close",
                    'payload': {
                        'price': row['price'],
                        'high_52w': row['high_52w'],
                        'low_52w': row['low_52w'],
                        'distance_from_low_pct': row['distance_from_low_pct'],
                        'distance_from_high_pct': row['distance_from_high_pct'],
                        'rank_from_high': i + 1
                    }
                }
                signals.append(signal)
            
            self.logger.info(f"Generated {len(signals)} stock signals: {len([s for s in signals if s['side'] == 'BUY'])} BUY, {len([s for s in signals if s['side'] == 'SELL'])} SELL")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating stock signals: {e}")
            raise
    
    def save_signals_to_database(self, signals: List[Dict[str, Any]], run_id: str, strategy_type: str, strategy_config: Dict[str, Any]):
        """Save stock signals to database (PostgreSQL)"""
        try:
            from Databases.app_data_db_connection import get_session
            from Databases.strategy_models import LiveStockSignal, LiveStockRun
            from sqlalchemy.dialects.postgresql import insert
            
            # Calculate the most recent Friday (same logic as ETF signal generator)
            today = datetime.now()
            days_since_friday = (today.weekday() - 4) % 7  # 4 is Friday
            last_friday = today - timedelta(days=days_since_friday)
            signal_date = last_friday.strftime('%Y-%m-%d')
            
            session = get_session()
            try:
                # Save run information
                buy_count = len([s for s in signals if s['side'] == 'BUY'])
                sell_count = len([s for s in signals if s['side'] == 'SELL'])
                
                run_data = {
                    'run_id': run_id,
                    'strategy_type': strategy_type,
                    'strategy_config': json.dumps(strategy_config) if strategy_config else None,
                    'signals_count': len(signals),
                    'buy_count': buy_count,
                    'sell_count': sell_count,
                    'run_date': signal_date,
                    'status': 'completed'
                }
                
                stmt = insert(LiveStockRun).values(**run_data)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['run_id'],
                    set_=dict(
                        strategy_type=stmt.excluded.strategy_type,
                        strategy_config=stmt.excluded.strategy_config,
                        signals_count=stmt.excluded.signals_count,
                        buy_count=stmt.excluded.buy_count,
                        sell_count=stmt.excluded.sell_count,
                        run_date=stmt.excluded.run_date,
                        status=stmt.excluded.status
                    )
                )
                session.execute(stmt)
                
                # Insert/update signals
                for signal in signals:
                    signal_data = {
                        'run_id': run_id,
                        'symbol': signal['symbol'],
                        'side': signal['side'],
                        'score': signal['score'],
                        'reason': signal.get('reason'),
                        'payload': json.dumps(signal['payload']) if 'payload' in signal else None,
                        'signal_date': signal_date
                    }
                    
                    stmt = insert(LiveStockSignal).values(**signal_data)
                    stmt = stmt.on_conflict_do_nothing()  # Skip if duplicate
                    session.execute(stmt)
                
                session.commit()
                self.logger.info(f"Saved {len(signals)} stock signals to PostgreSQL database with run_id: {run_id}")
                
            except Exception as e:
                session.rollback()
                raise
            finally:
                session.close()
            
        except Exception as e:
            self.logger.error(f"Error saving stock signals: {e}")
            raise
    
    def run_weekly_signal_generation(self, run_id: str = None, strategy_type: str = 'StockSurfTrend', strategy_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run weekly stock signal generation
        
        Args:
            run_id: Optional run_id from deployment. If not provided, generates one.
            strategy_type: Strategy type name
            strategy_config: Strategy configuration parameters
        """
        start_time = datetime.now()
        
        try:
            # Handle run_id - use provided or generate new one
            if run_id:
                self.logger.info(f"Using run_id: {run_id}")
            else:
                run_id = f"{start_time.strftime('%Y%m%d')}_Stock{strategy_type}"
                self.logger.warning(f"No run_id provided, generated: {run_id}")
            
            self.logger.info("Starting weekly stock signal generation...")
            
            # Get stock symbols - try from deployment first, fallback to all stocks
            try:
                stock_symbols = self.get_symbols_from_deployment(run_id)
                self.logger.info(f"Using {len(stock_symbols)} stocks from deployment: {stock_symbols}")
            except Exception as e:
                self.logger.error(f"Error getting stocks from deployment: {e}")
                self.logger.warning("Falling back to all available stocks")
                stock_symbols = self.get_stock_data_for_signals()
            
            if not stock_symbols:
                raise ValueError(f"No valid stock symbols found for run_id: {run_id}")
            
            # Calculate metrics for all stocks
            metrics_df = self.calculate_stock_metrics(stock_symbols)
            
            if metrics_df.empty:
                raise ValueError("No valid stock metrics calculated")
            
            # Print detailed momentum calculation
            self.print_detailed_momentum_calculation(metrics_df)
            
            # Generate signals
            signals = self.generate_signals(metrics_df, strategy_config)
            
            # Print signal details only (no execution simulation)
            self.print_signal_details(signals)
            
            # Save to database
            self.save_signals_to_database(signals, run_id, strategy_type, strategy_config)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = {
                'success': True,
                'run_id': run_id,
                'signals': signals,
                'signals_count': len(signals),
                'buy_count': len([s for s in signals if s['side'] == 'BUY']),
                'sell_count': len([s for s in signals if s['side'] == 'SELL']),
                'duration_seconds': duration,
                'strategy_type': strategy_type,
                'strategy_config': strategy_config
            }
            
            self.logger.info(f"Stock signal generation completed successfully in {duration:.2f}s")
            self.logger.info(f"Generated {result['buy_count']} BUY and {result['sell_count']} SELL signals")
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.error(f"Stock signal generation failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'duration_seconds': duration,
                'run_id': run_id if 'run_id' in locals() else None
            }
    
    def get_all_running_strategies(self) -> List[Dict[str, Any]]:
        """
        Get all running stock strategies from stock_saved_strategy table (PostgreSQL)
        
        Returns:
            List of dictionaries with run_id, user_id, and strategy info
        """
        try:
            from Databases.app_data_db_connection import get_session
            from Databases.strategy_models import StockSavedStrategy
            
            session = get_session()
            try:
                # Query all active strategies with run_id and status = 'running'
                strategies_query = session.query(StockSavedStrategy).filter(
                    StockSavedStrategy.run_id.isnot(None),
                    StockSavedStrategy.run_id != '',
                    StockSavedStrategy.status == 'running'
                ).order_by(StockSavedStrategy.created_timestamp.desc())
                
                strategies = []
                for strategy in strategies_query.all():
                    # Parse tickers
                    try:
                        tickers = json.loads(strategy.tickers) if strategy.tickers else []
                    except:
                        tickers = []
                    
                    if tickers:  # Only include if we have tickers
                        strategies.append({
                            'run_id': strategy.run_id,
                            'user_id': strategy.user_id,
                            'strategy_name': strategy.strategy_name,
                            'strategy_type': strategy.strategy_type,
                            'tickers': tickers
                        })
                
                self.logger.info(f"Found {len(strategies)} running stock strategies")
                return strategies
            finally:
                session.close()
            
        except Exception as e:
            self.logger.error(f"Error getting running stock strategies: {e}")
            return []
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'backtester') and self.backtester:
                # Cleanup backtester resources if needed
                pass
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    # Test the stock signal generator
    generator = LiveStockSignalGenerator()
    
    try:
        result = generator.run_weekly_signal_generation()
        print(f"\nStock Signal Generation Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        generator.cleanup()

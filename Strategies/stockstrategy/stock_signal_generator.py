import sys
import os
import sqlite3
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
import pytz
import json

# Add the etf-strategy path for backtester import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'etf-strategy'))

try:
    from backtester_core import ETFRotationBacktester
except ImportError:
    ETFRotationBacktester = None

class LiveStockSignalGenerator:
    """
    Live Stock Signal Generator using the same logic as ETF generator
    but filtering for stocks (excluding the 20 specific ETF symbols)
    """
    
    def __init__(self, db_path: str = None):
        """Initialize the stock signal generator"""
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'unified_etf_data.sqlite')
        
        # Ensure absolute path
        self.db_path = os.path.abspath(db_path)
        self.backtester = ETFRotationBacktester() if ETFRotationBacktester else None
        self.user_id = None  # Will be set from deployment
        self.setup_logging()
        self.create_tables()
    
    def setup_logging(self):
        """Setup logging for stock signal generation"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/live_stock_signals.log', encoding='utf-8'),
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
        """Create tables for stock signals if they don't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create live_stock_signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS live_stock_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    score REAL NOT NULL,
                    reason TEXT,
                    payload TEXT,
                    signal_date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT (datetime('now', '+05:30'))
                )
            ''')
            
            # Create live_stock_runs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS live_stock_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    strategy_type TEXT NOT NULL,
                    strategy_config TEXT,
                    signals_count INTEGER DEFAULT 0,
                    buy_count INTEGER DEFAULT 0,
                    sell_count INTEGER DEFAULT 0,
                    run_date DATE NOT NULL,
                    duration_seconds REAL,
                    status TEXT DEFAULT 'completed',
                    created_at TIMESTAMP DEFAULT (datetime('now', '+05:30'))
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("Live stock signals database tables created/verified successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating stock signal tables: {e}")
            raise
    
    def get_symbols_from_deployment(self, run_id: str) -> List[str]:
        """
        Get stock symbols from deploy_details table for a specific run_id
        Similar to ETF signal generator logic
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Fetch deployment details for the given run_id
            # Note: Stock deployments also use etf_names column (same column for both strategies)
            cursor.execute('''
                SELECT user_email, etf_names, client_information_json, strategy_type
                FROM deploy_details 
                WHERE run_id = ? AND strategy_type = 'Stock Strategy'
            ''', (run_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                raise ValueError(f"No Stock Strategy deployment found for run_id: {run_id}")
            
            user_email, symbols_json, client_info_json, strategy_type = result
            
            # Store user_id for later use
            self.user_id = user_email
            
            # Parse symbols JSON (stored in etf_names column for both ETF and Stock strategies)
            if symbols_json:
                try:
                    symbols = json.loads(symbols_json)
                    if isinstance(symbols, list):
                        stock_symbols = symbols
                    else:
                        self.logger.warning(f"Symbols is not a list: {symbols}")
                        stock_symbols = []
                except json.JSONDecodeError as e:
                    self.logger.error(f"Error parsing symbols JSON: {e}")
                    stock_symbols = []
            else:
                self.logger.warning(f"No symbols found for run_id: {run_id}")
                stock_symbols = []
            
            self.logger.info(f"Found {len(stock_symbols)} stock symbols for run_id {run_id}: {stock_symbols}")
            self.logger.info(f"Using stocks from deployment: {stock_symbols}")
            
            return stock_symbols
            
        except Exception as e:
            self.logger.error(f"Error getting symbols from deployment: {e}")
            raise
    
    def get_stock_data_for_signals(self, days_back: int = 365) -> List[str]:
        """
        Get stock symbols for signal generation (exclude the 20 ETF symbols)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Define the 20 ETF symbols to exclude
            etf_symbols = [
                'BANKBEES', 'GOLDBEES', 'GOLDSHARE', 'HDFCGOLD', 'HNGSNGBEES', 
                'ICICIGOLD', 'INFRABEES', 'ITBEES', 'JUNIORBEES', 'KOTAKGOLD', 
                'LIQUIDBEES', 'MON100', 'NIFTY1', 'NIFTYBEES', 'SHARMBEES', 
                'PSUBNKBEES', 'QNIFTY', 'SENSEX1', 'SETFGOLD', 'SHARIABEES'
            ]
            
            # Calculate the most recent Friday (same logic as ETF signal generator)
            today = datetime.now()
            days_since_friday = (today.weekday() - 4) % 7  # 4 is Friday (0 is Monday)
            last_friday = today - timedelta(days=days_since_friday)
            test_date = last_friday
            
            end_date = test_date.strftime('%Y-%m-%d')
            start_date = (test_date - timedelta(days=days_back)).strftime('%Y-%m-%d')
            self.logger.info(f"Using last Friday's date: {end_date} for stock signal generation")
            
            # Get all symbols from etf_unified table
            cursor.execute('SELECT DISTINCT symbol FROM etf_unified ORDER BY symbol')
            all_symbols = [row[0] for row in cursor.fetchall()]
            
            # Filter out ETF symbols to get only stocks
            stock_symbols = [symbol for symbol in all_symbols if symbol not in etf_symbols]
            
            # Filter symbols that have data in the specified date range
            valid_stock_symbols = []
            for symbol in stock_symbols:
                cursor.execute('''
                    SELECT COUNT(*) FROM etf_unified 
                    WHERE symbol = ? AND date >= ? AND date <= ?
                ''', (symbol, start_date, end_date))
                
                count = cursor.fetchone()[0]
                if count >= 100:  # At least 100 trading days for testing
                    valid_stock_symbols.append(symbol)
            
            conn.close()
            
            self.logger.info(f"Found {len(valid_stock_symbols)} stock symbols for signal generation")
            return valid_stock_symbols
            
        except Exception as e:
            self.logger.error(f"Error getting stock symbols: {e}")
            raise
    
    def load_stock_data_from_database(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load stock data from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT date, open, high, low, close, volume
                FROM etf_unified
                WHERE symbol = ? AND date >= ? AND date <= ?
                ORDER BY date
            ''', (symbol, start_date, end_date))
            
            data = cursor.fetchall()
            conn.close()
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df['symbol'] = symbol
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()
    
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
        """Save stock signals to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate the most recent Friday (same logic as ETF signal generator)
            today = datetime.now()
            days_since_friday = (today.weekday() - 4) % 7  # 4 is Friday
            last_friday = today - timedelta(days=days_since_friday)
            signal_date = last_friday.strftime('%Y-%m-%d')
            
            # Get current Indian time
            indian_tz = pytz.timezone('Asia/Kolkata')
            current_indian_time = datetime.now(indian_tz).strftime('%Y-%m-%d %H:%M:%S')
            
            for signal in signals:
                cursor.execute('''
                    INSERT INTO live_stock_signals 
                    (run_id, symbol, side, score, reason, payload, signal_date, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    run_id,
                    signal['symbol'],
                    signal['side'],
                    signal['score'],
                    signal['reason'],
                    str(signal['payload']),
                    signal_date,
                    current_indian_time
                ))
            
            # Save run information
            buy_count = len([s for s in signals if s['side'] == 'BUY'])
            sell_count = len([s for s in signals if s['side'] == 'SELL'])
            
            cursor.execute('''
                INSERT INTO live_stock_runs 
                (run_id, strategy_type, strategy_config, signals_count, buy_count, sell_count, run_date, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id,
                strategy_type,
                str(strategy_config),
                len(signals),
                buy_count,
                sell_count,
                signal_date,
                current_indian_time
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Saved {len(signals)} stock signals to database with run_id: {run_id}")
            
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
        Get all running stock strategies from stock_saved_strategy table
        
        Returns:
            List of dictionaries with run_id, user_id, and strategy info
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query all active strategies with run_id and status = 'running'
            cursor.execute('''
                SELECT run_id, user_id, strategy_name, strategy_type, tickers
                FROM stock_saved_strategy 
                WHERE run_id IS NOT NULL 
                AND run_id != ''
                AND status = 'running'
                ORDER BY created_timestamp DESC
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            strategies = []
            for row in rows:
                run_id, user_id, strategy_name, strategy_type, tickers_json = row
                
                # Parse tickers
                try:
                    tickers = json.loads(tickers_json) if tickers_json else []
                except:
                    tickers = []
                
                if tickers:  # Only include if we have tickers
                    strategies.append({
                        'run_id': run_id,
                        'user_id': user_id,
                        'strategy_name': strategy_name,
                        'strategy_type': strategy_type,
                        'tickers': tickers
                    })
            
            self.logger.info(f"Found {len(strategies)} running stock strategies")
            return strategies
            
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

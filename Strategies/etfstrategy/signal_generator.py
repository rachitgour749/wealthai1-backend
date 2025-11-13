"""
Live Market Engine - Phase 2: Signal Generation
Generates weekly BUY/SELL signals using ETFRotationBacktester core
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import os
import sys

# Get the absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

# Add paths to Python path
sys.path.insert(0, os.path.join(project_root, 'wealthai1-backend'))  # Add inner project directory
etf_strategy_path = os.path.join(project_root, 'wealthai1-backend', 'etf-strategy')
sys.path.insert(0, etf_strategy_path)

# Add Databases path for market data connection
databases_path = os.path.join(project_root, 'Databases')
sys.path.insert(0, databases_path)

print(f"Project root: {project_root}")
print(f"Inner project dir: {os.path.join(project_root, 'wealthai1-backend')}")
print(f"ETF strategy path: {etf_strategy_path}")
print(f"Python path: {sys.path}")

# Try multiple import paths for backtester_core
try:
    # First try relative import (when used as part of the package)
    from .backtester_core import ETFRotationBacktester
except ImportError:
    try:
        # Try absolute import from current directory
        from backtester_core import ETFRotationBacktester
    except ImportError as e:
        print(f"Error importing backtester_core: {e}")
        print("\nCurrent working directory:", os.getcwd())
        print("Available directories:")
        # Try to find it in the project structure
        for root, dirs, files in os.walk(os.path.join(project_root, 'wealthai1-backend')):
            if 'backtester_core.py' in files:
                print(f"Found backtester_core.py in: {root}")
                sys.path.insert(0, root)
                from backtester_core import ETFRotationBacktester
                print("Successfully imported ETFRotationBacktester after path correction")
                break
        else:
            # If all else fails, try importing from Strategies.etfstrategy
            try:
                from Strategies.etfstrategy.backtester_core import ETFRotationBacktester
            except ImportError:
                raise ImportError("Could not import ETFRotationBacktester from any known location")

# Try to import monitoring module, use fallback if not available
try:
    from monitoring import log_signal_generation, log_performance
except ImportError:
    print("Warning: monitoring module not found, using fallback functions")
    def log_signal_generation(*args, **kwargs):
        """Fallback function when monitoring module is not available"""
        pass
    
    def log_performance(*args, **kwargs):
        """Fallback function when monitoring module is not available"""
        pass

class LiveSignalGenerator:
    """
    Generates live trading signals using the ETF rotation backtester core
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize Live Signal Generator
        
        Args:
            db_path: Deprecated - kept for compatibility. Market data now uses PostgreSQL.
                    If provided, will be used for live_signals table (SQLite).
        """
        # For live_signals table, use SQLite if db_path provided, otherwise use default
        if db_path is None:
            # Get the root directory (2 levels up from the current file)
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.db_path = os.path.join(root_dir, "unified_etf_data.sqlite")
        else:
            self.db_path = os.path.abspath(db_path) if db_path else None
            
        # Initialize backtester (now uses PostgreSQL for market data)
        self.backtester = ETFRotationBacktester(None)
        self.setup_logging()
        if self.db_path:
            self.create_tables()
        
    def setup_logging(self):
        """Setup logging for signal generation"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/live_signals.log', encoding='utf-8'),
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
        """Initialize live_signals and live_runs tables in PostgreSQL (db_path parameter ignored, kept for compatibility)"""
        try:
            # Import database connection and models
            from Databases.neon_db_connection import create_connection, init_database
            
            # Ensure connection is established
            if not create_connection():
                self.logger.error("Failed to connect to PostgreSQL database")
                raise RuntimeError("Failed to connect to PostgreSQL database")
            
            # Initialize all tables (including live_signals and live_runs)
            if not init_database():
                self.logger.error("Failed to initialize database tables")
                raise RuntimeError("Failed to initialize database tables")
            
            self.logger.info("Live signals database tables initialized successfully in PostgreSQL")
            
        except Exception as e:
            self.logger.error(f"Error initializing live signals tables: {e}")
            raise
    
    def generate_run_id(self, strategy_type: str = "SurfTrend") -> str:
        """
        Generate run_id in format: YYYYMMDD_<strategy_type>
        Example: 20250912_SurfTrend
        """
        today = datetime.now().strftime("%Y%m%d")
        return f"{today}_{strategy_type}"
    
    def get_symbols_from_deployment(self, run_id: str) -> Dict[str, Any]:
        """
        Get ETF symbols and user info from etf_saved_strategy table based on run_id
        Falls back to deploy_details table if not found in etf_saved_strategy
        
        Args:
            run_id: Deployment run ID (format: run_etf_strategy_2025-10-15_1760544030144)
            
        Returns:
            Dictionary with user_id, etf_symbols list, and etf_names
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # First try etf_saved_strategy table (primary source)
            cursor.execute('''
                SELECT user_id, tickers, etf_names, etf_count 
                FROM etf_saved_strategy 
                WHERE run_id = ?
            ''', (run_id,))
            
            result = cursor.fetchone()
            
            if result:
                # Found in etf_saved_strategy table
                user_id, tickers_json, etf_names_json, etf_count = result
                
                # Parse tickers from JSON (preferred source)
                try:
                    etf_symbols = json.loads(tickers_json) if tickers_json else []
                except:
                    etf_symbols = []
                
                # If tickers not found, try etf_names
                if not etf_symbols or len(etf_symbols) == 0:
                    try:
                        etf_symbols = json.loads(etf_names_json) if etf_names_json else []
                    except:
                        etf_symbols = []
                
                conn.close()
                
                if not etf_symbols or len(etf_symbols) == 0:
                    raise ValueError(f"No ETF symbols found in tickers or etf_names for run_id: {run_id}")
                
                self.logger.info(f"Found {len(etf_symbols)} ETF symbols from etf_saved_strategy for run_id {run_id}: {etf_symbols}")
                
                return {
                    'user_id': user_id,
                    'etf_symbols': etf_symbols,
                    'etf_names': etf_symbols,
                    'etf_count': etf_count if etf_count else len(etf_symbols)
                }
            
            # Fallback to deploy_details table (for backward compatibility)
            try:
                cursor.execute('''
                    SELECT user_email, etf_names, etf_count 
                    FROM deploy_details 
                    WHERE run_id = ?
                ''', (run_id,))
                
                result = cursor.fetchone()
            except sqlite3.OperationalError as e:
                # Table might not exist
                conn.close()
                raise ValueError(f"No deployment found for run_id: {run_id} in etf_saved_strategy or deploy_details tables. Error: {str(e)}")
            
            if not result:
                conn.close()
                raise ValueError(f"No deployment found for run_id: {run_id} in etf_saved_strategy or deploy_details tables")
            
            user_email, etf_names_json, etf_count = result
            conn.close()
            
            # Parse ETF names from JSON
            try:
                etf_names = json.loads(etf_names_json) if etf_names_json else []
            except:
                etf_names = []
            
            if not etf_names or len(etf_names) == 0:
                raise ValueError(f"No ETF symbols found for run_id: {run_id}")
            
            self.logger.info(f"Found {len(etf_names)} ETF symbols from deploy_details for run_id {run_id}: {etf_names}")
            
            return {
                'user_id': user_email,
                'etf_symbols': etf_names,
                'etf_names': etf_names,
                'etf_count': etf_count
            }
            
        except Exception as e:
            self.logger.error(f"Error getting symbols from deployment: {str(e)}")
            raise
    
    def get_all_running_strategies(self) -> List[Dict[str, Any]]:
        """
        Get all running ETF strategies from etf_saved_strategy table
        
        Returns:
            List of dictionaries with run_id, user_id, and strategy info
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query all active strategies with run_id
            # Include 'running' and 'deploy' status (deploy is the default active status)
            # Include NULL and empty string status as active (for backward compatibility)
            # Exclude only explicitly 'stopped' strategies
            cursor.execute('''
                SELECT run_id, user_id, strategy_name, strategy_type, tickers, etf_names, etf_count
                FROM etf_saved_strategy 
                WHERE run_id IS NOT NULL 
                AND run_id != ''
                AND (status IS NULL OR status = '' OR status = 'running' OR status = 'deploy')
                ORDER BY created_timestamp DESC
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            strategies = []
            for row in rows:
                run_id, user_id, strategy_name, strategy_type, tickers_json, etf_names_json, etf_count = row
                
                # Parse tickers
                try:
                    tickers = json.loads(tickers_json) if tickers_json else []
                except:
                    tickers = []
                
                # Parse etf_names if tickers is empty
                if not tickers:
                    try:
                        tickers = json.loads(etf_names_json) if etf_names_json else []
                    except:
                        tickers = []
                
                if tickers:  # Only include if we have tickers
                    strategies.append({
                        'run_id': run_id,
                        'user_id': user_id,
                        'strategy_name': strategy_name,
                        'strategy_type': strategy_type,
                        'tickers': tickers,
                        'etf_count': etf_count if etf_count else len(tickers)
                    })
            
            self.logger.info(f"Found {len(strategies)} running strategies with run_id and tickers")
            return strategies
            
        except Exception as e:
            self.logger.error(f"Error getting running strategies: {str(e)}")
            return []
    
    def get_etf_data_for_signals(self, run_id: str = None, days_back: int = 400) -> Dict[str, pd.DataFrame]:
        """
        Get ETF data for signal generation with historical buffer
        
        Args:
            run_id: Deployment run ID to get specific ETFs
            days_back: Number of days to look back for historical data
            
        Returns:
            Dictionary of ETF dataframes
        """
        try:
            # Get ETF symbols from deploy_details table if run_id provided
            if run_id:
                deployment_info = self.get_symbols_from_deployment(run_id)
                etf_symbols = deployment_info['etf_symbols']
                self.user_id = deployment_info['user_id']  # Store user_id for later use
                self.logger.info(f"Using {len(etf_symbols)} ETFs from deployment: {etf_symbols}")
            else:
                # Fallback to default 12 ETFs if no run_id
                etf_symbols = [
                    'BANKBEES', 'GOLDBEES', 'HNGSNGBEES', 'INFRABEES', 
                    'ITBEES', 'JUNIORBEES', 'LIQUIDBEES', 'MON100', 'NIFTYBEES', 
                    'PHARMABEES', 'PSUBNKBEES', 'SENSEX1.BO'
                ]
                self.user_id = None
                self.logger.warning("No run_id provided, using default 12 ETFs")
            
            # Symbol mapping (clean symbols without .NS/.BO suffix)
            symbol_mapping = {}
            for symbol in etf_symbols:
                clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
                symbol_mapping[clean_symbol] = clean_symbol
            
            # Verify which ETFs exist in PostgreSQL database
            from market_data_db_connection import get_session as get_market_data_session
            from sqlalchemy import text
            
            session = None
            try:
                session = get_market_data_session()
                query = text("SELECT DISTINCT symbol FROM etf_data ORDER BY symbol")
                result = session.execute(query)
                available_symbols = [row[0] for row in result.fetchall()]
                
                # Filter to only include ETFs that exist in database
                symbols = [s.replace('.NS', '').replace('.BO', '') for s in etf_symbols if s.replace('.NS', '').replace('.BO', '') in available_symbols]
                
                # Store the symbol mapping for later use
                self.symbol_mapping = symbol_mapping
                self.etf_symbols_for_signals = symbols
                
                if not symbols:
                    error_msg = f"No symbols found in database for run_id: {run_id}. Available symbols: {available_symbols[:10]}"
                    raise ValueError(error_msg)
                
                self.logger.info(f"Found {len(symbols)} ETF symbols in database: {symbols}")
                
                # Calculate the most recent Friday
                today = datetime.now()
                days_since_friday = (today.weekday() - 4) % 7  # 4 is Friday (0 is Monday)
                last_friday = today - timedelta(days=days_since_friday)
                test_date = last_friday
                
                end_date = test_date.strftime('%Y-%m-%d')
                start_date = (test_date - timedelta(days=days_back)).strftime('%Y-%m-%d')
                
                self.logger.info(f"Using last Friday's date: {end_date} for signal generation")
                
                self.logger.info(f"Using test date: {end_date} (Friday) for signal generation")
                
                # Load data directly from PostgreSQL database
                etf_data = self.load_etf_data_from_database(session, symbols, start_date, end_date)
            finally:
                if session:
                    session.close()
            
            self.logger.info(f"Loaded data for {len(etf_data)} ETFs from {start_date} to {end_date}")
            
            return etf_data
            
        except Exception as e:
            self.logger.error(f"Error getting ETF data: {e}")
            raise
    
    def load_etf_data_from_database(self, session, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Load ETF data directly from PostgreSQL database
        
        Args:
            session: SQLAlchemy database session
            symbols: List of ETF symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary with keys 'open', 'high', 'low', 'close', 'volume' containing DataFrames
        """
        try:
            from sqlalchemy import text
            import pandas as pd
            
            # Initialize data containers
            data_dict = {
                'open': {},
                'high': {},
                'low': {},
                'close': {},
                'volume': {}
            }
            
            for symbol in symbols:
                try:
                    # Query data for this symbol from PostgreSQL
                    query = text("""
                        SELECT date, open, high, low, close, volume, adjusted_close
                        FROM etf_data
                        WHERE symbol = :symbol AND date >= :start_date::date AND date <= :end_date::date
                        ORDER BY date
                    """)
                    
                    df = pd.read_sql(query, session.bind, params={"symbol": symbol, "start_date": start_date, "end_date": end_date})
                    
                    if df.empty:
                        self.logger.warning(f"No data found for {symbol}")
                        continue
                    
                    # Convert date column to datetime
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    
                    # Convert to pandas series
                    data_dict['open'][symbol] = df['open']
                    data_dict['high'][symbol] = df['high']
                    data_dict['low'][symbol] = df['low']
                    data_dict['close'][symbol] = df['close']
                    data_dict['volume'][symbol] = df['volume']
                    
                except Exception as e:
                    self.logger.error(f"Error loading data for {symbol}: {e}")
                    continue
            
            # Convert to DataFrames
            result = {}
            for key, symbol_data in data_dict.items():
                if symbol_data:
                    result[key] = pd.DataFrame(symbol_data)
                else:
                    result[key] = pd.DataFrame()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error loading ETF data from database: {e}")
            raise
    
    def calculate_etf_metrics(self, etf_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate ETF metrics for signal generation using backtester core logic
        
        Args:
            etf_data: Dictionary with keys 'open', 'high', 'low', 'close', 'volume' 
                     where each value is a DataFrame with symbols as columns
            
        Returns:
            DataFrame with calculated metrics
        """
        try:
            metrics_list = []
            
            # Extract the close prices DataFrame to get available symbols
            if 'close' not in etf_data:
                raise ValueError("No close price data available")
            
            close_df = etf_data['close']
            available_symbols = close_df.columns.tolist()
            
            self.logger.info(f"Processing {len(available_symbols)} symbols: {available_symbols[:5]}...")
            
            for symbol in available_symbols:
                try:
                    # Get data for this symbol
                    close_prices = close_df[symbol].dropna()
                    high_prices = etf_data.get('high', pd.DataFrame())[symbol].dropna() if 'high' in etf_data else close_prices
                    low_prices = etf_data.get('low', pd.DataFrame())[symbol].dropna() if 'low' in etf_data else close_prices
                    volume_data = etf_data.get('volume', pd.DataFrame())[symbol].dropna() if 'volume' in etf_data else None
                    
                    if len(close_prices) < 52:  # Need at least 52 weeks of data
                        self.logger.warning(f"Insufficient data for {symbol}: {len(close_prices)} rows")
                        continue
                    
                    # Get latest date and price
                    latest_date = close_prices.index[-1]
                    current_price = close_prices.iloc[-1]
                    
                    # Calculate 52-week high and low from CLOSE prices (use 252 trading days ≈ 52 weeks)
                    high_52w = close_prices.rolling(window=252).max().iloc[-1]
                    low_52w = close_prices.rolling(window=252).min().iloc[-1]
                    
                    # Calculate relative strength (RS) score
                    rs_score = (current_price - low_52w) / (high_52w - low_52w) if high_52w != low_52w else 0
                    
                    # Calculate momentum indicators
                    price_20d_ago = close_prices.iloc[-20] if len(close_prices) >= 20 else close_prices.iloc[0]
                    momentum_20d = (current_price - price_20d_ago) / price_20d_ago if price_20d_ago != 0 else 0
                    
                    # Volume analysis
                    if volume_data is not None and len(volume_data) >= 20:
                        avg_volume_20d = volume_data.rolling(window=20).mean().iloc[-1]
                        current_volume = volume_data.iloc[-1]
                        volume_ratio = current_volume / avg_volume_20d if avg_volume_20d != 0 else 1
                    else:
                        volume_ratio = 1.0  # Default if no volume data
                    
                    # Determine phase (accumulation/churning)
                    if rs_score > 0.8 and momentum_20d > 0.05:
                        phase = "accumulation"
                    elif rs_score < 0.2 and momentum_20d < -0.05:
                        phase = "distribution"
                    else:
                        phase = "churning"
                    
                    metrics = {
                        'symbol': symbol,
                        'date': latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date),
                        'price': current_price,
                        'high_52w': high_52w,
                        'low_52w': low_52w,
                        'rs_score': rs_score,
                        'momentum_20d': momentum_20d,
                        'volume_ratio': volume_ratio,
                        'phase': phase,
                        'distance_from_high': (high_52w - current_price) / high_52w if high_52w != 0 else 0,
                        'distance_from_low': (current_price - low_52w) / low_52w if low_52w != 0 else 0
                    }
                    
                    metrics_list.append(metrics)
                    
                except Exception as e:
                    self.logger.error(f"Error calculating metrics for {symbol}: {e}")
                    continue
            
            if not metrics_list:
                raise ValueError("No valid metrics calculated for any ETF")
            
            metrics_df = pd.DataFrame(metrics_list)
            self.logger.info(f"Calculated metrics for {len(metrics_df)} ETFs")
            
            return metrics_df
            
        except Exception as e:
            self.logger.error(f"Error calculating ETF metrics: {e}")
            raise
    
    def print_detailed_momentum_calculation(self, metrics_df: pd.DataFrame):
        """
        Print detailed momentum calculation in the requested format
        """
        try:
            # Calculate the most recent Friday
            today = datetime.now()
            days_since_friday = (today.weekday() - 4) % 7
            last_friday = today - timedelta(days=days_since_friday)
            signal_date = last_friday.strftime('%Y-%m-%d')
            
            print(f"Momentum Calculation for {signal_date}:")
            
            # Filter ETFs with valid 52-week data
            valid_etfs = metrics_df[
                (metrics_df['high_52w'] > 0) & 
                (metrics_df['low_52w'] > 0) & 
                (metrics_df['price'] > 0)
            ].copy()
            
            if len(valid_etfs) == 0:
                print("   Available historical data: 0 ETFs")
                return
            
            print(f"   Available historical data: {len(valid_etfs)} ETFs")
            print(f"   Required for 52-week calc: 252 days")
            print(f"   ✅ Using exactly 252 trading days for momentum")
            
            # Calculate distances from 52-week high/low
            valid_etfs['distance_from_low_pct'] = ((valid_etfs['price'] - valid_etfs['low_52w']) / valid_etfs['low_52w']) * 100
            valid_etfs['distance_from_high_pct'] = ((valid_etfs['high_52w'] - valid_etfs['price']) / valid_etfs['high_52w']) * 100
            
            print(f"   📈 52-week highs calculated for {len(valid_etfs)} ETFs")
            print(f"   📉 52-week lows calculated for {len(valid_etfs)} ETFs")
            print(f"   💰 Current prices available for {len(valid_etfs)} ETFs")
            
            # Display detailed metrics for each ETF
            for _, row in valid_etfs.iterrows():
                print(f"   ✅ {row['symbol']}: Max_Close=₹{row['high_52w']:.2f}, Min_Close=₹{row['low_52w']:.2f}, Current=₹{row['price']:.2f}")
                print(f"       Distance from Min Close: {row['distance_from_low_pct']:.2f}%, Distance from Max Close: {row['distance_from_high_pct']:.2f}%")
            
            print(f"   🎯 Final momentum result: {len(valid_etfs)} valid ETFs with proper 52-week data")
            
            # Rank ETFs by distance from 52-week low (closest first)
            ranked_from_low = valid_etfs.sort_values('distance_from_low_pct', ascending=True)
            
            print(f"   📊 ETF Rankings from 52-week MIN CLOSE (closest to min close first):")
            for i, (_, row) in enumerate(ranked_from_low.iterrows(), 1):
                print(f"      {i}. {row['symbol']}: {row['distance_from_low_pct']:.2f}% from min close")
            
            # Rank ETFs by distance from 52-week max close (closest first)
            ranked_from_high = valid_etfs.sort_values('distance_from_high_pct', ascending=True)
            
            print(f"   📈 ETF Rankings from 52-week MAX CLOSE (closest to max close first):")
            for i, (_, row) in enumerate(ranked_from_high.iterrows(), 1):
                print(f"      {i}. {row['symbol']}: {row['distance_from_high_pct']:.2f}% from max close")
            
        except Exception as e:
            self.logger.error(f"Error in detailed momentum calculation: {e}")
    
    def print_signal_details(self, signals: List[Dict[str, Any]]):
        """
        Print signal details in the requested format
        """
        try:
            if not signals:
                print("\n[WARNING] No signals generated for this period")
                return
            
            buy_signals = [s for s in signals if s['side'] == 'BUY']
            sell_signals = [s for s in signals if s['side'] == 'SELL']
            
            print(f"\n📊 Final Results:")
            
            if buy_signals:
                print(f"\nBUY -> Buy ETF details according to the calculation:")
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
                    print(f"      Distance from 52-week low: {distance_from_low:.2f}%")
                    print(f"      Distance from 52-week high: {distance_from_high:.2f}%")
                    print(f"      Rank from low: #{rank}")
                    print(f"      Reason: {reason}")
            
            if sell_signals:
                print(f"\nSELL -> Sell ETF details according to the calculation:")
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
                    print(f"      Distance from 52-week low: {distance_from_low:.2f}%")
                    print(f"      Distance from 52-week high: {distance_from_high:.2f}%")
                    print(f"      Rank from high: #{rank}")
                    print(f"      Reason: {reason}")
            
        except Exception as e:
            self.logger.error(f"Error in signal details: {e}")
    
    def generate_signals(self, metrics_df: pd.DataFrame, strategy_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Generate BUY/SELL signals based on simple ranking logic:
        BUY: Closest to 52-week low
        SELL: Closest to 52-week high
        
        Args:
            metrics_df: DataFrame with ETF metrics
            strategy_config: Strategy configuration parameters (not used in simple logic)
            
        Returns:
            List of signal dictionaries
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
                # Map database symbol to display symbol
                display_symbol = self.symbol_mapping.get(row['symbol'], row['symbol'])
                signal = {
                    'symbol': display_symbol,
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
                # Map database symbol to display symbol
                display_symbol = self.symbol_mapping.get(row['symbol'], row['symbol'])
                signal = {
                    'symbol': display_symbol,
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
            
            self.logger.info(f"Generated {len(signals)} signals: {len([s for s in signals if s['side'] == 'BUY'])} BUY, {len([s for s in signals if s['side'] == 'SELL'])} SELL")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            raise
    
    def save_signals_to_database(self, run_id: str, signals: List[Dict[str, Any]], strategy_version: str = "SurfTrend-v1.2") -> bool:
        """
        Save generated signals to database with updated schema (now using PostgreSQL)
        
        Args:
            run_id: Unique run identifier (format: run_etf_strategy_2025-10-15_1760544030144)
            signals: List of signal dictionaries
            strategy_version: Strategy version string
            
        Returns:
            Boolean indicating success
        """
        try:
            from Databases.strategy_db_helpers import save_live_signal, save_live_run
            from Databases.neon_db_connection import get_session
            from Databases.strategy_models import LiveRun, LiveSignal
            from sqlalchemy.dialects.postgresql import insert
            
            # Calculate the most recent Friday (same as signal generation logic)
            today = datetime.now()
            days_since_friday = (today.weekday() - 4) % 7  # 4 is Friday
            last_friday = today - timedelta(days=days_since_friday)
            signal_date = last_friday.strftime('%Y-%m-%d')
            
            # Create run record
            summary = {
                'buy_count': len([s for s in signals if s['side'] == 'BUY']),
                'sell_count': len([s for s in signals if s['side'] == 'SELL']),
                'total_signals': len(signals)
            }
            
            session = get_session()
            try:
                # Upsert run record
                run_data = {
                    'run_id': run_id,
                    'strategy_version': strategy_version,
                    'signal_date': signal_date,
                    'status': 'generated',
                    'summary_json': json.dumps(summary)
                }
                
                stmt = insert(LiveRun).values(**run_data)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['run_id'],
                    set_=dict(
                        strategy_version=stmt.excluded.strategy_version,
                        signal_date=stmt.excluded.signal_date,
                        status=stmt.excluded.status,
                        summary_json=stmt.excluded.summary_json
                    )
                )
                session.execute(stmt)
                
                # Insert/update signals
                user_id = getattr(self, 'user_id', None)
                for signal in signals:
                    signal_data = {
                        'run_id': run_id,
                        'user_id': user_id,
                        'strategy_version': strategy_version,
                        'signal_date': signal_date,
                        'signal_symbol': signal['symbol'],
                        'etf_name': signal.get('etf_name', signal['symbol']),
                        'side': signal['side'],
                        'score': signal['score'],
                        'reason': signal['reason'],
                        'payload_json': json.dumps(signal['payload'])
                    }
                    
                    stmt = insert(LiveSignal).values(**signal_data)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=['run_id', 'signal_symbol', 'side'],
                        set_=dict(
                            user_id=stmt.excluded.user_id,
                            strategy_version=stmt.excluded.strategy_version,
                            signal_date=stmt.excluded.signal_date,
                            etf_name=stmt.excluded.etf_name,
                            score=stmt.excluded.score,
                            reason=stmt.excluded.reason,
                            payload_json=stmt.excluded.payload_json
                        )
                    )
                    session.execute(stmt)
                
                session.commit()
                self.logger.info(f"Saved {len(signals)} signals to PostgreSQL database with run_id: {run_id}")
                return True
                
            except Exception as e:
                session.rollback()
                raise
            finally:
                session.close()
            
        except Exception as e:
            self.logger.error(f"Error saving signals to database: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def run_weekly_signal_generation(self, run_id: str = None, strategy_type: str = "SurfTrend", strategy_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main method to run weekly signal generation
        
        Args:
            run_id: Deployment run ID (format: run_etf_strategy_2025-10-15_1760544030144)
            strategy_type: Type of strategy (e.g., "SurfTrend")
            strategy_config: Strategy configuration parameters
            
        Returns:
            Dictionary with generation results
        """
        try:
            start_time = datetime.now()
            self.logger.info("Starting weekly signal generation...")
            
            # Use provided run_id or generate one
            if not run_id:
                run_id = self.generate_run_id(strategy_type)
                self.logger.warning(f"No run_id provided, generated: {run_id}")
            
            self.logger.info(f"Using run_id: {run_id}")
            
            # Get ETF data (will fetch symbols from deploy_details if run_id provided)
            try:
                etf_data = self.get_etf_data_for_signals(run_id=run_id)
            except ValueError as e:
                # No symbols found for this run_id
                self.logger.error(f"No symbols found for run_id {run_id}: {str(e)}")
                return {
                    'success': False,
                    'run_id': run_id,
                    'signals_count': 0,
                    'error': f'No symbols found for run_id: {run_id}',
                    'message': 'No ETF symbols associated with this deployment'
                }
            
            # Calculate metrics
            metrics_df = self.calculate_etf_metrics(etf_data)
            
            # Print detailed momentum calculation
            self.print_detailed_momentum_calculation(metrics_df)
            
            # Generate signals
            signals = self.generate_signals(metrics_df, strategy_config)
            
            # Print signal details only (no execution simulation)
            self.print_signal_details(signals)
            
            if not signals:
                self.logger.warning("No signals generated")
                return {
                    'success': False,
                    'run_id': run_id,
                    'signals_count': 0,
                    'message': 'No signals generated'
                }
            
            # Save to database
            success = self.save_signals_to_database(run_id, signals)
            
            if not success:
                raise Exception("Failed to save signals to database")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = {
                'success': True,
                'run_id': run_id,
                'signals_count': len(signals),
                'buy_count': len([s for s in signals if s['side'] == 'BUY']),
                'sell_count': len([s for s in signals if s['side'] == 'SELL']),
                'duration_seconds': duration,
                'signals': signals
            }
            
            self.logger.info(f"Signal generation completed successfully in {duration:.2f}s")
            self.logger.info(f"Generated {result['buy_count']} BUY and {result['sell_count']} SELL signals")
            
            # Log to monitoring system
            log_signal_generation(
                run_id=run_id,
                signals_count=result['signals_count'],
                buy_count=result['buy_count'],
                sell_count=result['sell_count'],
                duration=duration
            )
            
            # Log performance metrics
            log_performance(
                operation='signal_generation',
                duration=duration,
                success=True,
                metadata={'strategy_type': strategy_type, 'run_id': run_id}
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in weekly signal generation: {e}")
            
            # Log performance metrics for failed operation
            if 'start_time' in locals():
                duration = (datetime.now() - start_time).total_seconds()
                log_performance(
                    operation='signal_generation',
                    duration=duration,
                    success=False,
                    error_message=str(e),
                    metadata={'strategy_type': strategy_type, 'run_id': run_id if 'run_id' in locals() else None}
                )
            
            return {
                'success': False,
                'error': str(e),
                'run_id': run_id if 'run_id' in locals() else None
            }
    
    def get_recent_signals(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get recent signals from database
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of recent signals
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            cursor.execute('''
                SELECT ls.*, lr.status, lr.summary_json
                FROM live_signals ls
                JOIN live_runs lr ON ls.run_id = lr.run_id
                WHERE ls.signal_date >= ?
                ORDER BY ls.signal_date DESC, ls.created_at DESC
            ''', (cutoff_date,))
            
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            signals = []
            for row in rows:
                signal_dict = dict(zip(columns, row))
                if signal_dict['payload_json']:
                    signal_dict['payload'] = json.loads(signal_dict['payload_json'])
                if signal_dict['summary_json']:
                    signal_dict['summary'] = json.loads(signal_dict['summary_json'])
                signals.append(signal_dict)
            
            conn.close()
            return signals
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error getting recent signals: {e}")
            return []
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'backtester'):
            self.backtester.cleanup()


def main():
    """Main function for testing signal generation"""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        generator = LiveSignalGenerator()
        
        # Test signal generation
        result = generator.run_weekly_signal_generation()
        
        if result['success']:
            print(f"[SUCCESS] Signal generation successful!")
            print(f"Run ID: {result['run_id']}")
            print(f"Signals: {result['signals_count']} ({result['buy_count']} BUY, {result['sell_count']} SELL)")
        else:
            print(f"[ERROR] Signal generation failed: {result.get('error', 'Unknown error')}")
        
        generator.cleanup()
        
    except Exception as e:
        print(f"[ERROR] Error in main: {e}")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sqlalchemy.orm import Session
try:
    from Strategies.RS_Stocks.database import StockData, IndexData, SavedRSStrategy as StrategyConfig, BacktestResult, TradeLog, PortfolioSnapshot
except ImportError:
    from database import StockData, IndexData, StrategyConfig, BacktestResult, TradeLog, PortfolioSnapshot
import json
from dataclasses import dataclass
import math
from enum import Enum

# Import benchmark calculator
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark_calculator import BenchmarkCalculator
from Strategies.utilities.logging_config import StrategyLogger

# Import RS configuration
try:
    from ..RS.rs_config_loader import get_rs_config
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'RS'))
    from rs_config_loader import get_rs_config

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

@dataclass
class DynamicParams:
    """Dynamic parameters that adjust based on market conditions"""
    rs_threshold: float
    position_size_pct: float
    stop_loss_pct: float
    max_positions: int
    rebalance_frequency_days: int
    volatility_threshold: float

def safe_float(value, default=0.0):
    """Convert value to float, handling infinity and NaN values safely for JSON serialization."""
    try:
        if value is None:
            return default
        float_value = float(value)
        if math.isinf(float_value) or math.isnan(float_value):
            return default
        return float_value
    except (ValueError, TypeError):
        return default

def safe_divide(numerator, denominator, default=0.0):
    """Safely divide two numbers, handling division by zero and infinity."""
    try:
        if denominator == 0 or math.isnan(denominator) or math.isinf(denominator):
            return default
        result = numerator / denominator
        if math.isinf(result) or math.isnan(result):
            return default
        return result
    except (ValueError, TypeError, ZeroDivisionError):
        return default

def safe_power(base, exponent, default=0.0):
    """Safely calculate power, handling edge cases."""
    try:
        if math.isnan(base) or math.isinf(base) or math.isnan(exponent) or math.isinf(exponent):
            return default
        result = math.pow(base, exponent)
        if math.isinf(result) or math.isnan(result):
            return default
        return result
    except (ValueError, TypeError, OverflowError):
        return default

def safe_sqrt(value, default=0.0):
    """Safely calculate square root, handling negative values."""
    try:
        if value < 0 or math.isnan(value) or math.isinf(value):
            return default
        result = math.sqrt(value)
        if math.isinf(result) or math.isnan(result):
            return default
        return result
    except (ValueError, TypeError):
        return default

@dataclass
class Trade:
    date: datetime
    symbol: str
    action: str  # BUY, SELL
    quantity: int
    price: float
    amount: float
    reason: str
    rs_score: Optional[float] = None
    rs_rank: Optional[int] = None
    
    # Detailed transaction cost breakdown
    transaction_value: Optional[float] = None
    brokerage: Optional[float] = None
    stt: Optional[float] = None
    stamp_duty: Optional[float] = None
    exchange_charges: Optional[float] = None
    sebi_charges: Optional[float] = None
    gst: Optional[float] = None
    total_costs: Optional[float] = None
    net_amount: Optional[float] = None
    
    # NAV and Capital Gains tracking
    portfolio_nav: Optional[float] = None  # Portfolio NAV at trade time
    buy_price: Optional[float] = None  # Original buy price (for SELL trades)
    capital_gain: Optional[float] = None  # Total gain/loss in ₹
    capital_gain_pct: Optional[float] = None  # Gain/loss percentage
    holding_period_days: Optional[int] = None  # Days held (for SELL trades)
    
    # Tax calculation (20% STCG for short-term)
    capital_gains_tax: Optional[float] = None  # Tax amount (20% of gain if positive)
    net_profit_after_tax: Optional[float] = None  # Profit after deducting tax

@dataclass
class Position:
    symbol: str
    quantity: int
    buy_price: float
    buy_date: datetime
    current_price: float
    unrealized_pnl: float
    stop_loss_price: Optional[float] = None

class RSStrategyBacktester:
    def __init__(self, db: Session, config_id: int = None, config_dict: Dict = None):
        # Initialize centralized logger
        self.logger = StrategyLogger('RS_Stocks')

        self.db = db
        
        # Support both config_id (existing) and config_dict (new custom config approach)
        if config_dict:
            # Use provided config dict directly
            self.main_index = config_dict.get('main_index', '^NSEI')
            self.lookback_weeks = config_dict.get('lookback_weeks', 5)
            self.lookback_months = config_dict.get('lookback_months', 20)
            self.lookback_quarters = config_dict.get('lookback_quarters', 60)
            self.max_positions = config_dict.get('max_positions', 20)
            
            # Calculate position_size_pct automatically if not provided
            # Formula: (1 - buffer_capital_pct) / max_positions
            if 'position_size_pct' in config_dict and config_dict.get('position_size_pct') is not None:
                # Use provided value (for backward compatibility)
                self.position_size_pct = config_dict.get('position_size_pct', 5.0) / 100
            else:
                # Auto-calculate based on max_positions and buffer_capital_pct
                buffer_pct = config_dict.get('buffer_capital_pct', 10.0) / 100
                max_pos = config_dict.get('max_positions', 20)
                if max_pos > 0:
                    available_capital_pct = 1 - buffer_pct
                    self.position_size_pct = available_capital_pct / max_pos
                else:
                    self.position_size_pct = 0.05  # Default 5% if max_positions is invalid
            
            self.buffer_capital_pct = config_dict.get('buffer_capital_pct', 10.0) / 100
            self.total_capital = config_dict.get('total_capital', 1000000.0)
            
            # Load RS configuration for stop loss behavior
            rs_config_obj = get_rs_config() # Use object for default
            sl_pct = config_dict.get('stop_loss_pct')
            if sl_pct is None:
                sl_pct = rs_config_obj.get_stop_loss_pct()
            self.stop_loss_pct = sl_pct / 100
            self.capital_reset_threshold_pct = config_dict.get('capital_reset_threshold_pct', 25.0) / 100
            # Removed max_holding_period - stocks held until RS ranking drops
            self.transaction_cost_pct = config_dict.get('transaction_cost_pct', 0.1) / 100
            # Removed min_price - no price filtering, all stocks eligible
            self.min_turnover = config_dict.get('min_turnover', 1000000.0)  # Default: ₹10L
            self.stock_universe = config_dict.get('stock_universe', 'NIFTY500')
            self.custom_stocks = config_dict.get('custom_stocks', None)  # Custom stock selection
            self.market = config_dict.get('market', 'INDIA').upper()
            self.config = None  # No database config object for custom configs
        elif config_id:
            # Load from database (existing behavior)
            self.config = db.query(StrategyConfig).filter(StrategyConfig.id == config_id).first()
            if not self.config:
                raise ValueError(f"Configuration with ID {config_id} not found")
            
            # Strategy parameters
            self.main_index = self.config.main_index
            self.lookback_weeks = self.config.lookback_weeks
            self.lookback_months = self.config.lookback_months
            self.lookback_quarters = self.config.lookback_quarters
            self.max_positions = self.config.max_positions
            self.position_size_pct = self.config.position_size_pct / 100
            self.buffer_capital_pct = self.config.buffer_capital_pct / 100
            self.total_capital = self.config.total_capital
            self.stop_loss_pct = self.config.stop_loss_pct / 100
            self.capital_reset_threshold_pct = self.config.capital_reset_threshold_pct / 100
            # Removed max_holding_period - stocks held until RS ranking drops
            self.transaction_cost_pct = self.config.transaction_cost_pct / 100
            # Removed min_price - no price filtering, all stocks eligible
            self.min_turnover = self.config.min_turnover
            self.stock_universe = getattr(self.config, 'stock_universe', 'NIFTY500')
            self.market = getattr(self.config, 'market', 'INDIA').upper()
        else:
            raise ValueError("Either config_id or config_dict must be provided")
        
        # Load RS configuration for stop loss behavior
        rs_config = get_rs_config()
        
        # Daily stop loss check setting - can be overridden by config_dict
        if config_dict and 'daily_stop_loss_check' in config_dict:
            self.daily_stop_loss_check = config_dict.get('daily_stop_loss_check')
        else:
            # Load from centralized RS config
            self.daily_stop_loss_check = rs_config.get_daily_stop_loss_check()
        
        self.logger.info(f"Stop Loss Mode: {'Daily Check' if self.daily_stop_loss_check else 'Weekly Check (Signal Day Only)'}")
        self.log_stop_loss_mode()
        
        # Calculated values for dynamic buffer system
        self.buffer_capital = self.total_capital * self.buffer_capital_pct  # Initial buffer amount
        # Full position size for maximum capital utilization
        self.per_trade_allocation = self.total_capital * self.position_size_pct  # Full position size
        
        # Backtest state
        self.positions: Dict[str, Position] = {}
        self.cash_balance = self.total_capital - self.buffer_capital  # Available cash for trading
        self.trades: List[Trade] = []
        self.portfolio_snapshots: List[Dict] = []
        self.last_trading_days: set = set()  # Pre-calculated last trading days of each week
        self.index_data: pd.DataFrame = pd.DataFrame()  # Store index data for beta calculation
        
        # Enhanced strategy steate
        self.current_regime = MarketRegime.BULL
        self.regime_history = []
        self.dynamic_params = self.get_default_dynamic_params()
        self.last_rebalance_date = None
        self.volatility_window = 20  # Days for volatility calculation
        
        # Capital reset state
        self.peak_portfolio_value = self.total_capital
        self.is_capital_reset_active = False
        self.capital_reset_start_date = None
        
        # Monday execution state
        self.pending_entries = None
        self.pending_exits = None
        self.signal_date = None
        
        # Weekly stop loss accumulation (for weekly mode)
        self.weekly_stop_loss_exits: List[str] = []
    
    @classmethod
    def from_config_dict(cls, db: Session, config_dict: Dict):
        """Create RSStrategyBacktester instance from config dictionary"""
        return cls(db=db, config_dict=config_dict)
        
    def get_default_dynamic_params(self) -> DynamicParams:
        """Get optimized dynamic parameters for better performance"""
        return DynamicParams(
            rs_threshold=0.1,  # Lower threshold for more trading opportunities
            position_size_pct=self.position_size_pct,
            stop_loss_pct=self.stop_loss_pct,
            max_positions=self.max_positions,  # Use frontend configuration
            rebalance_frequency_days=7,  # Weekly rebalancing (could be frontend configurable)
            volatility_threshold=0.5  # Higher threshold for more opportunities
        )
        
    def load_stock_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load stock data for Nifty 500 constituents with optimization for long periods"""
        # Ensure dates are timezone-naive
        if start_date.tzinfo:
            start_date = start_date.replace(tzinfo=None)
        if end_date.tzinfo:
            end_date = end_date.replace(tzinfo=None)
            
        # Calculate buffer start date - 400 calendar days BEFORE backtest start
        # This provides ~252 trading days for 60-day quarter calculations
        # buffer_start_date = (pd.to_datetime(start_date) - timedelta(days=400)).strftime('%Y-%m-%d')
        data_start_date = start_date - timedelta(days=100)
        
        self.logger.progress(f"Loading stock data from {data_start_date} to {end_date} (backtest period: {start_date} to {end_date})")
        
        # Calculate period length
        period_days = (end_date - start_date).days
        total_data_days = (end_date - data_start_date).days
        self.logger.info(f"Backtest period: {period_days} days, Total data period: {total_data_days} days")
        
        # Get custom stock universe
        stock_symbols = self.get_custom_stock_universe()
        if not stock_symbols:
            # Fallback if universe is empty - maybe log a warning?
            self.logger.info("WARNING: Custom stock universe is empty. Defaulting to empty dataframe.")
            return pd.DataFrame()
            
        self.logger.info(f"Restricting data load to {len(stock_symbols)} configured stocks")
        
        # Use SQLAlchemy text() for PostgreSQL-compatible queries
        from sqlalchemy import text
        
        # Determine table name based on market
        table_name = "us_stock_market" if self.market == 'US' else "stock_market"
        adj_close_col = "adj_close" # Both tables use adj_close
        
        # Query using SQLAlchemy text() with PostgreSQL parameter binding and symbol filtering
        # Using ANY(:symbols) for efficient array filtering in Postgres
        query = text(f"""
            SELECT symbol, date, {adj_close_col} as adjusted_close
            FROM {table_name} 
            WHERE symbol = ANY(:symbols) 
            AND date >= :start_date AND date <= :end_date
            ORDER BY symbol, date
        """)
        
        # Execute query and convert to DataFrame
        result = self.db.execute(query, {
            "symbols": stock_symbols,
            "start_date": data_start_date,
            "end_date": end_date
        })
        
        # Convert to DataFrame
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        if df.empty:
            self.logger.info("No data found for the selected stocks in the given date range.")
            return df

        # Set date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Set index
        df = df.set_index(['symbol', 'date'])
        
        self.logger.performance(f"Raw stock data query returned: {len(df)} records")
        
        # Convert timezone-aware dates to timezone-naive
        if df.index.get_level_values('date').tz is not None:
            df.index = df.index.set_levels(df.index.levels[1].tz_convert(None), level='date')
        
        self.logger.info(f"Final stock data: {len(df)} records, {len(df.index.get_level_values('symbol').unique())} symbols")
        if not df.empty:
            self.logger.info(f"Date range: {df.index.get_level_values('date').min()} to {df.index.get_level_values('date').max()}")
        
        return df
    
    def calculate_common_date_range(self, selected_stocks: List[str]) -> Tuple[Optional[str], Optional[str], float]:
        """Calculate common date range for selected stocks by querying the database"""
        if not selected_stocks:
            return None, None, 0.0
        
        try:
            # Query database for min/max dates for all stocks in one query (more efficient)
            stock_ranges = {}
            
            # Use SQLAlchemy text() for PostgreSQL-compatible queries
            from sqlalchemy import text
            
            # Build placeholders for IN clause
            placeholders = ','.join([f':symbol_{i}' for i in range(len(selected_stocks))])
            
            # Determine table name based on market
            table_name = "us_stock_market" if self.market == 'US' else "stock_market"
            
            # Single query to get min/max dates for all stocks
            query = text(f"""
                SELECT 
                    symbol,
                    MIN(date) as min_date, 
                    MAX(date) as max_date
                FROM {table_name}
                WHERE symbol IN ({placeholders})
                GROUP BY symbol
            """)
            
            # Build parameters dict
            params = {f'symbol_{i}': stock for i, stock in enumerate(selected_stocks)}
            
            # Execute query
            results = self.db.execute(query, params).fetchall()
            
            # Process results
            for row in results:
                if row.min_date is not None and row.max_date is not None:
                    min_date = pd.to_datetime(row.min_date)
                    max_date = pd.to_datetime(row.max_date)
                    stock_ranges[row.symbol] = {
                        'start_date': min_date,
                        'end_date': max_date
                    }
            
            if not stock_ranges:
                self.logger.info(f"No data found for selected stocks: {selected_stocks}")
                return None, None, 0.0
            
            # --- IMPROVED LOGIC: Filter out "bad" stocks before intersection ---
            
            # 1. Determine Global Latest Date (reference point)
            all_end_dates = [data['end_date'] for data in stock_ranges.values()]
            if not all_end_dates:
                 return None, None, 0.0
            global_latest_date = max(all_end_dates)
            
            valid_stocks = []
            stale_stocks = []
            short_history_stocks = []
            
            MIN_HISTORY_DAYS = 365  # Stocks must have at least 1 year of data
            STALE_THRESHOLD_DAYS = 10 # Data shouldn't be older than 10 days from latest available
            
            for stock, data in stock_ranges.items():
                start_date = data['start_date']
                end_date = data['end_date']
                
                # Check for Staleness
                days_lag = (global_latest_date - end_date).days
                if days_lag > STALE_THRESHOLD_DAYS:
                    stale_stocks.append(stock)
                    continue
                    
                # Check for History Length
                history_days = (end_date - start_date).days
                if history_days < MIN_HISTORY_DAYS:
                    short_history_stocks.append(stock)
                    continue
                    
                valid_stocks.append(stock)
            
            self.logger.info(f"Filtered out {len(stale_stocks)} stale stocks: {stale_stocks}")
            self.logger.info(f"Filtered out {len(short_history_stocks)} short-history stocks: {short_history_stocks}")
            self.logger.info(f"Remaining valid stocks: {len(valid_stocks)}")
            
            if not valid_stocks:
                self.logger.info("No valid stocks remained after filtering! Using all stocks as fallback (may fail).")
                # Fallback: Just use the original list if everything was filtered out, to avoid crash, 
                # but it will likely produce the same negative result.
                valid_ranges = stock_ranges.values()
            else:
                valid_ranges = [stock_ranges[s] for s in valid_stocks]

            # Find the common intersection of VALID stocks
            start_dates = [data['start_date'] for data in valid_ranges]
            end_dates = [data['end_date'] for data in valid_ranges]
            
            latest_start = max(start_dates)
            earliest_end = min(end_dates)
            
            self.logger.info(f"Common data range (Valid Subset): {latest_start.strftime('%Y-%m-%d')} to {earliest_end.strftime('%Y-%m-%d')}")
            
            # Add buffer for RS strategy calculations (similar to stock strategy)
            # RS strategy needs lookback periods, so add buffer
            buffer_weeks = 15  # 90 weeks = ~630 calendar days
            buffer_days = buffer_weeks * 7
            strategy_start = latest_start + timedelta(days=buffer_days)
            
            self.logger.info(f"RS Strategy Buffer:")
            self.logger.info(f"   Buffer period: {buffer_weeks} weeks ({buffer_days} calendar days)")
            self.logger.info(f"   Strategy start (with buffer): {strategy_start.strftime('%Y-%m-%d')}")
            
            # Ensure we have valid range
            if strategy_start >= earliest_end:
                self.logger.info(f"Insufficient data with buffer, using latest_start as strategy start")
                # If still invalid, it means the intersection is fundamentally broken even after filtering.
                # Try to salvage by just taking a 1-year lookback from earliest_end if possible
                if (earliest_end - latest_start).days > 365:
                     strategy_start = latest_start
                else: 
                     self.logger.performance("Range is still invalid even with valid subset. Returning default.")
                     # Return 0 years to signal failure gracefully
                     return latest_start.strftime('%Y-%m-%d'), earliest_end.strftime('%Y-%m-%d'), 0.0

            
            # Calculate years available for backtesting
            years_available = (earliest_end - strategy_start).days / 365.25
            
            # Format dates as strings
            start_date_str = strategy_start.strftime('%Y-%m-%d')
            end_date_str = earliest_end.strftime('%Y-%m-%d')
            
            self.logger.info(f"Final date range: {start_date_str} to {end_date_str} ({years_available:.1f} years)")
            
            return start_date_str, end_date_str, years_available
            
        except Exception as e:
            self.logger.progress(f"Error calculating date range: {e}")
            import traceback
            traceback.print_exc()
            return None, None, 0.0
    
    def load_index_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load index data for main index (Nifty 50)"""
        # Ensure dates are timezone-naive
        if start_date.tzinfo:
            start_date = start_date.replace(tzinfo=None)
        if end_date.tzinfo:
            end_date = end_date.replace(tzinfo=None)
            
        # Use same buffer logic as stock data - 400 calendar days BEFORE backtest start
        # buffer_start_date = (pd.to_datetime(start_date) - timedelta(days=400)).strftime('%Y-%m-%d')
        data_start_date = start_date - timedelta(days=400)
        
        self.logger.progress(f"Loading index data for {self.main_index} from {data_start_date} to {end_date} (backtest period: {start_date} to {end_date})")
        
        # Use SQLAlchemy text() for PostgreSQL-compatible queries
        from sqlalchemy import text
        
        # Determine table name and symbol variants based on market
        if self.market == 'US':
            symbol_variants = [self.main_index, 'S&P_500', '^GSPC', 'SPY']
            table_name = "s_p_500_index_market"
        else:
            symbol_variants = [self.main_index]
            if self.main_index in ['^NSEI', 'NSEI', 'NIFTY50', 'NIFTY_50']:
                symbol_variants = ['^NSEI', 'NSEI', 'NIFTY50', 'NIFTY_50']
            elif self.main_index in ['^NIFTY50', 'NIFTY50', 'NIFTY_50']:
                symbol_variants = ['^NIFTY50', 'NIFTY50', 'NSEI', 'NIFTY_50']
            table_name = "nifty_50_index_market"
        
        # Build placeholders for IN clause
        placeholders = ','.join([f':symbol_{i}' for i in range(len(symbol_variants))])
        
        # Query using SQLAlchemy text() with PostgreSQL parameter binding
        query = text(f"""
            SELECT symbol, date, open, high, low, close, COALESCE(adj_close, close) AS adjusted_close, volume
            FROM {table_name} 
            WHERE symbol IN ({placeholders}) AND date >= :start_date AND date <= :end_date
            ORDER BY date
        """)
        
        # Build parameters dict
        params = {f'symbol_{i}': symbol for i, symbol in enumerate(symbol_variants)}
        params.update({
            "start_date": data_start_date,
            "end_date": end_date
        })
        
        # Execute query and convert to DataFrame
        result = self.db.execute(query, params)
        
        # Convert to DataFrame
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        # Set date to datetime and index
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        self.logger.performance(f"Raw index data query returned: {len(df)} records")
        
        # Rename column to match our expected schema
        if 'adj_close' in df.columns:
            df = df.rename(columns={'adj_close': 'adjusted_close'})
        
        # Convert timezone-aware dates to timezone-naive
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        
        self.logger.info(f"Final index data: {len(df)} records")
        if not df.empty:
            self.logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df


    def get_nifty50_custom_stocks(self) -> List[str]:
        """Your best selected Nifty 50 stocks from actual database"""
        return [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR',
            'ICICIBANK', 'KOTAKBANK', 'ITC', 'BHARTIARTL','M&M',
            'SBIN', 'BAJFINANCE', 'ASIANPAINT', 'MARUTI', 'LT',
            'AXISBANK', 'NESTLEIND', 'ULTRACEMCO', 'SUNPHARMA', 'TITAN',
            'POWERGRID', 'NTPC', 'ONGC', 'TECHM', 'WIPRO',
            'HCLTECH', 'BAJAJFINSV', 'DRREDDY', 'CIPLA', 'GRASIM',
            'JSWSTEEL', 'TATAMOTORS', 'INDUSINDBK', 'COALINDIA', 'BPCL',
            'TATASTEEL', 'EICHERMOT', 'HEROMOTOCO', 'ADANIPORTS', 'SHREECEM',
            'BRITANNIA', 'DIVISLAB', 'APOLLOHOSP', 'HINDALCO', 'UPL',
            'BAJAJ-AUTO', 'TATACONSUM', 'DABUR', 'GODREJCP', 'PIDILITIND'
        ]
    
    def get_nifty100_custom_stocks(self) -> List[str]:
        """Your best selected Nifty 100 stocks"""
        nifty50 = self.get_nifty50_custom_stocks()
        additional_stocks = [
            'VEDL', 'GAIL', 'IOC', 'PETRONET', 'BANDHANBNK', 
            'MOTHERSON', 'BOSCHLTD', 'BERGEPAINT', 'GODREJPROP', 'MFSL',
            'BIOCON', 'COLPAL', 'HDFCLIFE', 'ICICIGI', 'ICICIPRULI',
            'MARICO', 'MPHASIS', 'MRF', 'SAIL', 'SIEMENS',
            'TATACHEM', 'TATAPOWER', 'TORNTPHARM', 'TVSMOTOR', 'UBL',
            'ZEEL', 'ZYDUSLIFE', 'ABBOTINDIA', 'ADANIENT', 'ADANIGREEN',
            'ALKEM', 'AMBUJACEM', 'APLLTD', 'ASHOKLEY', 'ASTRAL',
            'ATUL', 'AUBANK', 'AUROPHARMA', 'BAJAJHLDNG', 'BALRAMCHIN',
            'BATAINDIA', 'BEL', 'BEML', 'BHARATFORG', 'BHEL',
            'BLUEDART', 'BLUESTARCO', 'CANBK', 'CENTRALBK', 'CESC'
        ]
        return nifty50 + additional_stocks
    
    def get_nifty200_custom_stocks(self) -> List[str]:
        """Your best selected Nifty 200 stocks"""
        nifty100 = self.get_nifty100_custom_stocks()
        additional_stocks = [
            'BEML', 'BHARATFORG', 'CANFINHOME', 'CHOLAFIN', 'CONCOR',
            'CROMPTON', 'CUMMINSIND', 'DALBHARAT', 'DEEPAKNTR', 'DLF',
            'ESCORTS', 'EXIDEIND', 'FEDERALBNK', 'GLENMARK', 'GRANULES',
            'GUJGASLTD', 'HAL', 'HAVELLS', 'HDFCAMC', 'HINDPETRO',
            'IBULHSGFIN', 'IDBI', 'IDFCFIRSTB', 'IGL', 'INDHOTEL',
            'INDIANB', 'INDIGO', 'IPCALAB', 'IRCTC', 'JINDALSTEL',
            'JKCEMENT', 'JUBLFOOD', 'JUSTDIAL', 'LICHSGFIN', 'LTTS',
            'LUPIN', 'M&M', 'M&MFIN', 'MANAPPURAM', 'MUTHOOTFIN',
            'NATIONALUM', 'NAUKRI', 'NMDC', 'OBEROI', 'OFSS',
            'PAGEIND', 'PEL', 'PNB', 'POLYCAB', 'PVR',
            'RBLBANK', 'RECLTD', 'SBICARD', 'SBILIFE', 'SRF',
            'SUNTV', 'TRENT', 'VOLTAS', 'YESBANK', '3MINDIA',
            'ABB', 'ACC', 'ADANIPOWER', 'ADANITRANS', 'APOLLOTYRE',
            'ASHOKLEY', 'ASTRAL', 'ATUL', 'AUBANK', 'AUROPHARMA',
            'BAJAJHLDNG', 'BALRAMCHIN', 'BANDHANBNK', 'BATAINDIA', 'BEL',
            'BEML', 'BERGEPAINT', 'BHARATFORG', 'BIOCON', 'BOSCHLTD',
            'CADILAHC', 'CANFINHOME', 'CHOLAFIN', 'CIPLA', 'COLPAL',
            'CONCOR', 'CROMPTON', 'CUMMINSIND', 'DABUR', 'DALBHARAT',
            'DEEPAKNTR', 'DIVISLAB', 'DLF', 'DRREDDY', 'EICHERMOT',
            'ESCORTS', 'EXIDEIND', 'FEDERALBNK', 'GAIL', 'GLENMARK',
            # 'GODREJCP', 'GODREJPROP', 'GRANULES', 'GRASIM', 'GUJGASLTD',
            # 'HAL', 'HAVELLS', 'HCLTECH', 'HDFCAMC', 'HDFCLIFE',
            # 'HEROMOTOCO', 'HINDALCO', 'HINDPETRO', 'HINDUNILVR', 'IBULHSGFIN',
            # 'ICICIGI', 'ICICIPRULI', 'IDBI', 'IDFCFIRSTB', 'IGL',
            # 'INDHOTEL', 'INDIANB', 'INDIGO', 'INDUSINDBK', 'INFY',
            # 'IOC', 'IPCALAB', 'IRCTC', 'ITC', 'JINDALSTEL',
            # 'JKCEMENT', 'JSWSTEEL', 'JUBLFOOD', 'JUSTDIAL', 'KOTAKBANK',
            # 'LALPATHLAB', 'LICHSGFIN', 'LT', 'LTTS', 'LUPIN',
            # 'M&M', 'M&MFIN', 'MANAPPURAM', 'MARICO', 'MARUTI',
            # 'MCDOWELL-N', 'MFSL', 'MINDTREE', 'MOTHERSON', 'MPHASIS',
            # 'MRF', 'MUTHOOTFIN', 'NATIONALUM', 'NAUKRI', 'NESTLEIND',
            # 'NMDC', 'NTPC', 'OBEROI', 'OFSS', 'ONGC',
            # 'PAGEIND', 'PEL', 'PETRONET', 'PIDILITIND', 'PNB',
            # 'POLYCAB', 'POWERGRID', 'PVR', 'RBLBANK', 'RECLTD',
            # 'RELAXO', 'RELIANCE', 'SAIL', 'SBICARD', 'SBILIFE',
            # 'SBIN', 'SHREECEM', 'SIEMENS', 'SRF', 'SUNPHARMA',
            # 'SUNTV', 'TATACHEM', 'TATACONSUM', 'TATAMOTORS', 'TATAPOWER',
            # 'TATASTEEL', 'TCS', 'TECHM', 'TITAN', 'TORNTPHARM',
            # 'TRENT', 'TVSMOTORS', 'UBL', 'ULTRACEMCO', 'UPL',
            # 'VEDL', 'VOLTAS', 'WIPRO', 'YESBANK', 'ZEEL',
            # 'ZYDUSLIFE'
        ]
        return nifty100 + additional_stocks
    
    def get_nifty300_custom_stocks(self) -> List[str]:
        """Your best selected Nifty 300 stocks"""
        nifty200 = self.get_nifty200_custom_stocks()
        additional_stocks = [
            '3MINDIA', 'ABB', 'ACC', 'ADANIENT', 'ADANIGREEN',
            'ADANIPORTS', 'ADANITRANS', 'ADANIPOWER', 'ALKEM', 'AMBUJACEM',
            'APLLTD', 'ASHOKLEY', 'ASTRAL', 'ATUL', 'AUBANK',
            'AUROPHARMA', 'BAJAJHLDNG', 'BALRAMCHIN', 'BANDHANBNK', 'BATAINDIA',
            'BEL', 'BEML', 'BERGEPAINT', 'BHARATFORG', 'BIOCON',
            'BOSCHLTD', 'CADILAHC', 'CANFINHOME', 'CHOLAFIN', 'CIPLA',
            'COLPAL', 'CONCOR', 'CROMPTON', 'CUMMINSIND', 'DABUR',
            'DALBHARAT', 'DEEPAKNTR', 'DIVISLAB', 'DLF', 'DRREDDY',
            'EICHERMOT', 'ESCORTS', 'EXIDEIND', 'FEDERALBNK', 'GAIL',
            'GLENMARK', 'GODREJCP', 'GODREJPROP', 'GRANULES', 'GRASIM',
            'GUJGASLTD', 'HAL', 'HAVELLS', 'HCLTECH', 'HDFCAMC',
            'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDPETRO', 'HINDUNILVR',
            'IBULHSGFIN', 'ICICIGI', 'ICICIPRULI', 'IDBI', 'IDFCFIRSTB',
            'IGL', 'INDHOTEL', 'INDIANB', 'INDIGO', 'INDUSINDBK',
            'INFY', 'IOC', 'IPCALAB', 'IRCTC', 'ITC',
            'JINDALSTEL', 'JKCEMENT', 'JSWSTEEL', 'JUBLFOOD', 'JUSTDIAL',
            'KOTAKBANK', 'LALPATHLAB', 'LICHSGFIN', 'LT', 'LTTS',
            'LUPIN', 'M&M', 'M&MFIN', 'MANAPPURAM', 'MARICO',
            'MARUTI', 'MCDOWELL-N', 'MFSL', 'MINDTREE', 'MOTHERSON',
            'MPHASIS', 'MRF', 'MUTHOOTFIN', 'NATIONALUM', 'NAUKRI',
            # 'NESTLEIND', 'NMDC', 'NTPC', 'OBEROI', 'OFSS',
            # 'ONGC', 'PAGEIND', 'PEL', 'PETRONET', 'PIDILITIND',
            # 'PNB', 'POLYCAB', 'POWERGRID', 'PVR', 'RBLBANK',
            # 'RECLTD', 'RELAXO', 'RELIANCE', 'SAIL', 'SBICARD',
            # 'SBILIFE', 'SBIN', 'SHREECEM', 'SIEMENS', 'SRF',
            # 'SUNPHARMA', 'SUNTV', 'TATACHEM', 'TATACONSUM', 'TATAMOTORS',
            # 'TATAPOWER', 'TATASTEEL', 'TCS', 'TECHM', 'TITAN',
            # 'TORNTPHARM', 'TRENT', 'TVSMOTORS', 'UBL', 'ULTRACEMCO',
            # 'UPL', 'VEDL', 'VOLTAS', 'WIPRO', 'YESBANK',
            # 'ZEEL', 'ZYDUSLIFE'
        ]
        return nifty200 + additional_stocks
    
    def get_nifty500_custom_stocks(self) -> List[str]:
        """Your best selected Nifty 500 stocks"""
        nifty300 = self.get_nifty300_custom_stocks()
        additional_stocks = [
            '3MINDIA', 'ABB', 'ACC', 'ADANIENT', 'ADANIGREEN',
            'ADANIPORTS', 'ADANITRANS', 'ADANIPOWER', 'ALKEM', 'AMBUJACEM',
            'APLLTD', 'ASHOKLEY', 'ASTRAL', 'ATUL', 'AUBANK',
            'AUROPHARMA', 'BAJAJHLDNG', 'BALRAMCHIN', 'BANDHANBNK', 'BATAINDIA',
            'BEL', 'BEML', 'BERGEPAINT', 'BHARATFORG', 'BIOCON',
            'BOSCHLTD', 'CADILAHC', 'CANFINHOME', 'CHOLAFIN', 'CIPLA',
            'COLPAL', 'CONCOR', 'CROMPTON', 'CUMMINSIND', 'DABUR',
            'DALBHARAT', 'DEEPAKNTR', 'DIVISLAB', 'DLF', 'DRREDDY',
            'EICHERMOT', 'ESCORTS', 'EXIDEIND', 'FEDERALBNK', 'GAIL',
            'GLENMARK', 'GODREJCP', 'GODREJPROP', 'GRANULES', 'GRASIM',
            'GUJGASLTD', 'HAL', 'HAVELLS', 'HCLTECH', 'HDFCAMC',
            'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDPETRO', 'HINDUNILVR',
            'IBULHSGFIN', 'ICICIGI', 'ICICIPRULI', 'IDBI', 'IDFCFIRSTB',
            'IGL', 'INDHOTEL', 'INDIANB', 'INDIGO', 'INDUSINDBK',
            'INFY', 'IOC', 'IPCALAB', 'IRCTC', 'ITC',
            'JINDALSTEL', 'JKCEMENT', 'JSWSTEEL', 'JUBLFOOD', 'JUSTDIAL',
            'KOTAKBANK', 'LALPATHLAB', 'LICHSGFIN', 'LT', 'LTTS',
            'LUPIN', 'M&M', 'M&MFIN', 'MANAPPURAM', 'MARICO',
            'MARUTI', 'MCDOWELL-N', 'MFSL', 'MINDTREE', 'MOTHERSON',
            'MPHASIS', 'MRF', 'MUTHOOTFIN', 'NATIONALUM', 'NAUKRI',
            'NESTLEIND', 'NMDC', 'NTPC', 'OBEROI', 'OFSS',
            'ONGC', 'PAGEIND', 'PEL', 'PETRONET', 'PIDILITIND',
            'PNB', 'POLYCAB', 'POWERGRID', 'PVR', 'RBLBANK',
            'RECLTD', 'RELAXO', 'RELIANCE', 'SAIL', 'SBICARD',
            'SBILIFE', 'SBIN', 'SHREECEM', 'SIEMENS', 'SRF',
            'SUNPHARMA', 'SUNTV', 'TATACHEM', 'TATACONSUM', 'TATAMOTORS',
            'TATAPOWER', 'TATASTEEL', 'TCS', 'TECHM', 'TITAN',
            'TORNTPHARM', 'TRENT', 'TVSMOTORS', 'UBL', 'ULTRACEMCO',
            'UPL', 'VEDL', 'VOLTAS', 'WIPRO', 'YESBANK',
            # 'ZEEL', 'ZYDUSLIFE'
        ]
        return nifty300 + additional_stocks
    
    
    def get_custom_stock_universe(self) -> List[str]:
        """Get custom list of stocks based on user selection or configuration"""
        try:
            # PRIORITY 1: Use custom_stocks if provided (Option B: stock_universe + custom_stocks)
            if hasattr(self, 'custom_stocks') and self.custom_stocks:
                self.logger.info(f"Using custom stock selection: {len(self.custom_stocks)} stocks")
                return self.custom_stocks
            
            # PRIORITY 2: Get from configuration (if you add a custom_stocks field)
            if hasattr(self.config, 'custom_stocks') and self.config.custom_stocks:
                custom_stocks = json.loads(self.config.custom_stocks)
                self.logger.info(f"Using custom stock universe from config: {len(custom_stocks)} stocks")
                return custom_stocks
            
            # PRIORITY 3: Use stock_universe to get predefined list
            # Get stock universe from instance or config
            stock_universe = None
            if hasattr(self, 'stock_universe'):
                stock_universe = self.stock_universe
            elif self.config:
                stock_universe = getattr(self.config, 'stock_universe', None)
            
            # Option 4: Use actual database symbols instead of hardcoded lists
            # This ensures we use symbols that actually exist in the database
            all_database_symbols = [
                '360ONE', '3MINDIA', 'AADHARHFC', 'AARTIIND', 'AAVAS',
                'ABB', 'ABBOTINDIA', 'ABCAPITAL', 'ABFRL', 'ABREL',
                'ABSLAMC', 'ACC', 'ACE', 'ACMESOLAR', 'ADANIENSOL',
                'ADANIENT', 'ADANIGREEN', 'ADANIPORTS', 'ADANIPOWER', 'AEGISLOG',
                'AFCONS', 'AFFLE', 'AIAENG', 'AIIL', 'AJANTPHARM',
                'AKUMS', 'ALIVUS', 'ALKEM', 'ALKYLAMINE', 'ALOKINDS',
                'AMBER', 'AMBUJACEM', 'ANANDRATHI', 'ANANTRAJ', 'ANGELONE',
                'APARINDS', 'APLAPOLLO', 'APLLTD', 'APOLLOHOSP', 'APOLLOTYRE',
                'APTUS', 'ARE&M', 'ASAHIINDIA', 'ASHOKLEY', 'ASIANPAINT',
                'ASTERDM', 'ASTRAL', 'ASTRAZEN', 'ATGL', 'ATUL',
                'AUBANK', 'AUROPHARMA', 'AWL', 'AXISBANK', 'BAJAJ-AUTO',
                'BAJAJFINSV', 'BAJAJHFL', 'BAJAJHLDNG', 'BAJFINANCE', 'BALKRISIND',
                'BALRAMCHIN', 'BANDHANBNK', 'BANKBARODA', 'BANKINDIA', 'BASF',
                'BATAINDIA', 'BAYERCROP', 'BBTC', 'BDL', 'BEL',
                'BEML', 'BERGEPAINT', 'BHARATFORG', 'BHARTIARTL', 'BHARTIHEXA',
                'BHEL', 'BIKAJI', 'BIOCON', 'BLS', 'BLUEDART',
                'BLUESTARCO', 'BOSCHLTD', 'BPCL', 'BRIGADE', 'BRITANNIA',
                'BSE', 'BSOFT', 'CAMPUS', 'CAMS', 'CANBK',
                'CANFINHOME', 'CAPLIPOINT', 'CARBORUNIV', 'CASTROLIND', 'CCL',
                'CDSL', 'CEATLTD', 'CENTRALBK', 'CENTURYPLY', 'CERA',
                'CESC', 'CGCL', 'CGPOWER', 'CHALET', 'CHAMBLFERT',
                'CHENNPETRO', 'CHOLAFIN', 'CHOLAHLDNG', 'CIPLA', 'CLEAN',
                'COALINDIA', 'COCHINSHIP', 'COFORGE', 'COLPAL', 'CONCOR',
                'CONCORDBIO', 'COROMANDEL', 'CRAFTSMAN', 'CREDITACC', 'CRISIL',
                'CROMPTON', 'CUB', 'CUMMINSIND', 'CYIENT', 'DABUR',
                'DALBHARAT', 'DATAPATTNS', 'DBREALTY', 'DCMSHRIRAM', 'DEEPAKFERT',
                'DEEPAKNTR', 'DELHIVERY', 'DEVYANI', 'DIVISLAB', 'DIXON',
                'DLF', 'DMART', 'DOMS', 'DRREDDY', 'ECLERX',
                'EICHERMOT', 'EIDPARRY', 'EIHOTEL', 'ELECON', 'ELGIEQUIP',
                'EMAMILTD', 'EMCURE', 'ENDURANCE', 'ENGINERSIN', 'ERIS',
                'ESCORTS', 'ETERNAL', 'EXIDEIND', 'FACT', 'FEDERALBNK',
                'FINCABLES', 'FINPIPE', 'FIRSTCRY', 'FIVESTAR', 'FLUOROCHEM',
                'FORTIS', 'FSL', 'GAIL', 'GESHIP', 'GICRE',
                'GILLETTE', 'GLAND', 'GLAXO', 'GLENMARK', 'GMDCLTD',
                'GMRAIRPORT', 'GNFC', 'GODFRYPHLP', 'GODIGIT', 'GODREJAGRO',
                'GODREJCP', 'GODREJIND', 'GODREJPROP', 'GPIL', 'GPPL',
                'GRANULES', 'GRAPHITE', 'GRASIM', 'GRAVITA', 'GRSE',
                'GSPL', 'GUJGASLTD', 'GVT&D', 'HAL', 'HAPPSTMNDS',
                'HAVELLS', 'HBLENGINE', 'HCLTECH', 'HDFCAMC', 'HDFCBANK',
                'HDFCLIFE', 'HEG', 'HEROMOTOCO', 'HFCL', 'HINDALCO',
                'HINDCOPPER', 'HINDPETRO', 'HINDUNILVR', 'HINDZINC', 'HOMEFIRST',
                'HONASA', 'HONAUT', 'HSCL', 'HUDCO', 'HYUNDAI',
                'ICICIBANK', 'ICICIGI', 'ICICIPRULI', 'IDBI', 'IDEA',
                'IDFCFIRSTB', 'IEX', 'IFCI', 'IGIL', 'IGL',
                'IIFL', 'IKS', 'INDGN', 'INDHOTEL', 'INDIACEM',
                'INDIAMART', 'INDIANB', 'INDIGO', 'INDUSINDBK', 'INDUSTOWER',
                'INFY', 'INOXINDIA', 'INOXWIND', 'INTELLECT', 'IOB',
                'IOC', 'IPCALAB', 'IRB', 'IRCON', 'IRCTC',
                'KPITTECH', 'KPRMILL', 'LALPATHLAB', 'LATENTVIEW', 'LAURUSLABS',
                'LEMONTREE', 'LICHSGFIN', 'LICI', 'LINDEINDIA', 'LLOYDSME',
                'LODHA', 'LT', 'LTF', 'LTFOODS', 'LTIM',
                'LTTS', 'LUPIN', 'M&M', 'M&MFIN', 'MAHABANK',
                'MAHSEAMLES', 'MANAPPURAM', 'MANKIND', 'MANYAVAR', 'MAPMYINDIA',
                'MARICO', 'MARUTI', 'MASTEK', 'MAXHEALTH', 'MAZDOCK',
                'MCX', 'MEDANTA', 'METROPOLIS', 'MFSL', 'MGL',
                'MINDACORP', 'MMTC', 'MOTHERSON', 'MOTILALOFS', 'MPHASIS',
                'MRF', 'MRPL', 'MSUMI', 'MUTHOOTFIN', 'NAM-INDIA',
                'NATCOPHARM', 'NATIONALUM', 'NAUKRI', 'NAVA', 'NAVINFLUOR',
                'NBCC', 'NCC', 'NESTLEIND', 'NETWEB', 'NETWORK18',
                'NEULANDLAB', 'NEWGEN', 'NH', 'NHPC', 'NIACL',
                'NIVABUPA', 'NLCINDIA', 'NMDC', 'NSLNISP', 'NTPC',
                'NTPCGREEN', 'NUVAMA', 'NYKAA', 'OBEROIRLTY', 'OFSS',
                'OIL', 'OLAELEC', 'OLECTRA', 'ONGC', 'PAGEIND',
                'PATANJALI', 'PAYTM', 'PCBL', 'PEL', 'PERSISTENT',
                'PETRONET', 'PFC', 'PFIZER', 'PGEL', 'PHOENIXLTD',
                'PIDILITIND', 'PIIND', 'PNB', 'PNBHOUSING', 'PNCINFRA',
                'POLICYBZR', 'POLYCAB', 'POLYMED', 'POONAWALLA', 'POWERGRID',
                'POWERINDIA', 'PPLPHARMA', 'PRAJIND', 'PREMIERENE', 'PRESTIGE',
                'PTCIL', 'PVRINOX', 'RADICO', 'RAILTEL', 'RAINBOW',
                'RAMCOCEM', 'RAYMOND', 'RAYMONDLSL', 'RBLBANK', 'RCF',
                'RECLTD', 'REDINGTON', 'RELIANCE', 'RENUKA', 'RHIM',
                'RITES', 'RKFORGE', 'ROUTE', 'RPOWER', 'RRKABEL',
                'RTNINDIA', 'RVNL', 'SAGILITY', 'SAIL', 'SAILIFE',
                'SAMMAANCAP', 'SAPPHIRE', 'SARDAEN', 'SAREGAMA', 'SBFC',
                'SBICARD', 'SBILIFE', 'SBIN', 'SCHAEFFLER', 'SCHNEIDER',
                'SCI', 'SHREECEM', 'SHRIRAMFIN', 'SHYAMMETL', 'SIEMENS',
                'SIGNATURE', 'SJVN', 'SKFINDIA', 'SOBHA', 'SOLARINDS',
                'SONACOMS', 'SONATSOFTW', 'SRF', 'STARHEALTH', 'SUMICHEM',
                'SUNDARMFIN', 'SUNDRMFAST', 'SUNPHARMA', 'SUNTV', 'SUPREMEIND',
                'SUZLON', 'SWANCORP', 'SWIGGY', 'SWSOLAR', 'SYNGENE',
                'SYRMA', 'TANLA', 'TARIL', 'TATACHEM', 'TATACOMM',
                'TATACONSUM', 'TATAELXSI', 'TATAINVEST', 'TATAMOTORS', 'TATAPOWER',
                'TATASTEEL', 'TATATECH', 'TBOTEK', 'TCS', 'TECHM',
                'TECHNOE', 'TEJASNET', 'THERMAX', 'TIINDIA', 'TIMKEN',
                'TITAGARH', 'TITAN', 'TORNTPHARM', 'TORNTPOWER', 'TRENT',
                'TRIDENT', 'TRITURBINE', 'TRIVENI', 'TTML', 'TVSMOTOR',
                'UBL', 'UCOBANK', 'ULTRACEMCO', 'UNIONBANK', 'UNITDSPR',
                'UNOMINDA', 'UPL', 'USHAMART', 'UTIAMC', 'VBL',
                'VEDL', 'VGUARD', 'VIJAYA', 'VMM', 'VOLTAS',
                'VTL', 'WAAREEENER', 'WELCORP', 'WELSPUNLIV', 'WESTLIFE',
                'WHIRLPOOL', 'WIPRO', 'WOCKPHARMA', 'YESBANK', 'ZEEL',
                'ZENSARTECH', 'ZENTEC', 'ZFCVINDIA', 'ZYDUSLIFE'
            ]
            
            # Initialize custom_stocks with default value
            custom_stocks = []
            
            # Select subset based on stock_universe
            if stock_universe == 'NIFTY50':
                custom_stocks = self.get_nifty50_custom_stocks()
            elif stock_universe == 'NIFTY100':
                custom_stocks = self.get_nifty100_custom_stocks()
            elif stock_universe == 'NIFTY200':
                custom_stocks = self.get_nifty200_custom_stocks()
            elif stock_universe == 'NIFTY300':
                custom_stocks = self.get_nifty300_custom_stocks()
            elif stock_universe == 'NIFTY500':
                custom_stocks = self.get_nifty500_custom_stocks()
            else:
                # Fallback: use main_index for backward compatibility
                if self.main_index in ['^NIFTY50', '^NSEI', 'NIFTY50']:
                    custom_stocks = self.get_nifty50_custom_stocks()
                elif self.main_index == '^NIFTY100':
                    custom_stocks = self.get_nifty100_custom_stocks()
                elif self.main_index == '^NIFTY200':
                    custom_stocks = self.get_nifty200_custom_stocks()
                elif self.main_index == '^NIFTY300':
                    custom_stocks = self.get_nifty300_custom_stocks()
                else:  # Default to NIFTY500
                    custom_stocks = self.get_nifty500_custom_stocks()
            
            universe_used = stock_universe if stock_universe else f"main_index:{self.main_index}"
            self.logger.info(f"Using stock universe '{universe_used}': {len(custom_stocks)} stocks")
            if custom_stocks:
                self.logger.info(f"Sample symbols: {custom_stocks[:10]}")
            return custom_stocks
            
        except Exception as e:
            self.logger.info(f"Error fetching custom stock universe: {e}")
            return []

    def load_metadata(self) -> Dict[str, Dict]:
        """Load stock metadata by calculating directly from stock_data table"""
        try:
            # Calculate metadata directly from stock_data table
            metadata = self.calculate_metadata_from_data(self.db)
            return metadata
        except Exception as e:
            self.logger.error(f"Error loading stock metadata: {e}")
            return {}

    def calculate_metadata_from_data(self, session) -> Dict[str, Dict]:
        """Calculate metadata directly from stock_data table"""
        try:
            from sqlalchemy import text
            
            # Query metadata from stock_data
            query_str = """
                SELECT 
                    symbol,
                    MIN(date)::text as start_date,
                    MAX(date)::text as end_date,
                    COUNT(*) as total_records,
                    ROUND(EXTRACT(EPOCH FROM (MAX(date)::timestamp - MIN(date)::timestamp)) / 86400.0 / 365.25, 1) as years_available
                FROM stock_market
                GROUP BY symbol
                ORDER BY symbol
            """
            
            result = session.execute(text(query_str))
            rows = result.fetchall()
            
            metadata = {}
            for row in rows:
                try:
                    metadata[row[0]] = {
                        'start_date': str(row[1]) if row[1] else None,
                        'end_date': str(row[2]) if row[2] else None,
                        'years_available': float(row[4]) if row[4] else 0,
                        'total_records': int(row[3]) if row[3] else 0,
                        'data_source': 'stock_market'
                    }
                except Exception:
                    continue
            
            return metadata
        except Exception as e:
            self.logger.error(f"Error calculating metadata: {e}")
            return {}

    def generate_asset_description(self, symbol: str) -> str:
        """Generate intelligent stock descriptions based on symbol names"""
        symbol_lower = symbol.lower()
        if 'bank' in symbol_lower: return f'{symbol} - Banking Sector Stock'
        elif 'pharma' in symbol_lower: return f'{symbol} - Pharmaceutical Sector Stock'
        elif 'it' in symbol_lower or 'tech' in symbol_lower: return f'{symbol} - Technology Sector Stock'
        elif 'steel' in symbol_lower or 'metal' in symbol_lower: return f'{symbol} - Materials Sector Stock'
        elif 'auto' in symbol_lower: return f'{symbol} - Automotive Sector Stock'
        elif 'power' in symbol_lower or 'energy' in symbol_lower: return f'{symbol} - Energy Sector Stock'
        else: return f'{symbol} Stock'

    def get_asset_sector_classification(self, symbol: str) -> str:
        """Classify stock into sector categories"""
        symbol_lower = symbol.lower()
        if 'bank' in symbol_lower or 'fin' in symbol_lower: return 'Financial Services'
        elif 'pharma' in symbol_lower or 'health' in symbol_lower: return 'Healthcare'
        elif 'it' in symbol_lower or 'tech' in symbol_lower: return 'Technology'
        elif 'auto' in symbol_lower: return 'Automobile'
        elif 'steel' in symbol_lower or 'metal' in symbol_lower: return 'Materials'
        elif 'power' in symbol_lower or 'energy' in symbol_lower or 'oil' in symbol_lower: return 'Energy'
        elif 'infra' in symbol_lower: return 'Infrastructure'
        elif 'consumer' in symbol_lower or 'fmcg' in symbol_lower: return 'Consumer Goods'
        else: return 'Other'
    
    
    
    def validate_data_range(self, start_date: datetime, end_date: datetime) -> bool:
        """Validate that we have sufficient data for the requested period"""
        try:
            # Check if we have data for the required lookback periods
            earliest_required = start_date - timedelta(days=self.lookback_quarters * 7)
            
            # Query the earliest date in our data
            earliest_data = self.db.query(StockData.date).order_by(StockData.date.asc()).first()
            
            if earliest_data and earliest_data[0] > earliest_required:
                self.logger.info(f"WARNING: Insufficient data. Need data from {earliest_required}, but only have from {earliest_data[0]}")
                self.logger.info(f"Consider using shorter lookback periods or getting more historical data")
                return False
            
            return True
        except Exception as e:
            self.logger.info(f"Error validating data range: {e}")
            return True  # Continue anyway
    
    def calculate_last_trading_days(self, trading_dates: List[datetime]) -> None:
        """Calculate the last trading day of each week from actual trading data"""
        # Group dates by week (ISO week)
        weeks = {}
        for date in trading_dates:
            # Get ISO week number and year
            iso_year, iso_week, _ = date.isocalendar()
            week_key = (iso_year, iso_week)
            
            if week_key not in weeks:
                weeks[week_key] = []
            weeks[week_key].append(date)
        
        # For each week, find the last trading day
        for (year, week_num), dates_in_week in weeks.items():
            # Sort dates in the week
            dates_in_week.sort()
            last_date = dates_in_week[-1]
            self.last_trading_days.add(last_date)
    
    def calculate_rs_score(self, stock_prices: pd.Series, index_prices: pd.Series, 
                          current_date: datetime) -> Optional[float]:
        """Calculate Relative Strength score for a stock with robust data validation"""
        try:
            # Get available trading dates from actual data
            available_stock_dates = sorted(stock_prices.index)
            available_index_dates = sorted(index_prices.index)
            
            # Find current date index in both datasets
            try:
                current_stock_index = available_stock_dates.index(current_date)
                current_index_index = available_index_dates.index(current_date)
            except ValueError:
                # Current date not found in data
                return None
            
            # Get current prices
            current_stock_price = stock_prices.loc[current_date]
            current_index_price = index_prices.loc[current_date]
            
            # Calculate required lookback periods
            max_lookback = max(self.lookback_weeks, self.lookback_months, self.lookback_quarters)
            
            # DEBUG: Print debug info for first few calculations
            if hasattr(self, '_debug_count'):
                self._debug_count = getattr(self, '_debug_count', 0) + 1
            else:
                self._debug_count = 1
                
            if self._debug_count <= 3:  # Print for first 3 calculations
                self.logger.info(f"  DEBUG RS Calculation:")
                self.logger.info(f"    Date: {current_date}")
                self.logger.info(f"    Stock data: {len(available_stock_dates)} dates, first: {available_stock_dates[0]}, last: {available_stock_dates[-1]}")
                self.logger.info(f"    Index data: {len(available_index_dates)} dates, first: {available_index_dates[0]}, last: {available_index_dates[-1]}")
                self.logger.info(f"    Current stock index: {current_stock_index}")
                self.logger.info(f"    Current index position: {current_index_index}")
                self.logger.info(f"    Max lookback required: {max_lookback}")
                self.logger.info(f"    Lookback periods: week={self.lookback_weeks}, month={self.lookback_months}, quarter={self.lookback_quarters}")
        
            # Check if we have enough historical data
            if current_stock_index < max_lookback or current_index_index < max_lookback:
                if self._debug_count <= 3:
                    self.logger.info(f"      FAILED: Not enough historical data (need {max_lookback}, have stock:{current_stock_index}, index:{current_index_index})")
                return None
            
            # Get historical dates using actual available data
            week_ago_date = available_stock_dates[current_stock_index - self.lookback_weeks]
            month_ago_date = available_stock_dates[current_stock_index - self.lookback_months]
            quarter_ago_date = available_stock_dates[current_stock_index - self.lookback_quarters]
            
            # Verify index dates are available for the same periods
            week_ago_index_date = available_index_dates[current_index_index - self.lookback_weeks]
            month_ago_index_date = available_index_dates[current_index_index - self.lookback_months]
            quarter_ago_index_date = available_index_dates[current_index_index - self.lookback_quarters]
            
            # DEBUG: Print current and historical prices for first calculation
            # DEBUG: Print current and historical prices
            self.logger.info(f"    Current prices:")
            self.logger.info(f"      Stock: ₹{current_stock_price:.2f}, Index: ₹{current_index_price:.2f}")
            self.logger.info(f"    Historical dates and prices:")
            self.logger.info(f"      Week ago: {week_ago_date} - Stock: ₹{stock_prices.loc[week_ago_date]:.2f}, Index: ₹{index_prices.loc[week_ago_index_date]:.2f}")
            self.logger.info(f"      Month ago: {month_ago_date} - Stock: ₹{stock_prices.loc[month_ago_date]:.2f}, Index: ₹{index_prices.loc[month_ago_index_date]:.2f}")
            self.logger.info(f"      Quarter ago: {quarter_ago_date} - Stock: ₹{stock_prices.loc[quarter_ago_date]:.2f}, Index: ₹{index_prices.loc[quarter_ago_index_date]:.2f}")
            
            # Calculate RS for each period using actual historical data
            self.logger.info(f"    RS Calculation Breakdown:")
            
            # WEEK Calculation
            stock_past_w = stock_prices.loc[week_ago_date]
            index_past_w = index_prices.loc[week_ago_index_date]
            stock_ret_w = current_stock_price / stock_past_w
            index_ret_w = current_index_price / index_past_w
            rs_w = self.calculate_single_rs(current_stock_price, stock_past_w, current_index_price, index_past_w)
            self.logger.info(f"      WEEK (5 days):")
            self.logger.performance(f"        Stock Return = Current/Past = ₹{current_stock_price:.2f} / ₹{stock_past_w:.2f} = {stock_ret_w:.6f}")
            self.logger.performance(f"        Index Return = Current/Past = ₹{current_index_price:.2f} / ₹{index_past_w:.2f} = {index_ret_w:.6f}")
            self.logger.performance(f"        RS = (Stock Return / Index Return) - 1 = ({stock_ret_w:.6f} / {index_ret_w:.6f}) - 1 = {rs_w:.6f}")

            # MONTH Calculation
            stock_past_m = stock_prices.loc[month_ago_date]
            index_past_m = index_prices.loc[month_ago_index_date]
            stock_ret_m = current_stock_price / stock_past_m
            index_ret_m = current_index_price / index_past_m
            rs_m = self.calculate_single_rs(current_stock_price, stock_past_m, current_index_price, index_past_m)
            self.logger.info(f"      MONTH (20 days):")
            self.logger.performance(f"        Stock Return = Current/Past = ₹{current_stock_price:.2f} / ₹{stock_past_m:.2f} = {stock_ret_m:.6f}")
            self.logger.performance(f"        Index Return = Current/Past = ₹{current_index_price:.2f} / ₹{index_past_m:.2f} = {index_ret_m:.6f}")
            self.logger.performance(f"        RS = (Stock Return / Index Return) - 1 = ({stock_ret_m:.6f} / {index_ret_m:.6f}) - 1 = {rs_m:.6f}")

            # QUARTER Calculation
            stock_past_q = stock_prices.loc[quarter_ago_date]
            index_past_q = index_prices.loc[quarter_ago_index_date]
            stock_ret_q = current_stock_price / stock_past_q
            index_ret_q = current_index_price / index_past_q
            rs_q = self.calculate_single_rs(current_stock_price, stock_past_q, current_index_price, index_past_q)
            self.logger.info(f"      QUARTER (60 days):")
            self.logger.performance(f"        Stock Return = Current/Past = ₹{current_stock_price:.2f} / ₹{stock_past_q:.2f} = {stock_ret_q:.6f}")
            self.logger.performance(f"        Index Return = Current/Past = ₹{current_index_price:.2f} / ₹{index_past_q:.2f} = {index_ret_q:.6f}")
            self.logger.performance(f"        RS = (Stock Return / Index Return) - 1 = ({stock_ret_q:.6f} / {index_ret_q:.6f}) - 1 = {rs_q:.6f}")
            
            self.logger.info(f"    RS values: week={rs_w:.3f}, month={rs_m:.3f}, quarter={rs_q:.3f}")
            
            # Return None if any RS calculation failed
            if any(rs is None for rs in [rs_w, rs_m, rs_q]):
                self.logger.performance(f"      FAILED: One or more RS calculations returned None")
                return None
            
            # Calculate composite RS score
            rs_score = (rs_w + rs_m + rs_q) / 3
            
            return rs_score
            
        except (KeyError, IndexError, ValueError, ZeroDivisionError) as e:
            if hasattr(self, '_debug_count') and self._debug_count <= 3:
                self.logger.info(f"      FAILED: Exception - {str(e)}")
            return None
    
    def calculate_single_rs(self, stock_current: float, stock_past: float,
                           index_current: float, index_past: float) -> Optional[float]:
        """Calculate single period RS"""
        try:
            if stock_past == 0 or index_past == 0:
                return None
            rs = (stock_current / stock_past) / (index_current / index_past) - 1
            return rs
        except (ZeroDivisionError, ValueError):
            return None
    

    # ========================================================================
    # VECTORIZED RS CALCULATION METHODS (20-30x faster)
    # Added by auto-integration script
    # ========================================================================
    
    def calculate_rs_scores_vectorized(self, stock_data: pd.DataFrame, 
                                       index_data: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized RS calculation for all stocks at once using NumPy/Pandas
        
        Performance: 20-30x faster than loop-based approach
        """
        self.logger.progress("🚀 Calculating RS scores (vectorized)...")
        start_time = pd.Timestamp.now()
        
        try:
            # Pivot stock data: rows=dates, columns=symbols, values=prices
            stock_pivot = stock_data.pivot(
                index='date', 
                columns='symbol', 
                values='adjusted_close'
            )
            
            # Prepare index data as Series
            if isinstance(index_data, pd.DataFrame):
                if 'adjusted_close' in index_data.columns:
                    index_series = index_data.set_index('date')['adjusted_close']
                elif 'adj_close' in index_data.columns:
                    index_series = index_data.set_index('date')['adj_close']
                else:
                    raise ValueError("Index data must have 'adjusted_close' or 'adj_close' column")
            else:
                index_series = index_data
            
            # Ensure same date range
            common_dates = stock_pivot.index.intersection(index_series.index)
            stock_pivot = stock_pivot.loc[common_dates].sort_index()
            index_series = index_series.loc[common_dates].sort_index()
            
            self.logger.info(f"   Data: {stock_pivot.shape[0]} dates × {stock_pivot.shape[1]} stocks")
            
            # Calculate returns for all periods (vectorized)
            stock_returns_w = stock_pivot.pct_change(periods=self.lookback_weeks)
            stock_returns_m = stock_pivot.pct_change(periods=self.lookback_months)
            stock_returns_q = stock_pivot.pct_change(periods=self.lookback_quarters)
            
            index_returns_w = index_series.pct_change(periods=self.lookback_weeks)
            index_returns_m = index_series.pct_change(periods=self.lookback_months)
            index_returns_q = index_series.pct_change(periods=self.lookback_quarters)
            
            # Calculate RS (vectorized with broadcasting)
            rs_w = stock_returns_w.sub(index_returns_w, axis=0)
            rs_m = stock_returns_m.sub(index_returns_m, axis=0)
            rs_q = stock_returns_q.sub(index_returns_q, axis=0)
            
            # Simple Equal-Weight Average (Week + Month + Quarter) / 3
            rs_scores = (rs_w + rs_m + rs_q) / 3
            
            # Clean up inf/NaN
            rs_scores = rs_scores.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Store component scores for debugging/logging
            self.rs_w_df = rs_w
            self.rs_m_df = rs_m
            self.rs_q_df = rs_q
            
            calc_time = (pd.Timestamp.now() - start_time).total_seconds()
            total_calcs = rs_scores.shape[0] * rs_scores.shape[1]
            
            self.logger.progress(f"   ✅ {total_calcs:,} scores in {calc_time:.2f}s ({total_calcs/calc_time:,.0f}/sec)")
            
            return rs_scores
            
        except Exception as e:
            self.logger.info(f"   ❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_top_rs_stocks_vectorized(self, date: datetime, 
                                     rs_scores: pd.DataFrame,
                                     n: int = None) -> List[Tuple[str, float]]:
        """Get top N stocks by RS score for a given date"""
        if n is None:
            n = self.max_positions
        
        if date not in rs_scores.index:
            return []
        
        date_scores = rs_scores.loc[date].dropna()
        top_stocks = date_scores.nlargest(n)
        
        return [(symbol, float(score)) for symbol, score in top_stocks.items()]
    
    def get_stock_rs_rank_vectorized(self, symbol: str, date: datetime,
                                     rs_scores: pd.DataFrame) -> Optional[int]:
        """Get RS rank for a stock on a date"""
        if date not in rs_scores.index or symbol not in rs_scores.columns:
            return None
        
        date_scores = rs_scores.loc[date].dropna()
        if symbol not in date_scores.index:
            return None
        
        sorted_scores = date_scores.sort_values(ascending=False)
        rank = list(sorted_scores.index).index(symbol) + 1
        
        return rank
    
    def load_stock_data_optimized(self, start_date: datetime, 
                                  end_date: datetime) -> pd.DataFrame:
        """
        Optimized stock data loading
        Returns flat DataFrame for easier pivoting
        """
        if start_date.tzinfo:
            start_date = start_date.replace(tzinfo=None)
        if end_date.tzinfo:
            end_date = end_date.replace(tzinfo=None)
        
        buffer_days = max(self.lookback_quarters * 7, 100)
        data_start_date = start_date - timedelta(days=buffer_days)
        
        self.logger.progress(f"Loading data: {data_start_date.date()} to {end_date.date()}")
        
        stock_symbols = self.get_custom_stock_universe()
        if not stock_symbols:
            raise ValueError("No stocks in universe")
        
        self.logger.info(f"  Universe: {len(stock_symbols)} symbols")
        
        from sqlalchemy import text
        
        query = text("""
            SELECT symbol, date, adj_close as adjusted_close, open
            FROM stock_market 
            WHERE symbol = ANY(:symbols)
            AND date BETWEEN :start_date AND :end_date
            ORDER BY date, symbol
        """)
        
        result = self.db.execute(query, {
            "symbols": stock_symbols,
            "start_date": data_start_date,
            "end_date": end_date
        })
        
        df = pd.DataFrame(result.fetchall(), columns=['symbol', 'date', 'adjusted_close', 'open'])
        
        if df.empty:
            raise ValueError(f"No data for {data_start_date} to {end_date}")
        
        # Optimize dtypes
        df['symbol'] = df['symbol'].astype('category')
        df['adjusted_close'] = df['adjusted_close'].astype('float32')
        df['date'] = pd.to_datetime(df['date'])
        
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
        
        self.logger.progress(f"  ✅ {len(df):,} records, {df['symbol'].nunique()} symbols")
        
        return df
    
    # ========================================================================
    # END OF VECTORIZED METHODS
    # ========================================================================
    def get_trading_date_before(self, date: datetime, days_back: int) -> Optional[datetime]:
        """Get trading date N days before given date"""
        # Calculate actual trading days (approximately 5 trading days per week)
        trading_days_back = int(days_back * 7 / 5)
        
        # Go back the calculated number of days
        target_date = date - timedelta(days=trading_days_back)
        
        return target_date
    
    def is_friday_or_last_trading_day(self, date: datetime) -> bool:
        """Check if date is the last trading day of the week"""
        # Check if this date is in our pre-calculated last trading days set
        return date in self.last_trading_days
    
    def get_next_monday(self, date: datetime) -> datetime:
        """Get next Monday after given date"""
        days_ahead = 7 - date.weekday()  # Monday is 0
        if days_ahead == 7:
            days_ahead = 0
        return date + timedelta(days=days_ahead)
    
    def rank_stocks(self, stock_data: pd.DataFrame, index_data: pd.DataFrame, 
                   signal_date: datetime) -> pd.DataFrame:
        """Rank all eligible stocks by RS score with enhanced error handling"""
        symbols = self.get_custom_stock_universe()  # Use custom stock selection
        rankings = []
        valid_symbols = 0
        failed_symbols = 0
        
        # Get universe size based on custom selection
        universe_size = len(symbols)
        self.logger.info(f"Custom stock universe size: {universe_size} stocks and date: {signal_date.strftime('%Y-%m-%d (%A)')} ,day:{signal_date.weekday()}")
        
        self.logger.progress(f"Processing {universe_size} custom selected symbols for RS calculation")
        
        for i, symbol in enumerate(symbols):
            try:
                stock_prices = stock_data.loc[symbol]['adjusted_close']
                rs_score = self.calculate_rs_score(stock_prices, index_data['adjusted_close'], signal_date)
                
                if rs_score is not None:
                    valid_symbols += 1
                    # Debug: Print RS scores for first few symbols only
                    if valid_symbols <= 3:
                        self.logger.info(f"    {symbol}: RS Score = {rs_score:.3f}")

                    # Get additional ranking criteria
                    current_price = stock_prices.loc[signal_date]
                    
                    # Calculate 20-day volatility (simplified)
                    try:
                        price_20d = stock_prices.loc[signal_date - timedelta(days=20):signal_date]
                        volatility = price_20d.pct_change().std() * np.sqrt(252) if len(price_20d) > 1 else 0
                    except:
                        volatility = 0
                    
                    # Get market cap (simplified - would need actual data)
                    market_cap = current_price * 1000000  # Placeholder
                    
                    rankings.append({
                        'symbol': symbol,
                        'rs_score': rs_score,
                        'current_price': current_price,
                        'volatility': volatility,
                        'market_cap': market_cap
                    })
                else:
                    failed_symbols += 1
                    
            except (KeyError, IndexError) as e:
                failed_symbols += 1
                if failed_symbols <= 3:  # Show first few failures for debugging
                    self.logger.info(f"    {symbol}: Failed - {str(e)}")
                continue
        
        self.logger.info(f"  Valid RS calculations: {valid_symbols}, Failed: {failed_symbols}")
        
        if valid_symbols == 0:
            self.logger.info("  No valid rankings - insufficient historical data or all calculations failed")
            return pd.DataFrame()
        
        # Create DataFrame and sort
        df = pd.DataFrame(rankings)
        if df.empty:
            return df
        
        # Sort by RS score (descending), then by tie-breakers
        df = df.sort_values([
            'rs_score',  # Descending
            'volatility'  # Ascending (lower volatility preferred)
        ], ascending=[False, True])
        
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def generate_signals(self, stock_data: pd.DataFrame, index_data: pd.DataFrame,
                        signal_date: datetime) -> Tuple[List[str], List[str]]:
        """Generate buy and sell signals for a given date"""
        # No price filtering - all stocks eligible for RS ranking
        rankings = self.rank_stocks(stock_data, index_data, signal_date)
        
        if rankings.empty:
            self.logger.performance("  No rankings available, returning empty signals")
            return [], []
        
        # Select top N stocks with positive RS scores (more opportunities)
        positive_rs = rankings[rankings['rs_score'] > 0]
        self.logger.info(f"  Stocks with positive RS: {len(positive_rs)}")
        
        # Select top positions for better diversification
        top_stocks = positive_rs.head(self.dynamic_params.max_positions)
        target_symbols = top_stocks['symbol'].tolist()
        self.logger.info(f"  Target symbols: {target_symbols}")
        
        # Current positions
        current_symbols = list(self.positions.keys())
        self.logger.info(f"  Current positions: {current_symbols}")
        
        # Determine exits (stocks not in targets)
        exits = []
        for symbol in current_symbols:
            if symbol not in target_symbols:
                exits.append(symbol)
                self.logger.info(f"    Exit signal: {symbol} (not in targets)")
        
        # Determine entries (new stocks to buy)
        entries = []
        for symbol in target_symbols:
            if symbol not in current_symbols:
                entries.append(symbol)
                self.logger.info(f"    Entry signal: {symbol}")
        
        self.logger.info(f"  Final signals: {len(entries)} entries, {len(exits)} exits")
        return entries, exits
    

    def generate_signals_vectorized(self, stock_data: pd.DataFrame, index_data: pd.DataFrame,
                                    date: datetime, rs_scores_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Generate entry and exit signals using pre-calculated RS scores (vectorized)
        
        This is 20-30x faster than the original generate_signals method
        """
        # DEBUG LOGGING: Print detailed RS breakdown for all stocks on this date
        # Matching user's requested "Momentum Calculation" format
        if date in rs_scores_df.index:
            date_scores = rs_scores_df.loc[date].dropna().sort_values(ascending=False)
            
            # Retrieve index price safely
            index_price = 0.0
            try:
                if isinstance(index_data, pd.DataFrame):
                    # Check if date is in columns or already index
                    if 'date' in index_data.columns:
                        temp_idx_df = index_data.set_index('date')
                    else:
                        temp_idx_df = index_data
                        
                    if 'adjusted_close' in temp_idx_df.columns:
                        idx_series = temp_idx_df['adjusted_close']
                    elif 'adj_close' in temp_idx_df.columns:
                        idx_series = temp_idx_df['adj_close']
                    elif 'close' in temp_idx_df.columns:
                        idx_series = temp_idx_df['close']
                    else:
                        idx_series = temp_idx_df.iloc[:, 0] # Fallback
                    
                    if date in idx_series.index:
                         index_price = float(idx_series.loc[date])
                else:
                    # Assuming series
                    if date in index_data.index:
                        index_price = float(index_data.loc[date])
            except:
                pass

            self.logger.progress(f"📊 RS Momentum Calculation for {date.strftime('%Y-%m-%d')}:")
            self.logger.info(f"   Note: Using Relative Strength (RS) score instead of 52-week High/Low distance")
            self.logger.info(f"   Benchmark Index Price: ₹{index_price:.2f}")
            self.logger.progress(f"   ✅ Stocks Ranked by RS Score (Equal Weight) - Highest to Lowest:")
            self.logger.performance(f"   Note: Breakdown values show Outperformance (Excess Return vs Index)")
            self.logger.performance(f"         Week: 1-week return (5 days) outperformance")
            self.logger.performance(f"         Month: 1-month return (20 days) outperformance")
            self.logger.performance(f"         Quarter: 1-quarter return (60 days) outperformance")
            
            rank = 1
            for symbol, score in date_scores.items():
                try:
                    # Get component RS scores
                    w_score = self.rs_w_df.loc[date, symbol] if hasattr(self, 'rs_w_df') and date in self.rs_w_df.index else 0
                    m_score = self.rs_m_df.loc[date, symbol] if hasattr(self, 'rs_m_df') and date in self.rs_m_df.index else 0
                    q_score = self.rs_q_df.loc[date, symbol] if hasattr(self, 'rs_q_df') and date in self.rs_q_df.index else 0
                    
                    # Get stock price safely
                    stock_price = 0.0
                    try:
                        if isinstance(stock_data.index, pd.MultiIndex):
                             if (symbol, date) in stock_data.index:
                                stock_price = float(stock_data.loc[(symbol, date), 'adj_close'])
                        else:
                            # Flat df
                            mask = (stock_data['date'] == date) & (stock_data['symbol'] == symbol)
                            if mask.any():
                                stock_price = float(stock_data.loc[mask, 'adjusted_close'].iloc[0])
                    except:
                        pass
                    

                    # Print score and breakdown
                    self.logger.info(f"      {rank}. {symbol}: RS Score={score:.4f}, Price=₹{stock_price:.2f}")
                    self.logger.info(f"         Breakdown (Outperf): Week (5d)={w_score:.3f}, Month (20d)={m_score:.3f}, Quarter (60d)={q_score:.3f}")
                    rank += 1
                    
                except Exception:
                    continue
        
        # Use vectorized method to get top stocks
        top_stocks = self.get_top_rs_stocks_vectorized(date, rs_scores_df, n=self.max_positions)
        
        if not top_stocks:
            return [], []
        
        # Extract just the symbols
        top_symbols = [symbol for symbol, score in top_stocks]
        
        # Determine entries and exits
        current_positions = set(self.positions.keys())
        target_positions = set(top_symbols)
        
        entries = list(target_positions - current_positions)
        exits = list(current_positions - target_positions)
        
        return entries, exits

    def calculate_transaction_costs(self, transaction_value: float, action: str, brokerage_pct: float = None):
        """Calculate detailed Indian market transaction costs"""
        if brokerage_pct is None:
            brokerage_pct = self.transaction_cost_pct * 100  # Convert to percentage
        
        # Base brokerage
        brokerage = transaction_value * (brokerage_pct / 100)
        
        # Indian market specific costs
        if action == "BUY":
            stamp_duty = transaction_value * 0.015 / 100 # 0.015%
            stt = transaction_value * 0.10 / 100 # 0.1%
        else:  # SELL
            stamp_duty = 0  # No stamp duty on sell
            stt = transaction_value * 0.10 / 100  # 0.1%
        
        # Common costs
        exchange_charges = (transaction_value * 0.00297) * 0.18  # 0.00345%
        sebi_charges = (transaction_value * 0.0001) * 0.18  # 0.0001%
        gst = brokerage * 0.18  # 18% on brokerage
        
        total_costs = brokerage + stamp_duty + stt + exchange_charges + sebi_charges + gst
        
        return {
            'transaction_value': transaction_value,
            'brokerage': brokerage,
            'stamp_duty': stamp_duty,
            'stt': stt,
            'exchange_charges': exchange_charges,
            'sebi_charges': sebi_charges,
            'gst': gst,
            'total_costs': total_costs,
            'net_amount': transaction_value + total_costs if action == "BUY" else transaction_value - total_costs
        }

    
    def update_positions(self, stock_data: pd.DataFrame, date: datetime):
        """Update current positions with latest prices"""
        for symbol, position in self.positions.items():
            try:
                current_price = stock_data.loc[symbol, date]['adjusted_close']
                position.current_price = current_price
                position.unrealized_pnl = (current_price - position.buy_price) * position.quantity
            except (KeyError, IndexError):
                continue
    
    def calculate_portfolio_value(self, stock_data: pd.DataFrame, date: datetime) -> float:
        """Calculate total portfolio value"""
        self.update_positions(stock_data, date)
        
        positions_value = sum(pos.current_price * pos.quantity for pos in self.positions.values())
        return self.cash_balance + positions_value
    
    def take_portfolio_snapshot(self, stock_data: pd.DataFrame, index_data: pd.DataFrame, date: datetime):
        """Take a snapshot of portfolio state with memory optimization"""
        total_value = self.calculate_portfolio_value(stock_data, date)
        
        # For long backtests, store simplified snapshots to save memory
        # Only store detailed positions for weekly snapshots
        store_positions = self.is_friday_or_last_trading_day(date) or len(self.portfolio_snapshots) < 100
        
        positions_data = {}
        if store_positions:
            for symbol, position in self.positions.items():
                positions_data[symbol] = {
                    'quantity': position.quantity,
                    'buy_price': position.buy_price,
                    'current_price': position.current_price,
                    'unrealized_pnl': position.unrealized_pnl
                }
        
        # Calculate daily P&L (simplified)
        if self.portfolio_snapshots:
            prev_value = self.portfolio_snapshots[-1]['total_value']
            daily_pnl = total_value - prev_value
            cumulative_pnl = self.portfolio_snapshots[-1]['cumulative_pnl'] + daily_pnl
        else:
            daily_pnl = 0
            cumulative_pnl = 0
        
        # Calculate drawdown efficiently
        if self.portfolio_snapshots:
            # Keep track of peak value to avoid recalculating every time
            if not hasattr(self, '_peak_value'):
                self._peak_value = max(snap['total_value'] for snap in self.portfolio_snapshots)
            self._peak_value = max(self._peak_value, total_value)
            drawdown_pct = safe_float(safe_divide((self._peak_value - total_value) * 100, self._peak_value, 0.0)) if self._peak_value > 0 else 0.0
        else:
            self._peak_value = total_value
            drawdown_pct = 0
        
        snapshot = {
            'date': date,
            'total_value': total_value,
            'cash_balance': self.cash_balance,
            'daily_pnl': daily_pnl,
            'cumulative_pnl': cumulative_pnl,
            'drawdown_pct': drawdown_pct
        }
        
        # Only include positions if storing detailed snapshot
        if positions_data:
            snapshot['positions'] = positions_data
        
        self.portfolio_snapshots.append(snapshot)
        
        # For very long backtests, periodically clean old snapshots to save memory
        if len(self.portfolio_snapshots) > 10000:  # Keep last 10k snapshots max
            self.portfolio_snapshots = self.portfolio_snapshots[-5000:]  # Keep last 5k
    
    # def run_backtest(self, start_date: datetime, end_date: datetime) -> Dict:
    #     """Run the complete backtest"""
    #     # Ensure dates are timezone-naive
    #     if start_date.tzinfo:
    #         start_date = start_date.replace(tzinfo=None)
    #     if end_date.tzinfo:
    #         end_date = end_date.replace(tzinfo=None)
            
        
    #     # Validate data range before starting
    #     if not self.validate_data_range(start_date, end_date):
        
    #     # Load data
    #     stock_data = self.load_stock_data(start_date, end_date)
        
    #     index_data = self.load_index_data(start_date, end_date)
        
    #     if stock_data.empty or index_data.empty:
    #         raise ValueError("No data available for backtest period")
        
    #     # Get all trading dates (should be timezone-naive now)
    #     trading_dates = sorted(stock_data.index.get_level_values('date').unique())
        
    #     # Calculate last trading days of each week
    #     self.calculate_last_trading_days(trading_dates)
        
    #     signal_count = 0
    #     trade_count = 0
    #     total_dates = len(trading_dates)
    #     progress_interval = max(1, total_dates // 20)  # Show progress every 5%
        
        
    #     for i, date in enumerate(trading_dates):
    #         # Progress reporting for long backtests
    #         if i % progress_interval == 0 or i == total_dates - 1:
    #             progress_pct = (i / total_dates) * 100
    #         # Take portfolio snapshot
    #         self.take_portfolio_snapshot(stock_data, index_data, date)
            
    #         # Check capital reset threshold
    #         current_portfolio_value = self.calculate_portfolio_value(stock_data, date)
    #         self.check_capital_reset_threshold(current_portfolio_value, date)
            
    #         # Check max holding period exits
    #         max_holding_exits = self.check_max_holding_period(date)
            
    #         # Generate signals on last trading day of week
    #         if self.is_friday_or_last_trading_day(date):
    #             signal_count += 1
                
    #             # Print Friday signal date and Monday execution date
    #             next_monday = self.get_next_monday(date)
                
    #             entries, exits = self.generate_signals_vectorized(stock_data, index_data, date, rs_scores_df)
                
    #             # Add max holding period exits
    #             exits.extend(max_holding_exits)
                
    #             # Apply capital reset logic
    #             entries, exits = self.apply_capital_reset_logic(entries, exits)
                
    #             # Execute trades on next Monday
    #             if next_monday <= end_date:
    #                 # Execute exits
    #                 for symbol in exits:
    #                     try:
    #                         price = stock_data.loc[symbol, next_monday]['adjusted_close']
    #                         self.execute_trade(next_monday, symbol, "SELL", price, "Exit Signal")
    #                         trade_count += 1
    #                     except (KeyError, IndexError):
    #                         continue
                    
    #                 # Execute entries
    #                 rankings = self.rank_stocks(stock_data, index_data, date)  # Calculate once per signal
    #                 for symbol in entries:
    #                     try:
    #                         price = stock_data.loc[symbol, next_monday]['adjusted_close']
    #                         if not rankings.empty and symbol in rankings['symbol'].values:
    #                             symbol_rank = rankings[rankings['symbol'] == symbol]
    #                             rs_score = symbol_rank['rs_score'].iloc[0]
    #                             rs_rank = symbol_rank['rank'].iloc[0]
    #                             self.execute_trade(next_monday, symbol, "BUY", price, "Entry Signal", 
    #                                              rs_score, rs_rank)
    #                             trade_count += 1
    #                     except (KeyError, IndexError):
    #                         continue
        
        
    #     # Calculate final metrics
    #     metrics = self.calculate_backtest_metrics()
    #     return metrics
    
    # def calculate_cagr(self, start_value: float, end_value: float, start_date: datetime, end_date: datetime) -> float:
    #     """Calculate Compound Annual Growth Rate (CAGR)"""
    #     if start_value <= 0 or end_value <= 0:
    #         return 0.0
        
    #     # Calculate years between dates
    #     years = (end_date - start_date).days / 365.25
        
    #     if years <= 0:
    #         return 0.0
        
    #     # CAGR formula: (End Value / Start Value)^(1/years) - 1 - SAFE VERSION
    #     if years > 0 and start_value > 0:
    #         cagr = safe_power(end_value / start_value, 1 / years, 1.0) - 1
    #     else:
    #         cagr = 0.0
    #     return safe_float(cagr * 100)  # Return as percentage
    
    # def calculate_rule_of_72_metrics(self, cagr_pct: float, years: float) -> Dict:
    #     """Calculate Rule of 72 metrics for compounding analysis"""
    #     if cagr_pct <= 0:
    #         return {
    #             'years_to_double': float('inf'),
    #             'expected_doublings': 0.0,
    #             'rule_of_72_return': 0.0,
    #             'compounding_factor': 1.0
    #         }
        
    #     # Rule of 72: Years to double = 72 / Annual Rate - SAFE VERSION
    #     years_to_double = safe_divide(72, cagr_pct, 0.0)
        
    #     # Calculate how many times the investment should double
    #     expected_doublings = safe_divide(years, years_to_double, 0.0)
        
    #     # Calculate expected return using Rule of 72
    #     # Each doubling = 2x, so 2^doublings
    #     compounding_factor = safe_power(2, expected_doublings, 1.0)
    #     rule_of_72_return = safe_float((compounding_factor - 1) * 100)
        
    #     return {
    #         'years_to_double': years_to_double,
    #         'expected_doublings': expected_doublings,
    #         'rule_of_72_return': rule_of_72_return,
    #         'compounding_factor': compounding_factor
    #     }
    
    # def calculate_xirr(self, cash_flows: List[Tuple[datetime, float]]) -> float:
    #     """Calculate Extended Internal Rate of Return (XIRR) using Newton-Raphson method"""
    #     if len(cash_flows) < 2:
    #         return 0.0
        
    #     # Sort cash flows by date
    #     cash_flows.sort(key=lambda x: x[0])
        
    #     # Extract dates and amounts
    #     dates = [cf[0] for cf in cash_flows]
    #     amounts = [cf[1] for cf in cash_flows]
        
    #     # Convert dates to years from first date
    #     first_date = dates[0]
    #     years = [(date - first_date).days / 365.25 for date in dates]
        
    #     try:
    #         # Simple IRR calculation using Newton-Raphson method
    #         def npv(rate):
    #             return sum(amount / (1 + rate) ** year for amount, year in zip(amounts, years))
            
    #         def npv_derivative(rate):
    #             return sum(-amount * year / (1 + rate) ** (year + 1) for amount, year in zip(amounts, years))
            
    #         # Initial guess
    #         rate = 0.1
            
    #         # Newton-Raphson iteration
    #         for _ in range(100):  # Max 100 iterations
    #             npv_val = npv(rate)
    #             if abs(npv_val) < 1e-6:  # Convergence
    #                 break
    #             derivative = npv_derivative(rate)
    #             if abs(derivative) < 1e-10:  # Avoid division by zero
    #                 break
    #             rate = rate - npv_val / derivative
            
    #         # Check if result is reasonable
    #         if np.isnan(rate) or np.isinf(rate) or rate < -0.99 or rate > 10:
    #             return 0.0
            
    #         return rate * 100  # Return as percentage
    #     except:
    #         return 0.0
    
    # def calculate_backtest_metrics(self) -> Dict:
    #     """Calculate backtest performance metrics"""
        
    #     if not self.portfolio_snapshots:
    #         return {}
        
    #     snapshots = pd.DataFrame(self.portfolio_snapshots)
    #     snapshots['returns'] = snapshots['total_value'].pct_change()
        
        
    #     # Basic metrics - SAFE VERSION
    #     start_val = snapshots['total_value'].iloc[0]
    #     end_val = snapshots['total_value'].iloc[-1]
    #     total_return = safe_float((safe_divide(end_val, start_val, 1.0) - 1) * 100)
        
    #     # Calculate CAGR
    #     start_date = snapshots['date'].iloc[0]
    #     end_date = snapshots['date'].iloc[-1]
    #     start_value = snapshots['total_value'].iloc[0]
    #     end_value = snapshots['total_value'].iloc[-1]
    #     cagr = self.calculate_cagr(start_value, end_value, start_date, end_date)
        
    #     # Calculate Rule of 72 metrics
    #     years = (end_date - start_date).days / 365.25
    #     rule_72_metrics = self.calculate_rule_of_72_metrics(cagr, years)
        
    #     # Calculate XIRR from cash flows
    #     cash_flows = []
    #     # Initial investment (negative cash flow)
    #     cash_flows.append((start_date, -self.total_capital))
        
    #     # Add trade cash flows
    #     for trade in self.trades:
    #         if trade.action == "BUY":
    #             cash_flows.append((trade.date, -trade.amount))  # Negative for outflow
    #         elif trade.action == "SELL":
    #             cash_flows.append((trade.date, trade.amount))   # Positive for inflow
        
    #     # Final portfolio value (positive cash flow)
    #     cash_flows.append((end_date, end_value))
        
    #     xirr = self.calculate_xirr(cash_flows)
        
    #     # Annualized return (legacy calculation) - SAFE VERSION
    #     days = (end_date - start_date).days
    #     if days > 0 and start_value > 0:
    #         annualized_return = safe_float(((end_value / start_value) ** (365 / days) - 1) * 100)
    #     else:
    #         annualized_return = 0.0
        
    #     # Max drawdown - SAFE VERSION
    #     peak = snapshots['total_value'].expanding().max()
    #     drawdown = (snapshots['total_value'] - peak) / peak * 100
    #     # Handle division by zero and infinity values
    #     drawdown = drawdown.replace([float('inf'), float('-inf')], 0.0).fillna(0.0)
    #     max_drawdown = safe_float(drawdown.min())
        
    #     # Sharpe ratio (simplified) - SAFE VERSION
    #     returns_std = snapshots['returns'].std()
    #     returns_mean = snapshots['returns'].mean()
    #     if returns_std > 0 and not math.isnan(returns_std) and not math.isinf(returns_std):
    #         sharpe_ratio = safe_float(returns_mean / returns_std * safe_sqrt(252))
    #     else:
    #         sharpe_ratio = 0.0
        
    #     # Win rate
    #     winning_trades = len([t for t in self.trades if t.action == "SELL" and t.amount > 0])
    #     total_sell_trades = len([t for t in self.trades if t.action == "SELL"])
    #     win_rate = (winning_trades / total_sell_trades * 100) if total_sell_trades > 0 else 0
        
    #     # Smart snapshot sampling for long backtests
    #     total_snapshots = len(self.portfolio_snapshots)
    #     if total_snapshots <= 500:
    #         # For short backtests, keep all snapshots
    #         snapshot_sample = self.portfolio_snapshots
    #     else:
    #         # For long backtests, sample intelligently
    #         # Keep first 50, last 50, and sample middle evenly
    #         step = max(1, (total_snapshots - 100) // 400)  # Sample ~400 from middle
    #         indices = (list(range(50)) + 
    #                   list(range(50, total_snapshots - 50, step)) + 
    #                   list(range(total_snapshots - 50, total_snapshots)))
    #         snapshot_sample = [self.portfolio_snapshots[i] for i in indices if i < total_snapshots]
        
    #     simplified_snapshots = []
    #     for snapshot in snapshot_sample:
    #         simplified_snapshot = {
    #             'date': snapshot['date'].isoformat() if hasattr(snapshot['date'], 'isoformat') else str(snapshot['date']),
    #             'total_value': float(snapshot.get('total_value', 0)),
    #             'cash_balance': float(snapshot.get('cash_balance', 0)),
    #             'daily_pnl': float(snapshot.get('daily_pnl', 0)),
    #             'cumulative_pnl': float(snapshot.get('cumulative_pnl', 0)),
    #             'drawdown_pct': float(snapshot.get('drawdown_pct', 0)),
    #             'position_count': len(snapshot.get('positions', {}))
    #         }
    #         simplified_snapshots.append(simplified_snapshot)
        
    #     metrics = {
    #         'total_return_pct': safe_float(total_return),
    #         'annualized_return_pct': safe_float(annualized_return),
    #         'cagr_pct': safe_float(cagr),
    #         'xirr_pct': safe_float(xirr),
    #         'max_drawdown_pct': safe_float(max_drawdown),
    #         'sharpe_ratio': safe_float(sharpe_ratio),
    #         'win_rate_pct': safe_float(win_rate),
    #         'total_trades': int(len(self.trades)),
    #         'final_capital': safe_float(snapshots['total_value'].iloc[-1]),
    #         'portfolio_snapshots': simplified_snapshots,
    #         'trades': [self.trade_to_dict(t) for t in self.trades],
    #         'rule_of_72': {
    #             'years_to_double': safe_float(rule_72_metrics['years_to_double']),
    #             'expected_doublings': safe_float(rule_72_metrics['expected_doublings']),
    #             'rule_of_72_return_pct': safe_float(rule_72_metrics['rule_of_72_return']),
    #             'compounding_factor': safe_float(rule_72_metrics['compounding_factor'])
    #         }
    #     }
        
    #     return metrics
    

    def convert_to_json_safe(self, obj):
        """Recursively convert any object to JSON-safe format"""
        if obj is None:
            return None
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_to_json_safe(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.convert_to_json_safe(item) for item in obj]
        elif hasattr(obj, '__dict__'):  # Custom objects
            return self.convert_to_json_safe(obj.__dict__)
        else:
            return obj
    
    def convert_snapshots_to_json_safe(self, snapshots: List[Dict]) -> List[Dict]:
        """Convert portfolio snapshots to JSON-safe format"""
        return self.convert_to_json_safe(snapshots)
    
    def trade_to_dict(self, trade: Trade) -> Dict:
        """Convert Trade object to dictionary"""
        trade_dict = {
            'date': trade.date.isoformat(),
            'symbol': trade.symbol,
            'action': trade.action,
            'quantity': trade.quantity,
            'price': trade.price,
            'amount': trade.amount,
            'reason': trade.reason,
            'rs_score': trade.rs_score,
            'rs_rank': trade.rs_rank,
            
            # Transaction cost fields
            'transaction_value': trade.transaction_value,
            'brokerage': trade.brokerage,
            'stt': trade.stt,
            'stamp_duty': trade.stamp_duty,
            'exchange_charges': trade.exchange_charges,
            'sebi_charges': trade.sebi_charges,
            'gst': trade.gst,
            'total_costs': trade.total_costs,
            'net_amount': trade.net_amount,
            
            # NAV and Capital Gains fields
            'portfolio_nav': trade.portfolio_nav,
            'buy_price': trade.buy_price,
            'capital_gain': trade.capital_gain,
            'capital_gain_pct': trade.capital_gain_pct,
            'holding_period_days': trade.holding_period_days,
            'capital_gains_tax': trade.capital_gains_tax,
            'net_profit_after_tax': trade.net_profit_after_tax
        }
        # Apply comprehensive JSON-safe conversion
        return self.convert_to_json_safe(trade_dict)
    
    def check_capital_reset_threshold(self, current_portfolio_value: float, current_date: datetime) -> bool:
        """Check if capital reset threshold is triggered and manage the reset process"""
        try:
            # Update peak value
            if current_portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = current_portfolio_value
                # If we're in reset mode and recovered, exit reset mode
                if self.is_capital_reset_active:
                    recovery_threshold = self.peak_portfolio_value * 0.8  # 20% recovery from peak
                    if current_portfolio_value >= recovery_threshold:
                        self.logger.info(f"  Capital reset deactivated - portfolio recovered to {current_portfolio_value:.0f}")
                        self.is_capital_reset_active = False
                        self.capital_reset_start_date = None
                        return False
            
            # Check if we need to trigger capital reset
            if not self.is_capital_reset_active:
                drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
                if drawdown >= self.capital_reset_threshold_pct:
                    self.logger.performance(f"  CAPITAL RESET TRIGGERED: Drawdown {drawdown:.1%} >= {self.capital_reset_threshold_pct:.1%}")
                    self.is_capital_reset_active = True
                    self.capital_reset_start_date = current_date
                    return True
            
            return self.is_capital_reset_active
            
        except Exception as e:
            self.logger.info(f"Error in capital reset check: {e}")
            return False
    
    def apply_capital_reset_logic(self, entries: List[str], exits: List[str]) -> Tuple[List[str], List[str]]:
        """Apply capital reset logic by reducing positions and being more conservative"""
        if not self.is_capital_reset_active:
            return entries, exits
        
        self.logger.info(f"⚠️  CAPITAL RESET ACTIVE - Reducing risk exposure")
        # Reduce entries by 50%
        reduced_entries = entries[:len(entries)//2] if len(entries) > 1 else []
        
        # Add more exits to reduce portfolio size
        additional_exits = []
        if len(self.positions) > 5:  # If we have more than 5 positions
            # Exit positions with lowest RS scores
            current_positions = list(self.positions.keys())
            additional_exits = current_positions[5:]  # Keep only top 5 positions
        
        all_exits = exits + additional_exits
        
        self.logger.info(f"   Entries: {len(entries)} → {len(reduced_entries)} | Exits: {len(exits)} → {len(all_exits)}")
        
        return reduced_entries, all_exits
    
    def check_max_holding_period(self, current_date: datetime) -> List[str]:
        """Check for positions that have exceeded max holding period"""
        exits = []
        
        for symbol, position in self.positions.items():
            try:
                holding_weeks = (current_date - position.buy_date).days / 7
                if holding_weeks >= self.max_holding_period:
                    exits.append(symbol)
                    self.logger.info(f"    Max holding period exit: {symbol} (held {holding_weeks:.1f} weeks)")
            except Exception as e:
                self.logger.info(f"Error checking holding period for {symbol}: {e}")
                continue
        
        return exits
    
    def apply_minimum_price_filter(self, stock_data: pd.DataFrame, signal_date: datetime) -> List[str]:
        """Filter out stocks below minimum price threshold"""
        filtered_symbols = []
        
        try:
            for symbol in stock_data.index.get_level_values('symbol').unique():
                try:
                    current_price = stock_data.loc[symbol, signal_date]['adjusted_close']
                    if current_price >= (self.min_price * 0.5):  # Relaxed minimum price filter
                        filtered_symbols.append(symbol)
                except (KeyError, IndexError):
                    continue
        except Exception as e:
            self.logger.info(f"Error applying minimum price filter: {e}")
            return list(stock_data.index.get_level_values('symbol').unique())
        
        self.logger.info(f"  Price filter: {len(stock_data.index.get_level_values('symbol').unique())} -> {len(filtered_symbols)} stocks (min price: ₹{self.min_price})")
        return filtered_symbols
     
    def run_backtest(self, start_date: datetime, end_date: datetime) -> Dict:
        """Run the complete backtest"""
        try:
             # Ensure dates are timezone-naive
            if start_date.tzinfo:
                start_date = start_date.replace(tzinfo=None)
            if end_date.tzinfo:
                end_date = end_date.replace(tzinfo=None)
            
            self.logger.info(f"=== BACKTEST START ===")
            self.logger.info(f"Starting backtest from {start_date} to {end_date}")
            self.logger.info(f"Configuration ID: {self.config.id if self.config else 'Custom Config'}")
            self.logger.info(f"Total Capital: {self.total_capital}")
            self.logger.info(f"Max Positions: {self.max_positions}")
            self.logger.info(f"Position Size %: {self.position_size_pct}")
            
            # Validate date range
            try:
                if start_date >= end_date:
                    raise ValueError("Start date must be before end date")
                if (end_date - start_date).days < 30:
                    raise ValueError("Backtest period must be at least 30 days")
            except Exception as e:
                self.logger.info(f"Error validating data range: {e}")
                raise
            
            # Load data
            self.logger.progress("Loading stock data...")
            stock_data = self.load_stock_data(start_date, end_date)
            if stock_data.empty:
                raise ValueError("No stock data available for the specified date range")
            
            self.logger.progress("Loading index data...")
            index_data = self.load_index_data(start_date, end_date)
            if index_data.empty:
                raise ValueError("No index data available for the specified date range")
            
            # Store index data for beta calculation
            self.index_data = index_data
            
            # Get trading dates - filter to backtest period only for processing
            all_trading_dates = sorted(stock_data.index.get_level_values('date').unique())
            trading_dates = [d for d in all_trading_dates if start_date <= d <= end_date]
            
            self.logger.info(f"Total loaded dates: {len(all_trading_dates)} (from {all_trading_dates[0]} to {all_trading_dates[-1]})")
            self.logger.info(f"Backtest trading dates: {len(trading_dates)} (from {trading_dates[0]} to {trading_dates[-1]})")
            
            # Calculate last trading days of each week for Friday signal generation
            self.calculate_last_trading_days(trading_dates)
            self.logger.info(f"Last trading days calculated: {len(self.last_trading_days)} days")
            
            # Initialize portfolio snapshot
            self.portfolio_snapshots = []
            self.trades = []
            self.positions = {}
            # Reset buffer and cash to initial values
            self.buffer_capital = self.total_capital * self.buffer_capital_pct
            self.cash_balance = self.total_capital - self.buffer_capital
            
            
            # Calculate RS scores (Vectorized) - CRITICAL FIX for zero trades issue
            self.logger.progress("Calculating RS scores...")
            rs_scores_df = None  # Initialize
            try:
                stock_data_flat = stock_data.reset_index()
                
                # Prepare index data for vectorization
                if isinstance(index_data.index, pd.DatetimeIndex):
                    index_data_flat = index_data.reset_index()
                    if 'index' in index_data_flat.columns:
                        index_data_flat = index_data_flat.rename(columns={'index': 'date'})
                else:
                    index_data_flat = index_data
                
                rs_scores_df = self.calculate_rs_scores_vectorized(stock_data_flat, index_data_flat)
                self.logger.progress(f"✅ Pre-calculated RS scores for {rs_scores_df.shape[0]} dates and {rs_scores_df.shape[1]} stocks")
            except Exception as e:
                self.logger.info(f"⚠️  Vectorized RS calculation failed: {e}")
                self.logger.info("   Falling back to per-date calculation")
                rs_scores_df = None
            
            # Run backtest
            self.logger.progress(f"Processing {len(trading_dates)} trading dates...")
            signal_count = 0
            
            for i, date in enumerate(trading_dates):
                if i % 20 == 0:  # Progress every 20 days
                    progress = (i / len(trading_dates)) * 100
                    self.logger.info(f"Progress: {progress:.0f}% ({i}/{len(trading_dates)}) - {date.strftime('%Y-%m-%d')}")
                
                try:
                    # Update positions with current prices
                    self.update_positions(stock_data, date)
                    
                    # Check for capital reset threshold
                    current_portfolio_value = self.calculate_portfolio_value(stock_data, date)
                    self.check_capital_reset_threshold(current_portfolio_value, date)
                    
                    # Removed max holding period check - stocks held until RS ranking drops
                    
                    # Stop Loss Check - Conditional based on configuration
                    if self.daily_stop_loss_check:
                        # DAILY MODE: Check stop loss every day and execute immediately
                        stop_loss_exits = self.check_daily_stop_loss(stock_data, date)
                        
                        # Execute stop loss exits immediately
                        for symbol in stop_loss_exits:
                            try:
                                price_data = stock_data.loc[symbol, date]['adjusted_close']
                                price = float(price_data.iloc[0]) if hasattr(price_data, 'iloc') else float(price_data)
                                self.execute_trade(date, symbol, "SELL", price, "Stop Loss (Daily)")
                            except (KeyError, IndexError):
                                continue
                    else:
                        # WEEKLY MODE: Check stop loss daily but accumulate for weekly execution
                        # Only check on non-signal days (signal day will check separately)
                        if not self.is_friday_or_last_trading_day(date):
                            stop_loss_exits = self.check_daily_stop_loss(stock_data, date)
                            # Accumulate stop loss exits for Monday execution
                            for symbol in stop_loss_exits:
                                if symbol not in self.weekly_stop_loss_exits:
                                    self.weekly_stop_loss_exits.append(symbol)
                    
                    # Initialize entries and exits for each day
                    entries, exits = [], []
                    
                    # Generate signals only on Friday (or last trading day of week)
                    if self.is_friday_or_last_trading_day(date):  # Generate signals only on Friday
                        
                        # WEEKLY MODE: Check stop loss on signal day and add to exits
                        if not self.daily_stop_loss_check:
                            stop_loss_exits = self.check_daily_stop_loss(stock_data, date)
                            # Combine weekly accumulated stop losses with current check
                            all_stop_loss_exits = list(set(self.weekly_stop_loss_exits + stop_loss_exits))
                            if all_stop_loss_exits:
                                self.logger.info(f"  📊 Weekly Stop Loss Summary: {len(all_stop_loss_exits)} position(s) to exit")
                                for symbol in all_stop_loss_exits:
                                    self.logger.info(f"    - {symbol}")
                        
                        entries, exits = self.generate_signals_vectorized(stock_data, index_data, date, rs_scores_df)
                        
                        # WEEKLY MODE: Add stop loss exits to regular exits
                        if not self.daily_stop_loss_check and all_stop_loss_exits:
                            # Combine stop loss exits with RS signal exits (avoid duplicates)
                            exits = list(set(exits + all_stop_loss_exits))
                            self.logger.info(f"  Combined exits: {len(exits)} total (RS signals + stop loss)")
                        
                        # Apply capital reset logic
                        entries, exits = self.apply_capital_reset_logic(entries, exits)
                        
                        # Removed max holding period exits - stocks only exit based on RS ranking
                        
                        # Store signals for Monday execution (don't execute immediately)
                        self.pending_entries = entries
                        self.pending_exits = exits
                        self.signal_date = date
                        
                        # Reset weekly stop loss accumulator
                        if not self.daily_stop_loss_check:
                            self.weekly_stop_loss_exits = []
                        
                        self.logger.info(f"\n[{date.strftime('%Y-%m-%d')}] SIGNAL (Fri) → {len(entries)} entries, {len(exits)} exits")
                    
                    # Execute trades on Monday (or next available trading day if Monday is holiday)
                    if hasattr(self, 'pending_entries') and self.pending_entries is not None:
                        # Check if this is the execution day (Monday after signal Friday)
                        next_monday = self.get_next_monday(self.signal_date)
                        if next_monday in trading_dates:
                            execution_day = next_monday
                        else:
                            # Find next available trading day after Monday if Monday is holiday
                            execution_day = self.find_next_available_trading_day(next_monday, trading_dates)
                        
                        if date == execution_day:
                            self.logger.info(f"[{date.strftime('%Y-%m-%d')}] EXECUTE (Mon) → {len(self.pending_exits)} exits, {len(self.pending_entries)} entries")
                            
                            # SELL FIRST to free up cash before buying new positions
                            for symbol in self.pending_exits:
                                try:
                                    # Use OPEN price for execution as per request
                                    if 'open' in stock_data.columns:
                                        price_data = stock_data.loc[symbol, execution_day]['open']
                                    else:
                                        price_data = stock_data.loc[symbol, execution_day]['adjusted_close']
                                        
                                    price = float(price_data.iloc[0]) if hasattr(price_data, 'iloc') else float(price_data)
                                    self.execute_trade(execution_day, symbol, "SELL", price, "RS Exit")
                                except (KeyError, IndexError):
                                    continue
                            
                            # THEN BUY using freed cash (and buffer only if needed)
                            for symbol in self.pending_entries:
                                try:
                                    # Use OPEN price for execution as per request
                                    if 'open' in stock_data.columns:
                                        price_data = stock_data.loc[symbol, execution_day]['open']
                                    else:
                                        price_data = stock_data.loc[symbol, execution_day]['adjusted_close']
                                        
                                    price = float(price_data.iloc[0]) if hasattr(price_data, 'iloc') else float(price_data)
                                    self.execute_trade(execution_day, symbol, "BUY", price, "RS Signal")
                                except (KeyError, IndexError):
                                    continue
                            
                            # Clear pending signals after execution
                            self.pending_entries = None
                            self.pending_exits = None
                    
                    # Record portfolio snapshot
                    portfolio_value = self.calculate_portfolio_value(stock_data, date)
                    snapshot = {
                        'date': date,
                        'total_value': portfolio_value,
                        'cash_balance': self.cash_balance,
                        'positions': {symbol: {
                            'quantity': pos.quantity,
                            'buy_price': pos.buy_price,
                            'current_price': pos.current_price,
                            'unrealized_pnl': pos.unrealized_pnl
                        } for symbol, pos in self.positions.items()},
                        'daily_pnl': 0,  # Will be calculated later
                        'cumulative_pnl': portfolio_value - self.total_capital,
                        'drawdown_pct': 0  # Will be calculated later
                    }
                    self.portfolio_snapshots.append(snapshot)
                    
                    # Weekly Summary Log (matching user request)
                    if self.is_friday_or_last_trading_day(date):
                        week_num = (i // 5) + 1
                        holdings_summary = [(s, p.quantity) for s, p in self.positions.items()]
                        self.logger.progress(f"📊 Week {week_num} summary:")
                        self.logger.info(f"   Date: {date.strftime('%Y-%m-%d')}")
                        self.logger.info(f"   NAV: ₹{portfolio_value:,.2f}")
                        self.logger.info(f"   Cash: ₹{self.cash_balance:,.2f}")
                        self.logger.info(f"   Holdings Value: ₹{portfolio_value - self.cash_balance:,.2f}")
                        self.logger.info(f"   Holdings: {holdings_summary}")
                        self.logger.info("============================================================")
                    
                    if entries or exits:
                        signal_count += 1
                        
                except Exception as e:
                    self.logger.progress(f"Error processing date {date}: {e}")
                    continue
            
            self.logger.info(f"=== BACKTEST COMPLETE ===")
            self.logger.info(f"Total signal generations: {signal_count}")
            self.logger.trade(f"Total trades executed: {len(self.trades)}")
            self.logger.info(f"Final cash balance: {self.cash_balance:.1f}")
            self.logger.info(f"Final positions: {len(self.positions)}")
            
            # Calculate metrics (with default risk_free_rate, will be overridden by API)
            self.logger.progress("=== CALCULATING METRICS ===")
            metrics = self.calculate_metrics(risk_free_rate=6.0)
            
            return metrics
            
        except Exception as e:
            self.logger.info(f"Backtest failed: {e}")
            raise
    
    def calculate_portfolio_value(self, stock_data: pd.DataFrame, date: datetime) -> float:
        """Calculate total portfolio value at given date (including buffer capital)"""
        try:
            # Start with cash balance
            total_value = self.cash_balance
            
            # Add buffer capital
            total_value += self.buffer_capital
            
            # Add value of all positions
            for symbol, position in self.positions.items():
                try:
                    price_data = stock_data.loc[symbol, date]['adjusted_close']
                    current_price = float(price_data.iloc[0]) if hasattr(price_data, 'iloc') else float(price_data)
                    position.current_price = current_price
                    position.unrealized_pnl = (current_price - position.buy_price) * position.quantity
                    total_value += current_price * position.quantity
                except (KeyError, IndexError):
                    # Use last known price if current price not available
                    total_value += position.buy_price * position.quantity
            
            return total_value
        except Exception as e:
            self.logger.progress(f"Error calculating portfolio value: {e}")
            return self.cash_balance + self.buffer_capital
    
    def calculate_metrics(self, risk_free_rate: float = 6.0) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            if not self.portfolio_snapshots:
                self.logger.info("ERROR: No portfolio snapshots available")
                return {}
            
            snapshots = pd.DataFrame(self.portfolio_snapshots)
            snapshots['returns'] = snapshots['total_value'].pct_change()
            
            self.logger.info(f"Snapshot data shape: {snapshots.shape}")
            self.logger.info(f"First portfolio value: {snapshots['total_value'].iloc[0]}")
            self.logger.info(f"Last portfolio value: {snapshots['total_value'].iloc[-1]}")
            
            # Basic metrics - SAFE VERSION
            start_val = snapshots['total_value'].iloc[0]
            end_val = snapshots['total_value'].iloc[-1]
            total_return = safe_float((safe_divide(end_val, start_val, 1.0) - 1) * 100)
            self.logger.performance(f"Total return: {total_return}%")
            
            # Calculate CAGR
            start_date = snapshots['date'].iloc[0]
            end_date = snapshots['date'].iloc[-1]
            start_value = snapshots['total_value'].iloc[0]
            end_value = snapshots['total_value'].iloc[-1]
            cagr = self.calculate_cagr(start_value, end_value, start_date, end_date)
            self.logger.performance(f"CAGR: {cagr}%")
            
            # Calculate Rule of 72 metrics
            years = (end_date - start_date).days / 365.25
            rule_72_metrics = self.calculate_rule_of_72_metrics(cagr, years)
            self.logger.info(f"Rule of 72 - Years to double: {rule_72_metrics['years_to_double']:.1f}")
            self.logger.performance(f"Rule of 72 - Expected return: {rule_72_metrics['rule_of_72_return']:.1f}%")
            self.logger.info(f"Rule of 72 - Doublings: {rule_72_metrics['expected_doublings']:.2f}")
            
            # Calculate XIRR from cash flows
            cash_flows = []
            # Initial investment (negative cash flow)
            cash_flows.append((start_date, -self.total_capital))
            
            # Add trade cash flows
            for trade in self.trades:
                if trade.action == "BUY":
                    cash_flows.append((trade.date, -trade.amount))  # Negative for outflow
                elif trade.action == "SELL":
                    cash_flows.append((trade.date, trade.amount))   # Positive for inflow
            
            # Final portfolio value (positive cash flow)
            cash_flows.append((end_date, end_value))
            
            xirr = self.calculate_xirr(cash_flows)
            self.logger.info(f"XIRR: {xirr}%")
            
            # Calculate annualized return
            annualized_return = safe_float((safe_power(safe_divide(end_value, start_value, 1.0), safe_divide(365.25, (end_date - start_date).days, 1.0)) - 1) * 100)
            self.logger.performance(f"Annualized return: {annualized_return}% (over {(end_date - start_date).days} days)")
            
            # Calculate maximum drawdown
            max_drawdown = self.calculate_max_drawdown(snapshots)
            self.logger.performance(f"Max drawdown: {max_drawdown}%")
            
            # Calculate Sharpe ratio
            sharpe_ratio = self.calculate_sharpe_ratio(snapshots)
            self.logger.performance(f"Sharpe ratio: {sharpe_ratio}")
            
            # Calculate Beta and Treynor ratio
            beta, treynor_ratio = self.calculate_beta_and_treynor(snapshots, risk_free_rate)
            self.logger.info(f"Beta: {beta:.2f}")
            self.logger.info(f"Treynor ratio: {treynor_ratio:.2f}%")
            
            # Calculate Calmar ratio: abs(CAGR / max_drawdown) if max_drawdown < 0
            calmar_ratio = abs(cagr / max_drawdown) if max_drawdown < 0 else 0.0
            self.logger.info(f"Calmar ratio: {calmar_ratio:.2f}")
            
            # Calculate win rate
            win_rate = self.calculate_win_rate()
            self.logger.trade(f"Win rate: {win_rate}% ({len([t for t in self.trades if t.action == 'SELL' and self.get_trade_pnl(t) > 0])}/{len([t for t in self.trades if t.action == 'SELL'])})")
            
            # Convert snapshots to JSON-safe format
            simplified_snapshots = []
            for snapshot in self.portfolio_snapshots:
                simplified_snapshot = {
                    'date': snapshot['date'].isoformat() if hasattr(snapshot['date'], 'isoformat') else str(snapshot['date']),
                    'total_value': float(snapshot.get('total_value', 0)),
                    'cash_balance': float(snapshot.get('cash_balance', 0)),
                    'daily_pnl': float(snapshot.get('daily_pnl', 0)),
                    'cumulative_pnl': float(snapshot.get('cumulative_pnl', 0)),
                    'drawdown_pct': float(snapshot.get('drawdown_pct', 0)),
                    'position_count': len(snapshot.get('positions', {}))
                }
                simplified_snapshots.append(simplified_snapshot)
            
            metrics = {
                'total_return_pct': safe_float(total_return),
                'annualized_return_pct': safe_float(annualized_return),
                'cagr_pct': safe_float(cagr),
                'xirr_pct': safe_float(xirr),
                'max_drawdown_pct': safe_float(max_drawdown),
                'sharpe_ratio': safe_float(sharpe_ratio),
                'beta': safe_float(beta),
                'treynor_ratio': safe_float(treynor_ratio),
                'calmar_ratio': safe_float(calmar_ratio),
                'win_rate_pct': safe_float(win_rate),
                'total_trades': int(len(self.trades)),
                'final_capital': safe_float(snapshots['total_value'].iloc[-1]),
                'portfolio_snapshots': simplified_snapshots,
                'trades': [self.trade_to_dict(t) for t in self.trades],
                'rule_of_72': {
                    'years_to_double': safe_float(rule_72_metrics['years_to_double']),
                    'expected_doublings': safe_float(rule_72_metrics['expected_doublings']),
                    'rule_of_72_return_pct': safe_float(rule_72_metrics['rule_of_72_return']),
                    'compounding_factor': safe_float(rule_72_metrics['compounding_factor'])
                }
            }
            
            # Calculate benchmark buy-and-hold metrics
            self.logger.progress("=== CALCULATING BENCHMARK METRICS ===")
            try:
                trading_dates = [snap['date'] for snap in self.portfolio_snapshots]
                benchmark_calc = BenchmarkCalculator(
                    initial_capital=self.total_capital,
                    index_data=self.index_data,
                    trading_dates=trading_dates
                )
                benchmark_metrics = benchmark_calc.calculate_benchmark_metrics(risk_free_rate)
                benchmark_values = benchmark_calc.get_benchmark_values_array()
                
                self.logger.info(f"DEBUG: Benchmark values array length: {len(benchmark_values)}")
                self.logger.info(f"DEBUG: First 3 benchmark values: {benchmark_values[:3] if len(benchmark_values) >= 3 else benchmark_values}")
                
                # Add benchmark data to metrics
                metrics['benchmark_metrics'] = benchmark_metrics
                metrics['benchmark_buyhold'] = benchmark_values
                metrics['alpha_pct'] = safe_float(cagr - benchmark_metrics['cagr_pct'])
                
                self.logger.performance(f"Benchmark Total Return: {benchmark_metrics['total_return_pct']:.2f}%")
                self.logger.performance(f"Benchmark CAGR: {benchmark_metrics['cagr_pct']:.2f}%")
                self.logger.info(f"Strategy Alpha: {metrics['alpha_pct']:.2f}%")
                self.logger.trade(f"DEBUG: benchmark_buyhold in metrics: {len(metrics.get('benchmark_buyhold', []))} values")
            except Exception as e:
                self.logger.progress(f"Error calculating benchmark metrics: {e}")
                import traceback
                traceback.print_exc()
                metrics['benchmark_metrics'] = {}
                metrics['benchmark_buyhold'] = []
                metrics['alpha_pct'] = 0.0
            
            self.logger.info(f"Final metrics calculated: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.progress(f"Error calculating metrics: {e}")
            return {
                'total_return_pct': 0.0,
                'annualized_return_pct': 0.0,
                'cagr_pct': 0.0,
                'xirr_pct': 0.0,
                'max_drawdown_pct': 0.0,
                'sharpe_ratio': 0.0,
                'beta': 1.0,
                'treynor_ratio': 0.0,
                'calmar_ratio': 0.0,
                'win_rate_pct': 0.0,
                'total_trades': 0,
                'final_capital': self.total_capital,
                'portfolio_snapshots': [],
                'trades': [],
                'rule_of_72': {
                    'years_to_double': 0.0,
                    'expected_doublings': 0.0,
                    'rule_of_72_return_pct': 0.0,
                    'compounding_factor': 1.0
                }
            }
    
    def calculate_cagr(self, start_value: float, end_value: float, start_date: datetime, end_date: datetime) -> float:
        """Calculate Compound Annual Growth Rate"""
        try:
            if start_value <= 0 or end_value <= 0:
                return 0.0
            
            years = (end_date - start_date).days / 365.25
            if years <= 0:
                return 0.0
            
            cagr = (safe_power(safe_divide(end_value, start_value, 1.0), safe_divide(1.0, years, 1.0)) - 1) * 100
            return safe_float(cagr)
        except Exception as e:
            self.logger.performance(f"Error calculating CAGR: {e}")
            return 0.0
    
    def calculate_rule_of_72_metrics(self, cagr: float, years: float) -> Dict:
        """Calculate Rule of 72 metrics"""
        try:
            if cagr <= 0:
                return {
                    'years_to_double': 0.0,
                    'expected_doublings': 0.0,
                    'rule_of_72_return': 0.0,
                    'compounding_factor': 1.0
                }
            
            years_to_double = safe_divide(72.0, cagr, 0.0)
            expected_doublings = safe_divide(years, years_to_double, 0.0)
            compounding_factor = safe_power(2.0, expected_doublings, 1.0)
            
            return {
                'years_to_double': safe_float(years_to_double),
                'expected_doublings': safe_float(expected_doublings),
                'rule_of_72_return': safe_float(cagr),
                'compounding_factor': safe_float(compounding_factor)
            }
        except Exception as e:
            self.logger.progress(f"Error calculating Rule of 72 metrics: {e}")
            return {
                'years_to_double': 0.0,
                'expected_doublings': 0.0,
                'rule_of_72_return': 0.0,
                'compounding_factor': 1.0
            }
    
    def calculate_xirr(self, cash_flows: List[Tuple[datetime, float]]) -> float:
        """Calculate XIRR (Extended Internal Rate of Return)"""
        try:
            if len(cash_flows) < 2:
                return 0.0
            
            # Simple approximation for XIRR
            total_investment = sum(cf[1] for cf in cash_flows if cf[1] < 0)
            total_return = sum(cf[1] for cf in cash_flows if cf[1] > 0)
            
            if total_investment == 0:
                return 0.0
            
            # Calculate time-weighted return
            start_date = cash_flows[0][0]
            end_date = cash_flows[-1][0]
            years = (end_date - start_date).days / 365.25
            
            if years <= 0:
                return 0.0
            
            xirr = (safe_power(safe_divide(total_return, abs(total_investment), 1.0), safe_divide(1.0, years, 1.0)) - 1) * 100
            return safe_float(xirr)
        except Exception as e:
            self.logger.progress(f"Error calculating XIRR: {e}")
            return 0.0
    
    def calculate_max_drawdown(self, snapshots: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        try:
            if snapshots.empty:
                return 0.0
            
            peak = snapshots['total_value'].expanding().max()
            drawdown = (snapshots['total_value'] - peak) / peak * 100
            max_drawdown = drawdown.min()
            
            return safe_float(max_drawdown)
        except Exception as e:
            self.logger.performance(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def calculate_sharpe_ratio(self, snapshots: pd.DataFrame) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(snapshots) < 2:
                return 0.0
            
            returns = snapshots['total_value'].pct_change().dropna()
            if len(returns) == 0:
                return 0.0
            
            mean_return = returns.mean()
            std_return = returns.std()
            
            if std_return == 0:
                return 0.0
            
            # Annualize the Sharpe ratio
            sharpe = (mean_return / std_return) * np.sqrt(252)
            return safe_float(sharpe)
        except Exception as e:
            self.logger.performance(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def calculate_beta_and_treynor(self, snapshots: pd.DataFrame, risk_free_rate: float = 6.0) -> tuple:
        """Calculate portfolio beta against market benchmark and Treynor ratio"""
        try:
            if snapshots.empty or self.index_data.empty:
                return 1.0, 0.0
            
            if len(snapshots) < 10:
                return 1.0, 0.0
            
            # Calculate portfolio returns from snapshots
            portfolio_df = snapshots.copy()
            portfolio_df['returns'] = portfolio_df['total_value'].pct_change().dropna()
            portfolio_returns = portfolio_df['returns'].dropna()
            
            if len(portfolio_returns) < 10:
                return 1.0, 0.0
            
            # Get index returns
            index_df = self.index_data.copy()
            if 'adjusted_close' not in index_df.columns:
                if 'close' in index_df.columns:
                    index_df['adjusted_close'] = index_df['close']
                else:
                    return 1.0, 0.0
            
            index_df['returns'] = index_df['adjusted_close'].pct_change().dropna()
            index_returns = index_df['returns'].dropna()
            
            if len(index_returns) < 10:
                return 1.0, 0.0
            
            # Align dates for portfolio and index returns
            # Portfolio snapshots have 'date' as a column, index_data has dates as index
            portfolio_df_indexed = portfolio_df.set_index('date')
            
            # Align portfolio returns to match index dates (daily)
            # Both should be indexed by date now
            portfolio_returns_series = portfolio_df_indexed['returns']
            index_returns_series = index_df['returns']
            
            # Align both series on common dates
            aligned_data = pd.DataFrame({
                'portfolio': portfolio_returns_series,
                'index': index_returns_series
            }).dropna()
            
            if len(aligned_data) < 10:
                return 1.0, 0.0
            
            # Calculate beta: covariance(portfolio, market) / variance(market)
            portfolio_var = aligned_data['portfolio'].var()
            index_var = aligned_data['index'].var()
            covariance = aligned_data['portfolio'].cov(aligned_data['index'])
            
            beta = covariance / index_var if index_var > 0 else 1.0
            beta = max(0.1, min(3.0, beta))  # Clamp beta between 0.1 and 3.0
            
            # Calculate CAGR for excess return calculation
            start_date = snapshots['date'].iloc[0]
            end_date = snapshots['date'].iloc[-1]
            start_value = snapshots['total_value'].iloc[0]
            end_value = snapshots['total_value'].iloc[-1]
            
            years = (end_date - start_date).days / 365.25
            if years > 0 and start_value > 0:
                portfolio_cagr = ((end_value / start_value) ** (1 / years) - 1) * 100
            else:
                portfolio_cagr = 0.0
            
            # Calculate Treynor ratio: (CAGR - risk_free_rate) / beta
            excess_return = portfolio_cagr - risk_free_rate
            treynor_ratio = excess_return / beta if beta > 0 else 0.0
            
            return beta, treynor_ratio
            
        except Exception as e:
            self.logger.progress(f"Error calculating beta and Treynor ratio: {e}")
            import traceback
            traceback.print_exc()
            return 1.0, 0.0
    
    def calculate_win_rate(self) -> float:
        """Calculate win rate from trades"""
        try:
            sell_trades = [t for t in self.trades if t.action == 'SELL']
            if not sell_trades:
                return 0.0
            
            winning_trades = 0
            for trade in sell_trades:
                # Find corresponding buy trade
                buy_trade = None
                for bt in self.trades:
                    if bt.symbol == trade.symbol and bt.action == 'BUY' and bt.date <= trade.date:
                        buy_trade = bt
                        break
                
                if buy_trade and trade.price > buy_trade.price:
                    winning_trades += 1
            
            win_rate = (winning_trades / len(sell_trades)) * 100
            return safe_float(win_rate)
        except Exception as e:
            self.logger.progress(f"Error calculating win rate: {e}")
            return 0.0
    
    def get_trade_pnl(self, trade: Trade) -> float:
        """Get P&L for a trade"""
        try:
            if trade.action != 'SELL':
                return 0.0
            
            # Find corresponding buy trade
            buy_trade = None
            for bt in self.trades:
                if bt.symbol == trade.symbol and bt.action == 'BUY' and bt.date <= trade.date:
                    buy_trade = bt
                    break
            
            if buy_trade:
                return (trade.price - buy_trade.price) * trade.quantity
            
            return 0.0
        except Exception as e:
            self.logger.trade(f"Error calculating trade P&L: {e}")
            return 0.0
    
    def calculate_portfolio_nav(self, date: datetime, stock_data: pd.DataFrame = None) -> float:
        """Calculate total portfolio NAV (positions + cash + buffer)"""
        try:
            positions_value = 0.0
            
            # Calculate value of all positions
            for symbol, position in self.positions.items():
                try:
                    # Try to get current price from stock_data if available
                    if stock_data is not None and date in stock_data.index and symbol in stock_data.columns:
                        current_price = stock_data.loc[date, symbol]['adjusted_close']
                    else:
                        # Fallback to position's current price
                        current_price = position.current_price
                    
                    positions_value += position.quantity * current_price
                except (KeyError, TypeError, AttributeError):
                    # Use position's stored price if data not available
                    positions_value += position.quantity * position.current_price
            
            # Total NAV = positions + cash + buffer
            total_nav = positions_value + self.cash_balance + self.buffer_capital
            return round(total_nav, 2)
            
        except Exception as e:
            self.logger.progress(f"Error calculating portfolio NAV: {e}")
            # Fallback to total capital
            return self.total_capital
    
    def execute_trade(self, date: datetime, symbol: str, action: str, price: float, reason: str, rs_score: float = None, rs_rank: int = None):
        """Execute a trade with proper Indian market cost calculation and buffer capital system"""
        try:
            # Validate price is scalar and valid
            if hasattr(price, 'iloc'):
                price = float(price.iloc[0])
            else:
                price = float(price)
                
            if pd.isna(price) or price <= 0:
                self.logger.info(f"  Invalid price for {symbol}: {price}")
                return False
            if action == "BUY":
                # Fixed position size (₹9,000 per stock)
                fixed_position_size = self.per_trade_allocation  # This should be ₹9,000
                
                # Check if we have enough capital (cash + buffer)
                total_available = self.cash_balance + self.buffer_capital
                
                if total_available < fixed_position_size:
                    self.logger.trade(f"  Insufficient capital to buy {symbol}: {total_available} < {fixed_position_size}")
                    return False
                
                # Calculate quantity
                quantity = int(fixed_position_size / price)
                if quantity <= 0:
                    self.logger.trade(f"  Cannot buy {symbol}: quantity would be {quantity}")
                    return False
                
                # Calculate transaction costs using proper Indian market calculation
                transaction_value = quantity * price
                cost_details = self.calculate_transaction_costs(transaction_value, "BUY")
                net_amount = cost_details['net_amount']
                
                # Use cash first, then buffer if needed
                if self.cash_balance >= net_amount:
                    # Use only cash
                    self.cash_balance -= net_amount
                    self.logger.info(f"  Used cash: ₹{net_amount:.2f}")
                else:
                    # Use cash + buffer
                    original_cash_used = self.cash_balance  # Store original cash before resetting
                    needed_from_buffer = net_amount - self.cash_balance
                    self.cash_balance = 0
                    self.buffer_capital -= needed_from_buffer
                    self.logger.info(f"  Used cash: ₹{original_cash_used:.2f}, buffer: ₹{needed_from_buffer:.2f}")
                
                # Calculate stop loss price
                if self.stop_loss_pct > 0:
                    stop_loss_price = price * (1 - self.stop_loss_pct)
                    self.logger.info(f"  🛑 Stoploss set to ₹{stop_loss_price:.2f} (Formula: ₹{price:.2f} * (1 - {self.stop_loss_pct*100:.1f}/100))")
                else:
                    stop_loss_price = None
                    self.logger.info(f"  ℹ️  Stoploss disabled (0%)")
                
                # Create position
                position = Position(
                    symbol=symbol,
                    quantity=quantity,
                    buy_price=price,
                    buy_date=date,
                    current_price=price,
                    unrealized_pnl=0,
                    stop_loss_price=stop_loss_price
                )
                
                self.positions[symbol] = position
                
                # Calculate current portfolio NAV
                portfolio_nav = self.calculate_portfolio_nav(date)
                
                # Record trade with detailed cost breakdown and NAV
                trade = Trade(
                    date=date,
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    price=price,
                    amount=net_amount,
                    reason=reason,
                    rs_score=rs_score,
                    rs_rank=rs_rank,
                    transaction_value=cost_details['transaction_value'],
                    brokerage=cost_details['brokerage'],
                    stt=cost_details['stt'],
                    stamp_duty=cost_details['stamp_duty'],
                    exchange_charges=cost_details['exchange_charges'],
                    sebi_charges=cost_details['sebi_charges'],
                    gst=cost_details['gst'],
                    total_costs=cost_details['total_costs'],
                    net_amount=net_amount,
                    portfolio_nav=portfolio_nav,
                    buy_price=price,  # Store buy price for reference
                    capital_gain=0.0,
                    capital_gain_pct=0.0,
                    holding_period_days=0,
                    capital_gains_tax=0.0,
                    net_profit_after_tax=0.0
                )
                self.trades.append(trade)
                
                self.logger.info(f"📋 Purchase calculation:")
                self.logger.info(f"   Gross amount: ₹{cost_details['transaction_value']:,.2f}")
                self.logger.info(f"   Transaction costs: ₹{cost_details['total_costs']:,.2f}")
                self.logger.info(f"   Net amount: ₹{net_amount:,.2f}")
                self.logger.trade(f"   Units to buy: {quantity}")
                self.logger.trade(f"✅ Purchase executed: {quantity} units of {symbol} for ₹{net_amount:,.2f}")
                self.logger.info(f"💳 Total cost (including fees): ₹{cost_details['total_costs']:,.2f}")
                self.logger.progress(f"📊 Portfolio NAV: ₹{portfolio_nav:,.2f}")
                self.logger.trade(f"💰 Buy Price: ₹{price:,.2f}")
                self.logger.info(f"💰 Remaining cash: ₹{self.cash_balance:,.2f}")
                return True
                
            elif action == "SELL":
                if symbol not in self.positions:
                    self.logger.trade(f"  Cannot sell {symbol}: not in positions")
                    return False
                
                position = self.positions[symbol]
                quantity = position.quantity
                
                # Calculate transaction costs using proper Indian market calculation
                transaction_value = quantity * price
                cost_details = self.calculate_transaction_costs(transaction_value, "SELL")
                net_amount = cost_details['net_amount']
                
                # Calculate P&L
                pnl = (price - position.buy_price) * quantity
                pnl_pct = ((price - position.buy_price) / position.buy_price) * 100
                
                # Adjust buffer capital based on P&L (Segregate Principal and Profit)
                # Strategy: 
                # 1. Principal (Cost Basis) -> Return to Cash Balance
                # 2. Net Profit/Loss (Net Amount - Cost Basis) -> Adjust Buffer Capital
                
                cost_basis = position.buy_price * quantity
                net_pnl = net_amount - cost_basis
                
                # Update Buffer (Profit/Loss)
                self.buffer_capital += net_pnl
                
                # Update Cash (Return Principal)
                self.cash_balance += cost_basis
                
                if net_pnl > 0:
                    self.logger.info(f"  Profit: ₹{net_pnl:.2f} added to buffer")
                else:
                    self.logger.info(f"  Loss: ₹{abs(net_pnl):.2f} subtracted from buffer")
                
                # Remove position
                del self.positions[symbol]
                
                # Calculate holding period only
                buy_price = position.buy_price
                holding_period_days = (date - position.buy_date).days
                
                # Capital gains calculation removed as per request
                capital_gain = 0.0
                capital_gain_pct = 0.0
                capital_gains_tax = 0.0
                net_profit_after_tax = 0.0
                
                # Calculate current portfolio NAV
                portfolio_nav = self.calculate_portfolio_nav(date)
                
                # Record trade with detailed cost breakdown, capital gains, and tax
                trade = Trade(
                    date=date,
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    price=price,
                    amount=net_amount,
                    reason=reason,
                    rs_score=rs_score,
                    rs_rank=rs_rank,
                    transaction_value=cost_details['transaction_value'],
                    brokerage=cost_details['brokerage'],
                    stt=cost_details['stt'],
                    stamp_duty=cost_details['stamp_duty'],
                    exchange_charges=cost_details['exchange_charges'],
                    sebi_charges=cost_details['sebi_charges'],
                    gst=cost_details['gst'],
                    total_costs=cost_details['total_costs'],
                    net_amount=net_amount,
                    portfolio_nav=portfolio_nav,
                    buy_price=buy_price,
                    capital_gain=round(capital_gain, 2),
                    capital_gain_pct=round(capital_gain_pct, 2),
                    holding_period_days=holding_period_days,
                    capital_gains_tax=round(capital_gains_tax, 2),
                    net_profit_after_tax=round(net_profit_after_tax, 2)
                )
                self.trades.append(trade)
                
                print(f"")
                self.logger.trade(f"💰 SELL EXECUTED: {symbol}")
                self.logger.info(f"   Quantity: {quantity} units")
                self.logger.trade(f"   Sell Price: ₹{price:,.2f}")
                self.logger.trade(f"   Buy Price: ₹{buy_price:,.2f}")
                self.logger.info(f"   Holding Period: {holding_period_days} days")
                print(f"")

                self.logger.info(f"📈 PORTFOLIO STATUS:")
                self.logger.info(f"   Portfolio NAV: ₹{portfolio_nav:,.2f}")
                self.logger.info(f"   Cash Balance: ₹{self.cash_balance:,.2f}")
                self.logger.info(f"   Buffer Capital: ₹{self.buffer_capital:,.2f}")
                self.logger.info(f"   Holdings Value: ₹{portfolio_nav - self.cash_balance - self.buffer_capital:,.2f}")
                print(f"")
                return True
                
        except Exception as e:
            self.logger.trade(f"Error executing trade {action} {symbol}: {e}")
            return False
    
    def log_stop_loss_mode(self):
        """Explicitly log the stop loss configuration"""
        mode = "DAILY Check" if self.daily_stop_loss_check else "WEEKLY Check (on Friday/Signal Day)"
        self.logger.info(f"\n{'='*40}")
        self.logger.info(f"🛡️  STOP LOSS CONFIGURATION")
        self.logger.info(f"   Mode: {mode}")
        self.logger.info(f"   Threshold: {self.stop_loss_pct * 100:.1f}%")
        self.logger.info(f"   Formula: Stoploss Price = Buy Price * (1 - {self.stop_loss_pct*100:.1f}/100)")
        self.logger.info(f"{'='*40}\n")

    def check_daily_stop_loss(self, stock_data: pd.DataFrame, current_date: datetime) -> List[str]:
        """Check stop loss for all positions and return stocks to sell"""
        stop_loss_exits = []
        
        # Early return if no positions (silent)
        if len(self.positions) == 0:
            return stop_loss_exits
        
        try:
            for symbol, position in self.positions.items():
                try:
                    # Get current price
                    price_data = stock_data.loc[symbol, current_date]['adjusted_close']
                    current_price = float(price_data.iloc[0]) if hasattr(price_data, 'iloc') else float(price_data)
                    
                    # Check if stop loss hit
                    if position.stop_loss_price is not None and current_price <= position.stop_loss_price:
                        stop_loss_exits.append(symbol)
                        loss_pct = ((current_price - position.buy_price) / position.buy_price * 100)
                        self.logger.info(f"  ⚠️  STOP LOSS HIT: {symbol} @ ₹{current_price:.2f} <= SL: ₹{position.stop_loss_price:.2f} (Drakedown: {loss_pct:+.2f}%)")
                    else:
                        # Explicitly log safe status
                        loss_pct = ((current_price - position.buy_price) / position.buy_price * 100)
                        self.logger.info(f"  ✓  Stoploss safe: {symbol} @ ₹{current_price:.2f} > SL: ₹{position.stop_loss_price:.2f} (Drakedown: {loss_pct:+.2f}%)")
                        
                except (KeyError, IndexError):
                    # Stock data not available - skip silently
                    pass
                    continue
                    
        except Exception as e:
            self.logger.info(f"  ❌ Error checking stop loss: {e}")
        
        # Only print if stop loss triggered
        if len(stop_loss_exits) > 0:
            self.logger.trade(f"⚠️  {len(stop_loss_exits)} position(s) will be sold due to stop loss")
        
        return stop_loss_exits
    
    def get_next_monday(self, date: datetime) -> datetime:
        """Get the next Monday from the given date"""
        days_ahead = 0 - date.weekday()  # Monday is 0
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        return date + timedelta(days=days_ahead)
    
    def find_next_available_trading_day(self, start_date: datetime, trading_dates: List[datetime]) -> datetime:
        """Find the next available trading day after the given start date"""
        for date in trading_dates:
            if date > start_date:
                return date
        return None
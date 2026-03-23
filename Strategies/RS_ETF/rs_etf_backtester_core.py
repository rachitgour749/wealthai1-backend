import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from sqlalchemy.orm import Session
try:
    from Strategies.RS_ETF.database import ETFData, IndexData, StrategyConfig, BacktestResult, TradeLog, PortfolioSnapshot
except ImportError:
    from database import ETFData, IndexData, StrategyConfig, BacktestResult, TradeLog, PortfolioSnapshot
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

    # NAV and Capital Gains Tracking
    portfolio_nav: Optional[float] = None
    buy_price: Optional[float] = None
    capital_gain: Optional[float] = None
    capital_gain_pct: Optional[float] = None
    holding_period_days: Optional[int] = None
    capital_gains_tax: Optional[float] = None
    net_profit_after_tax: Optional[float] = None

@dataclass
class Position:
    symbol: str
    quantity: int
    buy_price: float
    buy_date: datetime
    current_price: float
    unrealized_pnl: float
    stop_loss_price: Optional[float] = None

class RSETFStrategyBacktester:
    def __init__(self, db: Session, config_id: int = None, config_dict: Dict = None):
        # Initialize centralized logger
        self.logger = StrategyLogger('RS_ETF')

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
            self.etf_universe = config_dict.get('etf_universe', 'ALL_ETFS')
            self.custom_etfs = config_dict.get('custom_etfs', None)  # Custom ETF selection
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
            # Removed max_holding_period - ETFs held until RS ranking drops
            self.transaction_cost_pct = self.config.transaction_cost_pct / 100
            # Removed min_price - no price filtering, all ETFs eligible
            self.min_turnover = self.config.min_turnover
            self.etf_universe = getattr(self.config, 'etf_universe', 'ALL_ETFS')
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
    
    def _load_logging_config(self):
        """Load logging configuration from JSON file"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'logging_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.logging_config = json.load(f)
                self.logger.info("Loaded logging configuration from logging_config.json")
            else:
                # Default configuration
                self.logging_config = {
                    "purchase_limit_config": {
                        "enabled": True,
                        "allocation_multiplier": 1.5
                    },
                    "logging": {
                        "limit": True,
                        "trade": True,
                        "error": True
                    }
                }
                self.logger.info("Using default logging configuration (file not found)")
        except Exception as e:
            self.logger.error(f"Error loading logging config: {e}")
            self.logging_config = {}
    
    def _log(self, category: str, message: str):
        """Log message if category is enabled in config"""
        logging_settings = self.logging_config.get("logging", {})
        if logging_settings.get(category, False):
            if category == "limit":
                self.logger.info(message)
            elif category == "trade":
                self.logger.info(message)
            elif category == "error":
                self.logger.error(message)
            else:
                self.logger.info(message)
    
    def initialize_purchase_limits(self, tickers: List[str], accumulation_weeks: int):
        """
        Initialize purchase limits for all ETFs based on allocation formula
        
        Formula:
        - allocation_pct = (100 / total_etfs) * allocation_multiplier
        - limit_per_etf = floor(allocation_pct * accumulation_weeks / 100)
        
        This prevents over-concentration and achieves better diversification.
        
        Args:
            tickers: List of ETF tickers
            accumulation_weeks: Number of accumulation weeks
        """
        # Get allocation multiplier from config (default 1.5)
        purchase_limit_config = self.logging_config.get("purchase_limit_config", {})
        allocation_multiplier = purchase_limit_config.get("allocation_multiplier", 1.5)
        limit_enabled = purchase_limit_config.get("enabled", True)
        
        self.total_etfs = len(tickers)
        self.accumulation_weeks = accumulation_weeks
        
        # Calculate allocation percentage (round down)
        allocation_percentage = (100 / self.total_etfs) * allocation_multiplier
        allocation_percentage = int(allocation_percentage)  # Round down
        
        # Calculate limit per ETF (round down)
        limit_per_etf = int((allocation_percentage * accumulation_weeks) / 100)  # Round down
        
        # Initialize limits and counters for all ETFs
        for ticker in tickers:
            # If limits disabled, set to infinity (no limit)
            self.etf_purchase_limits[ticker] = limit_per_etf if limit_enabled else float('inf')
            self.etf_purchase_counts[ticker] = 0
        
        self._log("limit", "=" * 80)
        self._log("limit", "📊 PURCHASE LIMIT INITIALIZATION")
        self._log("limit", "=" * 80)
        self._log("limit", f"   Total ETFs: {self.total_etfs}")
        self._log("limit", f"   Accumulation Weeks: {accumulation_weeks}")
        self._log("limit", f"   Allocation Multiplier: {allocation_multiplier}")
        self._log("limit", f"   Calculation: (100 / {self.total_etfs}) × {allocation_multiplier} = {allocation_percentage}%")
        self._log("limit", f"   Limit Formula: {allocation_percentage}% × {accumulation_weeks} weeks / 100")
        if limit_enabled:
            self._log("limit", f"   ✅ Limit per ETF: {limit_per_etf} purchases maximum")
        else:
            self._log("limit", f"   ⚠️ Purchase limits DISABLED (no limit)")
        self._log("limit", "")
        self._log("limit", "   ETF Purchase Limits:")
        for ticker in sorted(tickers):
            if limit_enabled:
                self._log("limit", f"      {ticker:12s}: 0/{limit_per_etf} purchases")
            else:
                self._log("limit", f"      {ticker:12s}: 0/∞ purchases (unlimited)")
        self._log("limit", "=" * 80)
    
    def get_purchase_limit_status(self) -> Dict[str, Any]:
        """Get current status of purchase limits for all ETFs"""
        # Get config values
        purchase_limit_config = self.logging_config.get("purchase_limit_config", {})
        allocation_multiplier = purchase_limit_config.get("allocation_multiplier", 1.5)
        limit_enabled = purchase_limit_config.get("enabled", True)
        
        status = {
            'total_etfs': self.total_etfs,
            'accumulation_weeks': self.accumulation_weeks,
            'allocation_multiplier': allocation_multiplier,
            'limit_enabled': limit_enabled,
            'etf_status': []
        }
        
        for ticker in sorted(self.etf_purchase_limits.keys()):
            limit = self.etf_purchase_limits[ticker]
            count = self.etf_purchase_counts.get(ticker, 0)
            
            # Handle infinity (unlimited)
            if limit == float('inf'):
                percentage = 0
                at_limit = False
                remaining = float('inf')
            else:
                percentage = (count / limit * 100) if limit > 0 else 0
                at_limit = count >= limit
                remaining = limit - count
            
            status['etf_status'].append({
                'ticker': ticker,
                'current_purchases': count,
                'limit': limit if limit != float('inf') else None,
                'remaining': remaining if remaining != float('inf') else None,
                'utilization_pct': round(percentage, 1),
                'at_limit': at_limit
            })
        
        return status
        
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
        
    def load_etf_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load ETF data from etf_data table"""
        # Ensure dates are timezone-naive
        if start_date.tzinfo:
            start_date = start_date.replace(tzinfo=None)
        if end_date.tzinfo:
            end_date = end_date.replace(tzinfo=None)
            
        # Calculate buffer start date - 400 calendar days BEFORE backtest start
        # This provides ~252 trading days for 60-day quarter calculations
        # buffer_start_date = (pd.to_datetime(start_date) - timedelta(days=400)).strftime('%Y-%m-%d')
        data_start_date = start_date - timedelta(days=100)
        
        self.logger.progress(f"Loading ETF data from {data_start_date} to {end_date} (backtest period: {start_date} to {end_date})")
        
        # Calculate period length
        period_days = (end_date - start_date).days
        total_data_days = (end_date - data_start_date).days
        self.logger.info(f"Backtest period: {period_days} days, Total data period: {total_data_days} days")
        
        # Get selected ETF symbols
        selected_etfs = self.get_custom_etf_universe()
        if not selected_etfs:
            raise ValueError("No ETFs selected for backtest")
        
        self.logger.info(f"Selected ETFs for backtest: {len(selected_etfs)} ETFs")
        self.logger.info(f"ETF symbols: {selected_etfs}")
        
        # Build symbol filter for SQL query
        placeholders = ','.join(['%s'] * len(selected_etfs))
        symbol_limit = f"AND symbol IN ({placeholders})"
        
        # Use raw SQL to query etf_market table with symbol filtering
        sql = f"""
        SELECT symbol, date, adj_close AS adjusted_close, open
        FROM etf_market 
        WHERE date >= %s AND date <= %s {symbol_limit}
        ORDER BY symbol, date
        """
        
        params = tuple([data_start_date, end_date] + selected_etfs)
        df = pd.read_sql(sql, self.db.bind, params=params, 
                        index_col=['symbol', 'date'], parse_dates=['date'])
        
        self.logger.performance(f"Raw ETF data query returned: {len(df)} records")
        
        # Convert timezone-aware dates to timezone-naive
        if df.index.get_level_values('date').tz is not None:
            df.index = df.index.set_levels(df.index.levels[1].tz_convert(None), level='date')
        
        self.logger.info(f"Final ETF data: {len(df)} records, {len(df.index.get_level_values('symbol').unique())} ETFs")
        if not df.empty:
            self.logger.info(f"Date range: {df.index.get_level_values('date').min()} to {df.index.get_level_values('date').max()}")
        
        return df
    
    def calculate_common_date_range(self, selected_etfs: List[str]) -> Tuple[Optional[str], Optional[str], float]:
        """Calculate common date range for selected ETFs by querying the database
        
        For RS_ETF Strategy: Returns maximum 5 years of data from current to past
        """
        if not selected_etfs:
            return None, None, 0.0
        
        try:
            # Filter to only allowed ETFs (optional validation)
            allowed_etfs = self._get_allowed_etf_list()
            valid_etfs = [etf for etf in selected_etfs if etf in allowed_etfs]
            
            if not valid_etfs:
                self.logger.info(f"Warning: No valid ETFs in selection. Allowed ETFs: {allowed_etfs}")
                return None, None, 0.0
            
            # Query database for min/max dates for each ETF
            etf_ranges = {}
            
            for etf in valid_etfs:
                # Query min and max dates for this ETF
                sql = """
                SELECT MIN(date) as min_date, MAX(date) as max_date
                FROM etf_market 
                WHERE symbol = %s
                """
                result = pd.read_sql(sql, self.db.bind, params=(etf,))
                
                if not result.empty and result.iloc[0]['min_date'] is not None:
                    min_date = pd.to_datetime(result.iloc[0]['min_date'])
                    max_date = pd.to_datetime(result.iloc[0]['max_date'])
                    etf_ranges[etf] = {
                        'start_date': min_date,
                        'end_date': max_date
                    }
            
            if not etf_ranges:
                self.logger.info(f"No data found for selected ETFs: {valid_etfs}")
                return None, None, 0.0
            
            # Find the latest start date and earliest end date (common range)
            start_dates = [data['start_date'] for data in etf_ranges.values()]
            end_dates = [data['end_date'] for data in etf_ranges.values()]
            
            latest_start = max(start_dates)
            earliest_end = min(end_dates)
            
            self.logger.info(f"📅 RS ETF Strategy Data Ranges:")
            for etf in valid_etfs:
                if etf in etf_ranges:
                    data = etf_ranges[etf]
                    days_available = (data['end_date'] - data['start_date']).days
                    years_available = days_available / 365.25
                    self.logger.info(f"   {etf:12s}: {data['start_date'].strftime('%Y-%m-%d')} to {data['end_date'].strftime('%Y-%m-%d')} ({years_available:.1f} years)")
            
            self.logger.progress(f"📊 Common data range: {latest_start.strftime('%Y-%m-%d')} to {earliest_end.strftime('%Y-%m-%d')}")
            
            # Add buffer for RS ETF strategy calculations
            # RS strategy needs lookback periods, so add buffer
            buffer_weeks = 15  # 90 weeks = ~630 calendar days
            buffer_days = buffer_weeks * 7
            strategy_start = latest_start + timedelta(days=buffer_days)
            
            # ===== NEW: LIMIT TO MAXIMUM 5 YEARS =====
            # Calculate 5 years before the end date
            five_years_ago = earliest_end - timedelta(days=int(5 * 365.25))
            
            # If the calculated start is earlier than 5 years ago, use 5 years ago instead
            if strategy_start < five_years_ago:
                self.logger.info(f"🔒 RS_ETF Strategy: Limiting to maximum 5 years of data")
                self.logger.info(f"   Original start: {strategy_start.strftime('%Y-%m-%d')}")
                self.logger.info(f"   Limited start (5 years ago): {five_years_ago.strftime('%Y-%m-%d')}")
                strategy_start = five_years_ago
            # ===== END NEW LOGIC =====
            
            self.logger.info(f"🎯 RS Strategy Buffer:")
            self.logger.info(f"   Buffer period: {buffer_weeks} weeks ({buffer_days} calendar days)")
            self.logger.info(f"   Strategy start (with buffer): {strategy_start.strftime('%Y-%m-%d')}")
            
            # Ensure we have valid range
            if strategy_start >= earliest_end:
                self.logger.info(f"⚠️ Insufficient data with buffer, using latest_start as strategy start")
                strategy_start = latest_start
            
            # Calculate years available for backtesting
            years_available = (earliest_end - strategy_start).days / 365.25
            
            # Format dates as strings
            start_date_str = strategy_start.strftime('%Y-%m-%d')
            end_date_str = earliest_end.strftime('%Y-%m-%d')
            
            self.logger.progress(f"✅ Final date range: {start_date_str} to {end_date_str} ({years_available:.1f} years)")
            
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
        
        # Use raw SQL to avoid SQLAlchemy model issues
        # Handle multiple symbol formats for NSEI index
        symbol_variants = [self.main_index]
        if self.main_index in ['^NSEI', 'NSEI', 'NIFTY50', 'NIFTY_50']:
            symbol_variants = ['^NSEI', 'NSEI', 'NIFTY50', 'NIFTY_50']
        elif self.main_index in ['^NIFTY50', 'NIFTY50', 'NIFTY_50']:
            symbol_variants = ['^NIFTY50', 'NIFTY50', 'NSEI', 'NIFTY_50']
        
        placeholders = ','.join(['%s' for _ in symbol_variants])
        sql = f"""
        SELECT symbol, date, open, high, low, close, COALESCE(adj_close, close) AS adjusted_close, volume
        FROM nifty_50_index_market 
        WHERE symbol IN ({placeholders}) AND date >= %s AND date <= %s
        ORDER BY date
        """
        
        params = tuple(symbol_variants + [data_start_date, end_date])
        df = pd.read_sql(sql, self.db.bind, params=params,
                        index_col='date', parse_dates=['date'])
        
        self.logger.performance(f"Raw index data query returned: {len(df)} records")
        
        
        # Rename adj_close to adjusted_close for consistency
        if 'adj_close' in df.columns:
            df = df.rename(columns={'adj_close': 'adjusted_close'})
        # Column is already named adjusted_close in MarketData.sqlite
        # No renaming needed
        
        # Convert timezone-aware dates to timezone-naive
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        
        self.logger.info(f"Final index data: {len(df)} records")
        if not df.empty:
            self.logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df
    
    def get_custom_etf_universe(self) -> List[str]:
        """Get custom list of ETFs based on user selection or configuration"""
        try:
            # PRIORITY 1: Use custom_etfs if provided (user-selected subset)
            if hasattr(self, 'custom_etfs') and self.custom_etfs:
                # Filter to ensure only valid ETFs from our allowed list
                allowed_etfs = self._get_allowed_etf_list()
                filtered_etfs = [etf for etf in self.custom_etfs if etf in allowed_etfs]
                if filtered_etfs:
                    self.logger.info(f"Using custom ETF selection: {len(filtered_etfs)} ETFs")
                    return filtered_etfs
                else:
                    self.logger.info(f"Warning: Custom ETFs not in allowed list, using default list")
            
            # PRIORITY 2: Get from configuration
            if self.config and hasattr(self.config, 'custom_etfs') and self.config.custom_etfs:
                custom_etfs = json.loads(self.config.custom_etfs) if isinstance(self.config.custom_etfs, str) else self.config.custom_etfs
                # Filter to ensure only valid ETFs
                allowed_etfs = self._get_allowed_etf_list()
                filtered_etfs = [etf for etf in custom_etfs if etf in allowed_etfs]
                if filtered_etfs:
                    self.logger.info(f"Using custom ETF universe from config: {len(filtered_etfs)} ETFs")
                    return filtered_etfs
            
            # PRIORITY 3: Return default allowed ETF list
            default_etfs = self._get_allowed_etf_list()
            self.logger.info(f"Using default allowed ETF list: {len(default_etfs)} ETFs")
            return default_etfs
            
        except Exception as e:
            self.logger.info(f"Error in get_custom_etf_universe: {e}")
            # Fallback to default list on error
            return self._get_allowed_etf_list()

    def load_metadata(self) -> Dict[str, Dict]:
        """Load ETF metadata by calculating directly from etf_data table"""
        try:
            # Calculate metadata directly from etf_data table
            metadata = self.calculate_metadata_from_data(self.db)
            return metadata
        except Exception as e:
            self.logger.error(f"Error loading ETF metadata: {e}")
            return {}

    def calculate_metadata_from_data(self, session) -> Dict[str, Dict]:
        """Calculate metadata directly from etf_data table"""
        try:
            from sqlalchemy import text
            
            # Query metadata from etf_data
            query_str = """
                SELECT 
                    symbol,
                    MIN(date)::text as start_date,
                    MAX(date)::text as end_date,
                    COUNT(*) as total_records,
                    ROUND(EXTRACT(EPOCH FROM (MAX(date)::timestamp - MIN(date)::timestamp)) / 86400.0 / 365.25, 1) as years_available
                FROM etf_market
                GROUP BY symbol
                ORDER BY symbol
            """
            
            result = session.execute(text(query_str))
            rows = result.fetchall()
            
            metadata = {}
            for row in rows:
                try:
                    symbol = row[0]
                    start_date = row[1]
                    end_date = row[2]
                    total_records = row[3]
                    years_available = row[4]
                    
                    metadata[symbol] = {
                        'start_date': str(start_date) if start_date else None,
                        'end_date': str(end_date) if end_date else None,
                        'years_available': float(years_available) if years_available else 0,
                        'total_records': int(total_records) if total_records else 0,
                        'data_source': 'etf_market'
                    }
                except Exception:
                    continue
            
            return metadata
        except Exception as e:
            self.logger.error(f"Error calculating metadata from etf_data: {e}")
            return {}

    def generate_asset_description(self, symbol: str) -> str:
        """Generate intelligent ETF descriptions based on symbol names"""
        etf_mappings = {
            'NIFTYBEES': 'Nifty 50 ETF - Broad Market',
            'BANKBEES': 'Banking Sector ETF',
            'JUNIORBEES': 'Nifty Next 50 ETF - Mid Cap',
            'ITBEES': 'Information Technology ETF',
            'PHARMABEES': 'Pharmaceutical Sector ETF',
            'INFRABEES': 'Infrastructure Sector ETF',
            'GOLDBEES': 'Gold Commodity ETF',
            'METALIETF': 'Metal & Mining Sector ETF',
            'OILIETF': 'Oil & Gas Sector ETF',
            'LIQUIDBEES': 'Liquid Fund ETF - Money Market',
            'CPSEETF': 'CPSE (Central PSE) ETF',
            'PSUBNKBEES': 'PSU Banking ETF',
            'MON100': 'NASDAQ 100 ETF - US Tech',
            'MODEFENCE': 'Defence Sector ETF',
            'MIDCAPETF': 'Mid Cap ETF',
        }

        if symbol in etf_mappings:
            return etf_mappings[symbol]

        symbol_lower = symbol.lower()
        if 'pharma' in symbol_lower: return 'Pharmaceutical Sector ETF'
        elif 'bank' in symbol_lower: return 'Banking Sector ETF'
        elif 'it' in symbol_lower or 'tech' in symbol_lower: return 'Technology Sector ETF'
        elif 'gold' in symbol_lower: return 'Gold Commodity ETF'
        elif 'oil' in symbol_lower or 'energy' in symbol_lower: return 'Oil & Gas Sector ETF'
        elif 'metal' in symbol_lower: return 'Metal & Mining Sector ETF'
        elif 'infra' in symbol_lower: return 'Infrastructure Sector ETF'
        elif 'defence' in symbol_lower or 'defense' in symbol_lower: return 'Defence Sector ETF'
        elif 'midcap' in symbol_lower or 'mid' in symbol_lower: return 'Mid Cap ETF'
        elif 'smallcap' in symbol_lower or 'small' in symbol_lower: return 'Small Cap ETF'
        elif 'liquid' in symbol_lower: return 'Liquid Fund ETF'
        elif 'nifty' in symbol_lower: return 'Nifty Index ETF'
        elif 'sensex' in symbol_lower: return 'Sensex Index ETF'
        elif 'psu' in symbol_lower: return 'PSU Sector ETF'
        elif 'cpse' in symbol_lower: return 'CPSE ETF'
        elif 'dividend' in symbol_lower: return 'Dividend ETF'
        elif 'momentum' in symbol_lower: return 'Momentum ETF'
        elif 'value' in symbol_lower: return 'Value ETF'
        elif 'quality' in symbol_lower: return 'Quality ETF'
        elif any(geo in symbol_lower for geo in ['us', 'usa', 'nasdaq', 'sp500', 'dow']): return 'International ETF'
        elif 'consumption' in symbol_lower or 'consumer' in symbol_lower: return 'Consumer Sector ETF'
        elif 'auto' in symbol_lower: return 'Automotive Sector ETF'
        elif 'realty' in symbol_lower or 'real' in symbol_lower: return 'Real Estate ETF'
        elif 'healthcare' in symbol_lower or 'health' in symbol_lower: return 'Healthcare Sector ETF'
        elif 'fmcg' in symbol_lower: return 'FMCG Sector ETF'
        elif 'pvt' in symbol_lower or 'private' in symbol_lower: return 'Private Bank ETF'
        else: return f'{symbol} ETF'

    def get_asset_sector_classification(self, symbol: str) -> str:
        """Classify ETF into sector categories"""
        symbol_lower = symbol.lower()
        if symbol in ['NIFTYBEES', 'JUNIORBEES', 'MIDCAPETF']: return 'Broad Market'
        elif 'bank' in symbol_lower or 'psu' in symbol_lower: return 'Financial'
        elif 'it' in symbol_lower or 'tech' in symbol_lower or 'mon100' in symbol_lower: return 'Technology'
        elif 'pharma' in symbol_lower or 'health' in symbol_lower: return 'Healthcare'
        elif 'gold' in symbol_lower: return 'Commodity'
        elif 'oil' in symbol_lower or 'energy' in symbol_lower: return 'Energy'
        elif 'metal' in symbol_lower: return 'Materials'
        elif 'infra' in symbol_lower: return 'Infrastructure'
        elif 'defence' in symbol_lower or 'defense' in symbol_lower: return 'Defence'
        elif 'liquid' in symbol_lower: return 'Cash/Liquid'
        elif 'cpse' in symbol_lower: return 'PSU'
        else: return 'Other'
    
    def _get_allowed_etf_list(self) -> List[str]:
        """Get the allowed list of ETFs for RS ETF Strategy - Dynamic from DB"""
        try:
            # Query the database for all available unique symbols in etf_data
            sql = "SELECT DISTINCT symbol FROM etf_market ORDER BY symbol"
            result = pd.read_sql(sql, self.db.bind)
            
            if not result.empty:
                dynamic_list = result['symbol'].tolist()
                self.logger.info(f"Dynamically loaded {len(dynamic_list)} ETFs from database")
                return dynamic_list
            else:
                self.logger.info("Warning: No ETFs found in database, falling back to default list")
                raise ValueError("Empty ETF table")
                
        except Exception as e:
            self.logger.progress(f"Error loading dynamic ETF list: {e}")
            # Fallback to hardcoded list if database query fails
            return [
                'NIFTYBEES',
                'JUNIORBEES',
                'GOLDBEES',
                'PSUBNKBEES',
                'HNGSNGBEES',
                'MON100',
                'BANKBEES',
                'CPSEETF',
                'INFRABEES',
                'ITBEES',
                'PHARMABEES'
            ]
    def validate_data_range(self, start_date: datetime, end_date: datetime) -> bool:
        """Validate that we have sufficient data for the requested period"""
        try:
            # Check if we have data for the required lookback periods
            earliest_required = start_date - timedelta(days=self.lookback_quarters * 7)
            
            # Query the earliest date in our data
            earliest_data = self.db.query(ETFData.date).order_by(ETFData.date.asc()).first()
            
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
    
    def calculate_rs_score(self, etf_prices: pd.Series, index_prices: pd.Series, 
                          current_date: datetime, symbol: str = None) -> Optional[Tuple[float, float, float, float]]:
        """Calculate Relative Strength score for an ETF with robust data validation"""
        try:
            # Get available trading dates from actual data
            available_etf_dates = sorted(etf_prices.index)
            
            # Find current date index
            try:
                current_etf_index = available_etf_dates.index(current_date) 
            except ValueError:
                return None

            # Get current prices
            current_etf_price = etf_prices.loc[current_date]
            current_index_price = index_prices.loc[current_date] if current_date in index_prices.index else None
            
            if current_index_price is None:
                return None
            
            # Use fixed lookback periods for index (assuming index data is complete/aligned or handled by caller)
            # For ETFs, we use the finding by index method since we have the list of available dates
            
            # Calculate required lookback periods
            max_lookback = max(self.lookback_weeks, self.lookback_months, self.lookback_quarters)
            
            # Check if we have enough historical data
            if current_etf_index < max_lookback:                                                                        
                return None

            # Get historical dates using actual available data
            week_ago_date = available_etf_dates[current_etf_index - self.lookback_weeks]                                                                    
            month_ago_date = available_etf_dates[current_etf_index - self.lookback_months]                                                                  
            quarter_ago_date = available_etf_dates[current_etf_index - self.lookback_quarters]                                                              

            # Get Past Prices using 'asof' for Index to handle mismatches (e.g. holidays, data gaps)
            # This finds the price at or immediately before the target date
            try:
                # Ensure index prices are sorted (they should be, but safety first)
                if not index_prices.index.is_monotonic_increasing:
                    index_prices = index_prices.sort_index()
                
                # Use get_indexer to find nearest dates efficiently or just use asof
                # Pandas asof is powerful for this. 
                # Note: index_prices is a Series with DatetimeIndex
                
                if week_ago_date not in index_prices.index:
                    index_past_w = float(index_prices.asof(week_ago_date))
                else:
                    index_past_w = float(index_prices.loc[week_ago_date])
                    
                if month_ago_date not in index_prices.index:
                    index_past_m = float(index_prices.asof(month_ago_date))
                else:
                    index_past_m = float(index_prices.loc[month_ago_date])
                    
                if quarter_ago_date not in index_prices.index:
                    index_past_q = float(index_prices.asof(quarter_ago_date))
                else:
                    index_past_q = float(index_prices.loc[quarter_ago_date])
                
                # Check if any lookups failed (returned NaN)
                if any(np.isnan(x) for x in [index_past_w, index_past_m, index_past_q]):
                    return None
                    
            except (KeyError, ValueError, IndexError):
                # If asof fails (e.g. date before first index date), we cannot calculate RS
                return None

            # WEEK (Outperformance)
            etf_past_w = float(etf_prices.loc[week_ago_date])
            # index_past_w already got above
            etf_ret_w = (current_etf_price / etf_past_w) - 1
            index_ret_w = (current_index_price / index_past_w) - 1
            rs_w = etf_ret_w - index_ret_w

            # MONTH
            etf_past_m = float(etf_prices.loc[month_ago_date])
            # index_past_m already got above
            etf_ret_m = (current_etf_price / etf_past_m) - 1
            index_ret_m = (current_index_price / index_past_m) - 1
            rs_m = etf_ret_m - index_ret_m

            # QUARTER
            etf_past_q = float(etf_prices.loc[quarter_ago_date])
            # index_past_q already got above
            etf_ret_q = (current_etf_price / etf_past_q) - 1
            index_ret_q = (current_index_price / index_past_q) - 1
            rs_q = etf_ret_q - index_ret_q
            
            # Simple Equal-Weight Average (Week + Month + Quarter) / 3
            rs_score = (rs_w + rs_m + rs_q) / 3
            
            return (rs_score, rs_w, rs_m, rs_q)
            
        except (KeyError, IndexError, ValueError, ZeroDivisionError) as e:
            return None
    

    

    
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
    
    def rank_etfs(self, etf_data: pd.DataFrame, index_data: pd.DataFrame, 
                   signal_date: datetime) -> pd.DataFrame:
        """Rank all eligible ETFs by RS score with enhanced error handling"""
        symbols = self.get_custom_etf_universe()  # Use custom ETF selection
        rankings = []
        valid_symbols = 0
        failed_symbols = 0
        
        # Get universe size based on custom selection
        universe_size = len(symbols)
        self.logger.info(f"Custom ETF universe size: {universe_size} ETFs and date: {signal_date.strftime('%Y-%m-%d (%A)')} ,day:{signal_date.weekday()}")
        
        self.logger.progress(f"📊 RS Momentum Calculation for {signal_date.strftime('%Y-%m-%d')}:")
        self.logger.info(f"   Note: Using Relative Strength (RS) score (Equal Weight)")
        self.logger.performance(f"   Note: Breakdown values show Outperformance (Excess Return vs Index)")
        self.logger.performance(f"         Week: 1-week return (5 days) outperformance")
        self.logger.performance(f"         Month: 1-month return (20 days) outperformance")
        self.logger.performance(f"         Quarter: 1-quarter return (60 days) outperformance")

        for i, symbol in enumerate(symbols):
            try:
                etf_prices = etf_data.loc[symbol]['adjusted_close']
                result = self.calculate_rs_score(etf_prices, index_data['adjusted_close'], signal_date, symbol)
                
                if result is not None:
                    rs_score, rs_w, rs_m, rs_q = result
                    valid_symbols += 1
                    
                    # Log the score immediately as we process
                    # We will re-print sorted later, but this confirms calculation
                    # Actually, better to store and print RANKED list at the end or here? 
                    # The user example shows Ranked list. The loop iterates symbols in arbitrary order.
                    # So we should collect then print. But we can print individual items as processed if debugging.
                    # User request: "Stocks Ranked by RS Score". The loop is NOT ranked yet.
                    # So I should silence per-item print here and print sorted list later?
                    # BUT `rank_etfs` returns a DF. The sorting happens at end of function.
                    # The previous code printed "    {symbol}: RS Score = ..." immediately.
                    # User wants: "   Stocks Ranked by RS Score..."
                    # This function constructs the rankings. The printing should ideally happen AFTER sorting.
                    # However, to avoid major structural changes, I will collect them and print top N entries at end of function or outside?
                    # The `generate_signals` calls this. 
                    # Let's keep it simple: Just calculate here.
                    
                    # Get additional ranking criteria
                    current_price = etf_prices.loc[signal_date]
                    
                    # Calculate 20-day volatility (simplified)
                    try:
                        price_20d = etf_prices.loc[signal_date - timedelta(days=20):signal_date]
                        volatility = price_20d.pct_change().std() * np.sqrt(252) if len(price_20d) > 1 else 0
                    except:
                        volatility = 0
                    
                    # Get market cap (simplified - would need actual data)
                    market_cap = current_price * 1000000  # Placeholder
                    
                    rankings.append({
                        'symbol': symbol,
                        'rs_score': rs_score,
                        'rs_w': rs_w,
                        'rs_m': rs_m,
                        'rs_q': rs_q,
                        'current_price': current_price,
                        'volatility': volatility,
                        'market_cap': market_cap
                    })
                else:
                    failed_symbols += 1
                    
            except (KeyError, IndexError) as e:
                failed_symbols += 1
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
        
        # Print Ranked List matching user request format
        self.logger.progress(f"   ✅ ETFs Ranked by RS Score (Equal Weight) - Highest to Lowest:")
        
        for index, row in df.iterrows():
            self.logger.info(f"      {int(row['rank'])}. {row['symbol']}: RS Score={row['rs_score']:.4f}, Price=₹{row['current_price']:.2f}")
            self.logger.info(f"         Breakdown (Outperf): Week (5d)={row['rs_w']:.3f}, Month (20d)={row['rs_m']:.3f}, Quarter (60d)={row['rs_q']:.3f}")
        
        return df
    
    def generate_signals(self, etf_data: pd.DataFrame, index_data: pd.DataFrame,
                        signal_date: datetime) -> Tuple[List[str], List[str]]:
        """Generate buy and sell signals for a given date"""
        # No price filtering - all ETFs eligible for RS ranking
        rankings = self.rank_etfs(etf_data, index_data, signal_date)
        
        if rankings.empty:
            self.logger.performance("  No rankings available, returning empty signals")
            return [], []
        
        # Select top N ETFs with positive RS scores (more opportunities)
        positive_rs = rankings[rankings['rs_score'] > 0]
        self.logger.info(f"  ETFs with positive RS: {len(positive_rs)}")
        
        # IMPROVEMENT: Filter out ETFs that have reached purchase limits
        eligible_indices = []
        for idx, row in positive_rs.iterrows():
            symbol = row['symbol']
            count = self.etf_purchase_counts.get(symbol, 0)
            limit = self.etf_purchase_limits.get(symbol, float('inf'))
            
            if count < limit:
                eligible_indices.append(idx)
            else:
                self._log("limit", f"⚠️ Signal filtered: {symbol} reached purchase limit ({count}/{limit})")
        
        # Create filtered dataframe
        positive_rs_filtered = positive_rs.loc[eligible_indices]
        if len(positive_rs) != len(positive_rs_filtered):
            self.logger.info(f"  Available after limit filter: {len(positive_rs_filtered)} (filtered {len(positive_rs) - len(positive_rs_filtered)})")
    
        # Select top positions for better diversification
        top_etfs = positive_rs_filtered.head(self.dynamic_params.max_positions)
        target_symbols = top_etfs['symbol'].tolist()
        self.logger.info(f"  Target symbols: {target_symbols}")
        
        # Current positions
        current_symbols = list(self.positions.keys())
        self.logger.info(f"  Current positions: {current_symbols}")
        
        # Determine exits (ETFs not in targets)
        exits = []
        for symbol in current_symbols:
            if symbol not in target_symbols:
                exits.append(symbol)
                self.logger.info(f"    Exit signal: {symbol} (not in targets)")
        
        # Determine entries (new ETFs to buy)
        entries = []
        for symbol in target_symbols:
            if symbol not in current_symbols:
                entries.append(symbol)
                self.logger.info(f"    Entry signal: {symbol}")
        
        self.logger.info(f"  Final signals: {len(entries)} entries, {len(exits)} exits")
        return entries, exits
    
    def calculate_transaction_costs(self, transaction_value: float, action: str, brokerage_pct: float = None):
        """Calculate detailed Indian market transaction costs"""
        if brokerage_pct is None:
            brokerage_pct = self.transaction_cost_pct * 100  # Convert to percentage
        
        # Base brokerage
        brokerage = transaction_value * (brokerage_pct / 100)
        
        # Indian market specific costs
        if action == "BUY":
            stamp_duty = transaction_value * 0.015 / 100 # 0.005%
            stt = transaction_value * 0.10 / 100
        else:  # SELL
            stamp_duty = 0  # No stamp duty on sell
            stt = transaction_value * 0.10 / 100  # 0.001%
        
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

    
    def update_positions(self, etf_data: pd.DataFrame, date: datetime):
        """Update current positions with latest prices"""
        for symbol, position in self.positions.items():
            try:
                current_price = etf_data.loc[symbol, date]['adjusted_close']
                position.current_price = current_price
                position.unrealized_pnl = (current_price - position.buy_price) * position.quantity
            except (KeyError, IndexError):
                continue
    
    def calculate_portfolio_value(self, etf_data: pd.DataFrame, date: datetime) -> float:
        """Calculate total portfolio value"""
        self.update_positions(etf_data, date)
        
        positions_value = sum(pos.current_price * pos.quantity for pos in self.positions.values())
        return self.cash_balance + positions_value
    
    def take_portfolio_snapshot(self, etf_data: pd.DataFrame, index_data: pd.DataFrame, date: datetime):
        """Take a snapshot of portfolio state with memory optimization"""
        total_value = self.calculate_portfolio_value(etf_data, date)
        
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
            
            # Transaction cost fields - CRITICAL FIX for saving cost data
            'transaction_value': trade.transaction_value,
            'brokerage': trade.brokerage,
            'stt': trade.stt,
            'stamp_duty': trade.stamp_duty,
            'exchange_charges': trade.exchange_charges,
            'sebi_charges': trade.sebi_charges,
            'gst': trade.gst,
            'total_costs': trade.total_costs,
            'net_amount': trade.net_amount
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
        
        self.logger.info(f"  Applying capital reset logic - reducing risk")
        
        # Reduce entries by 50%
        reduced_entries = entries[:len(entries)//2] if len(entries) > 1 else []
        
        # Add more exits to reduce portfolio size
        additional_exits = []
        if len(self.positions) > 5:  # If we have more than 5 positions
            # Exit positions with lowest RS scores
            current_positions = list(self.positions.keys())
            additional_exits = current_positions[5:]  # Keep only top 5 positions
        
        all_exits = exits + additional_exits
        
        self.logger.info(f"  Capital reset: {len(entries)} -> {len(reduced_entries)} entries, {len(exits)} -> {len(all_exits)} exits")
        
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
    
    def apply_minimum_price_filter(self, etf_data: pd.DataFrame, signal_date: datetime) -> List[str]:
        """Filter out ETFs below minimum price threshold"""
        filtered_symbols = []
        
        try:
            for symbol in etf_data.index.get_level_values('symbol').unique():
                try:
                    current_price = etf_data.loc[symbol, signal_date]['adjusted_close']
                    if current_price >= (self.min_price * 0.5):  # Relaxed minimum price filter
                        filtered_symbols.append(symbol)
                except (KeyError, IndexError):
                    continue
        except Exception as e:
            self.logger.info(f"Error applying minimum price filter: {e}")
            return list(etf_data.index.get_level_values('symbol').unique())
        
        self.logger.info(f"  Price filter: {len(etf_data.index.get_level_values('symbol').unique())} -> {len(filtered_symbols)} ETFs (min price: ₹{self.min_price})")
        
        # Add filtered list to allowed list if custom universe is used
        # This helps in universe management
        return filtered_symbols
     
    def run_backtest(self, start_date: datetime, end_date: datetime) -> Dict:
        """Run the complete backtest"""
        try:
            # Ensure dates are timezone-naive
            if start_date.tzinfo:
                start_date = start_date.replace(tzinfo=None)
            if end_date.tzinfo:
                end_date = end_date.replace(tzinfo=None)
            
            # IMPROVEMENT: Initialize purchase limits at start of backtest
            # Using a default of 13 weeks for accumulation calculation if not specified in config
            acc_weeks = getattr(self.config, 'accumulation_weeks', 13) if self.config else 13
            # Use the full potential universe for limit calculation
            self.initialize_purchase_limits(self._get_allowed_etf_list(), acc_weeks)

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
            self.logger.progress("Loading ETF data...")
            etf_data = self.load_etf_data(start_date, end_date)
            if etf_data.empty:
                raise ValueError("No ETF data available for the specified date range")
            
            self.logger.progress("Loading index data...")
            index_data = self.load_index_data(start_date, end_date)
            if index_data.empty:
                raise ValueError("No index data available for the specified date range")
            
            # Store index data for beta calculation
            self.index_data = index_data
            
            # Get trading dates - filter to backtest period only for processing
            all_trading_dates = sorted(etf_data.index.get_level_values('date').unique())
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
            
            # Run backtest
            self.logger.progress(f"Processing {len(trading_dates)} trading dates...")
            signal_count = 0
            
            for i, date in enumerate(trading_dates):
                if i % 20 == 0:  # Progress every 20 days
                    progress = (i / len(trading_dates)) * 100
                    self.logger.info(f"Progress: {progress:.1f}% ({i}/{len(trading_dates)}) - Date: {date}")
                
                try:
                    # Update positions with current prices
                    self.update_positions(etf_data, date)
                    
                    # Check for capital reset threshold
                    current_portfolio_value = self.calculate_portfolio_value(etf_data, date)
                    self.check_capital_reset_threshold(current_portfolio_value, date)
                    
                    # Removed max holding period check - stocks held until RS ranking drops
                    
                    # Stop Loss Check
                    if self.daily_stop_loss_check:
                        stop_loss_exits = self.check_daily_stop_loss(etf_data, date)
                        for symbol in stop_loss_exits:
                            try:
                                price_data = etf_data.loc[symbol, date]['adjusted_close']
                                price = float(price_data.iloc[0]) if hasattr(price_data, 'iloc') else float(price_data)
                                self.execute_trade(date, symbol, "SELL", price, "Stop Loss (Daily)")
                            except (KeyError, IndexError):
                                continue
                    else:
                        if not self.is_friday_or_last_trading_day(date):
                            stop_loss_exits = self.check_daily_stop_loss(etf_data, date)
                            for symbol in stop_loss_exits:
                                if symbol not in self.weekly_stop_loss_exits:
                                    self.weekly_stop_loss_exits.append(symbol)
                    
                    # Initialize entries and exits
                    entries, exits = [], []
                    
                    # Generate signals only on Friday (or last trading day of week)
                    if self.is_friday_or_last_trading_day(date): 

                        
                        # WEEKLY MODE: Check stop loss on signal day and add to exits
                        if not self.daily_stop_loss_check:
                            stop_loss_exits = self.check_daily_stop_loss(etf_data, date)
                            all_stop_loss_exits = list(set(self.weekly_stop_loss_exits + stop_loss_exits))
                            if all_stop_loss_exits:
                                self.logger.info(f"  📊 Weekly Stop Loss Summary: {len(all_stop_loss_exits)} position(s) to exit")
                        
                        entries, exits = self.generate_signals(etf_data, index_data, date)


                        # WEEKLY MODE: Add stop loss exits to regular exits
                        if not self.daily_stop_loss_check and all_stop_loss_exits:
                            exits = list(set(exits + all_stop_loss_exits))
                        
                        entries, exits = self.apply_capital_reset_logic(entries, exits)
                        
                        self.pending_entries = entries
                        self.pending_exits = exits
                        self.signal_date = date
                        
                        if not self.daily_stop_loss_check:
                            self.weekly_stop_loss_exits = []
                        
                        self.logger.info(f"📅 SIGNAL DAY: Friday {date.strftime('%Y-%m-%d (%A)')} - Generated {len(entries)} entries, {len(exits)} exits")
                        
                        # Calculate next execution day
                        next_monday = self.get_next_monday(date)

                        # Next line will be handled in next iteration or if date == next_monday (impossible if Fri != Mon)

                    # Execute trades on Monday (or next available trading day)
                    if hasattr(self, 'pending_entries') and self.pending_entries is not None:
                        next_monday = self.get_next_monday(self.signal_date)
                        if next_monday in trading_dates:
                            execution_day = next_monday
                        else:
                            execution_day = self.find_next_available_trading_day(next_monday, trading_dates)
                            #print(f"DEBUG: Next available trading day for execution: {execution_day} (Target: {next_monday})") # Too noisy
                        
                        #print(f"DEBUG: Checking execution for {date}: Target {execution_day}")
                        
                        if date == execution_day:

                            self.logger.progress(f"🔄 EXECUTION DAY: Monday {date.strftime('%Y-%m-%d (%A)')} - Executing {len(self.pending_exits)} exits and {len(self.pending_entries)} entries")
                            
                            # SELL FIRST
                            for symbol in self.pending_exits:
                                try:
                                    if 'open' in etf_data.columns:
                                        price_data = etf_data.loc[symbol, execution_day]['open']
                                    else:
                                        price_data = etf_data.loc[symbol, execution_day]['adjusted_close']
                                    price = float(price_data.iloc[0]) if hasattr(price_data, 'iloc') else float(price_data)
                                    self.execute_trade(execution_day, symbol, "SELL", price, "RS Exit")
                                except (KeyError, IndexError) as e:

                                    continue
                            
                            # THEN BUY
                            for symbol in self.pending_entries:
                                try:
                                    if 'open' in etf_data.columns:
                                        price_data = etf_data.loc[symbol, execution_day]['open']
                                    else:
                                        price_data = etf_data.loc[symbol, execution_day]['adjusted_close']
                                    price = float(price_data.iloc[0]) if hasattr(price_data, 'iloc') else float(price_data)
                                    self.execute_trade(execution_day, symbol, "BUY", price, "RS Signal")
                                except (KeyError, IndexError) as e:

                                    continue
                            
                            self.pending_entries = None
                            self.pending_exits = None
                    
                    # Record portfolio snapshot
                    portfolio_value = self.calculate_portfolio_value(etf_data, date)
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
                        'daily_pnl': 0,
                        'cumulative_pnl': portfolio_value - self.total_capital,
                        'drawdown_pct': 0
                    }
                    self.portfolio_snapshots.append(snapshot)
                    
                    if entries or exits:
                        signal_count += 1
                        
                except Exception as e:
                    self.logger.progress(f"Error processing date {date}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            self.logger.info(f"=== BACKTEST COMPLETE ===")
            self.logger.info(f"Total signal generations: {signal_count}")
            self.logger.trade(f"Total trades executed: {len(self.trades)}")
            self.logger.info(f"Final cash balance: {self.cash_balance:.1f}")
            self.logger.info(f"Final positions: {len(self.positions)}")
            
            # NEW: Print Purchase Limit Summary to Terminal
            print("\n" + "=" * 60)
            print("PURCHASE LIMIT SUMMARY")
            print("=" * 60)
            print(f"{'Ticker':<15} {'Purchases':<10} {'Limit':<10} {'Status':<15}")
            print("-" * 60)
            
            sorted_tickers = sorted(self.etf_purchase_limits.keys())
            for ticker in sorted_tickers:
                count = self.etf_purchase_counts.get(ticker, 0)
                limit = self.etf_purchase_limits.get(ticker, 0)
                status = "REACHED" if count >= limit else "OK"
                print(f"{ticker:<15} {count:<10} {limit:<10} {status:<15}")
            print("=" * 60 + "\n")
            
            # Calculate metrics (with default risk_free_rate, will be overridden by API)

            self.logger.progress("=== CALCULATING METRICS ===")
            metrics = self.calculate_metrics(risk_free_rate=6.0)
            
            return metrics
            
        except Exception as e:
            self.logger.info(f"Backtest failed: {e}")
            raise
    
    def calculate_portfolio_value(self, etf_data: pd.DataFrame, date: datetime) -> float:
        """Calculate total portfolio value at given date (including buffer capital)"""
        try:
            # Start with cash balance
            total_value = self.cash_balance
            
            # Add buffer capital
            total_value += self.buffer_capital
            
            # Add value of all positions
            for symbol, position in self.positions.items():
                try:
                    price_data = etf_data.loc[symbol, date]['adjusted_close']
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
    
    def trade_to_dict(self, trade: Trade) -> Dict:
        """Convert Trade object to dictionary for API response"""
        return {
            'date': trade.date.isoformat(),
            'symbol': trade.symbol,
            'action': trade.action,
            'quantity': trade.quantity,
            'price': trade.price,
            'amount': trade.amount,
            'reason': trade.reason,
            'rs_score': trade.rs_score,
            'rs_rank': trade.rs_rank,
            'transaction_value': trade.transaction_value,
            'brokerage': trade.brokerage,
            'stt': trade.stt,
            'stamp_duty': trade.stamp_duty,
            'exchange_charges': trade.exchange_charges,
            'sebi_charges': trade.sebi_charges,
            'gst': trade.gst,
            'total_costs': trade.total_costs,
            'net_amount': trade.net_amount,
            # New fields
            'portfolio_nav': trade.portfolio_nav,
            'buy_price': trade.buy_price,
            'capital_gain': trade.capital_gain,
            'capital_gain_pct': trade.capital_gain_pct,
            'holding_period_days': trade.holding_period_days,
            'capital_gains_tax': trade.capital_gains_tax,
            'net_profit_after_tax': trade.net_profit_after_tax
        }

    def calculate_metrics(self, risk_free_rate: float = 6.0) -> Dict:
        """Calculate comprehensive performance metrics with smart sampling"""
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
            
            # Annualized return (legacy calculation) - SAFE VERSION
            days = (end_date - start_date).days
            if days > 0 and start_value > 0:
                annualized_return = safe_float(((end_value / start_value) ** (365 / days) - 1) * 100)
            else:
                annualized_return = 0.0
            
            # Max drawdown - SAFE VERSION
            peak = snapshots['total_value'].expanding().max()
            drawdown = (snapshots['total_value'] - peak) / peak * 100
            # Handle division by zero and infinity values
            drawdown = drawdown.replace([float('inf'), float('-inf')], 0.0).fillna(0.0)
            max_drawdown = safe_float(drawdown.min())
            
            # Sharpe ratio (simplified) - SAFE VERSION
            returns_std = snapshots['returns'].std()
            returns_mean = snapshots['returns'].mean()
            if returns_std > 0 and not math.isnan(returns_std) and not math.isinf(returns_std):
                sharpe_ratio = safe_float(returns_mean / returns_std * safe_sqrt(252))
            else:
                sharpe_ratio = 0.0
            
            # Calculate Beta and Treynor ratio
            beta, treynor_ratio = self.calculate_beta_and_treynor(snapshots, risk_free_rate)
            
            # Calculate Calmar ratio: abs(CAGR / max_drawdown) if max_drawdown < 0
            calmar_ratio = abs(cagr / max_drawdown) if max_drawdown < 0 else 0.0
            
            # Calculate win rate
            win_rate = self.calculate_win_rate()
            
            # Smart snapshot sampling implementation
            # This reduces JSON payload size by >80% for long backtests while maintaining chart fidelity
            total_snapshots = len(self.portfolio_snapshots)
            if total_snapshots <= 500:
                # For short backtests, keep all snapshots
                snapshot_sample = self.portfolio_snapshots
            else:
                # For long backtests, sample intelligently
                # Keep first 50 (start details), last 50 (end details), and sample middle evenly
                step = max(1, (total_snapshots - 100) // 400)  # Target ~400 samples from middle
                
                indices = (list(range(50)) + 
                          list(range(50, total_snapshots - 50, step)) + 
                          list(range(total_snapshots - 50, total_snapshots)))
                          
                # Ensure unique, valid indices in order
                indices = sorted(list(set([i for i in indices if i < total_snapshots])))
                snapshot_sample = [self.portfolio_snapshots[i] for i in indices]
                
                self.logger.info(f"Smart sampling: Reduced snapshots from {total_snapshots} to {len(snapshot_sample)} for optimization")

            # Convert snapshots to JSON-safe format
            simplified_snapshots = []
            for snapshot in snapshot_sample:
                simplified_snapshot = {
                    'date': snapshot['date'].isoformat() if hasattr(snapshot['date'], 'isoformat') else str(snapshot['date']),
                    'total_value': float(snapshot.get('total_value', 0)),
                    'cash_balance': float(snapshot.get('cash_balance', 0)),
                    'drawdown_pct': float(snapshot.get('drawdown_pct', 0)),
                    # Omit detailed positions in list to save space, only send count
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
                # Use ALL original dates for accurate benchmark calculation, not sampled ones
                all_dates = [snap['date'] for snap in self.portfolio_snapshots]
                benchmark_calc = BenchmarkCalculator(
                    initial_capital=self.total_capital,
                    index_data=self.index_data,
                    trading_dates=all_dates
                )
                benchmark_metrics = benchmark_calc.calculate_benchmark_metrics(risk_free_rate)
                
                # Sample benchmark values to match snapshot sampling for charts
                full_benchmark_values = benchmark_calc.get_benchmark_values_array()
                if total_snapshots > 500 and len(full_benchmark_values) == total_snapshots:
                     benchmark_values = [full_benchmark_values[i] for i in indices]
                else:
                    benchmark_values = full_benchmark_values
                
                # Add benchmark data to metrics
                metrics['benchmark_metrics'] = benchmark_metrics
                metrics['benchmark_buyhold'] = benchmark_values
                metrics['alpha_pct'] = safe_float(cagr - benchmark_metrics['cagr_pct'])
                
                self.logger.performance(f"Benchmark Total Return: {benchmark_metrics['total_return_pct']:.2f}%")
                self.logger.info(f"Strategy Alpha: {metrics['alpha_pct']:.2f}%")
            except Exception as e:
                self.logger.progress(f"Error calculating benchmark metrics: {e}")
                metrics['benchmark_metrics'] = {}
                metrics['benchmark_buyhold'] = []
                metrics['alpha_pct'] = 0.0
            
            self.logger.info(f"Final metrics calculated: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.progress(f"Error calculating metrics: {e}")
            import traceback
            traceback.print_exc()
            return {
                'total_return_pct': 0.0,
                'final_capital': self.total_capital,
                'portfolio_snapshots': [],
                'trades': []
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

    def calculate_transaction_costs(self, transaction_value: float, action: str, brokerage_pct: float = None):
        """Calculate detailed Indian market transaction costs"""
        if brokerage_pct is None:
            # Default brokerage if not provided - use self.transaction_cost_pct which is set in __init__
            # self.transaction_cost_pct is stored as a fraction (e.g. 0.001), so multiply by 100 to get percentage
            if hasattr(self, 'transaction_cost_pct'):
                brokerage_pct = self.transaction_cost_pct * 100
            else:
                brokerage_pct = self.config.transaction_cost_pct * 100 if hasattr(self, 'config') and hasattr(self.config, 'transaction_cost_pct') else 0.1
        
        # Base brokerage
        brokerage = transaction_value * (brokerage_pct / 100)
        
        # Indian market specific costs
        if action == "BUY":
            stamp_duty = transaction_value * 0.015 / 100 # 0.015%
            stt = transaction_value * 0.001 / 100 # 0.001% as requested
        else:  # SELL
            stamp_duty = 0  # No stamp duty on sell
            stt = transaction_value * 0.001 / 100  # 0.001% as requested
        
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
                # Fixed position size
                fixed_position_size = self.per_trade_allocation
                
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
                
                 # NEW: Increment purchase counter
                self.etf_purchase_counts[symbol] = self.etf_purchase_counts.get(symbol, 0) + 1
                current_count = self.etf_purchase_counts[symbol]
                limit = self.etf_purchase_limits.get(symbol, 0)
                self._log("limit", f"📈 Purchase count incremented: {symbol} = {current_count}/{limit}")
                
                 # Print to terminal directly as requested
                print(f"📊 Purchase Count: {current_count}/{limit} ({ (current_count/limit)*100 if limit > 0 else 0:.0f}% used)")
                
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
                print(f"")
                
                # NEW: Decrement purchase counter
                if symbol in self.etf_purchase_counts:
                    self.etf_purchase_counts[symbol] = max(0, self.etf_purchase_counts[symbol] - 1)
                    new_count = self.etf_purchase_counts[symbol]
                    limit = self.etf_purchase_limits.get(symbol, 0)
                    self._log("limit", f"📉 Purchase count decremented: {symbol} = {new_count}/{limit}")
                
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

    def check_daily_stop_loss(self, etf_data: pd.DataFrame, current_date: datetime) -> List[str]:
        """Check stop loss for all positions and return ETFs to sell"""
        stop_loss_exits = []
        
        # Print daily check header
        if len(self.positions) > 0:
            self.logger.info(f"🛡️  Daily Stop Loss Check: {current_date.strftime('%Y-%m-%d (%A)')} - Checking {len(self.positions)} position(s)")
        else:
            self.logger.info(f"🛡️  Daily Stop Loss Check: {current_date.strftime('%Y-%m-%d (%A)')} - No positions to check")
            return stop_loss_exits  # Early return if no positions
        
        try:
            for symbol, position in self.positions.items():
                try:
                    # Get current price
                    price_data = etf_data.loc[symbol, current_date]['adjusted_close']
                    current_price = float(price_data.iloc[0]) if hasattr(price_data, 'iloc') else float(price_data)
                    
                    # Check if stop loss hit
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
                    # Stock data not available for this date
                    self.logger.info(f"  ⚠️  {symbol}: Data not available for stop loss check")
                    continue
                    
        except Exception as e:
            self.logger.info(f"  ❌ Error checking stop loss: {e}")
        
        # Print summary
        if len(stop_loss_exits) == 0:
            self.logger.progress(f"  ✅ No stop loss reached - All {len(self.positions)} position(s) safe")
        else:
            self.logger.trade(f"  ⚠️  {len(stop_loss_exits)} position(s) hit stop loss and will be sold")
        
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
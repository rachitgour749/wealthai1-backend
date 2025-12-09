import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sqlalchemy.orm import Session
try:
    from .database import ETFData, IndexData, StrategyConfig, BacktestResult, TradeLog, PortfolioSnapshot
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
    stop_loss_price: float

class RSETFStrategyBacktester:
    def __init__(self, db: Session, config_id: int = None, config_dict: Dict = None):
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
            self.stop_loss_pct = config_dict.get('stop_loss_pct', 15.0) / 100
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
        
        print(f"Loading ETF data from {data_start_date} to {end_date} (backtest period: {start_date} to {end_date})")
        
        # Calculate period length
        period_days = (end_date - start_date).days
        total_data_days = (end_date - data_start_date).days
        print(f"Backtest period: {period_days} days, Total data period: {total_data_days} days")
        
        # Get selected ETF symbols
        selected_etfs = self.get_custom_etf_universe()
        if not selected_etfs:
            raise ValueError("No ETFs selected for backtest")
        
        print(f"Selected ETFs for backtest: {len(selected_etfs)} ETFs")
        print(f"ETF symbols: {selected_etfs}")
        
        # Build symbol filter for SQL query
        placeholders = ','.join(['%s'] * len(selected_etfs))
        symbol_limit = f"AND symbol IN ({placeholders})"
        
        # Use raw SQL to query etf_data table with symbol filtering
        sql = f"""
        SELECT symbol, date, adjusted_close
        FROM etf_data 
        WHERE date >= %s AND date <= %s {symbol_limit}
        ORDER BY symbol, date
        """
        
        params = tuple([data_start_date, end_date] + selected_etfs)
        df = pd.read_sql(sql, self.db.bind, params=params, 
                        index_col=['symbol', 'date'], parse_dates=['date'])
        
        print(f"Raw ETF data query returned: {len(df)} records")
        
        # Convert timezone-aware dates to timezone-naive
        if df.index.get_level_values('date').tz is not None:
            df.index = df.index.set_levels(df.index.levels[1].tz_convert(None), level='date')
        
        print(f"Final ETF data: {len(df)} records, {len(df.index.get_level_values('symbol').unique())} ETFs")
        if not df.empty:
            print(f"Date range: {df.index.get_level_values('date').min()} to {df.index.get_level_values('date').max()}")
        
        return df
    
    def calculate_common_date_range(self, selected_etfs: List[str]) -> Tuple[Optional[str], Optional[str], float]:
        """Calculate common date range for selected ETFs by querying the database"""
        if not selected_etfs:
            return None, None, 0.0
        
        try:
            # Filter to only allowed ETFs (optional validation)
            allowed_etfs = self._get_allowed_etf_list()
            valid_etfs = [etf for etf in selected_etfs if etf in allowed_etfs]
            
            if not valid_etfs:
                print(f"Warning: No valid ETFs in selection. Allowed ETFs: {allowed_etfs}")
                return None, None, 0.0
            
            # Query database for min/max dates for each ETF
            etf_ranges = {}
            
            for etf in valid_etfs:
                # Query min and max dates for this ETF
                sql = """
                SELECT MIN(date) as min_date, MAX(date) as max_date
                FROM etf_data 
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
                print(f"No data found for selected ETFs: {valid_etfs}")
                return None, None, 0.0
            
            # Find the latest start date and earliest end date (common range)
            start_dates = [data['start_date'] for data in etf_ranges.values()]
            end_dates = [data['end_date'] for data in etf_ranges.values()]
            
            latest_start = max(start_dates)
            earliest_end = min(end_dates)
            
            print(f"📅 RS ETF Strategy Data Ranges:")
            for etf in valid_etfs:
                if etf in etf_ranges:
                    data = etf_ranges[etf]
                    days_available = (data['end_date'] - data['start_date']).days
                    years_available = days_available / 365.25
                    print(f"   {etf:12s}: {data['start_date'].strftime('%Y-%m-%d')} to {data['end_date'].strftime('%Y-%m-%d')} ({years_available:.1f} years)")
            
            print(f"📊 Common data range: {latest_start.strftime('%Y-%m-%d')} to {earliest_end.strftime('%Y-%m-%d')}")
            
            # Add buffer for RS ETF strategy calculations
            # RS strategy needs lookback periods, so add buffer
            buffer_weeks = 15  # 90 weeks = ~630 calendar days
            buffer_days = buffer_weeks * 7
            strategy_start = latest_start + timedelta(days=buffer_days)
            
            print(f"🎯 RS Strategy Buffer:")
            print(f"   Buffer period: {buffer_weeks} weeks ({buffer_days} calendar days)")
            print(f"   Strategy start (with buffer): {strategy_start.strftime('%Y-%m-%d')}")
            
            # Ensure we have valid range
            if strategy_start >= earliest_end:
                print(f"⚠️ Insufficient data with buffer, using latest_start as strategy start")
                strategy_start = latest_start
            
            # Calculate years available for backtesting
            years_available = (earliest_end - strategy_start).days / 365.25
            
            # Format dates as strings
            start_date_str = strategy_start.strftime('%Y-%m-%d')
            end_date_str = earliest_end.strftime('%Y-%m-%d')
            
            print(f"✅ Final date range: {start_date_str} to {end_date_str} ({years_available:.1f} years)")
            
            return start_date_str, end_date_str, years_available
            
        except Exception as e:
            print(f"Error calculating date range: {e}")
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
        
        print(f"Loading index data for {self.main_index} from {data_start_date} to {end_date} (backtest period: {start_date} to {end_date})")
        
        # Use raw SQL to avoid SQLAlchemy model issues
        # Handle multiple symbol formats for NSEI index
        symbol_variants = [self.main_index]
        if self.main_index in ['^NSEI', 'NSEI', 'NIFTY50']:
            symbol_variants = ['^NSEI', 'NSEI', 'NIFTY50']
        elif self.main_index in ['^NIFTY50', 'NIFTY50']:
            symbol_variants = ['^NIFTY50', 'NIFTY50', 'NSEI']
        
        placeholders = ','.join(['%s' for _ in symbol_variants])
        sql = f"""
        SELECT symbol, date, open, high, low, close, adj_close, volume
        FROM index_data 
        WHERE symbol IN ({placeholders}) AND date >= %s AND date <= %s
        ORDER BY date
        """
        
        params = tuple(symbol_variants + [data_start_date, end_date])
        df = pd.read_sql(sql, self.db.bind, params=params,
                        index_col='date', parse_dates=['date'])
        
        print(f"Raw index data query returned: {len(df)} records")
        
        
        # Rename adj_close to adjusted_close for consistency
        if 'adj_close' in df.columns:
            df = df.rename(columns={'adj_close': 'adjusted_close'})
        # Column is already named adjusted_close in MarketData.sqlite
        # No renaming needed
        
        # Convert timezone-aware dates to timezone-naive
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        
        print(f"Final index data: {len(df)} records")
        if not df.empty:
            print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df


    def get_nifty50_custom_stocks(self) -> List[str]:
        """Deprecated: This method is not used for ETF strategy. Use get_custom_etf_universe() instead."""
        # Return empty list - this method should not be called for ETF strategy
        return []
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
    
    # def get_nifty100_custom_stocks(self) -> List[str]:
    #     """Your best selected Nifty 100 stocks"""
    #     nifty50 = self.get_nifty50_custom_stocks()
    #     additional_stocks = [
    #         'VEDL', 'GAIL', 'IOC', 'PETRONET', 'BANDHANBNK', 
    #         'MOTHERSON', 'BOSCHLTD', 'BERGEPAINT', 'GODREJPROP', 'MFSL',
    #         'BIOCON', 'COLPAL', 'HDFCLIFE', 'ICICIGI', 'ICICIPRULI',
    #         'MARICO', 'MPHASIS', 'MRF', 'SAIL', 'SIEMENS',
    #         'TATACHEM', 'TATAPOWER', 'TORNTPHARM', 'TVSMOTOR', 'UBL',
    #         'ZEEL', 'ZYDUSLIFE', 'ABBOTINDIA', 'ADANIENT', 'ADANIGREEN',
    #         'ALKEM', 'AMBUJACEM', 'APLLTD', 'ASHOKLEY', 'ASTRAL',
    #         'ATUL', 'AUBANK', 'AUROPHARMA', 'BAJAJHLDNG', 'BALRAMCHIN',
    #         'BATAINDIA', 'BEL', 'BEML', 'BHARATFORG', 'BHEL',
    #         'BLUEDART', 'BLUESTARCO', 'CANBK', 'CENTRALBK', 'CESC'
    #     ]
    #     return nifty50 + additional_stocks
    
    # def get_nifty200_custom_stocks(self) -> List[str]:
    #     """Your best selected Nifty 200 stocks"""
    #     nifty100 = self.get_nifty100_custom_stocks()
    #     additional_stocks = [
    #         'BEML', 'BHARATFORG', 'CANFINHOME', 'CHOLAFIN', 'CONCOR',
    #         'CROMPTON', 'CUMMINSIND', 'DALBHARAT', 'DEEPAKNTR', 'DLF',
    #         'ESCORTS', 'EXIDEIND', 'FEDERALBNK', 'GLENMARK', 'GRANULES',
    #         'GUJGASLTD', 'HAL', 'HAVELLS', 'HDFCAMC', 'HINDPETRO',
    #         'IBULHSGFIN', 'IDBI', 'IDFCFIRSTB', 'IGL', 'INDHOTEL',
    #         'INDIANB', 'INDIGO', 'IPCALAB', 'IRCTC', 'JINDALSTEL',
    #         'JKCEMENT', 'JUBLFOOD', 'JUSTDIAL', 'LICHSGFIN', 'LTTS',
    #         'LUPIN', 'M&M', 'M&MFIN', 'MANAPPURAM', 'MUTHOOTFIN',
    #         'NATIONALUM', 'NAUKRI', 'NMDC', 'OBEROI', 'OFSS',
    #         'PAGEIND', 'PEL', 'PNB', 'POLYCAB', 'PVR',
    #         'RBLBANK', 'RECLTD', 'SBICARD', 'SBILIFE', 'SRF',
    #         'SUNTV', 'TRENT', 'VOLTAS', 'YESBANK', '3MINDIA',
    #         'ABB', 'ACC', 'ADANIPOWER', 'ADANITRANS', 'APOLLOTYRE',
    #         'ASHOKLEY', 'ASTRAL', 'ATUL', 'AUBANK', 'AUROPHARMA',
    #         'BAJAJHLDNG', 'BALRAMCHIN', 'BANDHANBNK', 'BATAINDIA', 'BEL',
    #         'BEML', 'BERGEPAINT', 'BHARATFORG', 'BIOCON', 'BOSCHLTD',
    #         'CADILAHC', 'CANFINHOME', 'CHOLAFIN', 'CIPLA', 'COLPAL',
    #         'CONCOR', 'CROMPTON', 'CUMMINSIND', 'DABUR', 'DALBHARAT',
    #         'DEEPAKNTR', 'DIVISLAB', 'DLF', 'DRREDDY', 'EICHERMOT',
    #         'ESCORTS', 'EXIDEIND', 'FEDERALBNK', 'GAIL', 'GLENMARK',
    #         # 'GODREJCP', 'GODREJPROP', 'GRANULES', 'GRASIM', 'GUJGASLTD',
    #         # 'HAL', 'HAVELLS', 'HCLTECH', 'HDFCAMC', 'HDFCLIFE',
    #         # 'HEROMOTOCO', 'HINDALCO', 'HINDPETRO', 'HINDUNILVR', 'IBULHSGFIN',
    #         # 'ICICIGI', 'ICICIPRULI', 'IDBI', 'IDFCFIRSTB', 'IGL',
    #         # 'INDHOTEL', 'INDIANB', 'INDIGO', 'INDUSINDBK', 'INFY',
    #         # 'IOC', 'IPCALAB', 'IRCTC', 'ITC', 'JINDALSTEL',
    #         # 'JKCEMENT', 'JSWSTEEL', 'JUBLFOOD', 'JUSTDIAL', 'KOTAKBANK',
    #         # 'LALPATHLAB', 'LICHSGFIN', 'LT', 'LTTS', 'LUPIN',
    #         # 'M&M', 'M&MFIN', 'MANAPPURAM', 'MARICO', 'MARUTI',
    #         # 'MCDOWELL-N', 'MFSL', 'MINDTREE', 'MOTHERSON', 'MPHASIS',
    #         # 'MRF', 'MUTHOOTFIN', 'NATIONALUM', 'NAUKRI', 'NESTLEIND',
    #         # 'NMDC', 'NTPC', 'OBEROI', 'OFSS', 'ONGC',
    #         # 'PAGEIND', 'PEL', 'PETRONET', 'PIDILITIND', 'PNB',
    #         # 'POLYCAB', 'POWERGRID', 'PVR', 'RBLBANK', 'RECLTD',
    #         # 'RELAXO', 'RELIANCE', 'SAIL', 'SBICARD', 'SBILIFE',
    #         # 'SBIN', 'SHREECEM', 'SIEMENS', 'SRF', 'SUNPHARMA',
    #         # 'SUNTV', 'TATACHEM', 'TATACONSUM', 'TATAMOTORS', 'TATAPOWER',
    #         # 'TATASTEEL', 'TCS', 'TECHM', 'TITAN', 'TORNTPHARM',
    #         # 'TRENT', 'TVSMOTORS', 'UBL', 'ULTRACEMCO', 'UPL',
    #         # 'VEDL', 'VOLTAS', 'WIPRO', 'YESBANK', 'ZEEL',
    #         # 'ZYDUSLIFE'
    #     ]
    #     return nifty100 + additional_stocks
    
    # def get_nifty300_custom_stocks(self) -> List[str]:
    #     """Your best selected Nifty 300 stocks"""
    #     nifty200 = self.get_nifty200_custom_stocks()
    #     additional_stocks = [
    #         '3MINDIA', 'ABB', 'ACC', 'ADANIENT', 'ADANIGREEN',
    #         'ADANIPORTS', 'ADANITRANS', 'ADANIPOWER', 'ALKEM', 'AMBUJACEM',
    #         'APLLTD', 'ASHOKLEY', 'ASTRAL', 'ATUL', 'AUBANK',
    #         'AUROPHARMA', 'BAJAJHLDNG', 'BALRAMCHIN', 'BANDHANBNK', 'BATAINDIA',
    #         'BEL', 'BEML', 'BERGEPAINT', 'BHARATFORG', 'BIOCON',
    #         'BOSCHLTD', 'CADILAHC', 'CANFINHOME', 'CHOLAFIN', 'CIPLA',
    #         'COLPAL', 'CONCOR', 'CROMPTON', 'CUMMINSIND', 'DABUR',
    #         'DALBHARAT', 'DEEPAKNTR', 'DIVISLAB', 'DLF', 'DRREDDY',
    #         'EICHERMOT', 'ESCORTS', 'EXIDEIND', 'FEDERALBNK', 'GAIL',
    #         'GLENMARK', 'GODREJCP', 'GODREJPROP', 'GRANULES', 'GRASIM',
    #         'GUJGASLTD', 'HAL', 'HAVELLS', 'HCLTECH', 'HDFCAMC',
    #         'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDPETRO', 'HINDUNILVR',
    #         'IBULHSGFIN', 'ICICIGI', 'ICICIPRULI', 'IDBI', 'IDFCFIRSTB',
    #         'IGL', 'INDHOTEL', 'INDIANB', 'INDIGO', 'INDUSINDBK',
    #         'INFY', 'IOC', 'IPCALAB', 'IRCTC', 'ITC',
    #         'JINDALSTEL', 'JKCEMENT', 'JSWSTEEL', 'JUBLFOOD', 'JUSTDIAL',
    #         'KOTAKBANK', 'LALPATHLAB', 'LICHSGFIN', 'LT', 'LTTS',
    #         'LUPIN', 'M&M', 'M&MFIN', 'MANAPPURAM', 'MARICO',
    #         'MARUTI', 'MCDOWELL-N', 'MFSL', 'MINDTREE', 'MOTHERSON',
    #         'MPHASIS', 'MRF', 'MUTHOOTFIN', 'NATIONALUM', 'NAUKRI',
    #         # 'NESTLEIND', 'NMDC', 'NTPC', 'OBEROI', 'OFSS',
    #         # 'ONGC', 'PAGEIND', 'PEL', 'PETRONET', 'PIDILITIND',
    #         # 'PNB', 'POLYCAB', 'POWERGRID', 'PVR', 'RBLBANK',
    #         # 'RECLTD', 'RELAXO', 'RELIANCE', 'SAIL', 'SBICARD',
    #         # 'SBILIFE', 'SBIN', 'SHREECEM', 'SIEMENS', 'SRF',
    #         # 'SUNPHARMA', 'SUNTV', 'TATACHEM', 'TATACONSUM', 'TATAMOTORS',
    #         # 'TATAPOWER', 'TATASTEEL', 'TCS', 'TECHM', 'TITAN',
    #         # 'TORNTPHARM', 'TRENT', 'TVSMOTORS', 'UBL', 'ULTRACEMCO',
    #         # 'UPL', 'VEDL', 'VOLTAS', 'WIPRO', 'YESBANK',
    #         # 'ZEEL', 'ZYDUSLIFE'
    #     ]
    #     return nifty200 + additional_stocks
    
    # def get_nifty500_custom_stocks(self) -> List[str]:
    #     """Your best selected Nifty 500 stocks"""
    #     nifty300 = self.get_nifty300_custom_stocks()
    #     additional_stocks = [
    #         '3MINDIA', 'ABB', 'ACC', 'ADANIENT', 'ADANIGREEN',
    #         'ADANIPORTS', 'ADANITRANS', 'ADANIPOWER', 'ALKEM', 'AMBUJACEM',
    #         'APLLTD', 'ASHOKLEY', 'ASTRAL', 'ATUL', 'AUBANK',
    #         'AUROPHARMA', 'BAJAJHLDNG', 'BALRAMCHIN', 'BANDHANBNK', 'BATAINDIA',
    #         'BEL', 'BEML', 'BERGEPAINT', 'BHARATFORG', 'BIOCON',
    #         'BOSCHLTD', 'CADILAHC', 'CANFINHOME', 'CHOLAFIN', 'CIPLA',
    #         'COLPAL', 'CONCOR', 'CROMPTON', 'CUMMINSIND', 'DABUR',
    #         'DALBHARAT', 'DEEPAKNTR', 'DIVISLAB', 'DLF', 'DRREDDY',
    #         'EICHERMOT', 'ESCORTS', 'EXIDEIND', 'FEDERALBNK', 'GAIL',
    #         'GLENMARK', 'GODREJCP', 'GODREJPROP', 'GRANULES', 'GRASIM',
    #         'GUJGASLTD', 'HAL', 'HAVELLS', 'HCLTECH', 'HDFCAMC',
    #         'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDPETRO', 'HINDUNILVR',
    #         'IBULHSGFIN', 'ICICIGI', 'ICICIPRULI', 'IDBI', 'IDFCFIRSTB',
    #         'IGL', 'INDHOTEL', 'INDIANB', 'INDIGO', 'INDUSINDBK',
    #         'INFY', 'IOC', 'IPCALAB', 'IRCTC', 'ITC',
    #         'JINDALSTEL', 'JKCEMENT', 'JSWSTEEL', 'JUBLFOOD', 'JUSTDIAL',
    #         'KOTAKBANK', 'LALPATHLAB', 'LICHSGFIN', 'LT', 'LTTS',
    #         'LUPIN', 'M&M', 'M&MFIN', 'MANAPPURAM', 'MARICO',
    #         'MARUTI', 'MCDOWELL-N', 'MFSL', 'MINDTREE', 'MOTHERSON',
    #         'MPHASIS', 'MRF', 'MUTHOOTFIN', 'NATIONALUM', 'NAUKRI',
    #         'NESTLEIND', 'NMDC', 'NTPC', 'OBEROI', 'OFSS',
    #         'ONGC', 'PAGEIND', 'PEL', 'PETRONET', 'PIDILITIND',
    #         'PNB', 'POLYCAB', 'POWERGRID', 'PVR', 'RBLBANK',
    #         'RECLTD', 'RELAXO', 'RELIANCE', 'SAIL', 'SBICARD',
    #         'SBILIFE', 'SBIN', 'SHREECEM', 'SIEMENS', 'SRF',
    #         'SUNPHARMA', 'SUNTV', 'TATACHEM', 'TATACONSUM', 'TATAMOTORS',
    #         'TATAPOWER', 'TATASTEEL', 'TCS', 'TECHM', 'TITAN',
    #         'TORNTPHARM', 'TRENT', 'TVSMOTORS', 'UBL', 'ULTRACEMCO',
    #         'UPL', 'VEDL', 'VOLTAS', 'WIPRO', 'YESBANK',
    #         # 'ZEEL', 'ZYDUSLIFE'
    #     ]
    #     return nifty300 + additional_stocks
    
    
    def get_custom_etf_universe(self) -> List[str]:
        """Get custom list of ETFs based on user selection or configuration"""
        try:
            # PRIORITY 1: Use custom_etfs if provided (user-selected subset)
            if hasattr(self, 'custom_etfs') and self.custom_etfs:
                # Filter to ensure only valid ETFs from our allowed list
                allowed_etfs = self._get_allowed_etf_list()
                filtered_etfs = [etf for etf in self.custom_etfs if etf in allowed_etfs]
                if filtered_etfs:
                    print(f"Using custom ETF selection: {len(filtered_etfs)} ETFs")
                    return filtered_etfs
                else:
                    print(f"Warning: Custom ETFs not in allowed list, using default list")
            
            # PRIORITY 2: Get from configuration
            if self.config and hasattr(self.config, 'custom_etfs') and self.config.custom_etfs:
                custom_etfs = json.loads(self.config.custom_etfs) if isinstance(self.config.custom_etfs, str) else self.config.custom_etfs
                # Filter to ensure only valid ETFs
                allowed_etfs = self._get_allowed_etf_list()
                filtered_etfs = [etf for etf in custom_etfs if etf in allowed_etfs]
                if filtered_etfs:
                    print(f"Using custom ETF universe from config: {len(filtered_etfs)} ETFs")
                    return filtered_etfs
            
            # PRIORITY 3: Return default allowed ETF list
            default_etfs = self._get_allowed_etf_list()
            print(f"Using default allowed ETF list: {len(default_etfs)} ETFs")
            return default_etfs
            
        except Exception as e:
            print(f"Error in get_custom_etf_universe: {e}")
            # Fallback to default list on error
            return self._get_allowed_etf_list()
    
    def _get_allowed_etf_list(self) -> List[str]:
        """Get the allowed list of ETFs for RS ETF Strategy"""
        # Hardcoded list of allowed ETFs only
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
                print(f"WARNING: Insufficient data. Need data from {earliest_required}, but only have from {earliest_data[0]}")
                print(f"Consider using shorter lookback periods or getting more historical data")
                return False
            
            return True
        except Exception as e:
            print(f"Error validating data range: {e}")
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
                          current_date: datetime, symbol: str = None) -> Optional[float]:
        """Calculate Relative Strength score for an ETF with robust data validation"""
        try:
            # Get available trading dates from actual data
            available_etf_dates = sorted(etf_prices.index)
            available_index_dates = sorted(index_prices.index)

            # Find current date index in both datasets
            try:
                current_etf_index = available_etf_dates.index(current_date) 
                current_index_index = available_index_dates.index(current_date) 
            except ValueError:
                # Current date not found in data
                return None

            # Get current prices
            current_etf_price = etf_prices.loc[current_date]
            current_index_price = index_prices.loc[current_date]
            
            # Calculate required lookback periods
            max_lookback = max(self.lookback_weeks, self.lookback_months, self.lookback_quarters)
            
            # DEBUG: Detailed Logging
            print(f"  DEBUG RS Calculation:")
            print(f"    Date: {current_date}")
            print(f"    Week: 1-week return (5 days) outperformance")
            print(f"    Month: 1-month return (20 days) outperformance")
            print(f"    Quarter: 1-quarter return (60 days) outperformance")
            print(f"    ETF data: {len(available_etf_dates)} dates, first: {available_etf_dates[0]}, last: {available_etf_dates[-1]}")              
            print(f"    Index data: {len(available_index_dates)} dates, first: {available_index_dates[0]}, last: {available_index_dates[-1]}")              
            print(f"    Current ETF index: {current_etf_index}")        
            print(f"    Current index position: {current_index_index}")     
            print(f"    Max lookback required: {max_lookback}")
            print(f"    Lookback periods: week={self.lookback_weeks}, month={self.lookback_months}, quarter={self.lookback_quarters}")                      

            # Check if we have enough historical data
            if current_etf_index < max_lookback or current_index_index < max_lookback:                                                                        
                print(f"      FAILED: Not enough historical data (need {max_lookback}, have ETF:{current_etf_index}, index:{current_index_index})")     
                return None

            # Get historical dates using actual available data
            week_ago_date = available_etf_dates[current_etf_index - self.lookback_weeks]                                                                    
            month_ago_date = available_etf_dates[current_etf_index - self.lookback_months]                                                                  
            quarter_ago_date = available_etf_dates[current_etf_index - self.lookback_quarters]                                                              

            # Verify index dates are available for the same periods
            week_ago_index = available_index_dates[current_index_index - self.lookback_weeks]
            month_ago_index = available_index_dates[current_index_index - self.lookback_months]
            quarter_ago_index = available_index_dates[current_index_index - self.lookback_quarters]
            
            # WEEK (Outperformance)
            etf_past_w = float(etf_prices.loc[week_ago_date])
            index_past_w = float(index_prices.loc[week_ago_index])
            etf_ret_w = (current_etf_price / etf_past_w) - 1
            index_ret_w = (current_index_price / index_past_w) - 1
            rs_w = etf_ret_w - index_ret_w
            print(f"      Week (5d): Date={week_ago_date.strftime('%Y-%m-%d')}, Price=₹{etf_past_w:.2f}, ETF={etf_ret_w:.2%}, Index={index_ret_w:.2%}, Outperf={rs_w:.4f}")

            # MONTH
            etf_past_m = float(etf_prices.loc[month_ago_date])
            index_past_m = float(index_prices.loc[month_ago_index])
            etf_ret_m = (current_etf_price / etf_past_m) - 1
            index_ret_m = (current_index_price / index_past_m) - 1
            rs_m = etf_ret_m - index_ret_m
            print(f"      Month (20d): Date={month_ago_date.strftime('%Y-%m-%d')}, Price=₹{etf_past_m:.2f}, ETF={etf_ret_m:.2%}, Index={index_ret_m:.2%}, Outperf={rs_m:.4f}")

            # QUARTER
            etf_past_q = float(etf_prices.loc[quarter_ago_date])
            index_past_q = float(index_prices.loc[quarter_ago_index])
            etf_ret_q = (current_etf_price / etf_past_q) - 1
            index_ret_q = (current_index_price / index_past_q) - 1
            rs_q = etf_ret_q - index_ret_q
            print(f"      Quarter (60d): Date={quarter_ago_date.strftime('%Y-%m-%d')}, Price=₹{etf_past_q:.2f}, ETF={etf_ret_q:.2%}, Index={index_ret_q:.2%}, Outperf={rs_q:.4f}")
            print(f"    RS values: week={rs_w:.3f}, month={rs_m:.3f}, quarter={rs_q:.3f}")
            
            # Return None if any RS calculation failed
            if any(rs is None for rs in [rs_w, rs_m, rs_q]):
                print(f"      FAILED: One or more RS calculations returned None")
                return None
            
            # Simple Equal-Weight Average (Week + Month + Quarter) / 3
            rs_score = (rs_w + rs_m + rs_q) / 3
            
            print(f"    Final RS score (Equal Weight): {rs_score:.4f}")
            
            return rs_score
            
        except (KeyError, IndexError, ValueError, ZeroDivisionError) as e:
            print(f"      FAILED: Exception - {str(e)}")
            return None
    
    def calculate_single_rs(self, etf_current: float, etf_past: float,
                           index_current: float, index_past: float) -> Optional[float]:
        """Calculate single period RS"""
        try:
            if etf_past == 0 or index_past == 0:
                return None
            rs = (etf_current / etf_past) / (index_current / index_past) - 1
            return rs
        except (ZeroDivisionError, ValueError):
            return None
    
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
    
    def rank_etfs(self, etf_data: pd.DataFrame, index_data: pd.DataFrame, 
                   signal_date: datetime) -> pd.DataFrame:
        """Rank all eligible ETFs by RS score with enhanced error handling"""
        symbols = self.get_custom_etf_universe()  # Use custom ETF selection
        rankings = []
        valid_symbols = 0
        failed_symbols = 0
        
        # Get universe size based on custom selection
        universe_size = len(symbols)
        print(f"Custom ETF universe size: {universe_size} ETFs and date: {signal_date.strftime('%Y-%m-%d (%A)')} ,day:{signal_date.weekday()}")
        
        print(f"Processing {universe_size} custom selected symbols for RS calculation")
        
        for i, symbol in enumerate(symbols):
            try:
                etf_prices = etf_data.loc[symbol]['adjusted_close']
                rs_score = self.calculate_rs_score(etf_prices, index_data['adjusted_close'], signal_date, symbol)
                
                if rs_score is not None:
                    valid_symbols += 1
                    # Print RS score for ALL ETFs (removed limitation)
                    print(f"    {symbol}: RS Score = {rs_score:.3f}")

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
                        'current_price': current_price,
                        'volatility': volatility,
                        'market_cap': market_cap
                    })
                else:
                    failed_symbols += 1
                    
            except (KeyError, IndexError) as e:
                failed_symbols += 1
                # Print failure for ALL ETFs (removed limitation)
                print(f"    {symbol}: Failed - '{str(e)}'")
                continue
        
        print(f"  Valid RS calculations: {valid_symbols}, Failed: {failed_symbols}")
        
        if valid_symbols == 0:
            print("  No valid rankings - insufficient historical data or all calculations failed")
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
    
    def generate_signals(self, etf_data: pd.DataFrame, index_data: pd.DataFrame,
                        signal_date: datetime) -> Tuple[List[str], List[str]]:
        """Generate buy and sell signals for a given date"""
        # No price filtering - all ETFs eligible for RS ranking
        rankings = self.rank_etfs(etf_data, index_data, signal_date)
        
        if rankings.empty:
            print("  No rankings available, returning empty signals")
            return [], []
        
        # Select top N ETFs with positive RS scores (more opportunities)
        positive_rs = rankings[rankings['rs_score'] > 0]
        print(f"  ETFs with positive RS: {len(positive_rs)}")
        
        # Select top positions for better diversification
        top_etfs = positive_rs.head(self.dynamic_params.max_positions)
        target_symbols = top_etfs['symbol'].tolist()
        print(f"  Target symbols: {target_symbols}")
        
        # Current positions
        current_symbols = list(self.positions.keys())
        print(f"  Current positions: {current_symbols}")
        
        # Determine exits (ETFs not in targets)
        exits = []
        for symbol in current_symbols:
            if symbol not in target_symbols:
                exits.append(symbol)
                print(f"    Exit signal: {symbol} (not in targets)")
        
        # Determine entries (new ETFs to buy)
        entries = []
        for symbol in target_symbols:
            if symbol not in current_symbols:
                entries.append(symbol)
                print(f"    Entry signal: {symbol}")
        
        print(f"  Final signals: {len(entries)} entries, {len(exits)} exits")
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
    
    # def run_backtest(self, start_date: datetime, end_date: datetime) -> Dict:
    #     """Run the complete backtest"""
    #     # Ensure dates are timezone-naive
    #     if start_date.tzinfo:
    #         start_date = start_date.replace(tzinfo=None)
    #     if end_date.tzinfo:
    #         end_date = end_date.replace(tzinfo=None)
            
    #     print(f"=== BACKTEST START ===")
    #     print(f"Starting backtest from {start_date} to {end_date}")
    #     print(f"Configuration ID: {self.config.id}")
    #     print(f"Total Capital: {self.total_capital}")
    #     print(f"Max Positions: {self.max_positions}")
    #     print(f"Position Size %: {self.position_size_pct}")
        
    #     # Validate data range before starting
    #     if not self.validate_data_range(start_date, end_date):
    #         print("WARNING: Proceeding with limited data range")
        
    #     # Load data
    #     print("Loading stock data...")
    #     etf_data = self.load_etf_data(start_date, end_date)
    #     print(f"ETF data loaded: {len(etf_data)} records, {len(etf_data.index.get_level_values('symbol').unique())} symbols")
        
    #     print("Loading index data...")
    #     index_data = self.load_index_data(start_date, end_date)
    #     print(f"Index data loaded: {len(index_data)} records")
        
    #     if etf_data.empty or index_data.empty:
    #         print("ERROR: No data available for backtest period")
    #         raise ValueError("No data available for backtest period")
        
    #     # Get all trading dates (should be timezone-naive now)
    #     trading_dates = sorted(etf_data.index.get_level_values('date').unique())
    #     print(f"Total trading dates: {len(trading_dates)}")
    #     print(f"First date: {trading_dates[0]}, Last date: {trading_dates[-1]}")
        
    #     # Calculate last trading days of each week
    #     self.calculate_last_trading_days(trading_dates)
        
    #     signal_count = 0
    #     trade_count = 0
    #     total_dates = len(trading_dates)
    #     progress_interval = max(1, total_dates // 20)  # Show progress every 5%
        
    #     print(f"Processing {total_dates} trading dates...")
        
    #     for i, date in enumerate(trading_dates):
    #         # Progress reporting for long backtests
    #         if i % progress_interval == 0 or i == total_dates - 1:
    #             progress_pct = (i / total_dates) * 100
    #             print(f"Progress: {progress_pct:.1f}% ({i}/{total_dates}) - Date: {date}")
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
    #             print(f"📅 SIGNAL DAY: Friday {date.strftime('%Y-%m-%d (%A)')} -> EXECUTION DAY: Monday {next_monday.strftime('%Y-%m-%d (%A)')}")
                
    #             entries, exits = self.generate_signals(stock_data, index_data, date)
                
    #             # Add max holding period exits
    #             exits.extend(max_holding_exits)
                
    #             # Apply capital reset logic
    #             entries, exits = self.apply_capital_reset_logic(entries, exits)
                
    #             # Execute trades on next Monday
    #             if next_monday <= end_date:
    #                 print(f"🔄 Executing {len(exits)} exits and {len(entries)} entries on Monday {next_monday.strftime('%Y-%m-%d')}")
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
        
    #     print(f"=== BACKTEST COMPLETE ===")
    #     print(f"Total signal generations: {signal_count}")
    #     print(f"Total trades executed: {trade_count}")
    #     print(f"Final cash balance: {self.cash_balance}")
    #     print(f"Final positions: {len(self.positions)}")
        
    #     # Calculate final metrics
    #     metrics = self.calculate_backtest_metrics()
    #     print(f"Final metrics: {metrics}")
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
    #     print(f"=== CALCULATING METRICS ===")
    #     print(f"Portfolio snapshots: {len(self.portfolio_snapshots)}")
    #     print(f"Total trades: {len(self.trades)}")
        
    #     if not self.portfolio_snapshots:
    #         print("ERROR: No portfolio snapshots available")
    #         return {}
        
    #     snapshots = pd.DataFrame(self.portfolio_snapshots)
    #     snapshots['returns'] = snapshots['total_value'].pct_change()
        
    #     print(f"Snapshot data shape: {snapshots.shape}")
    #     print(f"First portfolio value: {snapshots['total_value'].iloc[0]}")
    #     print(f"Last portfolio value: {snapshots['total_value'].iloc[-1]}")
        
    #     # Basic metrics - SAFE VERSION
    #     start_val = snapshots['total_value'].iloc[0]
    #     end_val = snapshots['total_value'].iloc[-1]
    #     total_return = safe_float((safe_divide(end_val, start_val, 1.0) - 1) * 100)
    #     print(f"Total return: {total_return}%")
        
    #     # Calculate CAGR
    #     start_date = snapshots['date'].iloc[0]
    #     end_date = snapshots['date'].iloc[-1]
    #     start_value = snapshots['total_value'].iloc[0]
    #     end_value = snapshots['total_value'].iloc[-1]
    #     cagr = self.calculate_cagr(start_value, end_value, start_date, end_date)
    #     print(f"CAGR: {cagr}%")
        
    #     # Calculate Rule of 72 metrics
    #     years = (end_date - start_date).days / 365.25
    #     rule_72_metrics = self.calculate_rule_of_72_metrics(cagr, years)
    #     print(f"Rule of 72 - Years to double: {rule_72_metrics['years_to_double']:.1f}")
    #     print(f"Rule of 72 - Expected return: {rule_72_metrics['rule_of_72_return']:.1f}%")
    #     print(f"Rule of 72 - Doublings: {rule_72_metrics['expected_doublings']:.2f}")
        
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
    #     print(f"XIRR: {xirr}%")
        
    #     # Annualized return (legacy calculation) - SAFE VERSION
    #     days = (end_date - start_date).days
    #     if days > 0 and start_value > 0:
    #         annualized_return = safe_float(((end_value / start_value) ** (365 / days) - 1) * 100)
    #     else:
    #         annualized_return = 0.0
    #     print(f"Annualized return: {annualized_return}% (over {days} days)")
        
    #     # Max drawdown - SAFE VERSION
    #     peak = snapshots['total_value'].expanding().max()
    #     drawdown = (snapshots['total_value'] - peak) / peak * 100
    #     # Handle division by zero and infinity values
    #     drawdown = drawdown.replace([float('inf'), float('-inf')], 0.0).fillna(0.0)
    #     max_drawdown = safe_float(drawdown.min())
    #     print(f"Max drawdown: {max_drawdown}%")
        
    #     # Sharpe ratio (simplified) - SAFE VERSION
    #     returns_std = snapshots['returns'].std()
    #     returns_mean = snapshots['returns'].mean()
    #     if returns_std > 0 and not math.isnan(returns_std) and not math.isinf(returns_std):
    #         sharpe_ratio = safe_float(returns_mean / returns_std * safe_sqrt(252))
    #     else:
    #         sharpe_ratio = 0.0
    #     print(f"Sharpe ratio: {sharpe_ratio}")
        
    #     # Win rate
    #     winning_trades = len([t for t in self.trades if t.action == "SELL" and t.amount > 0])
    #     total_sell_trades = len([t for t in self.trades if t.action == "SELL"])
    #     win_rate = (winning_trades / total_sell_trades * 100) if total_sell_trades > 0 else 0
    #     print(f"Win rate: {win_rate}% ({winning_trades}/{total_sell_trades})")
        
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
        
    #     print(f"Final metrics calculated: {metrics}")
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
                        print(f"  Capital reset deactivated - portfolio recovered to {current_portfolio_value:.0f}")
                        self.is_capital_reset_active = False
                        self.capital_reset_start_date = None
                        return False
            
            # Check if we need to trigger capital reset
            if not self.is_capital_reset_active:
                drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
                if drawdown >= self.capital_reset_threshold_pct:
                    print(f"  CAPITAL RESET TRIGGERED: Drawdown {drawdown:.1%} >= {self.capital_reset_threshold_pct:.1%}")
                    self.is_capital_reset_active = True
                    self.capital_reset_start_date = current_date
                    return True
            
            return self.is_capital_reset_active
            
        except Exception as e:
            print(f"Error in capital reset check: {e}")
            return False
    
    def apply_capital_reset_logic(self, entries: List[str], exits: List[str]) -> Tuple[List[str], List[str]]:
        """Apply capital reset logic by reducing positions and being more conservative"""
        if not self.is_capital_reset_active:
            return entries, exits
        
        print(f"  Applying capital reset logic - reducing risk")
        
        # Reduce entries by 50%
        reduced_entries = entries[:len(entries)//2] if len(entries) > 1 else []
        
        # Add more exits to reduce portfolio size
        additional_exits = []
        if len(self.positions) > 5:  # If we have more than 5 positions
            # Exit positions with lowest RS scores
            current_positions = list(self.positions.keys())
            additional_exits = current_positions[5:]  # Keep only top 5 positions
        
        all_exits = exits + additional_exits
        
        print(f"  Capital reset: {len(entries)} -> {len(reduced_entries)} entries, {len(exits)} -> {len(all_exits)} exits")
        
        return reduced_entries, all_exits
    
    def check_max_holding_period(self, current_date: datetime) -> List[str]:
        """Check for positions that have exceeded max holding period"""
        exits = []
        
        for symbol, position in self.positions.items():
            try:
                holding_weeks = (current_date - position.buy_date).days / 7
                if holding_weeks >= self.max_holding_period:
                    exits.append(symbol)
                    print(f"    Max holding period exit: {symbol} (held {holding_weeks:.1f} weeks)")
            except Exception as e:
                print(f"Error checking holding period for {symbol}: {e}")
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
            print(f"Error applying minimum price filter: {e}")
            return list(etf_data.index.get_level_values('symbol').unique())
        
        print(f"  Price filter: {len(etf_data.index.get_level_values('symbol').unique())} -> {len(filtered_symbols)} ETFs (min price: ₹{self.min_price})")
        return filtered_symbols
     
    def run_backtest(self, start_date: datetime, end_date: datetime) -> Dict:
        """Run the complete backtest"""
        try:
             # Ensure dates are timezone-naive
            if start_date.tzinfo:
                start_date = start_date.replace(tzinfo=None)
            if end_date.tzinfo:
                end_date = end_date.replace(tzinfo=None)
            
            print(f"=== BACKTEST START ===")
            print(f"Starting backtest from {start_date} to {end_date}")
            print(f"Configuration ID: {self.config.id if self.config else 'Custom Config'}")
            print(f"Total Capital: {self.total_capital}")
            print(f"Max Positions: {self.max_positions}")
            print(f"Position Size %: {self.position_size_pct}")
            
            # Validate date range
            try:
                if start_date >= end_date:
                    raise ValueError("Start date must be before end date")
                if (end_date - start_date).days < 30:
                    raise ValueError("Backtest period must be at least 30 days")
            except Exception as e:
                print(f"Error validating data range: {e}")
                raise
            
            # Load data
            print("Loading ETF data...")
            etf_data = self.load_etf_data(start_date, end_date)
            if etf_data.empty:
                raise ValueError("No ETF data available for the specified date range")
            
            print("Loading index data...")
            index_data = self.load_index_data(start_date, end_date)
            if index_data.empty:
                raise ValueError("No index data available for the specified date range")
            
            # Store index data for beta calculation
            self.index_data = index_data
            
            # Get trading dates - filter to backtest period only for processing
            all_trading_dates = sorted(etf_data.index.get_level_values('date').unique())
            trading_dates = [d for d in all_trading_dates if start_date <= d <= end_date]
            
            print(f"Total loaded dates: {len(all_trading_dates)} (from {all_trading_dates[0]} to {all_trading_dates[-1]})")
            print(f"Backtest trading dates: {len(trading_dates)} (from {trading_dates[0]} to {trading_dates[-1]})")
            
            # Calculate last trading days of each week for Friday signal generation
            self.calculate_last_trading_days(trading_dates)
            print(f"Last trading days calculated: {len(self.last_trading_days)} days")
            
            # Initialize portfolio snapshot
            self.portfolio_snapshots = []
            self.trades = []
            self.positions = {}
            # Reset buffer and cash to initial values
            self.buffer_capital = self.total_capital * self.buffer_capital_pct
            self.cash_balance = self.total_capital - self.buffer_capital
            
            # Run backtest
            print(f"Processing {len(trading_dates)} trading dates...")
            signal_count = 0
            
            for i, date in enumerate(trading_dates):
                if i % 20 == 0:  # Progress every 20 days
                    progress = (i / len(trading_dates)) * 100
                    print(f"Progress: {progress:.1f}% ({i}/{len(trading_dates)}) - Date: {date}")
                
                try:
                    # Update positions with current prices
                    self.update_positions(etf_data, date)
                    
                    # Check for capital reset threshold
                    current_portfolio_value = self.calculate_portfolio_value(etf_data, date)
                    self.check_capital_reset_threshold(current_portfolio_value, date)
                    
                    # Removed max holding period check - stocks held until RS ranking drops
                    
                    # Check for daily stop loss exits
                    stop_loss_exits = self.check_daily_stop_loss(etf_data, date)
                    
                    # Execute stop loss exits immediately
                    for symbol in stop_loss_exits:
                        try:
                            price_data = etf_data.loc[symbol, date]['adjusted_close']
                            price = float(price_data.iloc[0]) if hasattr(price_data, 'iloc') else float(price_data)
                            self.execute_trade(date, symbol, "SELL", price, "Stop Loss")
                        except (KeyError, IndexError):
                            continue
                    
                    # Initialize entries and exits for each day
                    entries, exits = [], []
                    
                    # Generate signals only on Friday (or last trading day of week)
                    if self.is_friday_or_last_trading_day(date):  # Generate signals only on Friday
                        entries, exits = self.generate_signals(etf_data, index_data, date)
                        
                        # Apply capital reset logic
                        entries, exits = self.apply_capital_reset_logic(entries, exits)
                        
                        # Removed max holding period exits - ETFs only exit based on RS ranking
                        
                        # Store signals for Monday execution (don't execute immediately)
                        self.pending_entries = entries
                        self.pending_exits = exits
                        self.signal_date = date
                        print(f"📅 SIGNAL DAY: Friday {date.strftime('%Y-%m-%d (%A)')} - Generated {len(entries)} entries, {len(exits)} exits")
                    
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
                            print(f"🔄 EXECUTION DAY: Monday {date.strftime('%Y-%m-%d (%A)')} - Executing {len(self.pending_exits)} exits and {len(self.pending_entries)} entries")
                            
                            # SELL FIRST to free up cash before buying new positions
                            for symbol in self.pending_exits:
                                try:
                                    price_data = etf_data.loc[symbol, execution_day]['adjusted_close']
                                    price = float(price_data.iloc[0]) if hasattr(price_data, 'iloc') else float(price_data)
                                    self.execute_trade(execution_day, symbol, "SELL", price, "RS Exit")
                                except (KeyError, IndexError):
                                    continue
                            
                            # THEN BUY using freed cash (and buffer only if needed)
                            for symbol in self.pending_entries:
                                try:
                                    price_data = etf_data.loc[symbol, execution_day]['adjusted_close']
                                    price = float(price_data.iloc[0]) if hasattr(price_data, 'iloc') else float(price_data)
                                    self.execute_trade(execution_day, symbol, "BUY", price, "RS Signal")
                                except (KeyError, IndexError):
                                    continue
                            
                            # Clear pending signals after execution
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
                        'daily_pnl': 0,  # Will be calculated later
                        'cumulative_pnl': portfolio_value - self.total_capital,
                        'drawdown_pct': 0  # Will be calculated later
                    }
                    self.portfolio_snapshots.append(snapshot)
                    
                    if entries or exits:
                        signal_count += 1
                        
                except Exception as e:
                    print(f"Error processing date {date}: {e}")
                    continue
            
            print(f"=== BACKTEST COMPLETE ===")
            print(f"Total signal generations: {signal_count}")
            print(f"Total trades executed: {len(self.trades)}")
            print(f"Final cash balance: {self.cash_balance:.1f}")
            print(f"Final positions: {len(self.positions)}")
            
            # Calculate metrics (with default risk_free_rate, will be overridden by API)
            print("=== CALCULATING METRICS ===")
            metrics = self.calculate_metrics(risk_free_rate=6.0)
            
            return metrics
            
        except Exception as e:
            print(f"Backtest failed: {e}")
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
            print(f"Error calculating portfolio value: {e}")
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
        """Calculate comprehensive performance metrics"""
        try:
            if not self.portfolio_snapshots:
                print("ERROR: No portfolio snapshots available")
                return {}
            
            snapshots = pd.DataFrame(self.portfolio_snapshots)
            snapshots['returns'] = snapshots['total_value'].pct_change()
            
            print(f"Snapshot data shape: {snapshots.shape}")
            print(f"First portfolio value: {snapshots['total_value'].iloc[0]}")
            print(f"Last portfolio value: {snapshots['total_value'].iloc[-1]}")
            
            # Basic metrics - SAFE VERSION
            start_val = snapshots['total_value'].iloc[0]
            end_val = snapshots['total_value'].iloc[-1]
            total_return = safe_float((safe_divide(end_val, start_val, 1.0) - 1) * 100)
            print(f"Total return: {total_return}%")
            
            # Calculate CAGR
            start_date = snapshots['date'].iloc[0]
            end_date = snapshots['date'].iloc[-1]
            start_value = snapshots['total_value'].iloc[0]
            end_value = snapshots['total_value'].iloc[-1]
            cagr = self.calculate_cagr(start_value, end_value, start_date, end_date)
            print(f"CAGR: {cagr}%")
            
            # Calculate Rule of 72 metrics
            years = (end_date - start_date).days / 365.25
            rule_72_metrics = self.calculate_rule_of_72_metrics(cagr, years)
            print(f"Rule of 72 - Years to double: {rule_72_metrics['years_to_double']:.1f}")
            print(f"Rule of 72 - Expected return: {rule_72_metrics['rule_of_72_return']:.1f}%")
            print(f"Rule of 72 - Doublings: {rule_72_metrics['expected_doublings']:.2f}")
            
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
            print(f"XIRR: {xirr}%")
            
            # Calculate annualized return
            annualized_return = safe_float((safe_power(safe_divide(end_value, start_value, 1.0), safe_divide(365.25, (end_date - start_date).days, 1.0)) - 1) * 100)
            print(f"Annualized return: {annualized_return}% (over {(end_date - start_date).days} days)")
            
            # Calculate maximum drawdown
            max_drawdown = self.calculate_max_drawdown(snapshots)
            print(f"Max drawdown: {max_drawdown}%")
            
            # Calculate Sharpe ratio
            sharpe_ratio = self.calculate_sharpe_ratio(snapshots)
            print(f"Sharpe ratio: {sharpe_ratio}")
            
            # Calculate Beta and Treynor ratio
            beta, treynor_ratio = self.calculate_beta_and_treynor(snapshots, risk_free_rate)
            print(f"Beta: {beta:.2f}")
            print(f"Treynor ratio: {treynor_ratio:.2f}%")
            
            # Calculate Calmar ratio: abs(CAGR / max_drawdown) if max_drawdown < 0
            calmar_ratio = abs(cagr / max_drawdown) if max_drawdown < 0 else 0.0
            print(f"Calmar ratio: {calmar_ratio:.2f}")
            
            # Calculate win rate
            win_rate = self.calculate_win_rate()
            print(f"Win rate: {win_rate}% ({len([t for t in self.trades if t.action == 'SELL' and self.get_trade_pnl(t) > 0])}/{len([t for t in self.trades if t.action == 'SELL'])})")
            
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
            print("=== CALCULATING BENCHMARK METRICS ===")
            try:
                trading_dates = [snap['date'] for snap in self.portfolio_snapshots]
                benchmark_calc = BenchmarkCalculator(
                    initial_capital=self.total_capital,
                    index_data=self.index_data,
                    trading_dates=trading_dates
                )
                benchmark_metrics = benchmark_calc.calculate_benchmark_metrics(risk_free_rate)
                benchmark_values = benchmark_calc.get_benchmark_values_array()
                
                # Add benchmark data to metrics
                metrics['benchmark_metrics'] = benchmark_metrics
                metrics['benchmark_buyhold'] = benchmark_values
                metrics['alpha_pct'] = safe_float(cagr - benchmark_metrics['cagr_pct'])
                
                print(f"Benchmark Total Return: {benchmark_metrics['total_return_pct']:.2f}%")
                print(f"Benchmark CAGR: {benchmark_metrics['cagr_pct']:.2f}%")
                print(f"Strategy Alpha: {metrics['alpha_pct']:.2f}%")
            except Exception as e:
                print(f"Error calculating benchmark metrics: {e}")
                metrics['benchmark_metrics'] = {}
                metrics['benchmark_buyhold'] = []
                metrics['alpha_pct'] = 0.0
            
            print(f"Final metrics calculated: {metrics}")
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
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
            print(f"Error calculating CAGR: {e}")
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
            print(f"Error calculating Rule of 72 metrics: {e}")
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
            print(f"Error calculating XIRR: {e}")
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
            print(f"Error calculating max drawdown: {e}")
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
            print(f"Error calculating Sharpe ratio: {e}")
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
            print(f"Error calculating beta and Treynor ratio: {e}")
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
            print(f"Error calculating win rate: {e}")
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
            print(f"Error calculating trade P&L: {e}")
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
            print(f"Error calculating portfolio NAV: {e}")
            # Fallback to total capital
            return self.total_capital

    def calculate_transaction_costs(self, transaction_value: float, action: str, brokerage_pct: float = None):
        """Calculate detailed Indian market transaction costs"""
        if brokerage_pct is None:
            # Default brokerage if not provided
            brokerage_pct = self.config.transaction_cost_pct * 100 if hasattr(self, 'config') and hasattr(self.config, 'transaction_cost_pct') else 0.0
        
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
                print(f"  Invalid price for {symbol}: {price}")
                return False
            if action == "BUY":
                # Fixed position size
                fixed_position_size = self.per_trade_allocation
                
                # Check if we have enough capital (cash + buffer)
                total_available = self.cash_balance + self.buffer_capital
                
                if total_available < fixed_position_size:
                    print(f"  Insufficient capital to buy {symbol}: {total_available} < {fixed_position_size}")
                    return False
                
                # Calculate quantity
                quantity = int(fixed_position_size / price)
                if quantity <= 0:
                    print(f"  Cannot buy {symbol}: quantity would be {quantity}")
                    return False
                
                # Calculate transaction costs using proper Indian market calculation
                transaction_value = quantity * price
                cost_details = self.calculate_transaction_costs(transaction_value, "BUY")
                net_amount = cost_details['net_amount']
                
                # Use cash first, then buffer if needed
                if self.cash_balance >= net_amount:
                    # Use only cash
                    self.cash_balance -= net_amount
                    print(f"  Used cash: ₹{net_amount:.2f}")
                else:
                    # Use cash + buffer
                    original_cash_used = self.cash_balance  # Store original cash before resetting
                    needed_from_buffer = net_amount - self.cash_balance
                    self.cash_balance = 0
                    self.buffer_capital -= needed_from_buffer
                    print(f"  Used cash: ₹{original_cash_used:.2f}, buffer: ₹{needed_from_buffer:.2f}")
                
                # Calculate stop loss price
                stop_loss_price = price * (1 - self.stop_loss_pct)
                
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
                
                print(f"📋 Purchase calculation:")
                print(f"   Gross amount: ₹{cost_details['transaction_value']:,.2f}")
                print(f"   Transaction costs: ₹{cost_details['total_costs']:,.2f}")
                print(f"   Net amount: ₹{net_amount:,.2f}")
                print(f"   Units to buy: {quantity}")
                print(f"✅ Purchase executed: {quantity} units of {symbol} for ₹{net_amount:,.2f}")
                print(f"💳 Total cost (including fees): ₹{cost_details['total_costs']:,.2f}")
                print(f"📊 Portfolio NAV: ₹{portfolio_nav:,.2f}")
                print(f"💰 Buy Price: ₹{price:,.2f}")
                print(f"💰 Remaining cash: ₹{self.cash_balance:,.2f}")
                return True
                
            elif action == "SELL":
                if symbol not in self.positions:
                    print(f"  Cannot sell {symbol}: not in positions")
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
                
                # Adjust buffer capital based on P&L
                if pnl > 0:
                    # Profit: Add to buffer
                    self.buffer_capital += pnl
                    print(f"  Profit: ₹{pnl:.2f} added to buffer")
                else:
                    # Loss: Subtract from buffer
                    self.buffer_capital += pnl  # pnl is negative, so this subtracts
                    print(f"  Loss: ₹{abs(pnl):.2f} subtracted from buffer")
                
                # Add proceeds to cash
                self.cash_balance += net_amount
                
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
                print(f"💰 SELL EXECUTED: {symbol}")
                print(f"   Quantity: {quantity} units")
                print(f"   Sell Price: ₹{price:,.2f}")
                print(f"   Buy Price: ₹{buy_price:,.2f}")
                print(f"   Holding Period: {holding_period_days} days")
                print(f"")

                print(f"📈 PORTFOLIO STATUS:")
                print(f"   Portfolio NAV: ₹{portfolio_nav:,.2f}")
                print(f"   Cash Balance: ₹{self.cash_balance:,.2f}")
                print(f"   Buffer Capital: ₹{self.buffer_capital:,.2f}")
                print(f"")
                return True
                
        except Exception as e:
            print(f"Error executing trade {action} {symbol}: {e}")
            return False

    def check_daily_stop_loss(self, etf_data: pd.DataFrame, current_date: datetime) -> List[str]:
        """Check stop loss for all positions and return ETFs to sell"""
        stop_loss_exits = []
        
        # Print daily check header
        if len(self.positions) > 0:
            print(f"🛡️  Daily Stop Loss Check: {current_date.strftime('%Y-%m-%d (%A)')} - Checking {len(self.positions)} position(s)")
        else:
            print(f"🛡️  Daily Stop Loss Check: {current_date.strftime('%Y-%m-%d (%A)')} - No positions to check")
            return stop_loss_exits  # Early return if no positions
        
        try:
            for symbol, position in self.positions.items():
                try:
                    # Get current price
                    price_data = etf_data.loc[symbol, current_date]['adjusted_close']
                    current_price = float(price_data.iloc[0]) if hasattr(price_data, 'iloc') else float(price_data)
                    
                    # Check if stop loss hit
                    if current_price <= position.stop_loss_price:
                        stop_loss_exits.append(symbol)
                        loss_pct = ((current_price - position.buy_price) / position.buy_price * 100)
                        print(f"  ⚠️  STOP LOSS HIT: {symbol} - Current: ₹{current_price:.2f} <= Stop Loss: ₹{position.stop_loss_price:.2f} (Loss: {loss_pct:.2f}%)")
                    else:
                        # Print that stop loss not reached for this stock
                        print(f"  ✓ {symbol}: Current ₹{current_price:.2f} > Stop Loss ₹{position.stop_loss_price:.2f} (Safe)")
                        
                except (KeyError, IndexError):
                    # Stock data not available for this date
                    print(f"  ⚠️  {symbol}: Data not available for stop loss check")
                    continue
                    
        except Exception as e:
            print(f"  ❌ Error checking stop loss: {e}")
        
        # Print summary
        if len(stop_loss_exits) == 0:
            print(f"  ✅ No stop loss reached - All {len(self.positions)} position(s) safe")
        else:
            print(f"  ⚠️  {len(stop_loss_exits)} position(s) hit stop loss and will be sold")
        
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
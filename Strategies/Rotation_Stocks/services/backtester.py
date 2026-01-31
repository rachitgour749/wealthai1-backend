import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import warnings
import os
import sys
import json
warnings.filterwarnings('ignore')

# Import market data database connection
current_dir = os.path.dirname(os.path.abspath(__file__))
# Adjust path to reach Databases from Strategies/Rotation_Stocks/services
# current_dir is .../Strategies/Rotation_Stocks/services
# parent_dir is .../Strategies/Rotation_Stocks
# grand_parent is .../Strategies
# great_grand is .../wealthai1-backend
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
# Actually, let's use relative path logic consistent with project structure
# Databases is in wealthai1-backend/Databases
# We are in wealthai1-backend/Strategies/Rotation_Stocks/services
# So we need to go up 3 levels
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
databases_path = os.path.join(project_root, 'Databases')
if databases_path not in sys.path:
    sys.path.insert(0, databases_path)

from app_data_db_connection import (
    create_connection as create_app_data_connection,
    get_session as get_app_data_session,
    init_database as init_app_data_database,
    StockMarket as StockData,  # Primary Stock data model (uses stock_market table)
    Nifty50IndexMarket as IndexData  # Index data for RS calculations (uses nifty_50_index_market table)
)

# Import base class for OOP refactoring
from Strategies.Rotation.RotationStrategy import RotationStrategy
from Strategies.utilities.logging_config import StrategyLogger

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Chart functionality will be disabled.")


class StockRotationBacktester(RotationStrategy):
    def __init__(self, db_path: str = None):
        """
        Initialize Stock Rotation Backtester
        
        Args:
            db_path: Deprecated - kept for compatibility. Now uses PostgreSQL MarketData database.
        """
        # Call base class constructor (initializes common attributes)
        super().__init__()
        
        # Initialize centralized logger AFTER parent init to prevent override
        self.logger = StrategyLogger('Rotation_Stocks')
        
        # Initialize PostgreSQL connection to ApplicationData
        if not create_app_data_connection():
            raise RuntimeError("Failed to connect to ApplicationData database")
        
        # Initialize database tables
        init_app_data_database()
        
        self.portfolio_log = []
        self.weekly_nav_df = pd.DataFrame()
        self.nifty50_df = pd.DataFrame()
        self.transaction_costs_log = []
        self.purchase_history = {}
        
        # Strategy-specific table configuration (must be set before method calls)
        # Strategy-specific table configuration (must be set before method calls)
        self.data_table = "stock_market"  # Stocks are stored in stock_market table
        
        self.available_databases = []  # Not needed for PostgreSQL
        self.stock_metadata = self.load_metadata()
        
        # Trade execution tracking
        self.skipped_days = []
        self.total_weeks = 0
        self.successful_signals = 0
        self.successful_executions = 0
        self.current_cash = 0
        self.current_holdings = {}
        
        # Performance optimizations
        self._data_cache = {}
        self._verbose = False  # Set to True for debugging


    # ============================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS (from BaseRotationBacktester)
    # ============================================================================
    
    def _get_data_model(self):
        """Return the SQLAlchemy model for Stock data"""
        return StockData

    def set_verbose(self, verbose: bool = True):
        """Enable or disable verbose logging for debugging"""
        self._verbose = verbose

    def _get_session(self):
        """Get database session for PostgreSQL (ApplicationData)"""
        return get_app_data_session()
    
    def load_data(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Implementation of abstract method from WealthAIBase.
        
        Loads data for all available stocks in metadata.
        For specific ticker loading, use load_data_from_database() instead.
        
        Args:
            start_date: Start date for data loading
            end_date: End date for data loading
            
        Returns:
            Dictionary with 'open', 'high', 'low', 'close', 'volume' DataFrames
        """
        # Get all available tickers from metadata
        if not self.stock_metadata:
            self.stock_metadata = self.load_metadata()
        
        tickers = list(self.stock_metadata.keys())
        
        # Convert datetime to string format expected by load_data_from_database
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Delegate to the more specific method
        return self.load_data_from_database(tickers, start_date_str, end_date_str)
    
    def get_trading_statistics(self) -> Dict:
        """Get trading statistics from portfolio log"""
        if not self.portfolio_log:
            return {
                "total_trades": 0,
                "buy_trades": 0,
                "sell_trades": 0,
                "churn_trades": 0,
                "no_trade_weeks": 0,
                "trading_frequency": "0%"
            }
        
        total_trades = len(self.portfolio_log)
        buy_trades = sum(1 for log in self.portfolio_log if log['action'] == 'buy')
        sell_trades = sum(1 for log in self.portfolio_log if log['action'] == 'sell')
        churn_trades = sum(1 for log in self.portfolio_log if log['action'] == 'churn')
        
        # Estimate total weeks from the backtest period
        if self.portfolio_log:
            first_week = self.portfolio_log[0]['week']
            last_week = self.portfolio_log[-1]['week']
            total_weeks = last_week - first_week + 1
            no_trade_weeks = total_weeks - total_trades
            trading_frequency = (total_trades / total_weeks * 100) if total_weeks > 0 else 0
        else:
            no_trade_weeks = 0
            trading_frequency = 0
        
        return {
            "total_trades": total_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "churn_trades": churn_trades,
            "no_trade_weeks": no_trade_weeks,
            "trading_frequency": f"{trading_frequency:.1f}%"
        }

    def check_available_databases(self) -> List[str]:
        """Check which database files are available - Not needed for PostgreSQL"""
        return []

    def load_metadata(self) -> Dict[str, Dict]:
        """Load stock metadata by calculating directly from stock_data table"""
        session = None
        try:
            session = self._get_session()
            
            # Calculate metadata directly from stock_data table
            metadata = self.calculate_metadata_from_data(session)
            
            return metadata
            
        except Exception as e:
            self.logger.progress(f"Error loading stock metadata: {e}")
            return {}
        finally:
            if session:
                session.close()

    def calculate_metadata_from_data(self, session) -> Dict[str, Dict]:
        """Calculate metadata directly from stock_data table"""
        try:
            from sqlalchemy import text
            
            # Use stock_data table directly
            table_name = 'stock_market'
            
            # Query metadata from stock_market - date column is of type DATE/TIMESTAMP
            # Fix: Cast dates to timestamp to get proper interval for EXTRACT
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
                except Exception as row_error:
                    continue

            return metadata

        except Exception as e:
            self.logger.progress(f"Error calculating metadata: {e}")
            
            # Rollback the session to clear the failed transaction
            try:
                session.rollback()
            except:
                pass
            
            return {}

    def generate_asset_description(self, symbol: str) -> str:
        """Generate intelligent stock descriptions based on symbol names"""
        # Dictionary of common stock descriptions
        desc_map = {
             # Nifty 50
            'RELIANCE': 'Reliance Industries Ltd.', 'HDFCBANK': 'HDFC Bank Ltd.', 'ICICIBANK': 'ICICI Bank Ltd.',
            'INFY': 'Infosys Ltd.', 'HDFC': 'Housing Development Finance Corp', 'TCS': 'Tata Consultancy Services',
            'ITC': 'ITC Ltd.', 'KOTAKBANK': 'Kotak Mahindra Bank', 'LT': 'Larsen & Toubro Ltd.',
            'AXISBANK': 'Axis Bank Ltd.', 'HINDUNILVR': 'Hindustan Unilever Ltd.', 'SBIN': 'State Bank of India',
            'BHARTIARTL': 'Bharti Airtel Ltd.', 'BAJFINANCE': 'Bajaj Finance Ltd.', 'ASIANPAINTS': 'Asian Paints Ltd.',
            'MARUTI': 'Maruti Suzuki India Ltd.', 'HCLTECH': 'HCL Technologies Ltd.', 'SUNPHARMA': 'Sun Pharmaceutical Inds.',
            'TITAN': 'Titan Company Ltd.', 'M&M': 'Mahindra & Mahindra Ltd.', 'ULTRACEMCO': 'UltraTech Cement Ltd.',
            'BAJAJFINSV': 'Bajaj Finserv Ltd.', 'WIPRO': 'Wipro Ltd.', 'NESTLEIND': 'Nestle India Ltd.',
            'POWERGRID': 'Power Grid Corp. of India', 'TATASTEEL': 'Tata Steel Ltd.', 'JSWSTEEL': 'JSW Steel Ltd.',
            'NTPC': 'NTPC Ltd.', 'GRASIM': 'Grasim Industries Ltd.', 'TECHM': 'Tech Mahindra Ltd.',
            'ADANIGREEN': 'Adani Green Energy Ltd.', 'ADANIENT': 'Adani Enterprises Ltd.', 'ADANIPORTS': 'Adani Ports & SEZ',
            'DIVISLAB': 'Divi\'s Laboratories Ltd.', 'HDFCLIFE': 'HDFC Life Insurance Co.', 'SBILIFE': 'SBI Life Insurance Co.',
            'BPCL': 'Bharat Petroleum Corp.', 'TATAMOTORS': 'Tata Motors Ltd.', 'ONGC': 'Oil & Natural Gas Corp.',
            'EICHERMOT': 'Eicher Motors Ltd.', 'DRREDDY': 'Dr. Reddy\'s Laboratories', 'BRITANNIA': 'Britannia Industries',
            'CIPLA': 'Cipla Ltd.', 'COALINDIA': 'Coal India Ltd.', 'TATACONSUM': 'Tata Consumer Products',
            'HINDALCO': 'Hindalco Industries Ltd.', 'UPL': 'UPL Ltd.', 'JIOFIN': 'Jio Financial Services'
        }
        
        if symbol in desc_map:
            return desc_map[symbol]

        # Artifact cleanup and fallbacks
        stock_mappings_legacy = {
            'NIFTYBEES': 'Nifty 50 ETF',
            'BANKBEES': 'Banking ETF',
            'JUNIORBEES': 'Nifty Next 50 ETF',
            'ITBEES': 'IT ETF',
            'GOLDBEES': 'Gold ETF',
            'LIQUIDBEES': 'Liquid ETF'
        }
        if symbol in stock_mappings_legacy:
            return stock_mappings_legacy[symbol]
            
        return f'{symbol} Stock'

    def get_asset_sector_classification(self, symbol: str) -> str:
        """Classify stock into sector categories"""
        
        # Robust Sector Mapping for Top Stocks
        sector_map = {
            # Financials
            'HDFCBANK': 'Financials', 'ICICIBANK': 'Financials', 'SBIN': 'Financials', 'AXISBANK': 'Financials',
            'KOTAKBANK': 'Financials', 'BAJFINANCE': 'Financials', 'BAJAJFINSV': 'Financials', 'HDFCLIFE': 'Financials',
            'SBILIFE': 'Financials', 'INDUSINDBK': 'Financials', 'BANKBARODA': 'Financials', 'PNB': 'Financials',
            'JIOFIN': 'Financials', 'PFC': 'Financials', 'RECLTD': 'Financials', 'SHRIRAMFIN': 'Financials',
            'CHOLAFIN': 'Financials', 'MUTHOOTFIN': 'Financials',
            
            # Technology
            'TCS': 'Technology', 'INFY': 'Technology', 'HCLTECH': 'Technology', 'WIPRO': 'Technology',
            'TECHM': 'Technology', 'LTIM': 'Technology', 'PERSISTENT': 'Technology', 'COFORGE': 'Technology',
            'MPHASIS': 'Technology', 'LTTS': 'Technology', 'KPITTECH': 'Technology', 'TATAELXSI': 'Technology',
            
            # Energy & Oil
            'RELIANCE': 'Energy', 'ONGC': 'Energy', 'BPCL': 'Energy', 'POWERGRID': 'Utilities',
            'NTPC': 'Utilities', 'ADANIGREEN': 'Utilities', 'TATAPOWER': 'Utilities', 'IOC': 'Energy',
            'GAIL': 'Energy', 'HPCL': 'Energy',
            
            # FMCG & Consumption
            'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'NESTLEIND': 'FMCG', 'BRITANNIA': 'FMCG',
            'TITAN': 'Consumer Discretionary', 'ASIANPAINTS': 'Materials', 'MARICO': 'FMCG',
            'DABUR': 'FMCG', 'GODREJCP': 'FMCG', 'TATACONSUM': 'FMCG', 'VARUN': 'FMCG',
            
            # Auto
            'TATAMOTORS': 'Automotive', 'MARUTI': 'Automotive', 'M&M': 'Automotive', 'BAJAJ-AUTO': 'Automotive',
            'EICHERMOT': 'Automotive', 'HEROMOTOCO': 'Automotive', 'TVSMOTOR': 'Automotive', 'BHARATFORG': 'Automotive',
            
            # Healthcare
            'SUNPHARMA': 'Healthcare', 'DRREDDY': 'Healthcare', 'CIPLA': 'Healthcare', 'DIVISLAB': 'Healthcare',
            'APOLLOHOSP': 'Healthcare', 'LUPIN': 'Healthcare', 'TORNTPHARM': 'Healthcare', 'MANKIND': 'Healthcare',
            
            # Materials & Metals
            'TATASTEEL': 'Materials', 'JSWSTEEL': 'Materials', 'HINDALCO': 'Materials', 'VEDANTA': 'Materials',
            'ULTRACEMCO': 'Materials', 'AMBUJACEM': 'Materials', 'PIDILITIND': 'Materials', 'JUBLFOOD': 'Consumer Services',
            
            # Infrastructure & Capital Goods
            'LT': 'Infrastructure', 'SIEMENS': 'Capital Goods', 'ABB': 'Capital Goods', 'HAL': 'Defence',
            'BEL': 'Defence', 'ADANIENT': 'Services', 'ADANIPORTS': 'Infrastructure',
            
            # Telecom
            'BHARTIARTL': 'Telecommunication', 'VODAFONE': 'Telecommunication', 'INDUSINDBK': 'Financials'
        }
        
        if symbol in sector_map:
            return sector_map[symbol]
            
        # Fallback substring logic
        symbol_lower = symbol.lower()

        if symbol in ['NIFTYBEES', 'JUNIORBEES', 'MIDCAPstock']:
            return 'Broad Market'
        elif 'bank' in symbol_lower or 'psu' in symbol_lower:
            return 'Financials'
        elif 'it' in symbol_lower or 'tech' in symbol_lower or 'mon100' in symbol_lower:
            return 'Technology'
        elif 'pharma' in symbol_lower or 'health' in symbol_lower:
            return 'Healthcare'
        elif 'gold' in symbol_lower:
            return 'Commodity'
        elif 'oil' in symbol_lower or 'energy' in symbol_lower:
            return 'Energy'
        elif 'metal' in symbol_lower:
            return 'Materials'
        elif 'infra' in symbol_lower:
            return 'Infrastructure'
        elif 'defence' in symbol_lower or 'defense' in symbol_lower:
            return 'Defence'
        elif 'liquid' in symbol_lower:
            return 'Cash/Liquid'
        elif 'cpse' in symbol_lower:
            return 'PSU'
        elif 'fmcg' in symbol_lower:
            return 'FMCG'
        elif 'auto' in symbol_lower:
            return 'Automotive'
        elif 'realty' in symbol_lower or 'real' in symbol_lower:
            return 'Real Estate'
        elif 'consumer' in symbol_lower:
            return 'Consumption'
        else:
            return 'Other'

    def get_default_stock_selection(self, available_stocks: List[str], max_count: int = 5) -> List[str]:
        """Intelligently select default stocks for diversified portfolio"""
        priority_symbols = [
            'RELIANCE', 'HDFCBANK', 'INFY', 'ICICIBANK', 'TCS',
            'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK'
        ]

        selected = []
        for symbol in priority_symbols:
            if symbol in available_stocks and len(selected) < max_count:
                selected.append(symbol)

        remaining = [stock for stock in available_stocks if stock not in selected]
        while len(selected) < max_count and remaining:
            selected.append(remaining.pop(0))

        return selected

    def calculate_common_date_range(self, selected_stocks: List[str]) -> Tuple[Optional[str], Optional[str], float]:
        """Calculate common date range for selected stocks by querying stock_data table directly"""
        if not selected_stocks:
            self.logger.error("No stocks provided for date range calculation")
            return None, None, 0.0
        
        session = None
        try:
            session = self._get_session()
            from sqlalchemy import text
            
            self.logger.debug(f"Querying stock_data table for date ranges of: {selected_stocks}")
            
            # Build placeholders for IN clause
            placeholders = ','.join([f':ticker_{i}' for i in range(len(selected_stocks))])
            
            # Query to get MIN and MAX dates for each stock directly from stock_data table
            query = text(f"""
                SELECT 
                    symbol,
                    MIN(date) as start_date,
                    MAX(date) as end_date,
                    COUNT(*) as total_records
                FROM stock_market
                WHERE symbol IN ({placeholders})
                GROUP BY symbol
                ORDER BY symbol
            """)
            
            # Build parameters dict
            params = {f'ticker_{i}': ticker for i, ticker in enumerate(selected_stocks)}
            
            result = session.execute(query, params)
            rows = result.fetchall()
            
            if not rows:
                self.logger.error(f"No data found in stock_data table for: {selected_stocks}")
                return None, None, 0.0
            
            # Parse results
            stock_date_ranges = {}
            for row in rows:
                symbol = row[0]
                start_date = row[1]
                end_date = row[2]
                total_records = row[3]
                
                # Convert to datetime if they're strings
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date)
                if isinstance(end_date, str):
                    end_date = pd.to_datetime(end_date)
                
                years_available = (end_date - start_date).days / 365.25
                
                stock_date_ranges[symbol] = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'total_records': total_records,
                    'years_available': years_available
                }
            
            # Check if all requested stocks were found
            missing_stocks = [stock for stock in selected_stocks if stock not in stock_date_ranges]
            if missing_stocks:
                self.logger.info(f"No data found for: {missing_stocks}")
            
            if not stock_date_ranges:
                self.logger.error(f"None of the requested stocks found in stock_data table: {selected_stocks}")
                return None, None, 0.0
            
            # Calculate common date range
            start_dates = [data['start_date'] for data in stock_date_ranges.values()]
            end_dates = [data['end_date'] for data in stock_date_ranges.values()]
            
            # The latest start date among all stocks (most restrictive)
            latest_start = max(start_dates)
            common_end = min(end_dates)
            
            self.logger.info(f"📅 Stock Data Ranges (from stock_data table):")
            for symbol, data in stock_date_ranges.items():
                years = data['years_available']
                self.logger.info(f"   {symbol:12s}: {data['start_date'].strftime('%Y-%m-%d')} to {data['end_date'].strftime('%Y-%m-%d')} ({years:.1f} years, {data['total_records']} records)")
            
            self.logger.progress(f"📊 Common data range: {latest_start.strftime('%Y-%m-%d')} to {common_end.strftime('%Y-%m-%d')}")
            
            # OPTIMIZED BUFFER: Use 53 weeks (371 days) for 252+ trading days
            buffer_weeks = 53
            buffer_days = buffer_weeks * 7  # 371 calendar days (~265 trading days)
            strategy_start = latest_start + timedelta(days=buffer_days)
            
            self.logger.info(f"🎯 Enhanced Buffer Strategy:")
            self.logger.info(f"   Buffer period: {buffer_weeks} weeks ({buffer_days} calendar days)")
            self.logger.info(f"   Expected trading days: ~{int(buffer_days * 5 / 7)} days")
            self.logger.info(f"   Required for momentum: 252 trading days")
            self.logger.info(f"   Safety margin: ~{int(buffer_days * 5 / 7) - 252} trading days")
            self.logger.info(f"   Strategy start: {strategy_start.strftime('%Y-%m-%d')}")
            
            # Ensure we have at least 1 year of backtest data after the strategy start
            if strategy_start >= common_end:
                self.logger.info(f"❌ Insufficient data even with {buffer_weeks}-week buffer:")
                self.logger.info(f"   Strategy start: {strategy_start.strftime('%Y-%m-%d')}")
                self.logger.info(f"   Data ends: {common_end.strftime('%Y-%m-%d')}")
                shortage_days = (strategy_start - common_end).days
                self.logger.info(f"   Shortage: {shortage_days} days")
                
                # Try with maximum possible buffer
                max_possible_days = (common_end - latest_start).days - 365  # Reserve 1 year for backtesting
                if max_possible_days > 252:  # At least need 252 trading days
                    alt_strategy_start = latest_start + timedelta(days=max_possible_days)
                    self.logger.progress(f"🔄 Alternative with maximum buffer:")
                    self.logger.info(f"   Maximum buffer: {max_possible_days} days ({max_possible_days / 7:.1f} weeks)")
                    self.logger.info(f"   Alternative start: {alt_strategy_start.strftime('%Y-%m-%d')}")
                    
                    years_available = (common_end - alt_strategy_start).days / 365.25
                    return alt_strategy_start.strftime('%Y-%m-%d'), common_end.strftime('%Y-%m-%d'), years_available
                else:
                    return None, None, 0.0
            
            # Check if we have sufficient backtest period
            years_available = (common_end - strategy_start).days / 365.25
            
            self.logger.info(f"📈 Backtest Feasibility:")
            if years_available >= 10:
                self.logger.progress(f"   ✅ Excellent: {years_available:.1f} years provides statistically robust results")
            elif years_available >= 5:
                self.logger.progress(f"   ✅ Good: {years_available:.1f} years allows reasonable strategy validation")
            elif years_available >= 2:
                self.logger.info(f"   ⚠️ Fair: {years_available:.1f} years provides basic validation")
            else:
                self.logger.info(f"   ⚠️ Limited: {years_available:.1f} years may not be reliable")
            
            return strategy_start.strftime('%Y-%m-%d'), common_end.strftime('%Y-%m-%d'), years_available
            
        except Exception as e:
            self.logger.error(f"Error calculating date range from stock_data table: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None, None, 0.0
        finally:
            if session:
                session.close()

    def load_data_from_database(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Load daily OHLCV data from PostgreSQL database for selected tickers with historical buffer for momentum calculations"""
        # Calculate buffer start date for cache key
        buffer_start_date = (pd.to_datetime(start_date) - timedelta(days=400)).strftime('%Y-%m-%d')
        
        # Check cache first (using buffer dates for cache key)
        cache_key = f"{','.join(sorted(tickers))}_{buffer_start_date}_{end_date}"
        if cache_key in self._data_cache:
            if self._verbose:
                self.logger.info(f"📦 Using cached data for {len(tickers)} stocks")
            return self._data_cache[cache_key]
        
        session = None
        try:
            session = self._get_session()
            from sqlalchemy import text

            if self._verbose:
                self.logger.progress(f"📊 Loading data with historical buffer:")
                self.logger.info(f"   Strategy period: {start_date} to {end_date}")
                self.logger.progress(f"   Data loading period: {buffer_start_date} to {end_date}")
                self.logger.info(f"   Buffer: 400 calendar days (~252 trading days) for momentum calculations")

            # Use stock_data table - column names match directly
            # Get available symbols using proper parameterization
            from sqlalchemy import bindparam
            placeholders = ','.join([f':ticker_{i}' for i in range(len(tickers))])
            
            query = text(f"""
                SELECT DISTINCT symbol 
                FROM stock_market 
                WHERE symbol IN ({placeholders})
                AND date >= CAST(:start_date AS DATE) AND date <= CAST(:end_date AS DATE)
            """)
            
            params = {f'ticker_{i}': t for i, t in enumerate(tickers)}
            params.update({
                "start_date": buffer_start_date,
                "end_date": end_date
            })
            
            result = session.execute(query, params)
            available_tickers = [row[0] for row in result.fetchall()]
            missing_tickers = [t for t in tickers if t not in available_tickers]

            if missing_tickers and self._verbose:
                self.logger.info(f"⚠️ Missing data for: {', '.join(missing_tickers)}")

            if not available_tickers:
                raise ValueError("No data available for any of the selected stocks")

            # Load data - column names match directly
            placeholders = ','.join([f':ticker_{i}' for i in range(len(available_tickers))])
            query = text(f"""
                SELECT 
                    date,
                    symbol,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    adj_close AS adjusted_close
                FROM stock_market
                WHERE symbol IN ({placeholders})
                AND date >= CAST(:start_date AS DATE) AND date <= CAST(:end_date AS DATE)
                ORDER BY date, symbol
            """)

            params = {f'ticker_{i}': t for i, t in enumerate(available_tickers)}
            params.update({
                "start_date": buffer_start_date,
                "end_date": end_date
            })
            
            # Use session.execute instead of pd.read_sql for better parameter binding
            result = session.execute(query, params)
            rows = result.fetchall()
            columns = result.keys()
            df = pd.DataFrame(rows, columns=columns)

            if df.empty:
                raise ValueError(f"No data found for the selected date range {start_date} to {end_date}")

            if self._verbose:
                self.logger.progress(f"✅ Loaded data for {len(available_tickers)} stocks: {', '.join(available_tickers)}")

            df['date'] = pd.to_datetime(df['date'])

            # Optimized pivot operations
            data_dict = {}
            for column in ['open', 'high', 'low', 'close', 'volume']:
                pivot_df = df.pivot(index='date', columns='symbol', values=column)
                data_dict[column] = pivot_df.ffill().bfill()

            # Cache the result
            self._data_cache[cache_key] = data_dict
            
            return data_dict

        except Exception as e:
            self.logger.progress(f"Error loading data: {e}")
            return {}
        finally:
            if session:
                session.close()

    def get_last_trading_day(self, close_df: pd.DataFrame, target_date: datetime, day: str = 'Friday') -> Optional[datetime]:
        """Return the nearest available trading day (e.g., Friday or fallback Thursday)"""
        target_weekday = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4}[day]

        days_ahead = target_weekday - target_date.weekday()
        if days_ahead <= 0:
            target_trading_date = target_date + timedelta(days=days_ahead)
        else:
            target_trading_date = target_date - timedelta(days=(7 - days_ahead))

        for i in range(5):
            check_date = target_trading_date - timedelta(days=i)
            if check_date in close_df.index:
                return check_date
        return None

    def get_next_trading_day(self, open_df: pd.DataFrame, target_date: datetime, day: str = 'Monday') -> Optional[datetime]:
        """Return Monday's open or fallback to Tuesday if holiday"""
        target_weekday = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4}[day]

        days_ahead = target_weekday - target_date.weekday()
        if days_ahead < 0:
            days_ahead += 7

        target_trading_date = target_date + timedelta(days=days_ahead)

        for i in range(5):
            check_date = target_trading_date + timedelta(days=i)
            if check_date in open_df.index:
                return check_date
        return None

    def compute_52_week_high_low(self, df: pd.DataFrame, current_date: datetime) -> pd.DataFrame:
        """Calculate rolling 52-week high/low for each ticker at signal date"""
        # Get data up to current date (signal date - typically Friday)
        historical_data = df[df.index <= current_date]

        # As per PDF specification: Requires minimum 252 trading days for accurate momentum signals
        required_trading_days = 252
        available_days = len(historical_data)

        self.logger.progress(f"📊 Momentum Calculation for {current_date.strftime('%Y-%m-%d')}:")
        self.logger.info(f"   Available historical data: {available_days} days")
        self.logger.info(f"   Required for 52-week calc: {required_trading_days} days")

        if available_days < required_trading_days:
            # Insufficient data for proper 52-week calculation
            self.logger.info(f"   ❌ Insufficient data: {available_days} < {required_trading_days} required")
            self.logger.info(f"   📋 Available stocks: {list(historical_data.columns)}")
            self.logger.progress(f"   🔄 Strategy will use fallback logic during accumulation")
            return pd.DataFrame()

        # Use exactly 252 trading days as per PDF specification
        window_data = historical_data.tail(252)
        self.logger.progress(f"   ✅ Using exactly {len(window_data)} trading days for momentum")

        if window_data.empty:
            self.logger.info("   ❌ Window data is empty after filtering")
            return pd.DataFrame()

        # Calculate 52-week highs and lows as per PDF specification
        highs_52w = window_data.max()  # Maximum closing price in trailing 252 trading days
        lows_52w = window_data.min()  # Minimum closing price in trailing 252 trading days
        current_prices = historical_data.iloc[-1] if not historical_data.empty else pd.Series()

        self.logger.info(f"   📈 52-week highs calculated for {len(highs_52w)} stocks")
        self.logger.info(f"   📉 52-week lows calculated for {len(lows_52w)} stocks")
        self.logger.info(f"   💰 Current prices available for {len(current_prices)} stocks")

        # Create result DataFrame with exact PDF specification calculations
        result_data = []
        valid_stocks = 0

        for symbol in highs_52w.index:
            if (not pd.isna(highs_52w[symbol]) and not pd.isna(lows_52w[symbol]) and
                    not pd.isna(current_prices[symbol]) and
                    highs_52w[symbol] > 0 and lows_52w[symbol] > 0 and current_prices[symbol] > 0):

                # Ensure data integrity: high >= low
                if highs_52w[symbol] >= lows_52w[symbol]:
                    # Calculate distances exactly as per PDF specification
                    distance_from_low = (current_prices[symbol] - lows_52w[symbol]) / lows_52w[symbol] * 100
                    distance_from_high = (highs_52w[symbol] - current_prices[symbol]) / highs_52w[symbol] * 100

                    result_data.append({
                        'symbol': symbol,
                        '52w_high': highs_52w[symbol],
                        '52w_low': lows_52w[symbol],
                        'current_price': current_prices[symbol],
                        'distance_from_low': distance_from_low,
                        'distance_from_high': distance_from_high
                    })

                    valid_stocks += 1
                    self.logger.progress(f"   ✅ {symbol}: High=₹{highs_52w[symbol]:.2f}, Low=₹{lows_52w[symbol]:.2f}, Current=₹{current_prices[symbol]:.2f}")
                    self.logger.info(f"       Distance from Low: {distance_from_low:.2f}%, Distance from High: {distance_from_high:.2f}%")
                else:
                    self.logger.info(f"   ❌ {symbol}: Data integrity issue - high ({highs_52w[symbol]:.2f}) < low ({lows_52w[symbol]:.2f})")
            else:
                self.logger.info(f"   ❌ {symbol}: Invalid data - contains NaN or zero values")

        result_df = pd.DataFrame(result_data)
        self.logger.info(f"   🎯 Final momentum result: {len(result_df)} valid stocks with proper 52-week data")

        if not result_df.empty:
            # Sort by distance from low for accumulation phase (buy closest to low)
            result_df_sorted = result_df.sort_values('distance_from_low')
            self.logger.progress(f"   📊 Ranked 52 week low stocks in ascending order:")
            for idx, row in result_df_sorted.head().iterrows():
                self.logger.info(f"      {idx + 1}. {row['symbol']}: {row['distance_from_low']:.2f}% from low")

        return result_df

    def calculate_transaction_costs(self, action: str, amount: float, brokerage_percent: float) -> Dict[str, float]:
        """
        Calculate Indian market transaction costs for STOCKS.
        
        Uses stock-specific cost calculation with STT on both buy and sell.
        """
        return self.calculate_stock_delivery_costs(action, amount, brokerage_percent)
    
    
    def add_purchase_record(self, ticker: str, units: int, price: float, date: datetime):
        """Add purchase record for FIFO tracking"""
        if ticker not in self.purchase_history:
            self.purchase_history[ticker] = []

        self.purchase_history[ticker].append({
            'date': date,
            'units': units,
            'price': price,
            'remaining_units': units
        })

    def calculate_capital_gains_tax(self, ticker: str, units_to_sell: int, sell_price: float,
                                    sell_date: datetime) -> Dict:
        """
        Calculate capital gains tax.
        DISABLED: Returns 0 tax as per requirement.
        """
        return {
            'short_term_gain': 0.0,
            'long_term_gain': 0.0,
            'capital_gains_tax': 0.0,
            'is_short_term': False
        }
    
    
    def log_transaction_costs(self, week: int, date: datetime, action: str, ticker: str, units: int, price: float, costs: Dict, capital_gains_tax: float = 0):
        """Log detailed transaction costs for analysis"""
        self.transaction_costs_log.append({
            'week': week,
            'date': date,
            'action': action,
            'ticker': ticker,
            'units': units,
            'price': price,
            'amount': units * price,
            'brokerage': costs.get('brokerage', 0),
            'stt': costs.get('stt', 0),
            'stamp_duty': costs.get('stamp_duty', 0),
            'exchange_charges': costs.get('exchange_charges', 0),
            'sebi_charges': costs.get('sebi_charges', 0),
            'gst': costs.get('gst', 0),
            'total_costs': costs.get('total_costs', 0),
            'capital_gains_tax': capital_gains_tax,
            'total_impact': costs.get('total_costs', 0) + capital_gains_tax
        })

    def calculate_dynamic_churn_amount(self, current_nav: float, cash: float, capital_per_week: float, accumulation_weeks: int, compounding_enabled: bool) -> float:
        """Calculate dynamic churn amount based on compounding logic"""
        if not compounding_enabled:
            return capital_per_week

        base_portfolio = accumulation_weeks * capital_per_week
        current_portfolio = current_nav
        portfolio_change_percent = (current_portfolio - base_portfolio) / base_portfolio * 100

        if portfolio_change_percent >= 0:
            thresholds_crossed = int(portfolio_change_percent // 20)
            dynamic_capital = capital_per_week * (1 + thresholds_crossed * 0.10)
        else:
            thresholds_crossed = int(abs(portfolio_change_percent) // 20)
            dynamic_capital = capital_per_week * (1 - thresholds_crossed * 0.10)

        min_churn = capital_per_week * 0.10
        dynamic_capital = max(dynamic_capital, min_churn)

        return dynamic_capital

    def execute_churning_phase(self, week_num: int, execution_date: datetime, high_low_data: pd.DataFrame,
                             open_prices: pd.Series, current_holdings: Dict[str, int], cash: float,
                             target_capital: float, brokerage_percent: float) -> Dict:
        """
        Execute complete churning phase logic as per Technical Specification PDF
        
        This method implements the churning phase with two main processes:
        1. Capital Raising Process: Sell from stocks closest to 52-week high
        2. Reallocation Process: Buy stock with smallest distance from 52-week low
        """
        self.logger.progress(f"🔄 Executing Churning Phase - Week {week_num}")
        self.logger.info(f"💰 Target capital to raise: ₹{target_capital:,.0f}")
        
        # Initialize tracking variables
        total_raised = 0
        sell_transactions = []
        total_capital_gains_tax = 0
        updated_holdings = current_holdings.copy()
        
        # ===== CAPITAL RAISING PROCESS =====
        self.logger.progress("📊 Starting Capital Raising Process...")
        capital_raising_result = self.execute_capital_raising_process(
            week_num=week_num,
            execution_date=execution_date,
            high_low_data=high_low_data,
            open_prices=open_prices,
            current_holdings=updated_holdings,
            cash=cash,
            target_capital=target_capital,
            brokerage_percent=brokerage_percent
        )
        
        # Extract results from capital raising
        total_raised = capital_raising_result['total_raised']
        sell_transactions = capital_raising_result['sell_transactions']
        total_capital_gains_tax = capital_raising_result['total_capital_gains_tax']
        updated_holdings = capital_raising_result['holdings']
        
        # Update cash with raised amount
        current_cash = cash + total_raised
        
        # ===== REALLOCATION PROCESS =====
        self.logger.progress(f"📊 Starting Reallocation Process (Cash available: ₹{current_cash:,.2f})...")
        
        # Only proceed if we have cash to deploy
        buy_transaction = {}
        if current_cash > 0:
            reallocation_result = self.execute_reallocation_process(
                week_num=week_num,
                execution_date=execution_date,
                high_low_data=high_low_data,
                open_prices=open_prices,
                current_holdings=updated_holdings,
                available_cash=current_cash,
                brokerage_percent=brokerage_percent,
                target_allocation=target_capital
            )
            
            buy_transaction = reallocation_result['buy_transaction']
            updated_holdings = reallocation_result['holdings']
            current_cash = reallocation_result['remaining_cash']
        
        # Log the complete churn transaction
        if sell_transactions or buy_transaction:
            self.portfolio_log.append({
                'week': week_num,
                'execution_date': execution_date,
                'action': 'churn',
                'sell_transactions': sell_transactions,
                'buy_transaction': buy_transaction,
                'total_raised': total_raised,
                'capital_gains_tax': total_capital_gains_tax,
                'nav': self.calculate_nav(updated_holdings, open_prices, current_cash),
                'cash': current_cash,
                'holdings': updated_holdings.copy(),
                'costs': {
                    'total_costs': sum(t.get('costs', {}).get('total_costs', 0) for t in sell_transactions) + 
                                  buy_transaction.get('costs', {}).get('total_costs', 0)
                }
            })
            
        return {
            'holdings': updated_holdings,
            'cash': current_cash,
            'total_raised': total_raised,
            'capital_gains_tax': total_capital_gains_tax
        }

    def execute_capital_raising_process(self, week_num: int, execution_date: datetime, high_low_data: pd.DataFrame,
                                      open_prices: pd.Series, current_holdings: Dict[str, int], cash: float,
                                      target_capital: float, brokerage_percent: float) -> Dict:
        """
        Execute Capital Raising Process: Sell from stocks closest to 52-week high
        """
        total_raised = 0
        sell_transactions = []
        total_capital_gains_tax = 0
        updated_holdings = current_holdings.copy()
        
        # Filter holdings that are in the high_low_data
        valid_holdings = [ticker for ticker in updated_holdings.keys() if ticker in high_low_data['symbol'].values]
        
        if not valid_holdings:
            self.logger.trade("   ⚠️ No valid holdings available for selling")
            return {
                'total_raised': 0,
                'sell_transactions': [],
                'total_capital_gains_tax': 0,
                'holdings': updated_holdings
            }
            
        # Get data for valid holdings
        holdings_data = high_low_data[high_low_data['symbol'].isin(valid_holdings)]
        
        # Sort by distance from high (ascending) - sell stocks closest to high first
        holdings_sorted = holdings_data.sort_values('distance_from_high')
        
        self.logger.trade("   📉 Sell candidates (closest to 52-week high):")
        for idx, row in holdings_sorted.iterrows():
            self.logger.info(f"      {row['symbol']}: {row['distance_from_high']:.2f}% from high")
            
        # Iterate through candidates and sell until target capital is raised
        for _, row in holdings_sorted.iterrows():
            if total_raised >= target_capital:
                break
                
            ticker = row['symbol']
            price = open_prices[ticker]
            
            # Keep selling from this stock until target is met or stock is exhausted
            while total_raised < target_capital and ticker in updated_holdings and updated_holdings[ticker] > 0:
                units_held = updated_holdings[ticker]
                remaining_target = target_capital - total_raised
                
                # Calculate value of current holding
                holding_value = units_held * price
                
                # Determine how many units to sell
                if holding_value <= remaining_target * 1.1:  # Sell all if close to target or less
                    units_to_sell = units_held
                else:
                    # Sell partial units to meet target (start with estimate, will iterate if needed)
                    units_to_sell = max(1, int(np.ceil(remaining_target / price)))
                    
                # Don't sell more than we have
                units_to_sell = min(units_to_sell, units_held)
                
                if units_to_sell > 0:
                    amount = units_to_sell * price
                    costs = self.calculate_transaction_costs('sell', amount, brokerage_percent)
                    
                    # Calculate capital gains tax
                    tax_info = self.calculate_capital_gains_tax(ticker, units_to_sell, price, execution_date)
                    capital_gains_tax = tax_info['capital_gains_tax']
                    
                    net_amount = costs['net_amount'] - capital_gains_tax
                    
                    # Update tracking
                    total_raised += net_amount
                    total_capital_gains_tax += capital_gains_tax
                    updated_holdings[ticker] -= units_to_sell
                    
                    if updated_holdings[ticker] == 0:
                        del updated_holdings[ticker]
                        
                    # Log transaction
                    self.log_transaction_costs(week_num, execution_date, 'sell', ticker, units_to_sell, price, costs, capital_gains_tax)
                    
                    sell_transactions.append({
                        'ticker': ticker,
                        'units': units_to_sell,
                        'price': price,
                        'amount': amount,
                        'net_amount': net_amount,
                        'costs': costs,
                        'capital_gains_tax': capital_gains_tax
                    })
                    
                    self.logger.info(f"📋 Sale calculation:")
                    self.logger.info(f"   Gross amount: ₹{amount:,.2f}")
                    self.logger.info(f"   Transaction costs: ₹{costs['total_costs']:,.2f}")
                    self.logger.info(f"   Capital Gains Tax: ₹{capital_gains_tax:,.2f}")
                    self.logger.info(f"   Net proceeds: ₹{net_amount:,.2f}")
                    self.logger.trade(f"   Units to sell: {units_to_sell}")
                    self.logger.trade(f"🔻 Sale executed: {units_to_sell} units of {ticker} for ₹{net_amount:,.2f}")
                    self.logger.info(f"💰 Remaining cash: ₹{cash + total_raised:,.2f}")
                
        return {
            'total_raised': total_raised,
            'sell_transactions': sell_transactions,
            'total_capital_gains_tax': total_capital_gains_tax,
            'holdings': updated_holdings
        }

    def execute_reallocation_process(self, week_num: int, execution_date: datetime, high_low_data: pd.DataFrame,
                                   open_prices: pd.Series, current_holdings: Dict[str, int], available_cash: float,
                                   brokerage_percent: float, target_allocation: float = None) -> Dict:
        """
        Execute Reallocation Process: Buy stock with smallest distance from 52-week low
        """
        updated_holdings = current_holdings.copy()
        remaining_cash = available_cash
        buy_transaction = {}
        
        # Sort all stocks by distance from low (ascending) - buy closest to low
        buy_candidates = high_low_data.sort_values('distance_from_low')
        
        if buy_candidates.empty:
            self.logger.trade("   ⚠️ No valid buy candidates found")
            return {
                'buy_transaction': {},
                'holdings': updated_holdings,
                'remaining_cash': remaining_cash
            }
            
        # Select top candidate
        top_candidate = buy_candidates.iloc[0]
        ticker = top_candidate['symbol']
        price = open_prices[ticker]
        
        self.logger.trade(f"   📈 Top Buy Candidate: {ticker} ({top_candidate['distance_from_low']:.2f}% from low)")
        
        # Determine investable capital
        # If target_allocation provided, use IT for sizing (matching ETF logic: try to buy fixed amount)
        # If insufficient cash, the subsequent check will prevent execution (All-or-Nothing)
        if target_allocation is not None:
             investable_capital = target_allocation
        else:
             investable_capital = available_cash
        
        # Calculate max units accurately (Iterative approach to maximize usage)
        # Start with theoretical max (ignoring costs) and decrement until it fits including actual costs
        units_to_buy = int(investable_capital / price)
        
        while units_to_buy > 0:
            amount = units_to_buy * price
            costs = self.calculate_transaction_costs('buy', amount, brokerage_percent)
            if costs['net_amount'] <= investable_capital:
                break
            units_to_buy -= 1

        if units_to_buy > 0:
            amount = units_to_buy * price
            costs = self.calculate_transaction_costs('buy', amount, brokerage_percent)
            
            if costs['net_amount'] <= remaining_cash:
                # Execute buy
                remaining_cash -= costs['net_amount']
                updated_holdings[ticker] = updated_holdings.get(ticker, 0) + units_to_buy
                
                # Add purchase record
                self.add_purchase_record(ticker, units_to_buy, price, execution_date)
                
                # Log transaction
                self.log_transaction_costs(week_num, execution_date, 'buy', ticker, units_to_buy, price, costs)
                
                buy_transaction = {
                    'ticker': ticker,
                    'units': units_to_buy,
                    'price': price,
                    'amount': amount,
                    'net_amount': costs['net_amount'],
                    'costs': costs
                }
                
                # Enhanced Logging for Buy (matching detailed style)
                self.logger.info(f"📋 Purchase calculation:")
                self.logger.info(f"   Gross amount: ₹{amount:,.0f}")
                self.logger.info(f"   Transaction costs: ₹{costs['total_costs']:,.2f}")
                self.logger.info(f"   Net for units: ₹{remaining_cash - costs['total_costs']:,.0f}")
                self.logger.trade(f"   Units to buy: {units_to_buy}")
                self.logger.trade(f"✅ Purchase executed: {units_to_buy} units of {ticker} for ₹{costs['net_amount']:,.2f}")
                self.logger.info(f"💳 Total cost (including fees): ₹{costs['total_costs']:,.2f}")
                self.logger.info(f"💰 Remaining cash: ₹{remaining_cash:,.2f}")
        else:
            self.logger.trade(f"      ⚠️ Insufficient cash to buy even 1 unit of {ticker}")
            
        return {
            'buy_transaction': buy_transaction,
            'holdings': updated_holdings,
            'remaining_cash': remaining_cash
        }

    def calculate_nav(self, holdings: Dict[str, int], prices: pd.Series, cash: float) -> float:
        """Calculate Net Asset Value"""
        holdings_value = sum(holdings[ticker] * prices.get(ticker, 0) for ticker in holdings)
        return holdings_value + cash

    def check_strategy_exists(self, strategy_name: str, user_id: str, tickers: List[str],
                            start_date: str, end_date: str, capital_per_week: float,
                            accumulation_weeks: int, brokerage_percent: float,
                            compounding_enabled: bool, risk_free_rate: float,
                            use_custom_dates: bool) -> Dict:
        """
        Check if a strategy with the same parameters already exists in the PostgreSQL database

        Database: PostgreSQL (Neon) - app_data_db_connection
        Table: stock_saved_strategy

        Returns:
            Dict with 'exists' boolean and 'existing_strategy' details if found
        """
        session = None
        try:
            # Import app_data_db_connection for Stock strategies
            from Databases.app_data_db_connection import get_session
            session = get_session()
            from sqlalchemy import text

            # Check if the stock_saved_strategy table exists
            table_check = session.execute(text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'stock_saved_strategy'
            """))
            if not table_check.fetchone():
                return {"exists": False, "message": "Strategy table not found"}

            # Convert tickers to JSON for comparison
            tickers_json = json.dumps(sorted(tickers))

            # Check for exact match of all parameters
            query = text("""
                SELECT id, strategy_name, created_at, created_timestamp
                FROM stock_saved_strategy
                WHERE user_id = :user_id
                AND strategy_name = :strategy_name
                AND tickers::text = :tickers
                AND start_date = :start_date
                AND end_date = :end_date
                AND capital_per_week = :capital_per_week
                AND accumulation_weeks = :accumulation_weeks
                AND brokerage_percent = :brokerage_percent
                AND compounding_enabled = :compounding_enabled
                AND risk_free_rate = :risk_free_rate
                AND use_custom_dates = :use_custom_dates
            """)

            result = session.execute(query, {
                "user_id": user_id,
                "strategy_name": strategy_name,
                "tickers": tickers_json,
                "start_date": start_date,
                "end_date": end_date,
                "capital_per_week": capital_per_week,
                "accumulation_weeks": accumulation_weeks,
                "brokerage_percent": brokerage_percent,
                "compounding_enabled": compounding_enabled,
                "risk_free_rate": risk_free_rate,
                "use_custom_dates": use_custom_dates
            })

            existing_strategy = result.fetchone()

            if existing_strategy:
                return {
                    "exists": True,
                    "existing_strategy": {
                        "id": existing_strategy[0],
                        "strategy_name": existing_strategy[1],
                        "created_at": str(existing_strategy[2]) if existing_strategy[2] else None,
                        "created_timestamp": str(existing_strategy[3]) if existing_strategy[3] else None
                    },
                    "message": f"Stock Strategy already exists"
                }
            else:
                return {
                    "exists": False,
                    "message": "No identical strategy found"
                }

        except Exception as e:
            return {
                "exists": False,
                "error": f"Error checking strategy existence: {str(e)}"
            }
        finally:
            if session:
                session.close()

    def diagnose_stock_data(self, tickers: List[str]) -> Dict[str, Any]:
        """Diagnose data availability for selected stocks"""
        try:
            if not tickers:
                return {"status": "error", "message": "No tickers provided"}
                
            session = self._get_session()
            from sqlalchemy import text
            
            diagnosis = {}
            
            # Check metadata first
            metadata = self.load_metadata()
            
            for ticker in tickers:
                ticker_diagnosis = {
                    "in_metadata": ticker in metadata,
                    "metadata_details": metadata.get(ticker, {}),
                    "data_count": 0,
                    "date_range": None
                }
                
                # Check actual data count
                query = text("SELECT COUNT(*), MIN(date), MAX(date) FROM stock_market WHERE symbol = :symbol")
                result = session.execute(query, {"symbol": ticker}).fetchone()
                
                if result:
                    ticker_diagnosis["data_count"] = result[0]
                    if result[1] and result[2]:
                        ticker_diagnosis["date_range"] = {
                            "start": str(result[1]),
                            "end": str(result[2])
                        }
                
                diagnosis[ticker] = ticker_diagnosis
                
            return {
                "status": "success",
                "diagnosis": diagnosis
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            if session:
                session.close()


    def compute_52_week_high_low(self, df: pd.DataFrame, current_date: datetime) -> pd.DataFrame:
        """Calculate rolling 52-week high/low for each ticker at signal date"""
        # Get data up to current date (signal date - typically Friday)
        historical_data = df[df.index <= current_date]

        # As per PDF specification: Requires minimum 252 trading days for accurate momentum signals
        required_trading_days = 252
        available_days = len(historical_data)

        self.logger.progress(f"📊 Momentum Calculation for {current_date.strftime('%Y-%m-%d')}:")
        self.logger.info(f"   Available historical data: {available_days} days")
        self.logger.info(f"   Required for 52-week calc: {required_trading_days} days")

        if available_days < required_trading_days:
            # Insufficient data for proper 52-week calculation
            self.logger.info(f"   ❌ Insufficient data: {available_days} < {required_trading_days} required")
            self.logger.info(f"   📋 Available Stocks: {list(historical_data.columns)}")
            self.logger.progress(f"   🔄 Strategy will use fallback logic during accumulation")
            return pd.DataFrame()

        # Use exactly 252 trading days as per PDF specification
        window_data = historical_data.tail(252)
        self.logger.progress(f"   ✅ Using exactly {len(window_data)} trading days for momentum")

        if window_data.empty:
            self.logger.info("   ❌ Window data is empty after filtering")
            return pd.DataFrame()

        # Calculate 52-week highs and lows as per PDF specification
        highs_52w = window_data.max()  # Maximum closing price in trailing 252 trading days
        lows_52w = window_data.min()  # Minimum closing price in trailing 252 trading days
        current_prices = historical_data.iloc[-1] if not historical_data.empty else pd.Series()

        self.logger.progress(f"   📈 52-week highs calculated for {len(highs_52w)} Stocks")
        self.logger.progress(f"   📉 52-week lows calculated for {len(lows_52w)} Stocks")
        self.logger.progress(f"   💰 Current prices available for {len(current_prices)} Stocks")

        # Create result DataFrame with exact PDF specification calculations
        result_data = []
        valid_items = 0

        for symbol in highs_52w.index:
            if (not pd.isna(highs_52w[symbol]) and not pd.isna(lows_52w[symbol]) and
                    not pd.isna(current_prices[symbol]) and
                    highs_52w[symbol] > 0 and lows_52w[symbol] > 0 and current_prices[symbol] > 0):

                # Ensure data integrity: high >= low
                if highs_52w[symbol] >= lows_52w[symbol]:
                    # Calculate distances exactly as per PDF specification
                    distance_from_low = (current_prices[symbol] - lows_52w[symbol]) / lows_52w[symbol] * 100
                    distance_from_high = (highs_52w[symbol] - current_prices[symbol]) / highs_52w[symbol] * 100

                    result_data.append({
                        'symbol': symbol,
                        '52w_high': highs_52w[symbol],
                        '52w_low': lows_52w[symbol],
                        'current_price': current_prices[symbol],
                        'distance_from_low': distance_from_low,
                        'distance_from_high': distance_from_high
                    })

                    valid_items += 1
                    # self.logger.progress(
                    #     f"   ✅ {symbol}: High=₹{highs_52w[symbol]:.2f}, Low=₹{lows_52w[symbol]:.2f}, Current=₹{current_prices[symbol]:.2f}")
                    # self.logger.progress(
                    #     f"       Distance from Low: {distance_from_low:.2f}%, Distance from High: {distance_from_high:.2f}%")
                else:
                    self.logger.info(
                        f"   ❌ {symbol}: Data integrity issue - high ({highs_52w[symbol]:.2f}) < low ({lows_52w[symbol]:.2f})")
            else:
                self.logger.info(f"   ❌ {symbol}: Invalid data - contains NaN or zero values")

        result_df = pd.DataFrame(result_data)
        self.logger.progress(f"   🎯 Final momentum result: {len(result_df)} valid Stocks with proper 52-week data")

        if not result_df.empty:
            # Sort by distance from low for accumulation phase (buy closest to low)
            result_df_sorted = result_df.sort_values('distance_from_low')
            self.logger.progress(f"   📊 Stock Rankings (closest to 52-week low first):")
            for idx, row in result_df_sorted.head().iterrows():
                self.logger.info(f"      {idx + 1}. {row['symbol']}: {row['distance_from_low']:.2f}% from low")

        return result_df

    def run_backtest(self, tickers: List[str], start_date: str, end_date: str,
                     capital_per_week: float, accumulation_weeks: int,
                     brokerage_percent: float, compounding_enabled: bool) -> Dict[str, Any]:
        """Run the full backtest simulation"""
        try:
            self.logger.info(f"🚀 Starting Stock Rotation Backtest...")
            self.logger.info(f"   Tickers: {tickers}")
            self.logger.info(f"   Period: {start_date} to {end_date}")
            
            # Load data
            data_dict = self.load_data_from_database(tickers, start_date, end_date)
            if not data_dict:
                return {"error": "Failed to load data"}
                
            open_df = data_dict['open']
            close_df = data_dict['close']
            
            # Initialize simulation variables
            current_date = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)
            
            # Find first trading Monday
            current_date = self.get_next_trading_day(open_df, current_date, 'Monday')
            if not current_date:
                return {"error": "Could not find valid start date"}
                
            week_num = 1
            cash = 0
            holdings = {}
            self.portfolio_log = []
            self.transaction_costs_log = []
            self.purchase_history = {}
            
            # Simulation Loop
            while current_date <= end_date_dt:
                # Phase Logging (matching detailed style)
                is_accumulation = week_num <= accumulation_weeks
                self.logger.progress(f"\n🔄 Week {week_num} - {'ACCUMULATION' if is_accumulation else 'CHURNING'} PHASE")
                
                # Get current prices (Execution Day Prices)
                try:
                    current_prices = open_df.loc[current_date]
                except KeyError:
                    self.logger.info(f"   ⚠️ No price data for {current_date}, skipping week")
                    current_date = current_date + timedelta(days=7)
                    continue

                # Signal Date Logic (Previous trading day, usually Friday)
                # Look back from execution date (Monday) to find previous Friday
                search_date = current_date - timedelta(days=1)
                signal_date = self.get_last_trading_day(close_df, search_date, 'Friday')
                
                # Log Dates (matching user request)
                if signal_date:
                    self.logger.info(f"📅 Signal Day: {signal_date.strftime('%A, %Y-%m-%d')}")
                self.logger.info(f"💼 Execution Day: {current_date.strftime('%A, %Y-%m-%d')}")
                if is_accumulation:
                    self.logger.info(f"📈 Accumulation Phase - Week {week_num}")
                    self.logger.info(f"💰 Weekly capital allocation: ₹{capital_per_week:,.0f}")
                    cash += capital_per_week
                
                # 2. Calculate Momentum (52-week High/Low)
                # Use signal_date already calculated above
                
                # ===== 52-WEEK MOMENTUM CALCULATION =====
                if signal_date:
                    self.logger.progress(f"📊 Computing 52-week momentum for signal date...")
                    high_low_data = self.compute_52_week_high_low(close_df, signal_date)
                else:
                    high_low_data = pd.DataFrame()

                # Update state
                self.current_cash = cash
                self.current_holdings = holdings
                
                # Calculate current NAV
                nav_display = self.calculate_nav(holdings, current_prices, cash)
                
                # Calculate dynamic capital per week (with compounding if enabled)
                dynamic_capital_per_week = self.calculate_dynamic_churn_amount(
                    nav_display, cash, capital_per_week, accumulation_weeks, compounding_enabled
                )
                
                # Initialize trade log for this week with all required fields
                trade_log = {
                    'week': week_num,
                    'signal_date': signal_date,
                    'execution_date': current_date,
                    'action': 'none',
                    'ticker': '',
                    'units': 0,
                    'price': 0,
                    'amount': 0,
                    'costs': {},
                    'capital_gains_tax': 0,
                    'cash_before': cash - (capital_per_week if is_accumulation else 0), # Approximate cash before allocation
                    'cash_after': cash,
                    'holdings': holdings.copy(),
                    'nav': nav_display,
                    'base_capital_per_week': capital_per_week,
                    'dynamic_capital_per_week': dynamic_capital_per_week,
                    'compounding_enabled': compounding_enabled,
                    'debug_info': {
                        'momentum_data_available': not high_low_data.empty if signal_date else False,
                        'momentum_stocks_count': len(high_low_data) if signal_date and not high_low_data.empty else 0,
                        'available_prices_count': len(current_prices),
                        'phase': 'accumulation' if is_accumulation else 'churning'
                    }
                }

                # 3. Execute Strategy
                if signal_date and not high_low_data.empty:
                    if week_num <= accumulation_weeks:
                        # Accumulation Phase: Buy stock closest to 52-week low
                        reallocation_result = self.execute_reallocation_process(
                            week_num=week_num,
                            execution_date=current_date,
                            high_low_data=high_low_data,
                            open_prices=current_prices,
                            current_holdings=holdings,
                            available_cash=cash,
                            brokerage_percent=brokerage_percent,
                            target_allocation=capital_per_week
                        )
                        
                        buy_transaction = reallocation_result['buy_transaction']
                        holdings = reallocation_result['holdings']
                        cash = reallocation_result['remaining_cash']
                        
                        if buy_transaction:
                            # Enhance with detailed buy info
                            buy_amount = buy_transaction['amount']
                            buy_costs = buy_transaction['costs']
                            total_cost = buy_transaction.get('net_amount', buy_amount + buy_costs.get('total_costs', 0))
                            
                            trade_log.update({
                                'action': 'buy',
                                'ticker': buy_transaction['ticker'],
                                'units': buy_transaction['units'],
                                'price': buy_transaction['price'],
                                'amount': buy_amount,
                                'costs': buy_costs,
                                'cash_after': cash,
                                'total_cost': total_cost,
                                'buy_details': {
                                    'gross_amount': buy_amount,
                                    'brokerage': buy_costs.get('brokerage', 0),
                                    'stt': buy_costs.get('stt', 0),
                                    'stamp_duty': buy_costs.get('stamp_duty', 0),
                                    'exchange_charges': buy_costs.get('exchange_charges', 0),
                                    'sebi_charges': buy_costs.get('sebi_charges', 0),
                                    'gst': buy_costs.get('gst', 0),
                                    'total_transaction_costs': buy_costs.get('total_costs', 0),
                                    'total_cost': total_cost,
                                    # Handle simplified costs dict that might lack total_costs key if old format
                                    'cost_percentage': (buy_costs.get('total_costs', 0) / buy_amount * 100) if buy_amount > 0 else 0
                                }
                            })
                            
                            self.log_transaction_costs(week_num, current_date, 'buy', buy_transaction['ticker'],
                                                       buy_transaction['units'], buy_transaction['price'],
                                                       buy_transaction['costs'])
                    else:
                        # Churning Phase
                        trade_log['target_capital'] = dynamic_capital_per_week
                        
                        churn_result = self.execute_churning_phase(
                            week_num=week_num,
                            execution_date=current_date,
                            high_low_data=high_low_data,
                            open_prices=current_prices,
                            current_holdings=holdings,
                            cash=cash,
                            target_capital=dynamic_capital_per_week,
                            brokerage_percent=brokerage_percent
                        )
                        
                        holdings = churn_result['holdings']
                        cash = churn_result['cash']
                        
                        # Generate summary similar to ETF backtester
                        churning_summary = self.get_churning_summary(churn_result)
                        
                        # Merge results
                        trade_log.update(churn_result)
                        trade_log.update(churning_summary)
                            
                # Update final state in log
                trade_log['holdings'] = holdings.copy()
                trade_log['cash_after'] = cash
                trade_log['nav'] = self.calculate_nav(holdings, current_prices, cash)
                
                self.portfolio_log.append(trade_log)
                
                # Weekly Summary Log (matching user request)
                nav_display = self.calculate_nav(holdings, current_prices, cash)
                holdings_summary = [(s, qty) for s, qty in holdings.items()]
                
                # Weekly Summary Log (matching user request)
                nav_display = self.calculate_nav(holdings, current_prices, cash)
                holdings_summary = [(s, qty) for s, qty in holdings.items() if qty > 0]
                
                self.logger.progress(f"📊 Week {week_num} summary:")
                self.logger.trade(f"   Action: {'buy' if is_accumulation else 'churn'}")
                self.logger.info(f"   NAV: ₹{nav_display:,.0f}")
                self.logger.info(f"   Cash: ₹{cash:,.0f}")
                self.logger.info(f"   Holdings: {holdings_summary}")
                self.logger.info("============================================================")
                self.total_weeks = week_num
                
                # Move to next week
                next_week = current_date + timedelta(days=7)
                current_date = self.get_next_trading_day(open_df, next_week, 'Monday')
                if not current_date:
                    break
                week_num += 1
            
            # Process results
            self.process_results(open_df, capital_per_week, accumulation_weeks)
            
            # Load benchmark data for comparison
            total_investment = accumulation_weeks * capital_per_week
            self.nifty50_df = self.calculate_benchmark_buyhold(
                start_date, end_date, total_investment, brokerage_percent
            )
            
            return {
                "success": True,
                "total_weeks": self.total_weeks,
                "final_nav": self.weekly_nav_df['nav'].iloc[-1] if not self.weekly_nav_df.empty else 0
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def get_churning_summary(self, churn_result: Dict) -> Dict:
        """
        Generate a comprehensive summary of the churning operation

        This method provides detailed analysis of both buy and sell operations
        including cost breakdown, efficiency metrics, and portfolio impact
        """
        if churn_result.get('action') != 'churn':
            return {'error': 'Not a churning operation'}

        sell_summary = churn_result.get('sell_summary', {})
        buy_summary = churn_result.get('buy_summary', {})
        portfolio_state = churn_result.get('portfolio_state', {})
        churning_metrics = churn_result.get('churning_metrics', {})

        # Calculate detailed metrics
        total_sell_amount = sum(txn.get('amount', 0) for txn in sell_summary.get('sell_transactions', []))
        total_sell_costs = sum(
            txn.get('costs', {}).get('total_costs', 0) for txn in sell_summary.get('sell_transactions', []))
        total_buy_amount = buy_summary.get('buy_amount', 0)
        total_buy_costs = buy_summary.get('buy_costs', {}).get('total_costs', 0)

        return {
            'churning_summary': {
                'week': churn_result.get('week'),
                'execution_date': churn_result.get('execution_date'),
                'target_capital': churn_result.get('target_capital'),

                # === SELL OPERATIONS SUMMARY ===
                'sell_operations': {
                    'total_transactions': sell_summary.get('number_of_sell_transactions', 0),
                    'total_gross_amount': total_sell_amount,
                    'total_transaction_costs': total_sell_costs,
                    'total_capital_gains_tax': sell_summary.get('total_capital_gains_tax', 0),
                    'total_deductions': total_sell_costs + sell_summary.get('total_capital_gains_tax', 0),
                    'net_proceeds': sell_summary.get('total_raised', 0),
                    'cost_efficiency': (total_sell_costs / total_sell_amount * 100) if total_sell_amount > 0 else 0,
                    'tax_efficiency': (sell_summary.get('total_capital_gains_tax',
                                                        0) / total_sell_amount * 100) if total_sell_amount > 0 else 0,
                    'net_efficiency': (sell_summary.get('total_raised',
                                                        0) / total_sell_amount * 100) if total_sell_amount > 0 else 0,
                    'transactions': sell_summary.get('sell_transactions', [])
                },

                # === BUY OPERATIONS SUMMARY ===
                'buy_operations': {
                    'target_ticker': buy_summary.get('target_ticker'),
                    'units_purchased': buy_summary.get('units_bought', 0),
                    'gross_amount': total_buy_amount,
                    'transaction_costs': total_buy_costs,
                    'total_cost': buy_summary.get('total_buy_cost', 0),
                    'cost_efficiency': (total_buy_costs / total_buy_amount * 100) if total_buy_amount > 0 else 0,
                    'effective_price_per_unit': buy_summary.get('total_buy_cost', 0) / buy_summary.get('units_bought',
                                                                                                       1) if buy_summary.get(
                        'units_bought', 0) > 0 else 0
                },

                # === OVERALL CHURNING METRICS ===
                'overall_metrics': {
                    'capital_raised_vs_target': (
                                sell_summary.get('total_raised', 0) / churn_result.get('target_capital',
                                                                                       1) * 100) if churn_result.get(
                        'target_capital', 0) > 0 else 0,
                    'capital_deployed': (buy_summary.get('total_buy_cost', 0) / sell_summary.get('total_raised',
                                                                                                 1) * 100) if sell_summary.get(
                        'total_raised', 0) > 0 else 0,
                    'total_costs_percentage': ((total_sell_costs + total_buy_costs + sell_summary.get(
                        'total_capital_gains_tax', 0)) / total_sell_amount * 100) if total_sell_amount > 0 else 0,
                    'net_cash_flow': portfolio_state.get('net_cash_flow', 0),
                    'reallocation_success': churning_metrics.get('reallocation_success', False)
                },

                # === PORTFOLIO IMPACT ===
                'portfolio_impact': {
                    'cash_change': portfolio_state.get('cash_after', 0) - portfolio_state.get('cash_before', 0),
                    'holdings_change': {
                        'before': portfolio_state.get('holdings_before', {}),
                        'after': portfolio_state.get('holdings_after', {})
                    },
                    'net_portfolio_value_change': portfolio_state.get('net_cash_flow', 0)
                }
            }
        }

    def process_results(self, price_df: pd.DataFrame, capital_per_week: float, accumulation_weeks: int):
        """Process backtest results into DataFrames"""
        nav_data = []
        
        # Reconstruct daily NAV
        # This is a simplified version, ideally we'd track daily
        # For now, we'll use the weekly log points
        
        cumulative_investment = 0
        for i in range(1, self.total_weeks + 1):
            if i <= accumulation_weeks:
                cumulative_investment += capital_per_week
            else:
                cumulative_investment = accumulation_weeks * capital_per_week
                
            # Find log entry for this week
            week_logs = [l for l in self.portfolio_log if l['week'] == i]
            
            if week_logs:
                last_log = week_logs[-1]
                nav = last_log.get('nav', 0)
                date = last_log['execution_date']
            else:
                # If no trade, NAV is roughly same as previous (ignoring price moves for simplicity here, 
                # but in reality we should update with current prices)
                if nav_data:
                    nav = nav_data[-1]['nav'] # Placeholder
                    date = nav_data[-1]['date'] + timedelta(days=7) # Approx
                else:
                    nav = 0
                    date = datetime.now() # Should not happen
            
            nav_data.append({
                'date': date,
                'nav': nav,
                'cumulative_investment': cumulative_investment,
                'week': i,
                'base_capital_per_week': capital_per_week
            })
            
        self.weekly_nav_df = pd.DataFrame(nav_data)
        
        # Calculate returns for metrics
        if len(self.weekly_nav_df) > 1:
            self.weekly_nav_df['returns'] = self.weekly_nav_df['nav'].pct_change()
        else:
            self.weekly_nav_df['returns'] = 0

    def calculate_benchmark_buyhold(self, start_date: str, end_date: str, total_investment: float,
                                    brokerage_percent: float) -> pd.DataFrame:
        """
        Calculate benchmark performance using SIP (Systematic Investment Plan) approach
        to match the strategy's weekly capital injection during accumulation phase.
        Uses NIFTY50 from index_data.
        """
        session = None
        try:
            session = self._get_session()
            from sqlalchemy import text
            
            benchmark_symbol = "NIFTY_50"

            # Load benchmark data for the entire period
            query = text("""
                SELECT date, COALESCE(adj_close, close) AS close
                FROM nifty_50_index_market
                WHERE symbol = :symbol
                AND date >= CAST(:start_date AS DATE)
                AND date <= CAST(:end_date AS DATE)
                ORDER BY date
            """)

            result = session.execute(query, {
                "symbol": benchmark_symbol,
                "start_date": start_date,
                "end_date": end_date
            })
            rows = result.fetchall()
            columns = result.keys()
            nifty_df = pd.DataFrame(rows, columns=columns)

            if nifty_df.empty:
                self.logger.info("⚠️ No Nifty50 benchmark data found in index_data")
                return pd.DataFrame()

            nifty_df['date'] = pd.to_datetime(nifty_df['date'])
            nifty_df = nifty_df.set_index('date')

            # Get strategy execution dates and capital flows from weekly_nav_df
            if hasattr(self, 'weekly_nav_df') and not self.weekly_nav_df.empty:
                weekly_data = self.weekly_nav_df.copy()
                weekly_data['date'] = pd.to_datetime(weekly_data['date'])
                
                # Align benchmark data to weekly dates
                benchmark_log = []
                
                # SIP Parameters
                units_held = 0
                cash_balance = 0
                total_invested_so_far = 0
                
                self.logger.progress(f"📊 Calculating Nift50 SIP Benchmark Performance...")
                
                for _, row in weekly_data.iterrows():
                    date = row['date']
                    week_num = row['week']
                    
                    # 1. Get weekly capital injection (if any)
                    capital_injection = 0
                    if 'base_capital_per_week' in row:
                        capital_per_week = row['base_capital_per_week']
                        # Assuming strategy injects capital as long as week_num <= accumulation_weeks
                        # We can derive accumulation_weeks roughly if not available
                        
                        if capital_per_week > 0:
                            # Heuristic: if total_investment is known, derive accumulation weeks
                            derived_acc_weeks = int(total_investment / capital_per_week)
                            if week_num <= derived_acc_weeks:
                                capital_injection = capital_per_week
                    
                    # 2. Add capital to cash
                    cash_balance += capital_injection
                    total_invested_so_far += capital_injection
                    
                    # 3. Find benchmark price for this date (or nearest prior)
                    price = 0
                    if date in nifty_df.index:
                        price = nifty_df.loc[date]['close']
                    else:
                        # Find nearest prior date
                        prior_dates = nifty_df.index[nifty_df.index <= date]
                        if not prior_dates.empty:
                            price = nifty_df.loc[prior_dates[-1]]['close']

                    if price is not None:
                        try:
                            price = float(price)
                        except (TypeError, ValueError):
                            price = 0

                    if price and price > 0:
                        # 4. Buy units if we have cash
                        units_to_buy = int(cash_balance / price)
                        
                        if units_to_buy > 0:
                            transaction_amt = units_to_buy * price
                            txn_costs = self.calculate_transaction_costs('buy', transaction_amt, brokerage_percent)
                            total_outflow = transaction_amt + txn_costs['total_costs']
                            
                            # Adjust if costs exceed cash
                            while total_outflow > cash_balance and units_to_buy > 0:
                                units_to_buy -= 1
                                transaction_amt = units_to_buy * price
                                txn_costs = self.calculate_transaction_costs('buy', transaction_amt, brokerage_percent)
                                total_outflow = transaction_amt + txn_costs['total_costs']
                            
                            if units_to_buy > 0:
                                units_held += units_to_buy
                                cash_balance -= total_outflow
                    
                    # 5. Calculate NAV
                    current_nav = (units_held * price) + cash_balance
                    
                    benchmark_log.append({
                        'date': date,
                        'nav': current_nav,
                        'cumulative_investment': total_invested_so_far
                    })
                
                # Create final dataframe
                nifty_weekly_df = pd.DataFrame(benchmark_log)
                if not nifty_weekly_df.empty:
                    nifty_weekly_df['returns'] = nifty_weekly_df['nav'].pct_change()
                
                self.logger.progress(f"✅ Calculated SIP Benchmark for {len(nifty_weekly_df)} weeks")
                return nifty_weekly_df
            
            else:
                 # Fallback to old simple calculation if no weekly data (should not happen in normal flows)
                 return pd.DataFrame()

        except Exception as e:
            self.logger.trade(f"Error calculating Nifty50 buy-hold benchmark: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
        finally:
            if session:
                session.close()

    def calculate_metrics(self, capital_per_week: float, accumulation_weeks: int, risk_free_rate: float = 8.0) -> Dict:
        """Calculate performance metrics including XIRR and Treynor ratio"""
        if self.weekly_nav_df.empty:
            return {}

        df = self.weekly_nav_df.copy()
        final_nav = df['nav'].iloc[-1]
        total_invested = accumulation_weeks * capital_per_week

        total_return = (final_nav - total_invested) / total_invested * 100

        # Calculate actual time period in years from first to last date
        if len(df) > 1:
            start_date = pd.to_datetime(df['date'].iloc[0])
            end_date = pd.to_datetime(df['date'].iloc[-1])
            years = (end_date - start_date).days / 365.25
        else:
            years = len(df) / 52  # Fallback to weekly calculation

        if years > 0:
            cagr = ((final_nav / total_invested) ** (1 / years) - 1) * 100
        else:
            cagr = 0

        xirr = self.calculate_xirr(capital_per_week, accumulation_weeks)

        weekly_returns = df['returns'].dropna()
        if len(weekly_returns) > 1:
            # Use consistent annualization
            volatility = weekly_returns.std() * np.sqrt(52) * 100
            sharpe = (cagr - risk_free_rate) / volatility if volatility > 0 else 0
        else:
            volatility = 0
            sharpe = 0

        beta, treynor = self.calculate_beta_and_treynor(risk_free_rate)

        df['peak'] = df['nav'].expanding().max()
        df['drawdown'] = (df['nav'] - df['peak']) / df['peak'] * 100
        max_drawdown = df['drawdown'].min()

        calmar = abs(cagr / max_drawdown) if max_drawdown < 0 else 0

        return {
            'Total Investment': f"₹{total_invested:,.0f}",
            'Final Value': f"₹{final_nav:,.0f}",
            'Total Return': f"{total_return:.2f}%",
            'CAGR': f"{cagr:.2f}%",
            'XIRR': f"{xirr:.2f}%",
            'Volatility': f"{volatility:.2f}%",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Beta': f"{beta:.2f}",
            'Treynor Ratio': f"{treynor:.2f}%",
            'Max Drawdown': f"{max_drawdown:.2f}%",
            'Calmar Ratio': f"{calmar:.2f}",
            'Total Weeks': len(df),
            'Total Trades': len(self.transaction_costs_log),
            'Win Rate': f"{(weekly_returns > 0).mean() * 100:.1f}%" if len(weekly_returns) > 0 else "N/A"
        }

    def calculate_xirr(self, capital_per_week: float, accumulation_weeks: int) -> float:
        """Calculate XIRR (Extended Internal Rate of Return) using cash flows"""
        if self.weekly_nav_df.empty:
            return 0.0

        try:
            cash_flows = []
            dates = []

            df = self.weekly_nav_df.copy()
            for _, row in df.iterrows():
                if row['week'] <= accumulation_weeks:
                    cash_flows.append(-capital_per_week)
                    dates.append(row['date'])

            if len(df) > 0:
                final_nav = df['nav'].iloc[-1]
                final_date = df['date'].iloc[-1]
                cash_flows.append(final_nav)
                dates.append(final_date)

            if len(cash_flows) < 2:
                return 0.0

            cash_flow_series = pd.Series(cash_flows, index=pd.to_datetime(dates))

            return self.xirr_calculation(cash_flow_series) * 100

        except Exception as e:
            return 0.0

    def xirr_calculation(self, cash_flows: pd.Series) -> float:
        """Calculate XIRR using Newton-Raphson method"""
        try:
            rate = 0.1
            tolerance = 1e-6
            max_iterations = 100

            dates = cash_flows.index
            values = cash_flows.values

            start_date = dates[0]
            days = [(date - start_date).days for date in dates]

            for _ in range(max_iterations):
                npv = sum(cf / ((1 + rate) ** (day / 365.0)) for cf, day in zip(values, days))
                npv_derivative = sum(
                    -cf * day / 365.0 / ((1 + rate) ** (day / 365.0 + 1)) for cf, day in zip(values, days))

                if abs(npv) < tolerance:
                    return rate

                if abs(npv_derivative) < tolerance:
                    break

                rate = rate - npv / npv_derivative

            return rate if rate > -0.99 else 0.0

        except:
            return 0.0

    def calculate_beta_and_treynor(self, risk_free_rate: float) -> Tuple[float, float]:
        """Calculate portfolio beta against market benchmark and Treynor ratio"""
        if self.weekly_nav_df.empty:
            return 0.0, 0.0

        try:
            df = self.weekly_nav_df.copy()

            portfolio_returns = df['returns'].dropna()

            if len(portfolio_returns) < 10:
                return 0.0, 0.0

            if not self.nifty50_df.empty:
                nifty_df = self.nifty50_df.copy()
                nifty_df['date'] = pd.to_datetime(nifty_df['date'])
                nifty_returns = nifty_df.set_index('date')['returns'].dropna()

                portfolio_df = df.set_index('date')

                nifty_weekly = nifty_returns.resample('W').apply(lambda x: (1 + x).prod() - 1)

                aligned_data = pd.concat([portfolio_returns, nifty_weekly], axis=1, join='inner')
                aligned_data.columns = ['portfolio', 'nifty']
                aligned_data = aligned_data.dropna()

                if len(aligned_data) > 10:
                    portfolio_var = aligned_data['portfolio'].var()
                    nifty_var = aligned_data['nifty'].var()
                    covariance = aligned_data['portfolio'].cov(aligned_data['nifty'])

                    beta = covariance / nifty_var if nifty_var > 0 else 1.0
                    beta = max(0.1, min(3.0, beta))
                else:
                    beta = 1.0
            else:
                beta = 1.0

            if len(df) > 0:
                years = len(df) / 52
                final_nav = df['nav'].iloc[-1]
                initial_investment = df['cumulative_investment'].iloc[-1]

                if years > 0 and initial_investment > 0:
                    portfolio_cagr = ((final_nav / initial_investment) ** (1 / years) - 1) * 100
                else:
                    portfolio_cagr = 0.0
            else:
                portfolio_cagr = 0.0

            # Use excess return for Treynor ratio calculation
            excess_return = portfolio_cagr - risk_free_rate
            treynor_ratio = excess_return / beta if beta > 0 else 0.0

            return beta, treynor_ratio

        except Exception as e:
            return 0.0, 0.0

    def calculate_nifty50_xirr(self, total_investment: float) -> float:
        """Calculate XIRR for Nifty50 buy-and-hold strategy (single investment at start)"""
        if self.nifty50_df.empty:
            return 0.0

        try:
            df = self.nifty50_df.copy()

            # Single investment at start, final value at end
            cash_flows = [-total_investment, df['nav'].iloc[-1]]
            dates = [df['date'].iloc[0], df['date'].iloc[-1]]

            cash_flow_series = pd.Series(cash_flows, index=pd.to_datetime(dates))
            return self.xirr_calculation(cash_flow_series) * 100

        except Exception as e:
            return 0.0

    def calculate_benchmark_metrics(self, total_investment: float, risk_free_rate: float = 8.0) -> Dict:
        """Calculate benchmark buy-and-hold metrics using market index or large cap stock"""
        if self.nifty50_df.empty:
            return {}

        df = self.nifty50_df.copy()
        final_nav = df['nav'].iloc[-1]

        total_return = (final_nav - total_investment) / total_investment * 100

        # Calculate actual time period in years from first to last date
        if len(df) > 1:
            start_date = pd.to_datetime(df['date'].iloc[0])
            end_date = pd.to_datetime(df['date'].iloc[-1])
            years = (end_date - start_date).days / 365.25
        else:
            years = len(df) / 252  # Fallback to daily calculation

        if years > 0:
            cagr = ((final_nav / total_investment) ** (1 / years) - 1) * 100
        else:
            cagr = 0

        # Calculate proper XIRR for Nifty50 (single investment at start)
        xirr = self.calculate_nifty50_xirr(total_investment)

        daily_returns = df['returns'].dropna()
        if len(daily_returns) > 1:
            # Use consistent annualization
            volatility = daily_returns.std() * np.sqrt(252) * 100
            sharpe = (cagr - risk_free_rate) / volatility if volatility > 0 else 0
        else:
            volatility = 0
            sharpe = 0

        beta = 1.0
        # Use excess return for Treynor ratio
        excess_return = cagr - risk_free_rate
        treynor_ratio = excess_return / beta if beta > 0 else 0.0

        df['peak'] = df['nav'].expanding().max()
        df['drawdown'] = (df['nav'] - df['peak']) / df['peak'] * 100
        max_drawdown = df['drawdown'].min()

        calmar = abs(cagr / max_drawdown) if max_drawdown < 0 else 0

        return {
            'Total Investment': f"₹{total_investment:,.0f}",
            'Final Value': f"₹{final_nav:,.0f}",
            'Total Return': f"{total_return:.2f}%",
            'CAGR': f"{cagr:.2f}%",
            'XIRR': f"{xirr:.2f}%",
            'Volatility': f"{volatility:.2f}%",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Beta': f"{beta:.2f}",
            'Treynor Ratio': f"{treynor_ratio:.2f}%",
            'Max Drawdown': f"{max_drawdown:.2f}%",
            'Calmar Ratio': f"{calmar:.2f}",
            'Total Days': len(df),
            'Win Rate': f"{(daily_returns > 0).mean() * 100:.1f}%" if len(daily_returns) > 0 else "N/A"
        }

    def create_formatted_metrics_table(self, stock_metrics: Dict, benchmark_metrics: Dict) -> pd.DataFrame:
        """Create formatted comparison table"""
        data = [
            {"Metric": "Total Return", "Stock Strategy": stock_metrics.get("total_return"), "Benchmark": benchmark_metrics.get("total_return")},
            {"Metric": "CAGR", "Stock Strategy": stock_metrics.get("cagr"), "Benchmark": benchmark_metrics.get("cagr")},
            {"Metric": "Final Value", "Stock Strategy": stock_metrics.get("final_value"), "Benchmark": benchmark_metrics.get("final_value")}
        ]
        return pd.DataFrame(data)

    def get_transaction_costs_summary(self) -> Dict:
        """Get summary of transaction costs grouped by year (Simplified Format)"""
        if not self.transaction_costs_log:
            return {}
            
        costs_df = pd.DataFrame(self.transaction_costs_log)
        
        # Robustly determine column to use for year
        date_col = None
        if 'date' in costs_df.columns:
            date_col = 'date'
        elif 'execution_date' in costs_df.columns:
            date_col = 'execution_date'
            
        if date_col:
            costs_df['year'] = pd.to_datetime(costs_df[date_col]).dt.year.astype(str)
        else:
             print(f"[DEBUG] Stock Rotation get_transaction_costs_summary: Missing date column! Columns found: {costs_df.columns.tolist()}")
             costs_df['year'] = "Total"

        summary = {}

        def calculate_metrics(df):
             transaction_costs = df['total_costs'].sum()
             capital_gains_tax = df['capital_gains_tax'].sum()
             total_costs = transaction_costs + capital_gains_tax
             brokerage = df['brokerage'].sum()
             transactions = df['week'].nunique()
             
             return {
                'transaction_costs': round(transaction_costs, 2),
                'capital_gains_tax': round(capital_gains_tax, 2),
                'total_costs': round(total_costs, 2),
                'total_brokerage': round(brokerage, 2),
                'transactions': transactions
             }
        
        # Yearly summary
        if 'year' in costs_df.columns:
             for year, group in costs_df.groupby('year'):
                 summary[str(year)] = calculate_metrics(group)
                 
        return summary

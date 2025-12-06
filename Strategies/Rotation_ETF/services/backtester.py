import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import warnings
import os
import json
from Databases.market_data_db_connection import (
    create_connection as create_market_data_connection,
    get_session as get_market_data_session,
    init_database as init_market_data_database,
    ETFData,  # Primary ETF data model (uses etf_data table)
    ETFUnified,  # Deprecated, kept for backward compatibility
    ETFMetadata  # Metadata model (uses etf_metadata table)
)

# Import base class for OOP refactoring
from Strategies.Rotation.RotationStrategy import RotationStrategy

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Chart functionality will be disabled.")


class ETFRotationBacktester(RotationStrategy):
    def __init__(self, db_path: str = None):
        """
        Initialize ETF Rotation Backtester

        Args:
            db_path: Deprecated - kept for compatibility. Now uses PostgreSQL MarketData database.
        """
        # Call base class constructor (initializes common attributes)
        super().__init__()
        
        # Initialize PostgreSQL connection
        if not create_market_data_connection():
            raise RuntimeError("Failed to connect to MarketData database")

        # Initialize database tables
        init_market_data_database()

        # Initialize attributes before calling load_metadata (which uses _verbose)
        self.portfolio_log = []
        self.weekly_nav_df = pd.DataFrame()
        self.nifty50_df = pd.DataFrame()
        self.transaction_costs_log = []
        self.purchase_history = {}
        
        # Performance optimizations
        self._data_cache = {}
        self._verbose = False  # Set to True for debugging
        
        # Strategy-specific table configuration (must be set before method calls)
        self.metadata_table = "etf_metadata"  # Use etf_metadata for ETF metadata
        self.data_table = "etf_data"  # ETFs are stored in etf_data table
        
        self.available_databases = []  # Not needed for PostgreSQL
        
        # Trade execution tracking
        self.skipped_days = []
        self.total_weeks = 0
        self.successful_signals = 0
        self.successful_executions = 0
        self.current_cash = 0
        self.current_holdings = {}
        
        # Load metadata (now safe to call since _verbose is initialized)
        self.etf_metadata = self.load_metadata()

    
    def _get_data_model(self):
        """Return the SQLAlchemy model for ETF data"""
        return ETFData
    
    def _get_metadata_model(self):
        """Return the SQLAlchemy model for ETF metadata"""
        return ETFMetadata
        
    def set_verbose(self, verbose: bool = True):
        """Enable or disable verbose logging for debugging"""
        self._verbose = verbose

    def _get_session(self):
        """Get database session for PostgreSQL"""
        return get_market_data_session()
    
    def load_data(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Implementation of abstract method from WealthAIBase.
        
        Loads data for all available ETFs in metadata.
        For specific ticker loading, use load_data_from_database() instead.
        
        Args:
            start_date: Start date for data loading
            end_date: End date for data loading
            
        Returns:
            Dictionary with 'open', 'high', 'low', 'close', 'volume' DataFrames
        """
        # Get all available tickers from metadata
        if not hasattr(self, 'etf_metadata') or not self.etf_metadata:
            self.etf_metadata = self.load_metadata()
        
        tickers = list(self.etf_metadata.keys())
        
        # Convert datetime to string format expected by load_data_from_database
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Delegate to the more specific method
        return self.load_data_from_database(tickers, start_date_str, end_date_str)

    def cleanup(self):
        """Clean up resources and clear cache"""
        self._data_cache.clear()

    def get_trading_summary(self):
        """Get a summary of trading activity"""
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
        """Load ETF metadata from PostgreSQL database"""
        session = None
        try:
            session = self._get_session()

            # Try to load from etf_metadata table first
            from sqlalchemy import text
            result = session.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND LOWER(table_name) = 'etf_metadata'
            """))

            metadata_table_exists = result.fetchone() is not None

            if metadata_table_exists:
                # Check if metadata table has any records
                count_result = session.execute(text("SELECT COUNT(*) as count FROM etf_metadata"))
                metadata_count = count_result.fetchone()[0]

                if metadata_count > 0:
                    # Load from metadata table using raw SQL (to avoid ORM model mismatch)
                    # Loading ETF metadata (silent)

                    # First, get the actual column names from the table
                    columns_result = session.execute(text("""
                        SELECT column_name 
                        FROM information_schema.columns
                        WHERE table_schema = 'public' 
                        AND table_name = 'etf_metadata'
                        ORDER BY ordinal_position
                    """))
                    column_names = [row[0] for row in columns_result.fetchall()]
                    # Metadata columns retrieved (silent)

                    # Build query based on actual columns
                    # Map common column name variations
                    symbol_col = None
                    start_date_col = None
                    end_date_col = None
                    years_col = None
                    records_col = None
                    source_col = None

                    for col in column_names:
                        col_lower = col.lower()
                        if 'symbol' in col_lower:
                            symbol_col = col
                        elif 'start' in col_lower and 'date' in col_lower:
                            start_date_col = col
                        elif 'end' in col_lower and 'date' in col_lower:
                            end_date_col = col
                        elif 'year' in col_lower:
                            years_col = col
                        elif 'record' in col_lower or 'count' in col_lower:
                            records_col = col
                        elif 'source' in col_lower:
                            source_col = col

                    if not symbol_col:
                        logger.error("Could not find symbol column in etf_metadata table")
                        # Fall back to calculating from data table
                    else:
                        # Build SELECT query with available columns
                        select_cols = [symbol_col]
                        if start_date_col:
                            select_cols.append(f"{start_date_col} as start_date")
                        if end_date_col:
                            select_cols.append(f"{end_date_col} as end_date")
                        if years_col:
                            select_cols.append(f"{years_col} as years_available")
                        if records_col:
                            select_cols.append(f"{records_col} as total_records")
                        if source_col:
                            select_cols.append(f"{source_col} as data_source")

                        query_str = f"SELECT {', '.join(select_cols)} FROM etf_metadata"
                        # Querying metadata (silent)

                        try:
                            metadata_result = session.execute(text(query_str))
                            metadata_rows = metadata_result.fetchall()
                            # Metadata rows retrieved (silent)

                            metadata = {}
                            for row in metadata_rows:
                                try:
                                    # Access by index (symbol is always first)
                                    symbol = row[0]

                                    # Build metadata dict with available fields
                                    meta_dict = {
                                        'start_date': None,
                                        'end_date': None,
                                        'years_available': 0,
                                        'total_records': 0,
                                        'data_source': 'database'
                                    }

                                    idx = 1
                                    if start_date_col:
                                        meta_dict['start_date'] = str(row[idx]) if idx < len(row) and row[idx] else None
                                        idx += 1
                                    if end_date_col:
                                        meta_dict['end_date'] = str(row[idx]) if idx < len(row) and row[idx] else None
                                        idx += 1
                                    if years_col:
                                        meta_dict['years_available'] = float(row[idx]) if idx < len(row) and row[
                                            idx] else 0
                                        idx += 1
                                    if records_col:
                                        meta_dict['total_records'] = int(row[idx]) if idx < len(row) and row[idx] else 0
                                        idx += 1
                                    if source_col:
                                        meta_dict['data_source'] = str(row[idx]) if idx < len(row) and row[
                                            idx] else 'database'

                                    metadata[symbol] = meta_dict
                                except Exception as row_error:
                                    # Skip invalid rows silently
                                    continue

                            # Metadata loaded successfully (silent - no log needed)
                            if len(metadata) > 0:
                                return metadata
                            else:
                                if self._verbose:
                                    print(
                                        "[DEBUG] etf_metadata table has records but couldn't parse them, falling back to calculating from data table")
                        except Exception as query_error:
                            # Query error - will fallback to data table calculation
                            pass
                            # Fall back to calculating from data table
                else:
                    if self._verbose:
                        print(
                            "[DEBUG] etf_metadata table exists but has 0 records, falling back to calculating from data table")
            else:
                if self._verbose:
                    print("[DEBUG] etf_metadata table not found, calculating from data table")

            # Fallback: calculate metadata from data table
            metadata = self.calculate_metadata_from_data(session)
            if self._verbose:
                print(f"[DEBUG] Calculated metadata for {len(metadata)} ETFs from data table")

            return metadata

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"[ERROR] Error loading ETF metadata: {e}")
            print(f"[ERROR] Traceback: {error_details}")
            return {}
        finally:
            if session:
                session.close()

    def calculate_metadata_from_data(self, session) -> Dict[str, Dict]:
        """Calculate metadata from ETF data table if metadata table doesn't exist"""
        try:
            from sqlalchemy import text, func
            from sqlalchemy.orm import Session

            # First, list all tables to see what's available
            all_tables = session.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE '%etf%'
                ORDER BY table_name
            """))
            available_tables = [row[0] for row in all_tables.fetchall()]
            print(f"[DEBUG] Available ETF-related tables: {available_tables}")

            # Try to find the correct table name (case-insensitive check)
            table_name = None
            for candidate in ['etf_data', 'ETF_DATA', 'etf_unified', 'ETF_UNIFIED']:
                table_check = session.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND LOWER(table_name) = LOWER(:table_name)
                """), {"table_name": candidate})
                if table_check.fetchone():
                    # Get the actual table name (case-sensitive)
                    actual_name = session.execute(text("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND LOWER(table_name) = LOWER(:table_name)
                    """), {"table_name": candidate}).fetchone()[0]
                    table_name = actual_name
                    print(f"[DEBUG] Found table: {table_name}")
                    break

            if not table_name:
                print("[ERROR] No ETF data table found. Available tables:", available_tables)
                # List all tables in the database
                all_db_tables = session.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """))
                all_table_names = [row[0] for row in all_db_tables.fetchall()]
                print(f"[DEBUG] All tables in database: {all_table_names}")
                return {}

            # Check column names to determine which table structure we're using
            columns_check = session.execute(text("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public' 
                AND table_name = :table_name
                ORDER BY ordinal_position
            """), {"table_name": table_name})
            columns = {row[0]: row[1] for row in columns_check.fetchall()}
            print(f"[DEBUG] Table {table_name} columns: {list(columns.keys())}")

            # Determine which columns to use based on table structure
            if 'open_price' in columns or 'close_price' in columns:
                # Using etf_unified structure
                date_col = 'date'
                symbol_col = 'symbol'
                open_col = 'open_price'
                close_col = 'close_price'
                print("[DEBUG] Using etf_unified table structure")
            else:
                # Using etf_data structure
                date_col = 'date'
                symbol_col = 'symbol'
                open_col = 'open'
                close_col = 'close'
                print("[DEBUG] Using etf_data table structure")

            # Check if table has any data
            count_check = session.execute(text(f"SELECT COUNT(*) as count FROM \"{table_name}\""))
            total_count = count_check.fetchone()[0]

            if total_count == 0:
                print(f"[WARNING] {table_name} table exists but is empty")
                return {}

            if self._verbose:
                print(f"[DEBUG] Found {total_count} total records in {table_name} table")

            # Determine date column type and build appropriate query
            date_type = str(columns.get('date', '')).lower()
            is_text_date = 'character' in date_type or 'text' in date_type or 'varchar' in date_type

            print(f"[DEBUG] Date column type: {date_type}, is_text_date: {is_text_date}")

            try:
                if is_text_date:
                    # Date is stored as text, need to convert
                    query_str = f"""
                        SELECT 
                            {symbol_col} as symbol,
                            MIN({date_col}) as start_date,
                            MAX({date_col}) as end_date,
                            COUNT(*) as total_records,
                            ROUND(EXTRACT(EPOCH FROM (MAX({date_col}::date) - MIN({date_col}::date))) / 86400.0 / 365.25, 1) as years_available
                        FROM "{table_name}"
                        GROUP BY {symbol_col}
                        ORDER BY {symbol_col}
                    """
                else:
                    # Date is stored as date/timestamp
                    query_str = f"""
                        SELECT 
                            {symbol_col} as symbol,
                            MIN({date_col}::text) as start_date,
                            MAX({date_col}::text) as end_date,
                            COUNT(*) as total_records,
                            ROUND(EXTRACT(EPOCH FROM (MAX({date_col}) - MIN({date_col}))) / 86400.0 / 365.25, 1) as years_available
                        FROM "{table_name}"
                        GROUP BY {symbol_col}
                        ORDER BY {symbol_col}
                    """

                print(f"[DEBUG] Executing query...")
                query = text(query_str)
                result = session.execute(query)
                rows = result.fetchall()

                print(f"[DEBUG] Query returned {len(rows)} distinct ETF symbols")

                if len(rows) == 0:
                    print("[WARNING] Query returned 0 rows. Trying simpler query...")
                    # Try a simpler query to see if we can get any data
                    simple_query = text(f'SELECT DISTINCT {symbol_col} FROM "{table_name}" LIMIT 10')
                    simple_result = session.execute(simple_query)
                    simple_rows = simple_result.fetchall()
                    print(f"[DEBUG] Simple query returned {len(simple_rows)} symbols: {[r[0] for r in simple_rows]}")
                    # If simple query works, use it as fallback
                    if len(simple_rows) > 0:
                        metadata = {}
                        for row in simple_rows:
                            symbol = row[0]
                            metadata[symbol] = {
                                'start_date': None,
                                'end_date': None,
                                'years_available': 0,
                                'total_records': 0,
                                'data_source': 'OHLCV'
                            }
                        print(f"[DEBUG] Using simple query result: {len(metadata)} symbols")
                        return metadata

                metadata = {}
                for idx, row in enumerate(rows):
                    try:
                        # Try attribute access first, then fall back to index
                        if hasattr(row, '_mapping') or hasattr(row, 'keys'):
                            # SQLAlchemy Row object with named access
                            try:
                                symbol = row.symbol
                                start_date = row.start_date
                                end_date = row.end_date
                                total_records = row.total_records
                                years_available = row.years_available
                            except AttributeError:
                                # Fall back to index access
                                symbol = row[0]
                                start_date = row[1]
                                end_date = row[2]
                                total_records = row[3]
                                years_available = row[4]
                        else:
                            # Tuple-like access
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
                            'data_source': 'OHLCV'
                        }
                    except Exception as row_error:
                        print(f"[ERROR] Error processing row {idx}: {row_error}, row type: {type(row)}, row: {row}")
                        import traceback
                        print(f"[ERROR] Row error traceback: {traceback.format_exc()}")
                        continue

                print(f"[DEBUG] Successfully processed {len(metadata)} ETFs into metadata dictionary")
                if len(metadata) == 0 and len(rows) > 0:
                    print(f"[ERROR] Failed to process any rows! First row sample: {rows[0] if rows else 'N/A'}")
                return metadata

            except Exception as query_error:
                print(f"[ERROR] Error executing metadata query: {query_error}")
                import traceback
                print(f"[ERROR] Traceback: {traceback.format_exc()}")
                # Try a fallback simple query
                try:
                    print("[DEBUG] Attempting fallback query...")
                    fallback_query = text(f"""
                        SELECT DISTINCT 
                            {symbol_col} as symbol
                        FROM "{table_name}"
                        ORDER BY {symbol_col}
                    """)
                    fallback_result = session.execute(fallback_query)
                    fallback_rows = fallback_result.fetchall()

                    metadata = {}
                    for row in fallback_rows:
                        symbol = row.symbol if hasattr(row, 'symbol') else row[0]
                        metadata[symbol] = {
                            'start_date': None,
                            'end_date': None,
                            'years_available': 0,
                            'total_records': 0,
                            'data_source': 'OHLCV'
                        }
                    print(f"[DEBUG] Fallback query returned {len(metadata)} symbols")
                    return metadata
                except Exception as fallback_error:
                    print(f"[ERROR] Fallback query also failed: {fallback_error}")
                    return {}

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"[ERROR] Error calculating metadata: {e}")
            print(f"[ERROR] Traceback: {error_details}")
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

        if 'pharma' in symbol_lower:
            return 'Pharmaceutical Sector ETF'
        elif 'bank' in symbol_lower:
            return 'Banking Sector ETF'
        elif 'it' in symbol_lower or 'tech' in symbol_lower:
            return 'Technology Sector ETF'
        elif 'gold' in symbol_lower:
            return 'Gold Commodity ETF'
        elif 'oil' in symbol_lower or 'energy' in symbol_lower:
            return 'Oil & Gas Sector ETF'
        elif 'metal' in symbol_lower:
            return 'Metal & Mining Sector ETF'
        elif 'infra' in symbol_lower:
            return 'Infrastructure Sector ETF'
        elif 'defence' in symbol_lower or 'defense' in symbol_lower:
            return 'Defence Sector ETF'
        elif 'midcap' in symbol_lower or 'mid' in symbol_lower:
            return 'Mid Cap ETF'
        elif 'smallcap' in symbol_lower or 'small' in symbol_lower:
            return 'Small Cap ETF'
        elif 'liquid' in symbol_lower:
            return 'Liquid Fund ETF'
        elif 'nifty' in symbol_lower:
            return 'Nifty Index ETF'
        elif 'sensex' in symbol_lower:
            return 'Sensex Index ETF'
        elif 'psu' in symbol_lower:
            return 'PSU Sector ETF'
        elif 'cpse' in symbol_lower:
            return 'CPSE ETF'
        elif 'dividend' in symbol_lower:
            return 'Dividend ETF'
        elif 'momentum' in symbol_lower:
            return 'Momentum ETF'
        elif 'value' in symbol_lower:
            return 'Value ETF'
        elif 'quality' in symbol_lower:
            return 'Quality ETF'
        elif any(geo in symbol_lower for geo in ['us', 'usa', 'nasdaq', 'sp500', 'dow']):
            return 'International ETF'
        elif 'consumption' in symbol_lower or 'consumer' in symbol_lower:
            return 'Consumer Sector ETF'
        elif 'auto' in symbol_lower:
            return 'Automotive Sector ETF'
        elif 'realty' in symbol_lower or 'real' in symbol_lower:
            return 'Real Estate ETF'
        elif 'healthcare' in symbol_lower or 'health' in symbol_lower:
            return 'Healthcare Sector ETF'
        elif 'fmcg' in symbol_lower:
            return 'FMCG Sector ETF'
        elif 'pvt' in symbol_lower or 'private' in symbol_lower:
            return 'Private Bank ETF'
        else:
            return f'{symbol} ETF'

    def get_asset_sector_classification(self, symbol: str) -> str:
        """Classify ETF into sector categories"""
        symbol_lower = symbol.lower()

        if symbol in ['NIFTYBEES', 'JUNIORBEES', 'MIDCAPETF']:
            return 'Broad Market'
        elif 'bank' in symbol_lower or 'psu' in symbol_lower:
            return 'Financial'
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
        else:
            return 'Other'

    def get_default_etf_selection(self, available_etfs: List[str], max_count: int = 5) -> List[str]:
        """Intelligently select default ETFs for diversified portfolio"""
        priority_symbols = [
            'RELIANCE', 'HDFCBANK', 'INFY', 'ICICIBANK', 'TCS',
            'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK'
        ]

        selected = []
        for symbol in priority_symbols:
            if symbol in available_etfs and len(selected) < max_count:
                selected.append(symbol)

        remaining = [etf for etf in available_etfs if etf not in selected]
        while len(selected) < max_count and remaining:
            selected.append(remaining.pop(0))

        return selected

    def calculate_common_date_range(self, selected_etfs: List[str]) -> Tuple[Optional[str], Optional[str], float]:
        """Calculate common date range for selected ETFs by querying etf_data table directly"""
        if not selected_etfs:
            print("[ERROR] No ETFs provided for date range calculation")
            return None, None, 0.0
        
        session = None
        try:
            session = self._get_session()
            from sqlalchemy import text
            
            print(f"[DEBUG] Querying etf_data table for date ranges of: {selected_etfs}")
            
            # Build placeholders for IN clause
            placeholders = ','.join([f':ticker_{i}' for i in range(len(selected_etfs))])
            
            # Query to get MIN and MAX dates for each ETF directly from etf_data table
            query = text(f"""
                SELECT 
                    symbol,
                    MIN(date) as start_date,
                    MAX(date) as end_date,
                    COUNT(*) as total_records
                FROM etf_data
                WHERE symbol IN ({placeholders})
                GROUP BY symbol
                ORDER BY symbol
            """)
            
            # Build parameters dict
            params = {f'ticker_{i}': ticker for i, ticker in enumerate(selected_etfs)}
            
            result = session.execute(query, params)
            rows = result.fetchall()
            
            if not rows:
                print(f"[ERROR] No data found in etf_data table for: {selected_etfs}")
                return None, None, 0.0
            
            # Parse results
            etf_date_ranges = {}
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
                
                etf_date_ranges[symbol] = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'total_records': total_records,
                    'years_available': years_available
                }
            
            # Check if all requested ETFs were found
            missing_etfs = [etf for etf in selected_etfs if etf not in etf_date_ranges]
            if missing_etfs:
                print(f"[WARNING] No data found for: {missing_etfs}")
            
            if not etf_date_ranges:
                print(f"[ERROR] None of the requested ETFs found in etf_data table: {selected_etfs}")
                return None, None, 0.0
            
            # Calculate common date range
            start_dates = [data['start_date'] for data in etf_date_ranges.values()]
            end_dates = [data['end_date'] for data in etf_date_ranges.values()]
            
            # The latest start date among all ETFs (most restrictive)
            latest_start = max(start_dates)
            common_end = min(end_dates)
            
            print(f"📅 ETF Data Ranges (from etf_data table):")
            for symbol, data in etf_date_ranges.items():
                years = data['years_available']
                print(f"   {symbol:12s}: {data['start_date'].strftime('%Y-%m-%d')} to {data['end_date'].strftime('%Y-%m-%d')} ({years:.1f} years, {data['total_records']} records)")
            
            print(f"📊 Common data range: {latest_start.strftime('%Y-%m-%d')} to {common_end.strftime('%Y-%m-%d')}")
            
            # OPTIMIZED BUFFER: Use 60 weeks (420 days) for 252+ trading days
            buffer_weeks = 53
            buffer_days = buffer_weeks * 7  # 420 calendar days (~300 trading days)
            strategy_start = latest_start + timedelta(days=buffer_days)
            
            print(f"🎯 Enhanced Buffer Strategy:")
            print(f"   Buffer period: {buffer_weeks} weeks ({buffer_days} calendar days)")
            print(f"   Expected trading days: ~{int(buffer_days * 5 / 7)} days")
            print(f"   Required for momentum: 252 trading days")
            print(f"   Safety margin: ~{int(buffer_days * 5 / 7) - 252} trading days")
            print(f"   Strategy start: {strategy_start.strftime('%Y-%m-%d')}")
            
            # Ensure we have at least 1 year of backtest data after the strategy start
            if strategy_start >= common_end:
                print(f"❌ Insufficient data even with {buffer_weeks}-week buffer:")
                print(f"   Strategy start: {strategy_start.strftime('%Y-%m-%d')}")
                print(f"   Data ends: {common_end.strftime('%Y-%m-%d')}")
                shortage_days = (strategy_start - common_end).days
                print(f"   Shortage: {shortage_days} days")
                
                # Try with maximum possible buffer
                max_possible_days = (common_end - latest_start).days - 365  # Reserve 1 year for backtesting
                if max_possible_days > 252:  # At least need 252 trading days
                    alt_strategy_start = latest_start + timedelta(days=max_possible_days)
                    print(f"🔄 Alternative with maximum buffer:")
                    print(f"   Maximum buffer: {max_possible_days} days ({max_possible_days / 7:.1f} weeks)")
                    print(f"   Alternative start: {alt_strategy_start.strftime('%Y-%m-%d')}")
                    
                    years_available = (common_end - alt_strategy_start).days / 365.25
                    return alt_strategy_start.strftime('%Y-%m-%d'), common_end.strftime('%Y-%m-%d'), years_available
                else:
                    return None, None, 0.0
            
            # Check if we have sufficient backtest period
            years_available = (common_end - strategy_start).days / 365.25
            
            print(f"📈 Backtest Feasibility:")
            if years_available >= 10:
                print(f"   ✅ Excellent: {years_available:.1f} years provides statistically robust results")
            elif years_available >= 5:
                print(f"   ✅ Good: {years_available:.1f} years allows reasonable strategy validation")
            elif years_available >= 2:
                print(f"   ⚠️ Fair: {years_available:.1f} years provides basic validation")
            else:
                print(f"   ⚠️ Limited: {years_available:.1f} years may not be reliable")
            
            return strategy_start.strftime('%Y-%m-%d'), common_end.strftime('%Y-%m-%d'), years_available
            
        except Exception as e:
            print(f"[ERROR] Error calculating date range from etf_data table: {e}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
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
                print(f"📦 Using cached data for {len(tickers)} ETFs")
            return self._data_cache[cache_key]

        session = None
        try:
            session = self._get_session()
            from sqlalchemy import text

            if self._verbose:
                print(f"📊 Loading data with historical buffer:")
                print(f"   Strategy period: {start_date} to {end_date}")
                print(f"   Data loading period: {buffer_start_date} to {end_date}")
                print(f"   Buffer: 400 calendar days (~252 trading days) for momentum calculations")

            # Use etf_data table - column names match directly
            # Get available symbols using proper parameterization
            # Build placeholders for IN clause
            placeholders = ','.join([f':ticker_{i}' for i in range(len(tickers))])

            query = text(f"""
                SELECT DISTINCT symbol 
                FROM etf_data 
                WHERE symbol IN ({placeholders})
                AND date >= CAST(:start_date AS DATE) AND date <= CAST(:end_date AS DATE)
            """)

            # Build parameters dict with all tickers and dates
            params = {f'ticker_{i}': t for i, t in enumerate(tickers)}
            params.update({
                "start_date": buffer_start_date,
                "end_date": end_date
            })

            result = session.execute(query, params)
            available_tickers = [row[0] for row in result.fetchall()]
            missing_tickers = [t for t in tickers if t not in available_tickers]

            if missing_tickers and self._verbose:
                print(f"⚠️ Missing data for: {', '.join(missing_tickers)}")

            if not available_tickers:
                raise ValueError("No data available for any of the selected ETFs")

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
                    adjusted_close
                FROM etf_data
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
                print(f"✅ Loaded data for {len(available_tickers)} ETFs: {', '.join(available_tickers)}")

            df['date'] = pd.to_datetime(df['date'])

            # Optimized pivot operations
            data_dict = {}
            for column in ['open', 'high', 'low', 'close', 'volume']:
                pivot_df = df.pivot(index='date', columns='symbol', values=column)
                data_dict[column] = pivot_df.fillna(method='ffill').fillna(method='bfill')

            # Cache the result
            self._data_cache[cache_key] = data_dict

            return data_dict

        except Exception as e:
            print(f"Error loading data: {e}")
            return {}
        finally:
            if session:
                session.close()

    def get_last_trading_day(self, close_df: pd.DataFrame, target_date: datetime, day: str = 'Friday') -> Optional[
        datetime]:
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

    def get_next_trading_day(self, open_df: pd.DataFrame, target_date: datetime, day: str = 'Monday') -> Optional[
        datetime]:
        """Return Monday's open or fallback to Tuesday if holiday"""
        target_weekday = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4}[day]

        days_ahead = target_weekday - target_date.weekday()
        if days_ahead <= 0:
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

        print(f"📊 Momentum Calculation for {current_date.strftime('%Y-%m-%d')}:")
        print(f"   Available historical data: {available_days} days")
        print(f"   Required for 52-week calc: {required_trading_days} days")

        if available_days < required_trading_days:
            # Insufficient data for proper 52-week calculation
            print(f"   ❌ Insufficient data: {available_days} < {required_trading_days} required")
            print(f"   📋 Available ETFs: {list(historical_data.columns)}")
            print(f"   🔄 Strategy will use fallback logic during accumulation")
            return pd.DataFrame()

        # Use exactly 252 trading days as per PDF specification
        window_data = historical_data.tail(252)
        print(f"   ✅ Using exactly {len(window_data)} trading days for momentum")

        if window_data.empty:
            print("   ❌ Window data is empty after filtering")
            return pd.DataFrame()

        # Calculate 52-week highs and lows as per PDF specification
        highs_52w = window_data.max()  # Maximum closing price in trailing 252 trading days
        lows_52w = window_data.min()  # Minimum closing price in trailing 252 trading days
        current_prices = historical_data.iloc[-1] if not historical_data.empty else pd.Series()

        print(f"   📈 52-week highs calculated for {len(highs_52w)} ETFs")
        print(f"   📉 52-week lows calculated for {len(lows_52w)} ETFs")
        print(f"   💰 Current prices available for {len(current_prices)} ETFs")

        # Create result DataFrame with exact PDF specification calculations
        result_data = []
        valid_etfs = 0

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

                    valid_etfs += 1
                    print(
                        f"   ✅ {symbol}: High=₹{highs_52w[symbol]:.2f}, Low=₹{lows_52w[symbol]:.2f}, Current=₹{current_prices[symbol]:.2f}")
                    print(
                        f"       Distance from Low: {distance_from_low:.2f}%, Distance from High: {distance_from_high:.2f}%")
                else:
                    print(
                        f"   ❌ {symbol}: Data integrity issue - high ({highs_52w[symbol]:.2f}) < low ({lows_52w[symbol]:.2f})")
            else:
                print(f"   ❌ {symbol}: Invalid data - contains NaN or zero values")

        result_df = pd.DataFrame(result_data)
        print(f"   🎯 Final momentum result: {len(result_df)} valid ETFs with proper 52-week data")

        if not result_df.empty:
            # Sort by distance from low for accumulation phase (buy closest to low)
            result_df_sorted = result_df.sort_values('distance_from_low')
            print(f"   📊 ETF Rankings (closest to 52-week low first):")
            for idx, row in result_df_sorted.head().iterrows():
                print(f"      {idx + 1}. {row['symbol']}: {row['distance_from_low']:.2f}% from low")

        return result_df

    def calculate_transaction_costs(self, action: str, amount: float, brokerage_percent: float) -> Dict[str, float]:
        """
        Calculate Indian market transaction costs using centralized calculator.
        
        This method now delegates to the EquitySegment implementation.
        """
        return self.calculate_delivery_costs(action, amount, brokerage_percent)

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
        Calculate capital gains tax using centralized calculator with FIFO logic.
        
        This method now delegates to the EquitySegment implementation.
        """
        return self.calculate_capital_gains(ticker, units_to_sell, sell_price, sell_date)

    def log_transaction_costs(self, week: int, date: datetime, action: str, ticker: str, units: int, price: float,
                              costs: Dict, capital_gains_tax: float = 0):
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

    def calculate_dynamic_churn_amount(self, current_nav: float, cash: float, capital_per_week: float,
                                       accumulation_weeks: int, compounding_enabled: bool) -> float:
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
        1. Capital Raising Process: Sell ETfs with smallest distance from 52-week high
        2. Reallocation Process: Buy ETF with smallest distance from 52-week low
        """
        print(f"🔄 Executing Churning Phase - Week {week_num}")
        print(f"💰 Target capital to raise: ₹{target_capital:,.0f}")

        # Initialize tracking variables
        total_raised = 0
        sell_transactions = []
        total_capital_gains_tax = 0
        updated_holdings = current_holdings.copy()

        # ===== CAPITAL RAISING PROCESS =====
        print("📊 Starting Capital Raising Process...")
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
        cash = capital_raising_result['cash']

        # ===== REALLOCATION PROCESS =====
        if total_raised > 0:
            print(f"💰 Total capital raised: ₹{total_raised:,.0f}")
            print("🎯 Starting Reallocation Process...")

            reallocation_result = self.execute_reallocation_process(
                week_num=week_num,
                execution_date=execution_date,
                high_low_data=high_low_data,
                open_prices=open_prices,
                current_holdings=updated_holdings,
                cash=cash,
                available_capital=total_raised,
                brokerage_percent=brokerage_percent
            )

            # Extract results from reallocation
            updated_holdings = reallocation_result['holdings']
            cash = reallocation_result['cash']
            buy_transaction = reallocation_result['buy_transaction']

            # Return complete churning result with comprehensive buy and sell information
            return {
                'action': 'churn',
                'week': week_num,
                'execution_date': execution_date,
                'target_capital': target_capital,

                # === SELL INFORMATION ===
                'sell_summary': {
                    'total_raised': total_raised,
                    'total_capital_gains_tax': total_capital_gains_tax,
                    'number_of_sell_transactions': len(sell_transactions),
                    'sell_transactions': sell_transactions
                },

                # === BUY INFORMATION ===
                'buy_summary': {
                    'target_etf': buy_transaction.get('ticker', 'N/A'),
                    'units_bought': buy_transaction.get('units', 0),
                    'buy_price': buy_transaction.get('price', 0),
                    'buy_amount': buy_transaction.get('amount', 0),
                    'buy_costs': buy_transaction.get('costs', {}),
                    'total_buy_cost': buy_transaction.get('total_cost', 0)
                },

                # === PORTFOLIO STATE ===
                'portfolio_state': {
                    'cash_before': cash - total_raised + buy_transaction.get('total_cost', 0),
                    'cash_after': cash,
                    'holdings_before': current_holdings,
                    'holdings_after': updated_holdings,
                    'net_cash_flow': total_raised - buy_transaction.get('total_cost', 0)
                },

                # === PERFORMANCE METRICS ===
                'churning_metrics': {
                    'capital_efficiency': (total_raised / target_capital * 100) if target_capital > 0 else 0,
                    'cost_ratio': ((total_capital_gains_tax + buy_transaction.get('total_cost',
                                                                                  0)) / total_raised * 100) if total_raised > 0 else 0,
                    'reallocation_success': total_raised > 0 and buy_transaction.get('units', 0) > 0
                },

                # === LEGACY FIELDS FOR COMPATIBILITY ===
                'ticker': buy_transaction.get('ticker', 'N/A'),
                'units': buy_transaction.get('units', 0),
                'price': buy_transaction.get('price', 0),
                'amount': buy_transaction.get('amount', 0),
                'costs': buy_transaction.get('costs', {}),
                'capital_gains_tax': total_capital_gains_tax,
                'sell_transactions': sell_transactions,
                'buy_transaction': buy_transaction,
                'cash_after': cash,
                'holdings': updated_holdings,
                'total_raised': total_raised
            }
        else:
            print("⚠️ No capital raised - skipping reallocation")
            return {
                'action': 'none',
                'ticker': 'N/A',
                'units': 0,
                'price': 0,
                'amount': 0,
                'costs': {},
                'capital_gains_tax': 0,
                'sell_transactions': [],
                'buy_transaction': {},
                'cash_after': cash,
                'holdings': updated_holdings,
                'failure_reason': 'no_capital_raised'
            }

    def execute_capital_raising_process(self, week_num: int, execution_date: datetime,
                                        high_low_data: pd.DataFrame, open_prices: pd.Series,
                                        current_holdings: Dict[str, int], cash: float,
                                        target_capital: float, brokerage_percent: float) -> Dict:
        """
        Execute Capital Raising Process as per pseudo-code:

        target_capital = weekly_capital_amount
        etfs_ranked_by_distance_from_high = sort_ascending(distance_from_52w_high)  # Closest to high first
        for each etf in etfs_ranked_by_distance_from_high:
            if total_raised >= target_capital:
                break
            sell_units = min(available_units, required_units_for_remaining_capital)
            execute_sell_with_costs_and_taxes()
        """
        print("🔴 CAPITAL RAISING PROCESS")

        total_raised = 0
        sell_transactions = []
        total_capital_gains_tax = 0
        updated_holdings = current_holdings.copy()

        # Step 1: Rank ETFs by distance from 52-week high (ascending - closest to high first)
        # PDF: etfs_ranked_by_distance_from_high = sort_ascending(distance_from_52w_high)  # Closest to high first
        sorted_for_sell = high_low_data.sort_values('distance_from_high', ascending=True).reset_index(drop=True)

        print(f"📊 Selling priority (closest to 52-week high first):")
        for idx, row in sorted_for_sell.head().iterrows():
            print(f"   {idx + 1}. {row['symbol']}: {row['distance_from_high']:.2f}% from high")

        # Step 2: Sell from ETFs closest to 52-week high until target capital is raised
        # NEW LOGIC: Continue selling from same ETF until target reached or ETF exhausted
        for _, etf_row in sorted_for_sell.iterrows():
            if total_raised >= target_capital:
                print(f"✅ Target capital reached: ₹{total_raised:,.0f} >= ₹{target_capital:,.0f}")
                break

            ticker = etf_row['symbol']

            # Check if we have holdings in this ETF
            if ticker not in updated_holdings or updated_holdings[ticker] <= 0:
                print(f"⚠️ No holdings in {ticker} - skipping to next ETF")
                continue

            # Check if we have valid price data
            if ticker not in open_prices.index or pd.isna(open_prices[ticker]):
                print(f"⚠️ No valid price for {ticker} - skipping")
                continue

            price = open_prices[ticker]
            print(f"🔄 Processing {ticker} - Available units: {updated_holdings[ticker]}")

            # CONTINUOUS SELLING LOOP - Keep selling from same ETF until target reached or ETF exhausted
            while total_raised < target_capital and updated_holdings.get(ticker, 0) > 0:
                available_units = updated_holdings[ticker]
                remaining_needed = target_capital - total_raised

                # Calculate units to sell (don't oversell)
                units_needed = int(remaining_needed / price)
                units_to_sell = min(available_units, max(1, units_needed))  # Sell at least 1 unit

                # Safety check: don't sell more than available
                if units_to_sell <= 0 or units_to_sell > available_units:
                    print(f"⚠️ Invalid units calculation for {ticker}: {units_to_sell} (available: {available_units})")
                    break

                # Execute sell transaction
                sell_result = self.execute_sell_transaction(
                    week_num=week_num,
                    execution_date=execution_date,
                    ticker=ticker,
                    units=units_to_sell,
                    price=price,
                    brokerage_percent=brokerage_percent
                )

                if sell_result['success']:
                    # Update portfolio state
                    updated_holdings[ticker] -= units_to_sell
                    cash += sell_result['net_proceeds']
                    total_raised += sell_result['net_proceeds']
                    total_capital_gains_tax += sell_result['capital_gains_tax']

                    # Record transaction
                    sell_transactions.append(sell_result)

                    print(f"🔴 Sold {units_to_sell} units of {ticker} @ ₹{price:.2f}")
                    print(f"   Net proceeds: ₹{sell_result['net_proceeds']:,.0f} (after costs & tax)")
                    print(f"   Capital gains tax: ₹{sell_result['capital_gains_tax']:,.2f}")
                    print(f"   Remaining units: {updated_holdings[ticker]}")
                    print(f"   Total raised so far: ₹{total_raised:,.0f}")

                    # Check if target reached
                    if total_raised >= target_capital:
                        print(f"✅ Target capital reached: ₹{total_raised:,.0f} >= ₹{target_capital:,.0f}")
                        break
                else:
                    print(f"❌ Failed to sell {ticker}: {sell_result.get('error', 'Unknown error')}")
                    break  # Exit the while loop if sell fails

            # Check if we need to move to next ETF
            if total_raised < target_capital and updated_holdings.get(ticker, 0) == 0:
                print(f"📦 {ticker} exhausted - moving to next priority ETF")
            elif total_raised >= target_capital:
                print(f"🎯 Target reached with {ticker} - stopping")
                break

        return {
            'total_raised': total_raised,
            'sell_transactions': sell_transactions,
            'total_capital_gains_tax': total_capital_gains_tax,
            'holdings': updated_holdings,
            'cash': cash
        }

    def execute_reallocation_process(self, week_num: int, execution_date: datetime,
                                     high_low_data: pd.DataFrame, open_prices: pd.Series,
                                     current_holdings: Dict[str, int], cash: float,
                                     available_capital: float, brokerage_percent: float) -> Dict:
        """
        Execute Reallocation Process as per pseudo-code:

        After raising capital:
        target_etf = ETF_with_smallest_distance_from_52w_low
        available_capital = total_raised_from_sells
        execute_purchase_with_costs()
        """
        print("🟢 REALLOCATION PROCESS")

        # Step 1: Select target ETF with smallest distance from 52-week low
        # PDF: target_etf = ETF_with_smallest_distance_from_52w_low
        sorted_for_buy = high_low_data.sort_values('distance_from_low', ascending=True)
        target_etf = sorted_for_buy.iloc[0]['symbol']
        distance_from_low = sorted_for_buy.iloc[0]['distance_from_low']

        print(f"🎯 Reallocation target: {target_etf} ({distance_from_low:.2f}% from 52-week low)")

        # Step 2: Check if target ETF has valid price
        if target_etf not in open_prices.index or pd.isna(open_prices[target_etf]):
            print(f"❌ Invalid price for target ETF {target_etf}")
            return {
                'holdings': current_holdings,
                'cash': cash,
                'buy_transaction': {'error': 'Invalid price for target ETF'}
            }

        price = open_prices[target_etf]

        # Step 3: Calculate purchase with available capital
        # PDF: available_capital = total_raised_from_sells
        buy_costs_estimate = self.calculate_transaction_costs('buy', available_capital, brokerage_percent)
        net_amount_for_units = available_capital - buy_costs_estimate['total_costs']
        units = int(net_amount_for_units / price) if price > 0 else 0

        if units <= 0:
            print(f"❌ Cannot buy {target_etf}: insufficient capital after costs")
            return {
                'holdings': current_holdings,
                'cash': cash,
                'buy_transaction': {'error': 'Insufficient capital after costs'}
            }

        # Step 4: Execute purchase transaction
        buy_result = self.execute_buy_transaction(
            week_num=week_num,
            execution_date=execution_date,
            ticker=target_etf,
            units=units,
            price=price,
            brokerage_percent=brokerage_percent
        )

        if buy_result['success']:
            # Update portfolio state
            updated_holdings = current_holdings.copy()
            updated_holdings[target_etf] = updated_holdings.get(target_etf, 0) + units
            cash -= buy_result['total_cost']

            print(f"🟢 Bought {units} units of {target_etf} @ ₹{price:.2f}")
            print(f"   Total cost: ₹{buy_result['total_cost']:,.0f}")

            return {
                'holdings': updated_holdings,
                'cash': cash,
                'buy_transaction': buy_result
            }
        else:
            print(f"❌ Failed to buy {target_etf}: {buy_result.get('error', 'Unknown error')}")
            return {
                'holdings': current_holdings,
                'cash': cash,
                'buy_transaction': buy_result
            }

    def execute_sell_transaction(self, week_num: int, execution_date: datetime, ticker: str,
                                 units: int, price: float, brokerage_percent: float) -> Dict:
        """
        Execute a sell transaction with all costs and taxes
        """
        try:
            # Calculate transaction details
            sell_amount = units * price
            sell_costs = self.calculate_transaction_costs('sell', sell_amount, brokerage_percent)

            # Calculate capital gains tax using FIFO logic
            capital_gains_info = self.calculate_capital_gains_tax(
                ticker, units, price, execution_date
            )
            capital_gains_tax = capital_gains_info['capital_gains_tax']

            # Calculate net proceeds
            net_proceeds = sell_costs['net_amount'] - capital_gains_tax

            # Log transaction costs
            self.log_transaction_costs(week_num, execution_date, 'sell', ticker,
                                       units, price, sell_costs, capital_gains_tax)

            return {
                'success': True,
                'ticker': ticker,
                'units': units,
                'price': price,
                'amount': sell_amount,
                'costs': sell_costs,
                'capital_gains_tax': capital_gains_tax,
                'capital_gains_info': capital_gains_info,
                'net_proceeds': net_proceeds,

                # === DETAILED SELL INFORMATION ===
                'sell_details': {
                    'gross_amount': sell_amount,
                    'brokerage': sell_costs.get('brokerage', 0),
                    'stt': sell_costs.get('stt', 0),
                    'stamp_duty': sell_costs.get('stamp_duty', 0),
                    'exchange_charges': sell_costs.get('exchange_charges', 0),
                    'sebi_charges': sell_costs.get('sebi_charges', 0),
                    'gst': sell_costs.get('gst', 0),
                    'total_transaction_costs': sell_costs.get('total_costs', 0),
                    'capital_gains_tax': capital_gains_tax,
                    'total_deductions': sell_costs.get('total_costs', 0) + capital_gains_tax,
                    'net_proceeds': net_proceeds,
                    'cost_percentage': (sell_costs.get('total_costs', 0) / sell_amount * 100) if sell_amount > 0 else 0,
                    'tax_percentage': (capital_gains_tax / sell_amount * 100) if sell_amount > 0 else 0
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

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
                    'target_etf': buy_summary.get('target_etf'),
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

    def execute_buy_transaction(self, week_num: int, execution_date: datetime, ticker: str,
                                units: int, price: float, brokerage_percent: float) -> Dict:
        """
        Execute a buy transaction with all costs
        """
        try:
            # Calculate transaction details
            buy_amount = units * price
            buy_costs = self.calculate_transaction_costs('buy', buy_amount, brokerage_percent)
            total_cost = buy_costs['net_amount']

            # Record purchase in FIFO tracking system
            self.add_purchase_record(ticker, units, price, execution_date)

            # Log transaction costs
            self.log_transaction_costs(week_num, execution_date, 'buy', ticker,
                                       units, price, buy_costs, 0)

            return {
                'success': True,
                'ticker': ticker,
                'units': units,
                'price': price,
                'amount': buy_amount,
                'costs': buy_costs,
                'total_cost': total_cost,

                # === DETAILED BUY INFORMATION ===
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
                    'cost_percentage': (buy_costs.get('total_costs', 0) / buy_amount * 100) if buy_amount > 0 else 0,
                    'effective_price_per_unit': total_cost / units if units > 0 else price
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def execute_weekly_trade(self, week_num: int, signal_date: datetime, execution_date: datetime,
                             high_low_data: pd.DataFrame, open_prices: pd.Series, close_prices: pd.Series,
                             current_holdings: Dict[str, int], cash: float, capital_per_week: float,
                             accumulation_weeks: int, brokerage_percent: float,
                             compounding_enabled: bool = False) -> Dict:
        """Execute weekly trading logic as per Technical Specification PDF"""
        # Calculate current NAV
        current_nav = cash
        for ticker, units in current_holdings.items():
            if ticker in close_prices.index and not pd.isna(close_prices[ticker]):
                current_nav += units * close_prices[ticker]

        # Calculate dynamic capital per week (with compounding if enabled)
        dynamic_capital_per_week = self.calculate_dynamic_churn_amount(
            current_nav, cash, capital_per_week, accumulation_weeks, compounding_enabled
        )

        # Initialize trade log
        trade_log = {
            'week': week_num,
            'signal_date': signal_date,
            'execution_date': execution_date,
            'action': 'none',
            'ticker': '',
            'units': 0,
            'price': 0,
            'amount': 0,
            'costs': {},
            'capital_gains_tax': 0,
            'cash_before': cash,
            'cash_after': cash,
            'holdings': current_holdings.copy(),
            'nav': current_nav,
            'base_capital_per_week': capital_per_week,
            'dynamic_capital_per_week': dynamic_capital_per_week,
            'compounding_enabled': compounding_enabled,
            'debug_info': {
                'momentum_data_available': not high_low_data.empty,
                'momentum_etfs_count': len(high_low_data),
                'available_prices_count': len(open_prices.dropna()),
                'phase': 'accumulation' if week_num <= accumulation_weeks else 'churning'
            }
        }

        is_accumulation = week_num <= accumulation_weeks
        print(f"🔄 Week {week_num} - {'ACCUMULATION' if is_accumulation else 'CHURNING'} PHASE")

        if is_accumulation:
            # ===== ACCUMULATION PHASE LOGIC =====
            print(f"📈 Accumulation Phase - Week {week_num}")
            print(f"💰 Weekly capital allocation: ₹{capital_per_week:,.0f}")

            # Selection Criteria: ETF with smallest distance from 52-week low
            target_etf = None

            if not high_low_data.empty:
                # PDF Specification: target_etf = ETF_with_smallest_distance_from_52w_low
                sorted_etfs = high_low_data.sort_values('distance_from_low')
                target_etf = sorted_etfs.iloc[0]['symbol']
                distance_from_low = sorted_etfs.iloc[0]['distance_from_low']
                print(f"📊 Momentum-based selection: {target_etf} ({distance_from_low:.2f}% from 52-week low)")
            else:
                # Fallback strategy when insufficient momentum data (early periods)
                print("⚠️ Insufficient momentum data - using round-robin fallback selection")
                available_etfs = [etf for etf in open_prices.index
                                  if not pd.isna(open_prices[etf]) and open_prices[etf] > 0]

                if available_etfs:
                    # Round-robin selection based on week number for proper rotation
                    target_etf = available_etfs[week_num % len(available_etfs)]
                    print(
                        f"🎯 Fallback selection: {target_etf} (round-robin week {week_num}, ETF {week_num % len(available_etfs) + 1}/{len(available_etfs)})")
                else:
                    target_etf = None
                    print("❌ No available ETFs for fallback selection")

            if target_etf and target_etf in open_prices.index and not pd.isna(open_prices[target_etf]):
                # PDF Specification: execution_price = monday_open_price
                price = open_prices[target_etf]
                print(f"💰 Execution price (Monday open): ₹{price:.2f}")

                # Purchase Calculation
                gross_amount = capital_per_week
                costs_estimate = self.calculate_transaction_costs('buy', gross_amount, brokerage_percent)

                # Determine maximum units purchasable with available cash
                net_amount_for_units = gross_amount - costs_estimate['total_costs']
                units = int(net_amount_for_units / price) if price > 0 else 0

                print(f"📋 Purchase calculation:")
                print(f"   Gross amount: ₹{gross_amount:,.0f}")
                print(f"   Transaction costs: ₹{costs_estimate['total_costs']:.2f}")
                print(f"   Net for units: ₹{net_amount_for_units:,.0f}")
                print(f"   Units to buy: {units}")

                if units > 0 and cash >= gross_amount:
                    # Execute purchase at Monday opening price
                    actual_amount = units * price
                    actual_costs = self.calculate_transaction_costs('buy', actual_amount, brokerage_percent)

                    # Update holdings and cash
                    current_holdings[target_etf] = current_holdings.get(target_etf, 0) + units
                    cash -= actual_costs['net_amount']

                    # Record purchase in FIFO tracking system
                    self.add_purchase_record(target_etf, units, price, execution_date)

                    # Log transaction costs
                    self.log_transaction_costs(week_num, execution_date, 'buy', target_etf,
                                               units, price, actual_costs, 0)

                    print(f"✅ Purchase executed: {units} units of {target_etf} for ₹{actual_amount:,.0f}")
                    print(f"💳 Total cost (including fees): ₹{actual_costs['net_amount']:,.0f}")
                    print(f"💰 Remaining cash: ₹{cash:,.0f}")

                    trade_log.update({
                        'action': 'buy',
                        'ticker': target_etf,
                        'units': units,
                        'price': price,
                        'amount': actual_amount,
                        'costs': actual_costs,
                        'capital_gains_tax': 0,
                        'cash_after': cash
                    })
                else:
                    print(f"❌ Cannot execute purchase: units={units}, cash=₹{cash:,.0f}, required=₹{gross_amount:,.0f}")
                    trade_log.update({
                        'action': 'none',
                        'ticker': target_etf if target_etf else 'N/A',
                        'units': 0,
                        'price': price if 'price' in locals() else 0,
                        'amount': 0,
                        'costs': {},
                        'capital_gains_tax': 0,
                        'cash_after': cash,
                        'failure_reason': 'insufficient_cash'
                    })
            else:
                print(f"❌ No valid target ETF found for purchase")
                trade_log.update({
                    'action': 'none',
                    'ticker': 'N/A',
                    'units': 0,
                    'price': 0,
                    'amount': 0,
                    'costs': {},
                    'capital_gains_tax': 0,
                    'cash_after': cash,
                    'failure_reason': 'no_valid_etf'
                })

        else:
            # ===== CHURNING PHASE LOGIC =====
            print(f"🔄 Churning Phase - Week {week_num}")

            if not high_low_data.empty:
                # Execute complete churning process
                churn_result = self.execute_churning_phase(
                    week_num=week_num,
                    execution_date=execution_date,
                    high_low_data=high_low_data,
                    open_prices=open_prices,
                    current_holdings=current_holdings,
                    cash=cash,
                    target_capital=dynamic_capital_per_week,
                    brokerage_percent=brokerage_percent
                )

                # Update portfolio state with churning results
                cash = churn_result['cash_after']
                current_holdings = churn_result['holdings']
                trade_log.update(churn_result)
            else:
                print("⚠️ No momentum data available for churning")
                trade_log.update({
                    'action': 'none',
                    'ticker': 'N/A',
                    'units': 0,
                    'price': 0,
                    'amount': 0,
                    'costs': {},
                    'capital_gains_tax': 0,
                    'cash_after': cash,
                    'failure_reason': 'no_momentum_data'
                })

        # Calculate final NAV
        nav = cash
        for ticker, units in current_holdings.items():
            if ticker in close_prices.index and not pd.isna(close_prices[ticker]):
                nav += units * close_prices[ticker]

        trade_log['nav'] = nav
        trade_log['holdings'] = current_holdings.copy()

        print(f"📊 Week {week_num} summary:")
        print(f"   Action: {trade_log['action']}")
        print(f"   NAV: ₹{nav:,.0f}")
        print(f"   Cash: ₹{cash:,.0f}")
        print(f"   Holdings: {[(k, v) for k, v in current_holdings.items() if v > 0]}")
        print("=" * 60)

        return trade_log

    def run_backtest(self, tickers: List[str], start_date: str, end_date: str,
                     capital_per_week: float, accumulation_weeks: int, brokerage_percent: float,
                     compounding_enabled: bool = False) -> Dict:
        """Run the complete backtest as per Technical Specification PDF"""
        if self._verbose:
            print("Loading data from database...")
        data_dict = self.load_data_from_database(tickers, start_date, end_date)

        if not data_dict:
            return {"error": "Failed to load data"}

        open_df = data_dict['open']
        high_df = data_dict['high']
        low_df = data_dict['low']
        close_df = data_dict['close']

        # Initialize portfolio state
        current_holdings = {ticker: 0 for ticker in tickers}
        cash = 0
        self.portfolio_log = []
        self.transaction_costs_log = []
        self.purchase_history = {}

        # Reset trade execution tracking
        self.skipped_days = []
        self.total_weeks = 0
        self.successful_signals = 0
        self.successful_executions = 0
        self.current_cash = cash
        self.current_holdings = current_holdings.copy()

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        current_date = start
        week_num = 1

        if self._verbose:
            print("Running backtest simulation...")

        # Performance tracking
        total_weeks = 0
        successful_signals = 0
        successful_executions = 0
        total_expected_weeks = int((end - start).days / 7) + 1

        if self._verbose:
            print(f"\n=== STARTING BACKTEST (PDF-ALIGNED) ===")
            print(f"Selected ETFs: {tickers}")
            print(f"Period: {start_date} to {end_date}")
            print(f"Expected weeks: ~{total_expected_weeks}")
            print(f"Capital per week: ₹{capital_per_week:,.0f}")
            print(f"Accumulation period: {accumulation_weeks} weeks")
            print(f"Compounding: {'Enabled' if compounding_enabled else 'Disabled'}")
            print(f"Data shape: {close_df.shape} (dates × ETFs)")
            print("=" * 80)

        while current_date <= end:
            total_weeks += 1

            if self._verbose:
                print(f"\n--- WEEK {week_num} ({current_date.strftime('%Y-%m-%d')}) ---")

            # ===== SIGNAL GENERATION =====
            # Signal Date: Friday of current week (or nearest trading day)
            signal_date = self.get_next_trading_day(close_df, current_date, 'Friday')

            if signal_date is not None:
                successful_signals += 1
                if self._verbose:
                    print(f"📅 Signal date: {signal_date.strftime('%Y-%m-%d')} (Friday)")

                # Execution Date: Monday of following week (or nearest trading day)
                execution_date = self.get_next_trading_day(open_df, signal_date, 'Monday')

                if execution_date is not None and execution_date <= end:
                    successful_executions += 1
                    if self._verbose:
                        print(f"💼 Execution date: {execution_date.strftime('%Y-%m-%d')} (Monday)")

                    # Add weekly capital during accumulation phase
                    if week_num <= accumulation_weeks:
                        cash += capital_per_week
                        if self._verbose:
                            print(f"💰 Added weekly capital: ₹{capital_per_week:,.0f} (Total cash: ₹{cash:,.0f})")

                    # ===== 52-WEEK MOMENTUM CALCULATION =====
                    if self._verbose:
                        print(f"📊 Computing 52-week momentum for signal date...")
                    high_low_data = self.compute_52_week_high_low(close_df, signal_date)

                    # Get execution prices (Monday opening prices)
                    open_prices = open_df.loc[execution_date]
                    close_prices = close_df.loc[execution_date]

                    if self._verbose:
                        print(f"💰 Available open prices: {len(open_prices.dropna())} ETFs")

                    # ===== TRADE EXECUTION =====
                    trade_result = self.execute_weekly_trade(
                        week_num, signal_date, execution_date, high_low_data,
                        open_prices, close_prices, current_holdings, cash,
                        capital_per_week, accumulation_weeks, brokerage_percent,
                        compounding_enabled
                    )

                    # Update portfolio state
                    cash = trade_result['cash_after']
                    current_holdings = trade_result['holdings']

                    # Update current state tracking
                    self.current_cash = cash
                    self.current_holdings = current_holdings.copy()

                    # Only log the trade if it's not a "none" action
                    if trade_result['action'] != 'none':
                        self.portfolio_log.append(trade_result)
                        if self._verbose:
                            print(f"📈 Trade completed: {trade_result['action']} {trade_result['ticker']}")
                            print(f"💰 Portfolio NAV: ₹{trade_result['nav']:,.0f}")
                    else:
                        # Track skipped trade execution
                        skip_reason = trade_result.get('skip_reason', 'No trade signal generated')
                        self.skipped_days.append({
                            'week': week_num,
                            'date': current_date.strftime('%Y-%m-%d'),
                            'signal_date': signal_date.strftime('%Y-%m-%d') if signal_date else 'N/A',
                            'reason': skip_reason
                        })
                        if self._verbose:
                            print(f"⏭️ No trade executed - {skip_reason}")

                else:
                    # Track skipped execution
                    skip_reason = "No valid execution date found or beyond end date"
                    self.skipped_days.append({
                        'week': week_num,
                        'date': current_date.strftime('%Y-%m-%d'),
                        'signal_date': signal_date.strftime('%Y-%m-%d') if signal_date else 'N/A',
                        'reason': skip_reason
                    })
                    if self._verbose:
                        print(f"❌ {skip_reason}")
            else:
                # Track skipped signal
                skip_reason = "No valid signal date found"
                self.skipped_days.append({
                    'week': week_num,
                    'date': current_date.strftime('%Y-%m-%d'),
                    'signal_date': 'N/A',
                    'reason': skip_reason
                })
                if self._verbose:
                    print(f"❌ {skip_reason}")

            # Move to next week
            current_date += timedelta(weeks=1)
            week_num += 1

        # Update final tracking variables
        self.total_weeks = total_weeks
        self.successful_signals = successful_signals
        self.successful_executions = successful_executions

        if self._verbose:
            print(f"\n=== BACKTEST COMPLETION SUMMARY ===")
            print(f"Total weeks processed: {total_weeks}")
            print(f"Successful signals: {successful_signals}")
            print(f"Successful executions: {successful_executions}")
            print(f"Portfolio log entries: {len(self.portfolio_log)}")
            print(f"Transaction cost entries: {len(self.transaction_costs_log)}")
            print(f"Skipped trades: {len(self.skipped_days)}")

        if successful_executions == 0:
            if self._verbose:
                print("⚠️ No successful trade executions found. Check date ranges and data availability.")
            return {"error": "No trade executions found"}

        # Enhanced validation checks
        if len(self.portfolio_log) < 10 and self._verbose:
            print(f"⚠️ Only {len(self.portfolio_log)} trades executed. Consider longer backtest period.")

        # Check for empty portfolio (no successful purchases)
        final_nav = self.portfolio_log[-1]['nav'] if self.portfolio_log else 0
        if final_nav == 0:
            if self._verbose:
                print("❌ Portfolio NAV is zero. Check ETF data quality and price availability.")
            return {"error": "Portfolio NAV is zero"}

        # Check for excessive transaction failures
        failed_trades = sum(1 for log in self.portfolio_log if log['action'] == 'none')
        success_rate = (len(self.portfolio_log) - failed_trades) / len(
            self.portfolio_log) * 100 if self.portfolio_log else 0
        if success_rate < 50 and self._verbose:
            print(f"⚠️ Low trade success rate: {success_rate:.1f}%. Data quality may be poor.")

        # Check for reasonable final value
        total_investment = accumulation_weeks * capital_per_week
        if final_nav < total_investment * 0.1 and self._verbose:  # Lost more than 90%
            print(
                f"⚠️ Significant portfolio decline detected. Final value: ₹{final_nav:,.0f} vs Investment: ₹{total_investment:,.0f}")

        # Check for data completeness
        if hasattr(self, 'transaction_costs_log') and self.transaction_costs_log:
            total_costs = sum(cost['total_costs'] for cost in self.transaction_costs_log)
            cost_ratio = total_costs / total_investment * 100
            if cost_ratio > 5 and self._verbose:  # More than 5% in costs
                print(f"⚠️ High transaction costs: {cost_ratio:.2f}% of total investment")

        # Create weekly NAV DataFrame
        if self.portfolio_log:
            self.weekly_nav_df = pd.DataFrame([
                {
                    'date': log['execution_date'],
                    'week': log['week'],
                    'nav': log['nav'],
                    'cash': log['cash_after'],
                    'action': log['action'],
                    'ticker': log['ticker'],
                    'capital_gains_tax': log.get('capital_gains_tax', 0),
                    'base_capital_per_week': log.get('base_capital_per_week', capital_per_week),
                    'dynamic_capital_per_week': log.get('dynamic_capital_per_week', capital_per_week),
                    'compounding_enabled': log.get('compounding_enabled', False)
                }
                for log in self.portfolio_log
            ])

            if not self.weekly_nav_df.empty:
                self.weekly_nav_df['returns'] = self.weekly_nav_df['nav'].pct_change()
                self.weekly_nav_df['cumulative_investment'] = self.weekly_nav_df.apply(
                    lambda x: min(x['week'], accumulation_weeks) * capital_per_week, axis=1
                )

        # Calculate benchmark comparison
        total_investment = accumulation_weeks * capital_per_week
        if self._verbose:
            print("Calculating benchmark buy-and-hold comparison...")
        self.nifty50_df = self.calculate_benchmark_buyhold(start_date, end_date, total_investment, brokerage_percent)

        if self._verbose:
            print(f"✅ Backtest completed successfully!")
            print(f"📊 Final portfolio value: ₹{self.portfolio_log[-1]['nav']:,.0f}")
            print(f"💰 Total investment: ₹{total_investment:,.0f}")

        return {
            "success": True,
            "weeks_traded": len(self.portfolio_log),
            "total_investment": total_investment,
            "final_nav": self.portfolio_log[-1]['nav'] if self.portfolio_log else 0
        }

    def calculate_benchmark_buyhold(self, start_date: str, end_date: str, total_investment: float,
                                    brokerage_percent: float) -> pd.DataFrame:
        """Calculate benchmark buy-and-hold performance using available market index or large cap stock"""
        session = None
        try:
            session = self._get_session()
            from sqlalchemy import text

            # Try multiple potential benchmark symbols in order of preference
            benchmark_symbols = [
                'NIFTYBEES.NS',  # Nifty ETF
                'NIFTYBEES',  # Nifty ETF (without suffix)
                'NIFTY50',  # NSE Nifty 50 Index
                'SENSEX',  # BSE Sensex
                'BANKBEES.NS',  # Banking ETF
                'BANKBEES',  # Banking ETF (without suffix)
                'JUNIORBEES.NS',  # Nifty Next 50 ETF
                'JUNIORBEES',  # Nifty Next 50 ETF (without suffix)
                'ITBEES.NS',  # IT Sector ETF
                'ITBEES'  # IT Sector ETF (without suffix)
            ]

            benchmark_symbol = None
            for symbol in benchmark_symbols:
                query = text("SELECT DISTINCT symbol FROM etf_data WHERE symbol = :symbol LIMIT 1")
                result = session.execute(query, {"symbol": symbol})
                if result.fetchone():
                    benchmark_symbol = symbol
                    print(f"✅ Using {benchmark_symbol} as market benchmark")
                    break

            if not benchmark_symbol:
                print("⚠️ No suitable benchmark found in database. Benchmark comparison will be skipped.")
                return pd.DataFrame()

            query = text("""
                SELECT date, close
                FROM etf_data
                WHERE symbol = :symbol
                AND date >= CAST(:start_date AS DATE)
                AND date <= CAST(:end_date AS DATE)
                ORDER BY date
            """)

            # Use session.execute for proper parameter binding
            result = session.execute(query, {
                "symbol": benchmark_symbol,
                "start_date": start_date,
                "end_date": end_date
            })
            rows = result.fetchall()
            columns = result.keys()
            nifty_df = pd.DataFrame(rows, columns=columns)

            if nifty_df.empty:
                return pd.DataFrame()

            nifty_df['date'] = pd.to_datetime(nifty_df['date'])
            nifty_df = nifty_df.set_index('date')

            start_price = nifty_df['close'].iloc[0]

            costs = self.calculate_transaction_costs('buy', total_investment, brokerage_percent)

            units = int(total_investment / costs['net_amount'] * total_investment / start_price)
            actual_investment = units * start_price
            actual_costs = self.calculate_transaction_costs('buy', actual_investment, brokerage_percent)

            nifty_df['nav'] = units * nifty_df['close']
            nifty_df['returns'] = nifty_df['close'].pct_change()

            nifty_df = nifty_df.reset_index()

            return nifty_df

        except Exception as e:
            print(f"Error calculating Nifty50 buy-hold: {e}")
            return pd.DataFrame()
        finally:
            if session:
                session.close()

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

            # Fix: Use excess return for Treynor ratio calculation
            excess_return = portfolio_cagr - risk_free_rate
            treynor_ratio = excess_return / beta if beta > 0 else 0.0

            return beta, treynor_ratio

        except Exception as e:
            return 1.0, 0.0

    def calculate_metrics(self, capital_per_week: float, accumulation_weeks: int, risk_free_rate: float = 8.0) -> Dict:
        """Calculate performance metrics including XIRR and Treynor ratio"""
        if self.weekly_nav_df.empty:
            return {}

        df = self.weekly_nav_df.copy()
        final_nav = df['nav'].iloc[-1]
        total_invested = accumulation_weeks * capital_per_week

        total_return = (final_nav - total_invested) / total_invested * 100

        # Fix: Use consistent time period calculation
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
            # Fix: Use consistent annualization
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
            'Win Rate': f"{(weekly_returns > 0).mean() * 100:.1f}%" if len(weekly_returns) > 0 else "N/A"
        }

    def calculate_benchmark_metrics(self, total_investment: float, risk_free_rate: float = 8.0) -> Dict:
        """Calculate benchmark buy-and-hold metrics using market index or large cap stock"""
        if self.nifty50_df.empty:
            return {}

        df = self.nifty50_df.copy()
        final_nav = df['nav'].iloc[-1]

        total_return = (final_nav - total_investment) / total_investment * 100

        # Fix: Use consistent time period calculation
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

        # Fix: Calculate proper XIRR for Nifty50 (single investment at start)
        xirr = self.calculate_nifty50_xirr(total_investment)

        daily_returns = df['returns'].dropna()
        if len(daily_returns) > 1:
            # Fix: Use consistent annualization
            volatility = daily_returns.std() * np.sqrt(252) * 100
            sharpe = (cagr - risk_free_rate) / volatility if volatility > 0 else 0
        else:
            volatility = 0
            sharpe = 0

        beta = 1.0
        # Fix: Use excess return for Treynor ratio
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

    def get_transaction_costs_summary(self) -> Dict:
        """Get summary of all transaction costs"""
        if not self.transaction_costs_log:
            return {}

        costs_df = pd.DataFrame(self.transaction_costs_log)

        total_brokerage = costs_df['brokerage'].sum()
        total_stt = costs_df['stt'].sum()
        total_stamp_duty = costs_df['stamp_duty'].sum()
        total_exchange_charges = costs_df['exchange_charges'].sum()
        total_sebi_charges = costs_df['sebi_charges'].sum()
        total_gst = costs_df['gst'].sum()
        total_capital_gains_tax = costs_df['capital_gains_tax'].sum()
        total_transaction_costs = costs_df['total_costs'].sum()
        total_all_costs = total_transaction_costs + total_capital_gains_tax

        total_buy_volume = costs_df[costs_df['action'] == 'buy']['amount'].sum()
        total_sell_volume = costs_df[costs_df['action'] == 'sell']['amount'].sum()
        total_volume = total_buy_volume + total_sell_volume

        buy_transactions = len(costs_df[costs_df['action'] == 'buy'])
        sell_transactions = len(costs_df[costs_df['action'] == 'sell'])
        total_transactions = buy_transactions + sell_transactions

        cost_percentage = (total_all_costs / total_volume * 100) if total_volume > 0 else 0

        return {
            'Total Transaction Costs': f"₹{total_transaction_costs:,.0f}",
            'Capital Gains Tax': f"₹{total_capital_gains_tax:,.0f}",
            'Total All Costs': f"₹{total_all_costs:,.0f}",
            'Cost as % of Volume': f"{cost_percentage:.3f}%",
            'Brokerage': f"₹{total_brokerage:,.0f}",
            'STT': f"₹{total_stt:,.0f}",
            'Stamp Duty': f"₹{total_stamp_duty:,.0f}",
            'Exchange Charges': f"₹{total_exchange_charges:,.0f}",
            'SEBI Charges': f"₹{total_sebi_charges:,.0f}",
            'GST': f"₹{total_gst:,.0f}",
            'Total Volume': f"₹{total_volume:,.0f}",
            'Buy Volume': f"₹{total_buy_volume:,.0f}",
            'Sell Volume': f"₹{total_sell_volume:,.0f}",
            'Total Transactions': total_transactions,
            'Buy Transactions': buy_transactions,
            'Sell Transactions': sell_transactions
        }

    def plot_equity_curve(self, show_benchmark: bool = True, show_etf_strategy: bool = True):
        """Create interactive equity curve plot with optional ETF strategy and benchmark comparison"""
        if not PLOTLY_AVAILABLE:
            print("Warning: Plotly not available. Cannot create equity curve plot.")
            return None

        if self.weekly_nav_df.empty:
            return go.Figure()

        df = self.weekly_nav_df.copy()
        fig = go.Figure()

        if show_etf_strategy:
            fig.add_trace(go.Scatter(
                x=df['date'], y=df['nav'],
                mode='lines', name='ETF Rotation Strategy',
                line=dict(color='#1f77b4', width=3),
                hovertemplate='<b>Date:</b> %{x}<br><b>ETF Strategy NAV:</b> ₹%{y:,.0f}<extra></extra>'
            ))

        if show_etf_strategy:
            fig.add_trace(go.Scatter(
                x=df['date'], y=df['cumulative_investment'],
                mode='lines', name='Cumulative Investment',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                hovertemplate='<b>Date:</b> %{x}<br><b>Invested:</b> ₹%{y:,.0f}<extra></extra>'
            ))

        if show_benchmark and not self.nifty50_df.empty:
            nifty_df = self.nifty50_df.copy()
            fig.add_trace(go.Scatter(
                x=nifty_df['date'], y=nifty_df['nav'],
                mode='lines', name='Benchmark Buy & Hold',
                line=dict(color='#d62728', width=3),
                hovertemplate='<b>Date:</b> %{x}<br><b>Benchmark NAV:</b> ₹%{y:,.0f}<extra></extra>'
            ))

        if len(df) > 0:
            accumulation_end = df[df['week'] <= 52]['date'].max() if any(df['week'] <= 52) else df['date'].min()

            fig.add_vrect(
                x0=df['date'].min(), x1=accumulation_end,
                fillcolor="green", opacity=0.1,
                annotation_text="Accumulation Phase", annotation_position="top left"
            )

            if accumulation_end < df['date'].max():
                fig.add_vrect(
                    x0=accumulation_end, x1=df['date'].max(),
                    fillcolor="blue", opacity=0.1,
                    annotation_text="Churning Phase", annotation_position="top right"
                )

        # Dynamic title based on what's shown
        if show_etf_strategy and show_benchmark:
            title = "ETF Rotation Strategy vs Benchmark Buy & Hold Performance"
        elif show_etf_strategy and not show_benchmark:
            title = "ETF Rotation Strategy Performance"
        elif not show_etf_strategy and show_benchmark:
            title = "Benchmark Buy & Hold Performance"
        else:
            title = "Performance Chart"

        fig.update_layout(
            title=title,
            xaxis_title="Date", yaxis_title="Portfolio Value (₹)",
            hovermode='x unified', template="plotly_white", height=450,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=0, r=0, t=40, b=0)
        )

        return fig

    def plot_transaction_costs_over_time(self):
        """Create transaction costs analysis chart"""
        if not PLOTLY_AVAILABLE:
            print("Warning: Plotly not available. Cannot create transaction costs plot.")
            return None

        if not self.transaction_costs_log:
            return go.Figure()

        costs_df = pd.DataFrame(self.transaction_costs_log)
        costs_df['date'] = pd.to_datetime(costs_df['date'])

        costs_df = costs_df.sort_values('date')
        costs_df['cumulative_transaction_costs'] = costs_df['total_costs'].cumsum()
        costs_df['cumulative_capital_gains_tax'] = costs_df['capital_gains_tax'].cumsum()
        costs_df['cumulative_total_costs'] = costs_df['total_impact'].cumsum()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=costs_df['date'], y=costs_df['cumulative_transaction_costs'],
            mode='lines', name='Cumulative Transaction Costs',
            line=dict(color='#ff7f0e', width=2),
            hovertemplate='<b>Date:</b> %{x}<br><b>Transaction Costs:</b> ₹%{y:,.0f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=costs_df['date'], y=costs_df['cumulative_capital_gains_tax'],
            mode='lines', name='Cumulative Capital Gains Tax',
            line=dict(color='#d62728', width=2),
            hovertemplate='<b>Date:</b> %{x}<br><b>Capital Gains Tax:</b> ₹%{y:,.0f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=costs_df['date'], y=costs_df['cumulative_total_costs'],
            mode='lines', name='Total Cumulative Costs',
            line=dict(color='#1f77b4', width=3),
            hovertemplate='<b>Date:</b> %{x}<br><b>Total Costs:</b> ₹%{y:,.0f}<extra></extra>'
        ))

        fig.update_layout(
            title="Transaction Costs Analysis Over Time",
            xaxis_title="Date", yaxis_title="Cumulative Costs (₹)",
            hovermode='x unified', template="plotly_white", height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        return fig

    def create_formatted_metrics_table(self, etf_metrics: Dict, benchmark_metrics: Dict) -> pd.DataFrame:
        """Create a beautifully formatted side-by-side metrics table"""
        if not etf_metrics or not benchmark_metrics:
            return pd.DataFrame()

        metrics_order = [
            ('💰 Investment & Returns', [
                ('Total Investment', 'Total Investment'),
                ('Final Value', 'Final Value'),
                ('Total Return', 'Total Return'),
                ('CAGR', 'CAGR'),
                ('XIRR', 'XIRR')
            ]),
            ('⚖️ Risk Metrics', [
                ('Volatility', 'Volatility'),
                ('Max Drawdown', 'Max Drawdown')
            ]),
            ('📊 Risk-Adjusted Returns', [
                ('Sharpe Ratio', 'Sharpe Ratio'),
                ('Treynor Ratio', 'Treynor Ratio'),
                ('Calmar Ratio', 'Calmar Ratio')
            ]),
            ('🎯 Success Rate', [
                ('Win Rate', 'Win Rate')
            ])
        ]

        formatted_data = {
            'Metric': [],
            '🔄 ETF Rotation Strategy': [],
            '📈 Benchmark Buy & Hold': []
        }

        for category, metrics in metrics_order:
            formatted_data['Metric'].append(f"**{category}**")
            formatted_data['🔄 ETF Rotation Strategy'].append("")
            formatted_data['📈 Benchmark Buy & Hold'].append("")

            for display_name, metric_key in metrics:
                if metric_key in etf_metrics and metric_key in benchmark_metrics:
                    etf_val = etf_metrics[metric_key]
                    benchmark_val = benchmark_metrics[metric_key]

                    formatted_data['Metric'].append(display_name)
                    formatted_data['🔄 ETF Rotation Strategy'].append(etf_val)
                    formatted_data['📈 Benchmark Buy & Hold'].append(benchmark_val)

        return pd.DataFrame(formatted_data)

    def check_strategy_exists(self, strategy_name: str, user_id: str, tickers: List[str], 
                            start_date: str, end_date: str, capital_per_week: float, 
                            accumulation_weeks: int, brokerage_percent: float, 
                            compounding_enabled: bool, risk_free_rate: float, 
                            use_custom_dates: bool) -> Dict:
        """
        Check if a strategy with the same parameters already exists in the PostgreSQL database
        
        Database: PostgreSQL (Neon) - app_data_db_connection
        Table: etf_saved_strategy
        
        Returns:
            Dict with 'exists' boolean and 'existing_strategy' details if found
        """
        session = None
        try:
            # Import app_data_db_connection for ETF strategies
            from Databases.app_data_db_connection import get_session
            session = get_session()
            from sqlalchemy import text
            
            # Check if the etf_saved_strategy table exists
            table_check = session.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'etf_saved_strategy'
            """))
            if not table_check.fetchone():
                return {"exists": False, "message": "Strategy table not found"}
            
            # Convert tickers to JSON for comparison
            tickers_json = json.dumps(sorted(tickers))
            
            # Check for exact match of all parameters
            query = text("""
                SELECT id, strategy_name, created_at, created_timestamp
                FROM etf_saved_strategy 
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
                    "message": f"ETF Strategy already exists"
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

    def diagnose_etf_data(self, selected_etfs: List[str]) -> Dict:
        """Diagnose ETF data availability and provide recommendations"""
        if not selected_etfs:
            return {"error": "No ETFs selected"}

        print(f"🔍 DIAGNOSING DATA FOR {len(selected_etfs)} ETFs")
        print("=" * 50)

        diagnosis = {
            "selected_etfs": selected_etfs,
            "data_ranges": {},
            "common_range": {},
            "recommendations": []
        }

        # Check each ETF's data range
        earliest_start = None
        latest_start = None
        earliest_end = None
        latest_end = None

        for etf in selected_etfs:
            if etf in self.etf_metadata:
                meta = self.etf_metadata[etf]
                start_date = pd.to_datetime(meta['start_date'])
                end_date = pd.to_datetime(meta['end_date'])

                diagnosis["data_ranges"][etf] = {
                    "start": meta['start_date'],
                    "end": meta['end_date'],
                    "years": meta['years_available'],
                    "records": meta['total_records']
                }

                print(
                    f"📊 {etf:12s}: {meta['start_date']} to {meta['end_date']} ({meta['years_available']:.1f} years, {meta['total_records']:,} records)")

                # Track extremes
                if earliest_start is None or start_date < earliest_start:
                    earliest_start = start_date
                if latest_start is None or start_date > latest_start:
                    latest_start = start_date
                if earliest_end is None or end_date < earliest_end:
                    earliest_end = end_date
                if latest_end is None or end_date > latest_end:
                    latest_end = end_date

        print("\n📈 DATA SUMMARY:")
        print(f"   Earliest ETF data starts: {earliest_start.strftime('%Y-%m-%d')}")
        print(f"   Latest ETF data starts:   {latest_start.strftime('%Y-%m-%d')}")
        print(f"   Earliest ETF data ends:   {earliest_end.strftime('%Y-%m-%d')}")
        print(f"   Latest ETF data ends:     {latest_end.strftime('%Y-%m-%d')}")

        # Calculate overlapping period
        overlap_start = latest_start
        overlap_end = earliest_end

        print(f"\n🎯 OVERLAPPING PERIOD:")
        print(f"   Common data range: {overlap_start.strftime('%Y-%m-%d')} to {overlap_end.strftime('%Y-%m-%d')}")

        if overlap_start >= overlap_end:
            print("   ❌ NO COMMON DATA PERIOD!")
            diagnosis["recommendations"].append("No overlapping data period - select different ETFs")
            return diagnosis

        overlap_days = (overlap_end - overlap_start).days
        overlap_years = overlap_days / 365.25
        print(f"   Overlap duration: {overlap_days} days ({overlap_years:.1f} years)")

        # Calculate strategy start date (after 90-week buffer)
        buffer_days = 630  # 90 weeks
        strategy_start = overlap_start + timedelta(days=buffer_days)

        print(f"\n🚀 STRATEGY FEASIBILITY:")
        print(f"   Required buffer: {buffer_days} days (90 weeks for enhanced 52-week momentum + safety)")
        print(f"   Strategy start: {strategy_start.strftime('%Y-%m-%d')}")

        if strategy_start >= overlap_end:
            print("   ❌ INSUFFICIENT DATA FOR STRATEGY!")
            shortage_days = (strategy_start - overlap_end).days
            print(f"   Shortage: {shortage_days} days")
            diagnosis["recommendations"].append(f"Need {shortage_days} more days of data")
            diagnosis["recommendations"].append("Consider selecting ETFs with longer history")
        else:
            backtest_days = (overlap_end - strategy_start).days
            backtest_years = backtest_days / 365.25
            print(f"   ✅ Available backtest period: {backtest_days} days ({backtest_years:.1f} years)")

            diagnosis["common_range"] = {
                "strategy_start": strategy_start.strftime('%Y-%m-%d'),
                "strategy_end": overlap_end.strftime('%Y-%m-%d'),
                "backtest_years": backtest_years
            }

            if backtest_years >= 10:
                diagnosis["recommendations"].append("Excellent: 10+ years provides statistically robust results")
            elif backtest_years >= 5:
                diagnosis["recommendations"].append("Good: 5+ years allows reasonable strategy validation")
            elif backtest_years >= 2:
                diagnosis["recommendations"].append("Fair: 2+ years provides basic validation")
            else:
                diagnosis["recommendations"].append("Limited: Less than 2 years may not be reliable")

        # ETF diversity check
        sectors = {}
        for etf in selected_etfs:
            sector = self.get_asset_sector_classification(etf)
            sectors[sector] = sectors.get(sector, 0) + 1

        print(f"\n📊 PORTFOLIO DIVERSIFICATION:")
        for sector, count in sectors.items():
            print(f"   {sector}: {count} ETF(s)")

        if len(sectors) >= 4:
            diagnosis["recommendations"].append("Excellent diversification across sectors")
        elif len(sectors) >= 3:
            diagnosis["recommendations"].append("Good diversification")
        else:
            diagnosis["recommendations"].append("Consider adding ETFs from different sectors")

        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(diagnosis["recommendations"], 1):
            print(f"   {i}. {rec}")

        return diagnosis

"""
RS Strategy EOD Data Fetcher
Fetches End-of-Day data for Nifty 500 stocks and Nifty 50 index
Uses the same logic as ETF scheduler but for RS strategy
"""

import yfinance as yf
import logging
import pandas as pd
from datetime import datetime, date, time as dt_time
import holidays
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Setup enhanced logging with rotation
from logging.handlers import RotatingFileHandler

class RSEODDataFetcher:
    """
    EOD Data Fetcher for RS Strategy - Nifty 500 stocks and Nifty 50 index
    """
    
    def __init__(self, db_path=None):
        """
        Initialize the RS EOD data fetcher
        
        Args:
            db_path: Deprecated - kept for compatibility. Now uses PostgreSQL MarketData database.
        """
        # Import market data database connection
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        databases_path = os.path.join(parent_dir, 'Databases')
        sys.path.insert(0, databases_path)
        
        from market_data_db_connection import (
            create_connection as create_market_data_connection,
            get_session as get_market_data_session,
            init_database as init_market_data_database
        )
        
        # Initialize PostgreSQL connection
        if not create_market_data_connection():
            raise RuntimeError("Failed to connect to MarketData database")
        
        # Initialize database tables
        init_market_data_database()
        
        self.get_session = get_market_data_session
        
        # Create logs directory
        logs_dir = os.path.join(parent_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging(logs_dir)
        
        # Nifty 500 stock symbols (499 stocks)
        self.stock_symbols = [
            "360ONE", "3MINDIA", "AADHARHFC", "AARTIIND", "AAVAS", "ABB", "ABBOTINDIA", "ABCAPITAL", "ABFRL", "ABREL",
            "ABSLAMC", "ACC", "ACE", "ACMESOLAR", "ADANIENSOL", "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANIPOWER", "AEGISLOG",
            "AFCONS", "AFFLE", "AIAENG", "AIIL", "AJANTPHARM", "AKUMS", "ALIVUS", "ALKEM", "ALKYLAMINE", "ALOKINDS",
            "AMBER", "AMBUJACEM", "ANANDRATHI", "ANANTRAJ", "ANGELONE", "APARINDS", "APLAPOLLO", "APLLTD", "APOLLOHOSP", "APOLLOTYRE",
            "APTUS", "ARE&M", "ASAHIINDIA", "ASHOKLEY", "ASIANPAINT", "ASTERDM", "ASTRAL", "ASTRAZEN", "ATGL", "ATUL",
            "AUBANK", "AUROPHARMA", "AWL", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV", "BAJAJHFL", "BAJAJHLDNG", "BAJFINANCE", "BALKRISIND",
            "BALRAMCHIN", "BANDHANBNK", "BANKBARODA", "BANKINDIA", "BASF", "BATAINDIA", "BAYERCROP", "BBTC", "BDL", "BEL",
            "BEML", "BERGEPAINT", "BHARATFORG", "BHARTIARTL", "BHARTIHEXA", "BHEL", "BIKAJI", "BIOCON", "BLS", "BLUEDART",
            "BLUESTARCO", "BOSCHLTD", "BPCL", "BRIGADE", "BRITANNIA", "BSE", "BSOFT", "CAMPUS", "CAMS", "CANBK",
            "CANFINHOME", "CAPLIPOINT", "CARBORUNIV", "CASTROLIND", "CCL", "CDSL", "CEATLTD", "CENTRALBK", "CENTURYPLY", "CERA",
            "CESC", "CGCL", "CGPOWER", "CHALET", "CHAMBLFERT", "CHENNPETRO", "CHOLAFIN", "CHOLAHLDNG", "CIPLA", "CLEAN",
            "COALINDIA", "COCHINSHIP", "COFORGE", "COLPAL", "CONCOR", "CONCORDBIO", "COROMANDEL", "CRAFTSMAN", "CREDITACC", "CRISIL",
            "CROMPTON", "CUB", "CUMMINSIND", "CYIENT", "DABUR", "DALBHARAT", "DATAPATTNS", "DBREALTY", "DCMSHRIRAM", "DEEPAKFERT",
            "DEEPAKNTR", "DELHIVERY", "DEVYANI", "DIVISLAB", "DIXON", "DLF", "DMART", "DOMS", "DRREDDY", "ECLERX",
            "EICHERMOT", "EIDPARRY", "EIHOTEL", "ELECON", "ELGIEQUIP", "EMAMILTD", "EMCURE", "ENDURANCE", "ENGINERSIN", "ERIS",
            "ESCORTS", "ETERNAL", "EXIDEIND", "FACT", "FEDERALBNK", "FINCABLES", "FINPIPE", "FIRSTCRY", "FIVESTAR", "FLUOROCHEM",
            "FORTIS", "FSL", "GAIL", "GESHIP", "GICRE", "GILLETTE", "GLAND", "GLAXO", "GLENMARK", "GMDCLTD",
            "GMRAIRPORT", "GNFC", "GODFRYPHLP", "GODIGIT", "GODREJAGRO", "GODREJCP", "GODREJIND", "GODREJPROP", "GPIL", "GPPL",
            "GRANULES", "GRAPHITE", "GRASIM", "GRAVITA", "GRSE", "GSPL", "GUJGASLTD", "GVT&D", "HAL", "HAPPSTMNDS",
            "HAVELLS", "HBLENGINE", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE", "HEG", "HEROMOTOCO", "HFCL", "HINDALCO",
            "HINDCOPPER", "HINDPETRO", "HINDUNILVR", "HINDZINC", "HOMEFIRST", "HONASA", "HONAUT", "HSCL", "HUDCO", "HYUNDAI",
            "ICICIBANK", "ICICIGI", "ICICIPRULI", "IDBI", "IDEA", "IDFCFIRSTB", "IEX", "IFCI", "IGIL", "IGL",
            "IIFL", "IKS", "INDGN", "INDHOTEL", "INDIACEM", "INDIAMART", "INDIANB", "INDIGO", "INDUSINDBK", "INDUSTOWER",
            "INFY", "INOXINDIA", "INOXWIND", "INTELLECT", "IOB", "IOC", "IPCALAB", "IRB", "IRCON", "IRCTC",
            "IREDA", "IRFC", "ITC", "ITI", "J&KBANK", "JBCHEPHARM", "JBMA", "JINDALSAW", "JINDALSTEL", "JIOFIN",
            "JKCEMENT", "JKTYRE", "JMFINANCIL", "JPPOWER", "JSL", "JSWENERGY", "JSWHL", "JSWINFRA", "JSWSTEEL", "JUBLFOOD",
            "JUBLINGREA", "JUBLPHARMA", "JUSTDIAL", "JWL", "JYOTHYLAB", "JYOTICNC", "KAJARIACER", "KALYANKJIL", "KANSAINER", "KARURVYSYA",
            "KAYNES", "KEC", "KEI", "KFINTECH", "KIMS", "KIRLOSBROS", "KIRLOSENG", "KNRCON", "KOTAKBANK", "KPIL",
            "KPITTECH", "KPRMILL", "LALPATHLAB", "LATENTVIEW", "LAURUSLABS", "LEMONTREE", "LICHSGFIN", "LICI", "LINDEINDIA", "LLOYDSME",
            "LODHA", "LT", "LTF", "LTFOODS", "LTIM", "LTTS", "LUPIN", "M&M", "M&MFIN", "MAHABANK",
            "MAHSEAMLES", "MANAPPURAM", "MANKIND", "MANYAVAR", "MAPMYINDIA", "MARICO", "MARUTI", "MASTEK", "MAXHEALTH", "MAZDOCK",
            "MCX", "MEDANTA", "METROPOLIS", "MFSL", "MGL", "MINDACORP", "MMTC", "MOTHERSON", "MOTILALOFS", "MPHASIS",
            "MRF", "MRPL", "MSUMI", "MUTHOOTFIN", "NAM-INDIA", "NATCOPHARM", "NATIONALUM", "NAUKRI", "NAVA", "NAVINFLUOR",
            "NBCC", "NCC", "NESTLEIND", "NETWEB", "NETWORK18", "NEULANDLAB", "NEWGEN", "NH", "NHPC", "NIACL",
            "NIVABUPA", "NLCINDIA", "NMDC", "NSLNISP", "NTPC", "NTPCGREEN", "NUVAMA", "NYKAA", "OBEROIRLTY", "OFSS",
            "OIL", "OLAELEC", "OLECTRA", "ONGC", "PAGEIND", "PATANJALI", "PAYTM", "PCBL", "PEL", "PERSISTENT",
            "PETRONET", "PFC", "PFIZER", "PGEL", "PHOENIXLTD", "PIDILITIND", "PIIND", "PNB", "PNBHOUSING", "PNCINFRA",
            "POLICYBZR", "POLYCAB", "POLYMED", "POONAWALLA", "POWERGRID", "POWERINDIA", "PPLPHARMA", "PRAJIND", "PREMIERENE", "PRESTIGE",
            "PTCIL", "PVRINOX", "RADICO", "RAILTEL", "RAINBOW", "RAMCOCEM", "RAYMOND", "RAYMONDLSL", "RBLBANK", "RCF",
            "RECLTD", "REDINGTON", "RELIANCE", "RENUKA", "RHIM", "RITES", "RKFORGE", "ROUTE", "RPOWER", "RRKABEL",
            "RTNINDIA", "RVNL", "SAGILITY", "SAIL", "SAILIFE", "SAMMAANCAP", "SAPPHIRE", "SARDAEN", "SAREGAMA", "SBFC",
            "SBICARD", "SBILIFE", "SBIN", "SCHAEFFLER", "SCHNEIDER", "SCI", "SHREECEM", "SHRIRAMFIN", "SHYAMMETL", "SIEMENS",
            "SIGNATURE", "SJVN", "SKFINDIA", "SOBHA", "SOLARINDS", "SONACOMS", "SONATSOFTW", "SRF", "STARHEALTH", "SUMICHEM",
            "SUNDARMFIN", "SUNDRMFAST", "SUNPHARMA", "SUNTV", "SUPREMEIND", "SUZLON", "SWANCORP", "SWIGGY", "SWSOLAR", "SYNGENE",
            "SYRMA", "TANLA", "TARIL", "TATACHEM", "TATACOMM", "TATACONSUM", "TATAELXSI", "TATAINVEST", "TATAMOTORS", "TATAPOWER",
            "TATASTEEL", "TATATECH", "TBOTEK", "TCS", "TECHM", "TECHNOE", "TEJASNET", "THERMAX", "TIINDIA", "TIMKEN",
            "TITAGARH", "TITAN", "TORNTPHARM", "TORNTPOWER", "TRENT", "TRIDENT", "TRITURBINE", "TRIVENI", "TTML", "TVSMOTOR",
            "UBL", "UCOBANK", "ULTRACEMCO", "UNIONBANK", "UNITDSPR", "UNOMINDA", "UPL", "USHAMART", "UTIAMC", "VBL",
            "VEDL", "VGUARD", "VIJAYA", "VMM", "VOLTAS", "VTL", "WAAREEENER", "WELCORP", "WELSPUNLIV", "WESTLIFE",
            "WHIRLPOOL", "WIPRO", "WOCKPHARMA", "YESBANK", "ZEEL", "ZENSARTECH", "ZENTEC", "ZFCVINDIA", "ZYDUSLIF"
        ]
        
        # Nifty 50 index symbol
        self.index_symbol = "NIFTY_50"
        
        # Indian holidays for market closure detection
        self.indian_holidays = holidays.India()
        
        # Setup database tables
        self.setup_database()
        
        self.logger.info(f"RS EOD Data Fetcher initialized with {len(self.stock_symbols)} stocks")
        self.logger.info("Using PostgreSQL MarketData database")
    
    def setup_logging(self, logs_dir):
        """Setup logging for the EOD data fetcher"""
        log_file = os.path.join(logs_dir, 'rs_eod_fetcher.log')
        
        self.logger = logging.getLogger('rs_eod_fetcher')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler with rotation
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def setup_database(self):
        """Setup database tables for stock and index data - tables are created via market_data_db_connection"""
        # Tables are automatically created by init_database() in market_data_db_connection
        # This method is kept for compatibility but does nothing
        self.logger.info("Database tables are managed by market_data_db_connection")
    
    def is_market_open(self, check_date=None):
        """
        Check if the market is open on the given date
        
        Args:
            check_date (date): Date to check (default: today)
            
        Returns:
            bool: True if market is open, False otherwise
        """
        if check_date is None:
            check_date = date.today()
        
        # Check if it's a weekend
        if check_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if it's a holiday
        if check_date in self.indian_holidays:
            return False
        
        return True
    
    def fetch_single_stock_data(self, symbol, start_date, end_date):
        """
        Fetch data for a single stock
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            dict: Result with success status and data
        """
        try:
            # Add .NS suffix for NSE stocks
            yf_symbol = f"{symbol}.NS"
            
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                return {"success": False, "error": "No data available", "symbol": symbol}
            
            # Prepare data for database
            records = []
            for date_idx, row in data.iterrows():
                records.append({
                    'symbol': symbol,
                    'date': date_idx.date(),
                    'open': float(row['Open']) if pd.notna(row['Open']) else None,
                    'high': float(row['High']) if pd.notna(row['High']) else None,
                    'low': float(row['Low']) if pd.notna(row['Low']) else None,
                    'close': float(row['Close']) if pd.notna(row['Close']) else None,
                    'volume': int(row['Volume']) if pd.notna(row['Volume']) else None,
                    'adj_close': float(row['Close']) if pd.notna(row['Close']) else None
                })
            
            return {"success": True, "data": records, "symbol": symbol}
            
        except Exception as e:
            return {"success": False, "error": str(e), "symbol": symbol}
    
    def fetch_index_data(self, start_date, end_date):
        """
        Fetch Nifty 50 index data
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            dict: Result with success status and data
        """
        try:
            # Fetch Nifty 50 data
            ticker = yf.Ticker("^NSEI")
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                return {"success": False, "error": "No data available", "symbol": self.index_symbol}
            
            # Prepare data for database
            records = []
            for date_idx, row in data.iterrows():
                records.append({
                    'symbol': self.index_symbol,
                    'date': date_idx.date(),
                    'open': float(row['Open']) if pd.notna(row['Open']) else None,
                    'high': float(row['High']) if pd.notna(row['High']) else None,
                    'low': float(row['Low']) if pd.notna(row['Low']) else None,
                    'close': float(row['Close']) if pd.notna(row['Close']) else None,
                    'volume': int(row['Volume']) if pd.notna(row['Volume']) else None,
                    'adj_close': float(row['Close']) if pd.notna(row['Close']) else None
                })
            
            return {"success": True, "data": records, "symbol": self.index_symbol}
            
        except Exception as e:
            return {"success": False, "error": str(e), "symbol": self.index_symbol}
    
    def save_stock_data(self, records):
        """
        Save stock data to PostgreSQL database
        
        Args:
            records (list): List of stock data records
        """
        session = None
        try:
            from market_data_db_connection import StockData
            from sqlalchemy.dialects.postgresql import insert
            from datetime import datetime
            
            session = self.get_session()
            
            for record in records:
                # Convert date string to datetime if needed
                if isinstance(record['date'], str):
                    record_date = datetime.strptime(record['date'], '%Y-%m-%d')
                else:
                    record_date = record['date']
                
                # Use PostgreSQL UPSERT
                stmt = insert(StockData).values(
                    symbol=record['symbol'],
                    date=record_date,
                    open=record['open'],
                    high=record['high'],
                    low=record['low'],
                    close=record['close'],
                    volume=record['volume'],
                    adj_close=record['adj_close']
                )
                
                stmt = stmt.on_conflict_do_update(
                    index_elements=['symbol', 'date'],
                    set_=dict(
                        open=stmt.excluded.open,
                        high=stmt.excluded.high,
                        low=stmt.excluded.low,
                        close=stmt.excluded.close,
                        volume=stmt.excluded.volume,
                        adj_close=stmt.excluded.adj_close
                    )
                )
                
                session.execute(stmt)
            
            session.commit()
                
        except Exception as e:
            if session:
                session.rollback()
            self.logger.error(f"Error saving stock data: {e}")
            raise
        finally:
            if session:
                session.close()
    
    def save_index_data(self, records):
        """
        Save index data to PostgreSQL database
        
        Args:
            records (list): List of index data records
        """
        session = None
        try:
            from market_data_db_connection import IndexData
            from sqlalchemy.dialects.postgresql import insert
            from datetime import datetime
            
            session = self.get_session()
            
            for record in records:
                # Convert date string to datetime if needed
                if isinstance(record['date'], str):
                    record_date = datetime.strptime(record['date'], '%Y-%m-%d')
                else:
                    record_date = record['date']
                
                # Use PostgreSQL UPSERT
                stmt = insert(IndexData).values(
                    symbol=record['symbol'],
                    date=record_date,
                    open=record['open'],
                    high=record['high'],
                    low=record['low'],
                    close=record['close'],
                    volume=record['volume'],
                    adj_close=record['adj_close']
                )
                
                stmt = stmt.on_conflict_do_update(
                    index_elements=['symbol', 'date'],
                    set_=dict(
                        open=stmt.excluded.open,
                        high=stmt.excluded.high,
                        low=stmt.excluded.low,
                        close=stmt.excluded.close,
                        volume=stmt.excluded.volume,
                        adj_close=stmt.excluded.adj_close
                    )
                )
                
                session.execute(stmt)
            
            session.commit()
                
        except Exception as e:
            if session:
                session.rollback()
            self.logger.error(f"Error saving index data: {e}")
            raise
        finally:
            if session:
                session.close()
    
    def fetch_daily_data(self, days_back=1):
        """
        Fetch daily EOD data for all stocks and index
        
        Args:
            days_back (int): Number of days back to fetch data
            
        Returns:
            dict: Summary of fetch results
        """
        try:
            # Calculate date range
            end_date = date.today()
            start_date = end_date - pd.Timedelta(days=days_back)
            
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            self.logger.info(f"Fetching EOD data from {start_date_str} to {end_date_str}")
            
            # Check if market was open
            if not self.is_market_open(end_date):
                self.logger.info(f"Market was closed on {end_date}, skipping data fetch")
                return {
                    "success": True,
                    "message": "Market was closed",
                    "successful_fetches": 0,
                    "failed_fetches": 0
                }
            
            successful_fetches = 0
            failed_fetches = 0
            
            # Fetch stock data in parallel
            self.logger.info(f"Fetching data for {len(self.stock_symbols)} stocks...")
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all stock fetch tasks
                future_to_symbol = {
                    executor.submit(self.fetch_single_stock_data, symbol, start_date_str, end_date_str): symbol
                    for symbol in self.stock_symbols
                }
                
                # Process completed tasks
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        if result["success"]:
                            self.save_stock_data(result["data"])
                            successful_fetches += 1
                            self.logger.debug(f"Successfully fetched data for {symbol}")
                        else:
                            failed_fetches += 1
                            self.logger.warning(f"Failed to fetch data for {symbol}: {result['error']}")
                    except Exception as e:
                        failed_fetches += 1
                        self.logger.error(f"Exception while fetching data for {symbol}: {e}")
            
            # Fetch index data
            self.logger.info("Fetching Nifty 50 index data...")
            index_result = self.fetch_index_data(start_date_str, end_date_str)
            
            if index_result["success"]:
                self.save_index_data(index_result["data"])
                self.logger.info("Successfully fetched Nifty 50 index data")
            else:
                self.logger.error(f"Failed to fetch Nifty 50 index data: {index_result['error']}")
            
            # Summary
            total_fetches = successful_fetches + failed_fetches
            self.logger.info(f"EOD data fetch completed: {successful_fetches}/{total_fetches} stocks successful")
            
            return {
                "success": True,
                "successful_fetches": successful_fetches,
                "failed_fetches": failed_fetches,
                "total_fetches": total_fetches,
                "date_range": f"{start_date_str} to {end_date_str}"
            }
            
        except Exception as e:
            self.logger.error(f"Error in fetch_daily_data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "successful_fetches": 0,
                "failed_fetches": 0
            }
    
    def fetch_historical_data(self, days_back=30):
        """
        Fetch historical EOD data for all stocks and index
        
        Args:
            days_back (int): Number of days back to fetch data
            
        Returns:
            dict: Summary of fetch results
        """
        try:
            # Calculate date range
            end_date = date.today()
            start_date = end_date - pd.Timedelta(days=days_back)
            
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            self.logger.info(f"Fetching historical data from {start_date_str} to {end_date_str}")
            
            successful_fetches = 0
            failed_fetches = 0
            
            # Fetch stock data in parallel
            self.logger.info(f"Fetching historical data for {len(self.stock_symbols)} stocks...")
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all stock fetch tasks
                future_to_symbol = {
                    executor.submit(self.fetch_single_stock_data, symbol, start_date_str, end_date_str): symbol
                    for symbol in self.stock_symbols
                }
                
                # Process completed tasks
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        if result["success"]:
                            self.save_stock_data(result["data"])
                            successful_fetches += 1
                            self.logger.debug(f"Successfully fetched historical data for {symbol}")
                        else:
                            failed_fetches += 1
                            self.logger.warning(f"Failed to fetch historical data for {symbol}: {result['error']}")
                    except Exception as e:
                        failed_fetches += 1
                        self.logger.error(f"Exception while fetching historical data for {symbol}: {e}")
            
            # Fetch index data
            self.logger.info("Fetching historical Nifty 50 index data...")
            index_result = self.fetch_index_data(start_date_str, end_date_str)
            
            if index_result["success"]:
                self.save_index_data(index_result["data"])
                self.logger.info("Successfully fetched historical Nifty 50 index data")
            else:
                self.logger.error(f"Failed to fetch historical Nifty 50 index data: {index_result['error']}")
            
            # Summary
            total_fetches = successful_fetches + failed_fetches
            self.logger.info(f"Historical data fetch completed: {successful_fetches}/{total_fetches} stocks successful")
            
            return {
                "success": True,
                "successful_fetches": successful_fetches,
                "failed_fetches": failed_fetches,
                "total_fetches": total_fetches,
                "date_range": f"{start_date_str} to {end_date_str}"
            }
            
        except Exception as e:
            self.logger.error(f"Error in fetch_historical_data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "successful_fetches": 0,
                "failed_fetches": 0
            }
    
    def cleanup_old_data(self, days_to_keep=365):
        """
        Cleanup old data from PostgreSQL database
        
        Args:
            days_to_keep (int): Number of days of data to keep
        """
        session = None
        try:
            from sqlalchemy import text
            from datetime import datetime
            
            cutoff_date = date.today() - pd.Timedelta(days=days_to_keep)
            cutoff_date_str = cutoff_date.strftime('%Y-%m-%d')
            
            session = self.get_session()
            
            # Delete old stock data
            query = text("DELETE FROM stock_data WHERE date < :cutoff_date")
            result = session.execute(query, {"cutoff_date": cutoff_date_str})
            stock_deleted = result.rowcount
            
            # Delete old index data
            query = text("DELETE FROM index_data WHERE date < :cutoff_date")
            result = session.execute(query, {"cutoff_date": cutoff_date_str})
            index_deleted = result.rowcount
            
            session.commit()
            
            self.logger.info(f"Cleaned up {stock_deleted} stock records and {index_deleted} index records older than {cutoff_date_str}")
            
        except Exception as e:
            if session:
                session.rollback()
            self.logger.error(f"Error cleaning up old data: {e}")
            raise
        finally:
            if session:
                session.close()

def main():
    """Main function to test the EOD data fetcher"""
    try:
        # Initialize fetcher
        fetcher = RSEODDataFetcher()
        
        # Fetch daily data
        result = fetcher.fetch_daily_data()
        print(f"Daily fetch result: {result}")
        
        # Fetch historical data (last 7 days)
        result = fetcher.fetch_historical_data(days_back=7)
        print(f"Historical fetch result: {result}")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

from datetime import datetime, time, timedelta
from typing import List, Optional
import pandas as pd
from CoreLogic.WealthAIBase import WealthAIBase

class IndianExchange(WealthAIBase):
    """
    Level 2: Exchange Layer
    
    Logic specific to Indian exchanges (NSE/BSE).
    Enforces Indian market rules:
    - Trading hours (09:15 - 15:30 IST)
    - Holidays & Weekends
    - INR Currency Formatting
    - NSE Symbol Validation
    """
    
    # Constants
    MARKET_START_TIME = time(9, 15)
    MARKET_END_TIME = time(15, 30)
    RISK_FREE_RATE = 0.07  # 7.0% (10Y Bond Yield approx)
    
    def get_trading_calendar(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """
        Returns valid NSE/BSE trading days (excludes weekends).
        TODO: Integrate with a real holiday list.
        """
        # Generate all days
        all_days = pd.date_range(start=start_date, end=end_date, freq='B') # 'B' is business days (Mon-Fri)
        # Convert to python datetime objects
        return [d.to_pydatetime() for d in all_days]

    def is_market_open(self, timestamp: datetime) -> bool:
        """
        Checks if a timestamp is within 09:15 - 15:30 IST on a weekday.
        """
        # Check weekend
        if timestamp.weekday() > 4: # 5=Sat, 6=Sun
            return False
            
        # Check time
        t = timestamp.time()
        return self.MARKET_START_TIME <= t <= self.MARKET_END_TIME

    def format_currency_inr(self, amount: float) -> str:
        """
        Formats numbers to Indian Lakhs/Crores (e.g., ₹1,50,000).
        """
        s, *d = str(amount).partition(".")
        r = ",".join([s[x-2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
        return "₹" + "".join([r] + d)

    def get_risk_free_rate(self) -> float:
        """
        Returns the current Indian 10Y Bond Yield (default 7.0%).
        """
        return self.RISK_FREE_RATE

    def validate_symbol_nse(self, symbol: str) -> bool:
        """
        Checks if a symbol follows NSE naming conventions.
        """
        # Basic check: Uppercase and no special chars except typical ones
        return symbol.isupper() and symbol.replace('-', '').replace('&', '').isalnum()

    def get_last_trading_day(self, close_df: pd.DataFrame, target_date: datetime) -> Optional[datetime]:
        """
        Return the nearest available trading day looking backwards.
        Useful for finding valid close prices.
        """
        for offset in range(7):
            check_date = target_date - timedelta(days=offset)
            if check_date in close_df.index:
                return check_date
        return None

    def get_next_trading_day(self, open_df: pd.DataFrame, target_date: datetime) -> Optional[datetime]:
        """
        Return the nearest available trading day looking forwards.
        Useful for finding valid open prices.
        """
        for offset in range(7):
            check_date = target_date + timedelta(days=offset)
            if check_date in open_df.index:
                return check_date
        return None

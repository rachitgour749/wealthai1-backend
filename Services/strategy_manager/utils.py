"""Utility functions for centralized strategy management system"""
from datetime import datetime, date, timedelta
import random
import string
import holidays
import pytz
from typing import Optional

def generate_run_id(strategy_type: str) -> str:
    """
    Generate a unique run_id for a strategy
    Format: {suffix}_{DDMMYY}_{uniquecode}
    """
    suffix_map = {
        "ETF_Rotation":             "ETF",
        "RS_ETF_Rotation":          "RS_ETF",
        "ETF_Payout":               "ETF_Payout",
        "International_ETF_Rotation": "ETF_INTL",
        "US_ETF_Swing_Strategy":    "US_ETF_SW",
        "ETF_Swing_Strategy":       "ETF_SW",
        "External_Strategy":        "EXT",
    }
    suffix   = suffix_map.get(strategy_type, "STRAT")
    date_str = datetime.now().strftime("%d%m%y")
    unique_code = ''.join(random.choices(string.digits, k=2))
    return f"{suffix}_{date_str}_{unique_code}"

# Markets that trade on the US calendar (NYSE/NASDAQ)
_US_CALENDAR_STRATEGIES = {
    'US_ETF_Swing_Strategy',
    'International_ETF_Rotation',
}

# Map strategy_type → market string for convenience
_STRATEGY_MARKET_MAP = {
    'US_ETF_Swing_Strategy':       'US',
    'International_ETF_Rotation':  'US',  # International ETFs trade on US exchanges
    'ETF_Rotation':                'INDIA',
    'RS_ETF_Rotation':             'INDIA',
    'ETF_Swing_Strategy':          'INDIA',
    'ETF_Payout':                  'INDIA',
    'Stock_Rotation':              'INDIA',
    'RS_Stocks':                   'INDIA',
}

def get_market_for_strategy(strategy_type: str) -> str:
    """Return 'US' or 'INDIA' for a given strategy type."""
    return _STRATEGY_MARKET_MAP.get(strategy_type, 'INDIA')


def get_next_trading_day(strategy_type: Optional[str] = None, market: Optional[str] = None) -> datetime:
    """
    Get the next valid trading day for the given market.

    Parameters
    ----------
    strategy_type : str, optional
        Used to auto-detect the market when `market` is not supplied.
    market : str, optional
        'US' or 'INDIA'. Overrides strategy_type-based detection.

    Returns
    -------
    datetime
        Timezone-aware datetime at the market's standard open time.
        INDIA → 9:00 AM IST (Asia/Kolkata)
        US    → 9:30 AM ET  (America/New_York)
    """
    # --- Resolve market ---
    if market is None:
        if strategy_type:
            market = get_market_for_strategy(strategy_type)
        else:
            market = 'INDIA'
    market = market.upper()

    if market == 'US':
        # ── US / International ETF calendar ──────────────────────────
        try:
            us_holidays = holidays.NYSE()  # NYSE holiday calendar
        except Exception:
            us_holidays = holidays.UnitedStates()

        tz      = pytz.timezone('America/New_York')
        now     = datetime.now(tz)
        # Start checking from tomorrow (signals generated after close → execute next open)
        start   = now + timedelta(days=1)
        current = start.replace(hour=0, minute=0, second=0, microsecond=0)

        for _ in range(14):  # safety: look at most 2 weeks ahead
            if current.weekday() < 5 and current.date() not in us_holidays:
                return current.replace(hour=9, minute=30, second=0, microsecond=0)
            current += timedelta(days=1)

        # Fallback
        return start.replace(hour=9, minute=30, second=0, microsecond=0)

    else:
        # ── India / NSE calendar ──────────────────────────────────────
        ind_holidays = holidays.India()
        tz           = pytz.timezone('Asia/Kolkata')
        now          = datetime.now(tz)

        # Find first trading day of the current/upcoming week (original logic)
        days_to_monday = now.weekday()
        monday = now - timedelta(days=days_to_monday)
        target_day = monday

        if now.weekday() >= 5:  # Sat or Sun → move to upcoming Monday
            target_day = now + timedelta(days=(7 - now.weekday()) % 7)

        current = target_day
        for _ in range(14):
            if current.weekday() < 5 and current.date() not in ind_holidays:
                return current.replace(hour=9, minute=0, second=0, microsecond=0)
            current += timedelta(days=1)

        # Fallback
        return target_day.replace(hour=9, minute=0, second=0, microsecond=0)

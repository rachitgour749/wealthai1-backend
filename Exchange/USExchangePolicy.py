from datetime import datetime, time
from typing import List, Dict, Any, Optional
import pandas as pd
from .ExchangePolicy import ExchangePolicy

class USExchangePolicy(ExchangePolicy):
    """
    Policy for US market (NYSE/NASDAQ).
    """
    
    @property
    def market_name(self) -> str:
        return "US"

    @property
    def currency_symbol(self) -> str:
        return "$"

    @property
    def market_start_time(self) -> time:
        return time(9, 30)

    @property
    def market_end_time(self) -> time:
        return time(16, 0)

    def is_trading_day(self, date_obj: datetime.date) -> bool:
        return date_obj.weekday() < 5

    def calculate_transaction_costs(self, action: str, asset_type: str, amount: float, brokerage_percent: float) -> Dict[str, float]:
        """
        Calculates transaction costs for US market.
        Only brokerage is applied (on both BUY and SELL) at the rate
        supplied via brokerage_percent from the payload.
        No SEC fee, TAF, GST, or any other charge.
        """
        brokerage = amount * (brokerage_percent / 100)

        costs = {
            'brokerage':    brokerage,
            'sec_fee':      0.0,
            'taf_fee':      0.0,
            'gst':          0.0,
            'total_costs':  brokerage,
        }
        # BUY  → cash out  = amount + brokerage
        # SELL → cash in   = amount - brokerage
        action = action.lower()
        costs['net_amount'] = (amount + brokerage) if action == 'buy' else (amount - brokerage)
        return costs

    def format_currency(self, amount: float) -> str:
        return f"{self.currency_symbol}{amount:,.2f}"

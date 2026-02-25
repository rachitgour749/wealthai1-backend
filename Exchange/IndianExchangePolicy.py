from datetime import datetime, time
from typing import List, Dict, Any, Optional
import pandas as pd
from .ExchangePolicy import ExchangePolicy

class IndianExchangePolicy(ExchangePolicy):
    """
    Policy for Indian market (NSE/BSE).
    """
    
    @property
    def market_name(self) -> str:
        return "INDIA"

    @property
    def currency_symbol(self) -> str:
        return "₹"

    @property
    def market_start_time(self) -> time:
        return time(9, 15)

    @property
    def market_end_time(self) -> time:
        return time(15, 30)

    def is_trading_day(self, date_obj: datetime.date) -> bool:
        """Checks if it's a weekday. Real holiday list could be added here."""
        return date_obj.weekday() < 5

    def calculate_transaction_costs(self, action: str, asset_type: str, amount: float, brokerage_percent: float) -> Dict[str, float]:
        """
        Calculates transaction costs for Indian market.
        Logic migrated from EquitySegment.py
        """
        costs = {}
        action = action.lower()
        asset_type = asset_type.upper()
        
        # Brokerage
        brokerage = amount * (brokerage_percent / 100)
        costs['brokerage'] = brokerage
        
        # STT (Securities Transaction Tax)
        if asset_type == 'STOCK':
            costs['stt'] = amount * 0.1 / 100 # 0.1% on BOTH Buy & Sell
        elif asset_type == 'ETF':
            costs['stt'] = (amount * 0.001 / 100) if action == 'sell' else 0 # 0.001% on Sell only
        else:
            costs['stt'] = 0
            
        # Stamp duty - 0.015% (Buy Only)
        costs['stamp_duty'] = (amount * 0.015 / 100) if action == 'buy' else 0
        
        # Exchange charges - 0.00297%
        exchange_charges = amount * 0.00297 / 100
        costs['exchange_charges'] = exchange_charges
        
        # SEBI charges - 0.0001%
        sebi_charges = amount * 0.0001 / 100
        costs['sebi_charges'] = sebi_charges
        
        # GST - 18% on Brokerage + Exchange + SEBI
        costs['gst'] = (brokerage + exchange_charges + sebi_charges) * 0.18
        
        # Totals
        total_costs = sum(costs.values())
        costs['total_costs'] = total_costs
        costs['net_amount'] = (amount + total_costs) if action == 'buy' else (amount - total_costs)
            
        return costs

    def format_currency(self, amount: float) -> str:
        """Indian numbering format (Lakhs/Crores)"""
        s, *d = f"{amount:.2f}".partition(".")
        r = ",".join([s[x-2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
        return self.currency_symbol + "".join([r] + d)

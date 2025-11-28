"""
Indian Market Cost Calculator

Handles transaction cost calculations for Indian stock/ETF markets.
All rates are hardcoded for Indian market regulations.
"""

from typing import Dict


class IndianMarketCostCalculator:
    """
    Calculate transaction costs for Indian markets (NSE/BSE).
    
    This calculator implements the complete Indian market cost structure:
    - Brokerage (variable)
    - STT (Securities Transaction Tax)
    - Exchange charges
    - GST (Goods and Services Tax)
    - SEBI charges
    - Stamp duty
    
    All rates are as per Indian regulations and are hardcoded.
    """
    
    # Indian Market Rates (Hardcoded)
    STT_RATE_SELL = 0.001 / 100  # 0.001% on sell
    STAMP_DUTY_BUY = 0.005 / 100  # 0.005% on buy
    EXCHANGE_CHARGES_RATE = 0.00297 / 100  # 0.00297%
    SEBI_CHARGES_RATE = 0.0001 / 100  # 0.0001%
    GST_RATE = 0.18  # 18% GST
    
    def calculate(self, action: str, amount: float, brokerage_percent: float) -> Dict[str, float]:
        """
        Calculate complete transaction costs for Indian markets.
        
        Args:
            action: 'buy' or 'sell'
            amount: Transaction amount in INR
            brokerage_percent: Brokerage percentage (e.g., 0.05 for 0.05%)
            
        Returns:
            Dictionary containing:
            - brokerage: Brokerage amount
            - stt: Securities Transaction Tax
            - stamp_duty: Stamp duty (buy only)
            - exchange_charges: Exchange charges with GST
            - sebi_charges: SEBI charges with GST
            - gst: GST on brokerage
            - total_costs: Sum of all costs
            - net_amount: Amount including costs
        """
        costs = {}
        
        # Brokerage
        brokerage = amount * (brokerage_percent / 100)
        costs['brokerage'] = brokerage
        
        # STT (Securities Transaction Tax) - only on sell
        if action == 'sell':
            costs['stt'] = amount * self.STT_RATE_SELL
        else:
            costs['stt'] = 0
        
        # Stamp duty - only on buy
        if action == 'buy':
            costs['stamp_duty'] = amount * self.STAMP_DUTY_BUY
        else:
            costs['stamp_duty'] = 0
        
        # Exchange charges (with GST)
        exchange_charges = amount * self.EXCHANGE_CHARGES_RATE
        costs['exchange_charges'] = exchange_charges * self.GST_RATE
        
        # SEBI charges (with GST)
        sebi_charges = amount * self.SEBI_CHARGES_RATE
        costs['sebi_charges'] = sebi_charges * self.GST_RATE
        
        # GST on brokerage
        costs['gst'] = brokerage * self.GST_RATE
        
        # Total costs
        total_costs = sum(costs.values())
        costs['total_costs'] = total_costs
        
        # Net amount (including costs)
        if action == 'buy':
            costs['net_amount'] = amount + total_costs
        else:
            costs['net_amount'] = amount - total_costs
        
        return costs


# For stocks, STT is different (applied on both buy and sell)
class IndianStockCostCalculator(IndianMarketCostCalculator):
    """
    Cost calculator specifically for Indian stocks.
    
    Stocks have STT on both buy and sell transactions (unlike ETFs).
    """
    
    STT_RATE_BUY = 0.10 / 100  # 0.10% on buy for stocks
    STT_RATE_SELL = 0.10 / 100  # 0.10% on sell for stocks
    STAMP_DUTY_BUY = 0.015 / 100  # 0.015% on buy for stocks
    
    def calculate(self, action: str, amount: float, brokerage_percent: float) -> Dict[str, float]:
        """
        Calculate transaction costs for Indian stocks.
        
        Stocks have STT on both buy and sell, unlike ETFs.
        """
        costs = {}
        
        # Brokerage
        brokerage = amount * (brokerage_percent / 100)
        costs['brokerage'] = brokerage
        
        # STT - on both buy and sell for stocks
        if action == 'sell':
            costs['stt'] = amount * self.STT_RATE_SELL
        else:
            costs['stt'] = amount * self.STT_RATE_BUY
        
        # Stamp duty - only on buy
        if action == 'buy':
            costs['stamp_duty'] = amount * self.STAMP_DUTY_BUY
        else:
            costs['stamp_duty'] = 0
        
        # Exchange charges (with GST)
        exchange_charges = amount * self.EXCHANGE_CHARGES_RATE
        costs['exchange_charges'] = exchange_charges * self.GST_RATE
        
        # SEBI charges (with GST)
        sebi_charges = amount * self.SEBI_CHARGES_RATE
        costs['sebi_charges'] = sebi_charges * self.GST_RATE
        
        # GST on brokerage
        costs['gst'] = brokerage * self.GST_RATE
        
        # Total costs
        total_costs = sum(costs.values())
        costs['total_costs'] = total_costs
        
        # Net amount (including costs)
        if action == 'buy':
            costs['net_amount'] = amount + total_costs
        else:
            costs['net_amount'] = amount - total_costs
        
        return costs

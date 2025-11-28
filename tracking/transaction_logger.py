"""
Transaction Logger

Handles detailed logging of transaction costs for analysis and reporting.
"""

from datetime import datetime
from typing import Dict, List


class TransactionLogger:
    """
    Log detailed transaction costs for analysis.
    
    This logger maintains a comprehensive record of all transactions
    including costs breakdown, taxes, and net amounts.
    """
    
    def __init__(self):
        """Initialize transaction logger with empty log"""
        self.transaction_costs_log: List[Dict] = []
    
    def log_transaction(self, week: int, date: datetime, action: str,
                       ticker: str, units: int, price: float,
                       costs: Dict, capital_gains_tax: float = 0):
        """
        Log detailed transaction costs for analysis.
        
        Args:
            week: Week number in backtest
            date: Transaction date
            action: 'buy' or 'sell'
            ticker: Asset symbol
            units: Number of units
            price: Price per unit
            costs: Cost breakdown from cost calculator
            capital_gains_tax: Capital gains tax amount (for sells)
        """
        self.transaction_costs_log.append({
            'week': week,
            'date': date,
            'action': action,
            'ticker': ticker,
            'units': units,
            'price': price,
            'gross_amount': units * price,
            **costs,  # Unpack all cost components
            'capital_gains_tax': capital_gains_tax,
            'total_cost_with_tax': costs.get('total_costs', 0) + capital_gains_tax
        })
    
    def get_all_transactions(self) -> List[Dict]:
        """
        Get all logged transactions.
        
        Returns:
            List of all transaction records
        """
        return self.transaction_costs_log
    
    def get_transactions_by_ticker(self, ticker: str) -> List[Dict]:
        """
        Get transactions for a specific ticker.
        
        Args:
            ticker: Asset symbol
            
        Returns:
            List of transactions for the ticker
        """
        return [t for t in self.transaction_costs_log if t['ticker'] == ticker]
    
    def get_transactions_by_action(self, action: str) -> List[Dict]:
        """
        Get transactions by action type.
        
        Args:
            action: 'buy' or 'sell'
            
        Returns:
            List of transactions matching the action
        """
        return [t for t in self.transaction_costs_log if t['action'] == action]
    
    def get_total_costs(self) -> Dict[str, float]:
        """
        Calculate total costs across all transactions.
        
        Returns:
            Dictionary with aggregated costs:
            - total_brokerage
            - total_stt
            - total_stamp_duty
            - total_exchange_charges
            - total_sebi_charges
            - total_gst
            - total_capital_gains_tax
            - grand_total
        """
        totals = {
            'total_brokerage': 0,
            'total_stt': 0,
            'total_stamp_duty': 0,
            'total_exchange_charges': 0,
            'total_sebi_charges': 0,
            'total_gst': 0,
            'total_capital_gains_tax': 0,
            'grand_total': 0
        }
        
        for transaction in self.transaction_costs_log:
            totals['total_brokerage'] += transaction.get('brokerage', 0)
            totals['total_stt'] += transaction.get('stt', 0)
            totals['total_stamp_duty'] += transaction.get('stamp_duty', 0)
            totals['total_exchange_charges'] += transaction.get('exchange_charges', 0)
            totals['total_sebi_charges'] += transaction.get('sebi_charges', 0)
            totals['total_gst'] += transaction.get('gst', 0)
            totals['total_capital_gains_tax'] += transaction.get('capital_gains_tax', 0)
            totals['grand_total'] += transaction.get('total_cost_with_tax', 0)
        
        return totals
    
    def clear(self):
        """Clear all transaction logs"""
        self.transaction_costs_log.clear()
    
    def get_transaction_count(self) -> Dict[str, int]:
        """
        Get count of transactions by type.
        
        Returns:
            Dictionary with counts:
            - total: Total transactions
            - buy: Buy transactions
            - sell: Sell transactions
        """
        total = len(self.transaction_costs_log)
        buy_count = sum(1 for t in self.transaction_costs_log if t['action'] == 'buy')
        sell_count = sum(1 for t in self.transaction_costs_log if t['action'] == 'sell')
        
        return {
            'total': total,
            'buy': buy_count,
            'sell': sell_count
        }

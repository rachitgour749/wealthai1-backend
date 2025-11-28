"""
FIFO Purchase Tracker

Handles First-In-First-Out tracking of purchase lots for accurate
capital gains tax calculation.
"""

from datetime import datetime
from typing import Dict, List, Optional


class FIFOTracker:
    """
    Track purchase lots using FIFO (First-In-First-Out) logic.
    
    This tracker maintains purchase history for each ticker and provides
    methods to add purchases and retrieve lots for tax calculations.
    """
    
    def __init__(self):
        """Initialize FIFO tracker with empty purchase history"""
        self.purchase_history: Dict[str, List[Dict]] = {}
    
    def add_purchase(self, ticker: str, units: int, price: float, date: datetime):
        """
        Add a purchase record for FIFO tracking.
        
        Args:
            ticker: Asset symbol
            units: Number of units purchased
            price: Purchase price per unit
            date: Purchase date
        """
        if ticker not in self.purchase_history:
            self.purchase_history[ticker] = []
        
        self.purchase_history[ticker].append({
            'date': date,
            'units': units,
            'price': price,
            'remaining_units': units  # Track remaining units for FIFO
        })
    
    def get_purchase_lots(self, ticker: str) -> List[Dict]:
        """
        Get all purchase lots for a ticker.
        
        Args:
            ticker: Asset symbol
            
        Returns:
            List of purchase records (in FIFO order)
        """
        return self.purchase_history.get(ticker, [])
    
    def update_remaining_units(self, ticker: str, units_sold: int):
        """
        Update remaining units after a sale (FIFO order).
        
        This method is called by tax calculator to track which lots
        have been sold.
        
        Args:
            ticker: Asset symbol
            units_sold: Number of units being sold
        """
        if ticker not in self.purchase_history:
            return
        
        units_remaining = units_sold
        
        for purchase in self.purchase_history[ticker]:
            if units_remaining <= 0:
                break
            
            if purchase['remaining_units'] <= 0:
                continue
            
            units_from_this_purchase = min(units_remaining, purchase['remaining_units'])
            purchase['remaining_units'] -= units_from_this_purchase
            units_remaining -= units_from_this_purchase
    
    def get_total_units(self, ticker: str) -> int:
        """
        Get total remaining units for a ticker.
        
        Args:
            ticker: Asset symbol
            
        Returns:
            Total remaining units across all purchase lots
        """
        if ticker not in self.purchase_history:
            return 0
        
        return sum(lot['remaining_units'] for lot in self.purchase_history[ticker])
    
    def clear_ticker(self, ticker: str):
        """
        Clear all purchase history for a ticker.
        
        Args:
            ticker: Asset symbol
        """
        if ticker in self.purchase_history:
            del self.purchase_history[ticker]
    
    def clear_all(self):
        """Clear all purchase history"""
        self.purchase_history.clear()
    
    def get_average_cost(self, ticker: str) -> Optional[float]:
        """
        Calculate average cost basis for a ticker.
        
        Args:
            ticker: Asset symbol
            
        Returns:
            Average cost per unit, or None if no purchases
        """
        lots = self.get_purchase_lots(ticker)
        if not lots:
            return None
        
        total_cost = sum(lot['units'] * lot['price'] for lot in lots)
        total_units = sum(lot['units'] for lot in lots)
        
        return total_cost / total_units if total_units > 0 else None

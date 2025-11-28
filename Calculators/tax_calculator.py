"""
Indian Capital Gains Tax Calculator

Handles capital gains tax calculations for Indian markets.
Tax rate is hardcoded as per Indian regulations (12.5% LTCG).
"""

from datetime import datetime
from typing import Dict, List


class IndianCapitalGainsTaxCalculator:
    """
    Calculate capital gains tax for Indian markets.
    
    As per Indian tax regulations:
    - Long Term Capital Gains (LTCG): 12.5% on profits
    - Holding period > 1 year for equity
    
    Tax rate is hardcoded for Indian market.
    """
    
    # Indian Tax Rates (Hardcoded)
    LTCG_TAX_RATE = 0.125  # 12.5% Long Term Capital Gains Tax
    
    def calculate(self, total_profit: float) -> float:
        """
        Calculate capital gains tax on profit.
        
        Args:
            total_profit: Total profit from sale (can be negative for loss)
            
        Returns:
            Tax amount (12.5% of profit, 0 if loss)
        """
        # Tax only on profits, not on losses
        return max(0, total_profit * self.LTCG_TAX_RATE) if total_profit > 0 else 0
    
    def calculate_with_details(self, purchase_lots: List[Dict], 
                               units_to_sell: int, sell_price: float,
                               sell_date: datetime) -> Dict:
        """
        Calculate capital gains tax with detailed FIFO transaction breakdown.
        
        This method processes purchase lots in FIFO order and calculates
        profit/loss and tax for each lot.
        
        Args:
            purchase_lots: List of purchase records (from FIFO tracker)
            units_to_sell: Number of units being sold
            sell_price: Selling price per unit
            sell_date: Sale date
            
        Returns:
            Dictionary containing:
            - total_profit: Total profit/loss
            - capital_gains_tax: Tax amount (12.5% of profit)
            - cost_basis: Total cost basis
            - transactions: List of FIFO transactions with details
        """
        if not purchase_lots:
            return {
                'total_profit': 0,
                'capital_gains_tax': 0,
                'cost_basis': sell_price * units_to_sell,
                'transactions': []
            }
        
        total_profit = 0
        total_cost_basis = 0
        units_remaining = units_to_sell
        tax_transactions = []
        
        # FIFO: Process purchases in order
        for purchase in purchase_lots:
            if units_remaining <= 0:
                break
            
            if purchase.get('remaining_units', 0) <= 0:
                continue
            
            # Determine how many units to sell from this purchase
            units_from_this_purchase = min(units_remaining, purchase['remaining_units'])
            
            # Calculate profit/loss for this lot
            cost_basis = units_from_this_purchase * purchase['price']
            sale_value = units_from_this_purchase * sell_price
            profit_loss = sale_value - cost_basis
            
            # Accumulate totals
            total_cost_basis += cost_basis
            total_profit += profit_loss
            units_remaining -= units_from_this_purchase
            
            # Record transaction for audit trail
            tax_transactions.append({
                'purchase_date': purchase['date'],
                'purchase_price': purchase['price'],
                'units_sold': units_from_this_purchase,
                'cost_basis': cost_basis,
                'sale_value': sale_value,
                'profit_loss': profit_loss
            })
        
        # Calculate tax
        capital_gains_tax = self.calculate(total_profit)
        
        return {
            'total_profit': total_profit,
            'capital_gains_tax': capital_gains_tax,
            'cost_basis': total_cost_basis,
            'transactions': tax_transactions
        }

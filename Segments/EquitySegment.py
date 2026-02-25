from datetime import datetime
from typing import Dict, List, Optional
from Exchange.IndianExchange import IndianExchange
from Exchange.ExchangePolicy import ExchangePolicy

class EquitySegment(IndianExchange):
    """
    Level 3: Segment Layer (Equity)
    
    Handles Delivery-based Equity trading (Stocks & ETFs).
    Injected with ExchangePolicy to handle market-specific costs.
    """
    
    def __init__(self, db_session=None, policy: Optional[ExchangePolicy] = None):
        super().__init__(db_session)
        # Default to Indian policy if none provided for backward compatibility
        if policy is None:
            from Exchange.IndianExchangePolicy import IndianExchangePolicy
            self.policy = IndianExchangePolicy()
        else:
            self.policy = policy
            
        # FIFO Inventory: {ticker: [{'date': date, 'units': qty, 'price': price, 'remaining_units': qty}]}
        self.purchase_history: Dict[str, List[Dict]] = {}
        
    def calculate_stock_delivery_costs(self, action: str, amount: float, 
                                       brokerage_percent: float = 0.1) -> Dict[str, float]:
        """Legacy wrapper - now uses policy logic"""
        return self.policy.calculate_transaction_costs(action, 'STOCK', amount, brokerage_percent)
    
    def calculate_etf_delivery_costs(self, action: str, amount: float, 
                                     brokerage_percent: float = 0.1) -> Dict[str, float]:
        """Legacy wrapper - now uses policy logic"""
        return self.policy.calculate_transaction_costs(action, 'ETF', amount, brokerage_percent)

    def calculate_delivery_costs(self, action: str, amount: float, 
                               brokerage_percent: float = 0.1) -> Dict[str, float]:
        """
        Calculates transaction costs for Delivery trades.
        Defaults to ETF logic for backward compatibility.
        Use calculate_stock_delivery_costs() or calculate_etf_delivery_costs() for specific needs.
        """
        return self.calculate_etf_delivery_costs(action, amount, brokerage_percent)

    def manage_fifo_inventory(self, ticker: str, units: int, price: float, date: datetime):
        """
        Adds a purchase record to the FIFO queue.
        """
        if ticker not in self.purchase_history:
            self.purchase_history[ticker] = []
        
        self.purchase_history[ticker].append({
            'date': date,
            'units': units,
            'price': price,
            'remaining_units': units
        })

    def calculate_capital_gains(self, ticker: str, units_to_sell: int, 
                              sell_price: float, sell_date: datetime) -> Dict:
        """
        Calculates Capital Gains Tax using FIFO method.
        Rate: 12.5% on Profits (LTCG assumption).
        """
        if ticker not in self.purchase_history or not self.purchase_history[ticker]:
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
        for purchase in self.purchase_history[ticker]:
            if units_remaining <= 0:
                break
            
            if purchase['remaining_units'] <= 0:
                continue
            
            # Determine how many units to sell from this purchase
            units_from_this_purchase = min(units_remaining, purchase['remaining_units'])
            
            # Calculate profit/loss for this lot
            cost_basis = units_from_this_purchase * purchase['price']
            sale_value = units_from_this_purchase * sell_price
            profit_loss = sale_value - cost_basis
            
            # Update remaining units in purchase record
            purchase['remaining_units'] -= units_from_this_purchase
            
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
        
        # Calculate capital gains tax (12.5% on profits only)
        capital_gains_tax = max(0, total_profit * 0.125) if total_profit > 0 else 0
        
        return {
            'total_profit': total_profit,
            'capital_gains_tax': capital_gains_tax,
            'cost_basis': total_cost_basis,
            'transactions': tax_transactions
        }

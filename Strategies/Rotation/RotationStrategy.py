import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from Segments.EquitySegment import EquitySegment

class RotationStrategy(EquitySegment):
    """
    Level 4: Strategy Implementation (Rotation)
    
    Base class for Momentum Rotation Strategies (ETF & Stocks).
    Inherits from EquitySegment to access Indian market logic and costs.
    
    Key Logic:
    - Momentum based on distance from 52-week high
    - Weekly rebalancing
    - Dynamic churning
    """
    
    def __init__(self, db_session=None, policy=None):
        super().__init__(db_session, policy=policy)
        # Strategy specific state
        self.current_holdings: Dict[str, int] = {} # Symbol -> Quantity
        self.current_cash: float = 0.0
        self.portfolio_value: float = 0.0
        self.trades: List[Dict] = []  # Trade log
        
    def calculate_momentum_score(self, df: pd.DataFrame, current_date: datetime) -> pd.DataFrame:
        """
        Calculates momentum score based on distance from 52-week high.
        
        Formula: ((Current Price - 52W High) / 52W High) * 100
        """
        lookback_start = current_date - timedelta(weeks=52)
        
        # Filter data for 52-week lookback
        mask = (df.index >= lookback_start) & (df.index <= current_date)
        lookback_df = df[mask]
        
        if lookback_df.empty:
            return pd.DataFrame()
        
        # Calculate 52-week high/low
        high_52w = lookback_df.max()
        low_52w = lookback_df.min()
        
        # Get current price (handle missing date by taking last available)
        if current_date in df.index:
            current_price = df.loc[current_date]
        else:
            current_price = df.iloc[-1]
            
        # Calculate distance from high
        distance_from_high = ((current_price - high_52w) / high_52w * 100)
        
        return pd.DataFrame({
            '52_week_high': high_52w,
            '52_week_low': low_52w,
            'current_price': current_price,
            'distance_from_high_pct': distance_from_high
        })

    def select_top_assets(self, momentum_scores: pd.DataFrame, top_n: int = 1) -> List[str]:
        """
        Selects the top N assets based on momentum score.
        """
        if momentum_scores.empty:
            return []
            
        # Sort by distance from high (descending - closer to 0 is better if negative)
        # Actually, standard momentum is usually higher is better.
        # But "Distance from High" is usually negative (e.g. -5% is better than -20%).
        # So sorting descending works.
        sorted_scores = momentum_scores.sort_values('distance_from_high_pct', ascending=False)
        
        return sorted_scores.head(top_n).index.tolist()

    def rebalance_portfolio(self, target_assets: List[str], current_date: datetime, 
                          prices: Dict[str, float], total_capital: float):
        """
        Rebalances the portfolio to hold the target assets.
        Sells assets not in target, Buys target assets.
        """
        # 1. Sell assets not in target
        holdings_to_sell = [s for s in self.current_holdings if s not in target_assets]
        
        for symbol in holdings_to_sell:
            qty = self.current_holdings[symbol]
            price = prices.get(symbol, 0)
            if price > 0:
                # Calculate costs
                amount = qty * price
                costs = self.calculate_delivery_costs('sell', amount)
                
                # Calculate tax
                # tax_info = self.calculate_capital_gains(symbol, qty, price, current_date)
                tax = 0.0 # Disabled as per request
                
                # Update cash
                net_proceeds = costs['net_amount'] - tax
                self.current_cash += net_proceeds
                
                # Log trade (Placeholder - needs implementation in subclass or here)
                self.trades.append({
                    'date': current_date,
                    'symbol': symbol,
                    'action': 'SELL',
                    'qty': qty,
                    'price': price,
                    'amount': amount,
                    'costs': costs['total_costs'],
                    'tax': tax
                })
                
                del self.current_holdings[symbol]

        # 2. Buy target assets
        # Simple equal weight allocation for now
        if not target_assets:
            return

        target_weight = 1.0 / len(target_assets)
        allocation_per_asset = total_capital * target_weight
        
        for symbol in target_assets:
            if symbol in self.current_holdings:
                continue # Already hold it
                
            price = prices.get(symbol, 0)
            if price > 0:
                # Calculate quantity
                # Estimate costs buffer (approx 0.5%)
                available_amount = allocation_per_asset * 0.995
                qty = int(available_amount // price)
                
                if qty > 0:
                    amount = qty * price
                    costs = self.calculate_delivery_costs('buy', amount)
                    
                    if self.current_cash >= costs['net_amount']:
                        self.current_cash -= costs['net_amount']
                        self.current_holdings[symbol] = qty
                        
                        # Add to FIFO inventory
                        self.manage_fifo_inventory(symbol, qty, price, current_date)
                        
                        # Log trade
                        self.trades.append({
                            'date': current_date,
                            'symbol': symbol,
                            'action': 'BUY',
                            'qty': qty,
                            'price': price,
                            'amount': amount,
                            'costs': costs['total_costs'],
                            'tax': 0
                        })

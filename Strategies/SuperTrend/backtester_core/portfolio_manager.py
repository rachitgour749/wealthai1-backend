"""
Portfolio Management and Rebalancing Logic
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class PortfolioManager:
    """
    Manages portfolio positions and rebalancing
    """
    
    def __init__(self, initial_capital: float = 1000000.0, max_holdings: int = 5, 
                 brokerage_pct: float = 0.1, buffer_pct: float = 10.0):
        """
        Initialize portfolio manager with buffer P&L absorption system
        
        Args:
            initial_capital: Starting capital
            max_holdings: Maximum number of holdings
            brokerage_pct: Brokerage percentage (e.g., 0.03 for 0.03%)
            buffer_pct: Buffer percentage for P&L absorption (e.g., 10.0 for 10%)
        """
        self.initial_capital = initial_capital
        self.max_holdings = max_holdings
        self.brokerage_pct = brokerage_pct
        self.buffer_pct = buffer_pct
        
        # Split capital: Buffer + Trading
        self.buffer_capital = initial_capital * (buffer_pct / 100.0)
        self.trading_capital = initial_capital * (1 - buffer_pct / 100.0)
        
        # Start with trading capital as available cash
        self.cash = self.trading_capital
        
        self.positions = {}  # {symbol: {'quantity': int, 'entry_price': float, 'entry_date': str}}
        self.portfolio_value = initial_capital
        self.trades = []
        self.equity_curve = []
        self.total_costs = 0  # Track cumulative transaction costs
    
    def get_position_size(self) -> float:
        """
        Calculate position size based on trading capital (excluding buffer)
        
        Buffer Logic:
        - Position sizing uses only trading capital (e.g., 90% of initial capital)
        - Buffer capital is reserved for P&L absorption
        
        Returns:
            Fixed position size per stock
        """
        # Use trading capital, NOT total portfolio value
        return self.trading_capital / self.max_holdings
    
    def buy(self, symbol: str, price: float, date: str) -> Dict:
        """
        Buy a stock with transaction costs
        
        Args:
            symbol: Stock symbol
            price: Entry price
            date: Trade date
        
        Returns:
            Trade details
        """
        if len(self.positions) >= self.max_holdings:
            return None
        
        from .utils import calculate_transaction_costs
        
        position_size = self.get_position_size()
        quantity = int(position_size / price)
        
        if quantity == 0:
            return None
        
        # Calculate transaction value
        transaction_value = quantity * price
        
        # Calculate transaction costs (BUY)
        costs = calculate_transaction_costs(transaction_value, self.brokerage_pct, is_buy=True)
        net_amount = costs['net_amount']  # Stock value + all costs
        
        # Check if we have enough cash
        if net_amount > self.cash:
            # Reduce quantity to fit available cash
            max_quantity = int(self.cash / (price * (1 + costs['total_costs'] / transaction_value)))
            quantity = max_quantity
            
            if quantity == 0:
                return None
            
            # Recalculate with adjusted quantity
            transaction_value = quantity * price
            costs = calculate_transaction_costs(transaction_value, self.brokerage_pct, is_buy=True)
            net_amount = costs['net_amount']
        
        # Deduct net amount from cash
        self.cash -= net_amount
        self.total_costs += costs['total_costs']
        
        self.positions[symbol] = {
            'quantity': quantity,
            'entry_price': price,
            'entry_date': date,
            'cost': transaction_value,
            'transaction_costs': costs['total_costs']
        }
        
        trade = {
            'date': date,
            'symbol': symbol,
            'action': 'BUY',
            'price': price,
            'quantity': quantity,
            'value': transaction_value,
            'brokerage': costs['brokerage'],
            'stamp_duty': costs['stamp_duty'],
            'exchange_charges': costs['exchange_charges'],
            'sebi_charges': costs['sebi_charges'],
            'gst': costs['gst'],
            'total_costs': costs['total_costs'],
            'net_amount': net_amount,
            'cash_after': self.cash
        }
        
        self.trades.append(trade)
        return trade
    
    def sell(self, symbol: str, price: float, date: str) -> Dict:
        """
        Sell stock with buffer-based P&L absorption
        
        Buffer Logic:
        - Profit → Add to buffer, return original position size as cash
        - Loss → Cover from buffer, restore full position size cash
        - This maintains consistent position sizing across all trades
        
        Args:
            symbol: Stock symbol
            price: Exit price
            date: Trade date
        
        Returns:
            Trade details
        """
        if symbol not in self.positions:
            return None
        
        from .utils import calculate_transaction_costs
        
        position = self.positions[symbol]
        quantity = position['quantity']
        original_cost = position['cost']  # Original position value
        
        # Calculate transaction value
        transaction_value = quantity * price
        
        # Calculate transaction costs (SELL)
        costs = calculate_transaction_costs(transaction_value, self.brokerage_pct, is_buy=False)
        net_proceeds = costs['net_amount']  # Proceeds after costs
        
        # Calculate P&L (after costs)
        pnl = net_proceeds - original_cost
        pnl_pct = (pnl / original_cost) * 100
        
        # **BUFFER LOGIC - P&L ABSORPTION**
        if pnl > 0:
            # PROFIT → Add to buffer, return original position size
            self.buffer_capital += pnl
            cash_to_add = original_cost  # Return original position size
            buffer_action = f"PROFIT_TO_BUFFER: +{pnl:.2f}"
        else:
            # LOSS → Cover from buffer, restore full position size
            loss_amount = abs(pnl)
            self.buffer_capital -= loss_amount  # Deduct loss from buffer
            cash_to_add = original_cost  # Restore original position size
            buffer_action = f"LOSS_FROM_BUFFER: -{loss_amount:.2f}"
        
        # Add cash for reinvestment
        self.cash += cash_to_add
        self.total_costs += costs['total_costs']
        
        trade = {
            'date': date,
            'symbol': symbol,
            'action': 'SELL',
            'price': price,
            'quantity': quantity,
            'value': transaction_value,
            'brokerage': costs['brokerage'],
            'stt': costs['stt'],
            'exchange_charges': costs['exchange_charges'],
            'sebi_charges': costs['sebi_charges'],
            'gst': costs['gst'],
            'total_costs': costs['total_costs'],
            'net_amount': net_proceeds,
            'original_cost': original_cost,
            'cash_returned': cash_to_add,  # Always original position size
            'cash_after': self.cash,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'entry_date': position['entry_date'],
            'entry_price': position['entry_price'],
            'buffer_capital': self.buffer_capital,
            'buffer_action': buffer_action
        }
        
        self.trades.append(trade)
        del self.positions[symbol]
        
        return trade
    
    def update_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Update portfolio value based on current prices (includes buffer)
        
        Args:
            current_prices: Dictionary of {symbol: price}
        
        Returns:
            Total portfolio value (cash + holdings + buffer)
        """
        holdings_value = 0
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                holdings_value += position['quantity'] * current_prices[symbol]
        
        # Total portfolio = cash + holdings + buffer
        self.portfolio_value = self.cash + holdings_value + self.buffer_capital
        return self.portfolio_value
    
    def get_positions_df(self, current_prices: Dict[str, float]) -> pd.DataFrame:
        """
        Get current positions as DataFrame
        
        Args:
            current_prices: Dictionary of {symbol: price}
        
        Returns:
            DataFrame with position details
        """
        if not self.positions:
            return pd.DataFrame()
        
        positions_list = []
        
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position['entry_price'])
            current_value = position['quantity'] * current_price
            pnl = current_value - position['cost']
            pnl_pct = (pnl / position['cost']) * 100
            
            positions_list.append({
                'symbol': symbol,
                'entry_date': position['entry_date'],
                'entry_price': position['entry_price'],
                'quantity': position['quantity'],
                'current_price': current_price,
                'current_value': current_value,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
        
        return pd.DataFrame(positions_list)
    
    def rebalance(self, candidates: pd.DataFrame, current_prices: Dict[str, float],
                  date: str, exit_symbols: List[str] = None) -> List[Dict]:
        """
        Rebalance portfolio
        
        Args:
            candidates: Top N candidates (sorted by RS)
            current_prices: Current prices
            date: Current date
            exit_symbols: Symbols to exit
        
        Returns:
            List of trades executed
        """
        trades = []
        
        # Exit positions
        if exit_symbols:
            for symbol in exit_symbols:
                if symbol in self.positions:
                    price = current_prices.get(symbol, self.positions[symbol]['entry_price'])
                    trade = self.sell(symbol, price, date)
                    if trade:
                        trades.append(trade)
        
        # Enter new positions - DEFENSIVE PROGRAMMING
        if not candidates.empty and 'symbol' in candidates.columns:
            candidate_symbols = candidates['symbol'].tolist()
            
            for symbol in candidate_symbols:
                if symbol not in self.positions and len(self.positions) < self.max_holdings:
                    price = current_prices.get(symbol)
                    if price:
                        trade = self.buy(symbol, price, date)
                        if trade:
                            trades.append(trade)
        
        # Update portfolio value
        self.update_portfolio_value(current_prices)
        
        return trades
    
    def record_equity(self, date: str):
        """Record equity curve point with buffer tracking"""
        # Calculate holdings value
        holdings_value = sum(
            p['quantity'] * p.get('current_price', p['entry_price']) 
            for p in self.positions.values()
        )
        
        self.equity_curve.append({
            'date': date,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'buffer_capital': self.buffer_capital,
            'trading_capital_deployed': holdings_value,
            'positions_count': len(self.positions)
        })
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get all trades as DataFrame"""
        return pd.DataFrame(self.trades)
    
    def get_equity_curve_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        return pd.DataFrame(self.equity_curve)


"""
Rotation ETF Payout Backtester

Extends the standard ETF Rotation strategy with systematic withdrawal feature.
During the churning phase, the strategy withdraws a specified amount weekly.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import json
import os

# Import base backtester
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Rotation_ETF.services.backtester import ETFRotationBacktester

# Import centralized logging
from Strategies.utilities.logging_config import StrategyLogger


class RotationETFPayoutBacktester(ETFRotationBacktester):
    """
    ETF Rotation Backtester with Payout/Withdrawal Feature
    
    Key Enhancement:
    - Adds 'withdraw_amount' parameter for systematic weekly withdrawals
    - During accumulation: Buys ETFs worth 'accumulation_per_week'
    - During churning: Sells ETFs worth 'accumulation_per_week + withdraw_amount'
    - Tracks total withdrawn amount throughout the backtest
    """
    
    def __init__(self, market: str = "INDIA", db_path: str = None, config_path: str = None):
        """
        Initialize Rotation ETF Payout Backtester
        
        Args:
            market: Market name (INDIA, US)
            db_path: Deprecated - kept for compatibility
            config_path: Path to configuration JSON file
        """
        # Call parent constructor
        super().__init__(market=market, db_path=db_path)
        
        # Override logger for this strategy
        self.logger = StrategyLogger('Rotation_ETF_Payout')
        
        # Load configuration
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), 'config.json'
        )
        self.load_config()
        
        # Initialize withdrawal tracking
        self.payout_start_week = 1
        self.current_week = 1
        self.total_withdrawn_amount = 0.0
        self.withdrawal_log = []
        
        self.logger.info(f"✅ Rotation ETF Payout Backtester initialized")
        self.logger.info(f"   Accumulation per week: ₹{self.accumulation_per_week:,.0f}")
        self.logger.info(f"   Withdraw amount: ₹{self.withdraw_amount:,.0f}")
        self.logger.info(f"   Total churning amount: ₹{self.accumulation_per_week + self.withdraw_amount:,.0f}")
    
    def load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Load strategy parameters
            self.accumulation_weeks = config.get('accumulation_weeks', 13)
            self.accumulation_per_week = config.get('accumulation_per_week', 50000)
            self.withdraw_amount = config.get('withdraw_amount', 0)
            self.payout_start_week = config.get('payout_start_week', 1)  # NEW
            self.selected_etfs = config.get('selected_etfs', [])
            self.brokerage_percent = config.get('brokerage_percent', 0.03)
            self.compounding_enabled = config.get('compounding_enabled', True)
            
            self.logger.info(f"📋 Configuration loaded from: {self.config_path}")
            
        except FileNotFoundError:
            self.logger.error(f"❌ Config file not found: {self.config_path}")
            # Set defaults
            self.accumulation_weeks = 13
            self.accumulation_per_week = 50000
            self.withdraw_amount = 0
            self.payout_start_week = 1
            self.selected_etfs = []
            self.brokerage_percent = 0.03
            self.compounding_enabled = True
        except Exception as e:
            self.logger.error(f"❌ Error loading config: {e}")
            raise
    
    def calculate_dynamic_churn_amount(self, current_nav: float, cash: float, 
                                      capital_per_week: float, accumulation_weeks: int, 
                                      compounding_enabled: bool) -> float:
        """
        Calculate dynamic churn amount with withdrawal logic
        
        Override parent method to add withdraw_amount to the churning capital.
        
        Returns:
            Base churning amount + withdraw_amount
        """
        # Get base churning amount from parent class
        base_churn_amount = super().calculate_dynamic_churn_amount(
            current_nav, cash, capital_per_week, accumulation_weeks, compounding_enabled
        )
        
        # Add withdrawal amount only if payout start week has been reached
        if self.current_week >= self.payout_start_week:
            total_churn_amount = base_churn_amount + self.withdraw_amount
            self.logger.info(f"💰 Churning calculation (WITHDRAWAL ACTIVE):")
        else:
            total_churn_amount = base_churn_amount
            self.logger.info(f"💰 Churning calculation (ACCUMULATION ONLY/BEFORE PAYOUT):")
        
        self.logger.info(f"   Current week: {self.current_week} (Payout starts: {self.payout_start_week})")
        self.logger.info(f"   Base churn amount: ₹{base_churn_amount:,.0f}")
        self.logger.info(f"   Withdrawal amount: ₹{self.withdraw_amount:,.0f}")
        self.logger.info(f"   Total churn amount: ₹{total_churn_amount:,.0f}")
        
        return total_churn_amount
    
    def execute_weekly_trade(self, *args, **kwargs) -> Dict:
        """
        Track current week number before executing trade
        """
        # The first argument is week_num
        if args:
            self.current_week = args[0]
        elif 'week_num' in kwargs:
            self.current_week = kwargs['week_num']
            
        return super().execute_weekly_trade(*args, **kwargs)

    def execute_churning_phase(self, week_num: int, execution_date: datetime, 
                               high_low_data: pd.DataFrame, open_prices: pd.Series, 
                               current_holdings: Dict[str, int], cash: float,
                               target_capital: float, brokerage_percent: float) -> Dict:
        """
        Execute churning phase with withdrawal tracking
        
        Override parent method to:
        1. Sell ETFs worth target_capital (base + withdrawal)
        2. Reinvest only base_capital
        3. Track the difference as actual withdrawal
        """
        # STEP 0: Calculate portions based on whether payout is active
        if self.current_week >= self.payout_start_week:
            base_capital = target_capital - self.withdraw_amount
            withdrawal_portion = self.withdraw_amount
        else:
            base_capital = target_capital
            withdrawal_portion = 0.0
        
        self.logger.progress(f"🔄 Churning Phase - Week {week_num}")
        self.logger.progress(f"   Current week: {self.current_week} (Payout starts: {self.payout_start_week})")
        self.logger.progress(f"   Total to raise: ₹{target_capital:,.0f}")
        self.logger.progress(f"   Reinvestment: ₹{base_capital:,.0f}")
        self.logger.progress(f"   Planned withdrawal: ₹{withdrawal_portion:,.0f}")
        
        # STEP 1: Raise capital (sell ETFs worth target_capital)
        # Get the selling priority (closest to 52-week high first)
        sorted_for_sell = high_low_data.sort_values('distance_from_high', ascending=True)
        
        total_raised = 0.0
        updated_holdings = current_holdings.copy()
        sell_transactions = []  # Track all sell transactions
        
        self.logger.progress(f"🔴 CAPITAL RAISING PROCESS (Target: ₹{target_capital:,.0f})")
        self.logger.debug(f"   Current holdings: {updated_holdings}")
        self.logger.debug(f"   High/Low DataFrame columns: {list(high_low_data.columns)}")
        self.logger.debug(f"   High/Low DataFrame:\n{high_low_data}")
        
        # Sell ETFs until we raise target_capital
        # Note: high_low_data has 'symbol' as a column, not as index
        for idx, row in sorted_for_sell.iterrows():
            # Get symbol from the row
            symbol = row['symbol']
            
            self.logger.debug(f"   Processing {symbol}...")
            
            if symbol not in updated_holdings or updated_holdings[symbol] <= 0:
                self.logger.debug(f"   Skipping {symbol}: no holdings ({updated_holdings.get(symbol, 0)} units)")
                continue
            
            # Get execution price
            if symbol not in open_prices or pd.isna(open_prices[symbol]):
                self.logger.debug(f"   Skipping {symbol}: no price data")
                continue
            
            price = open_prices[symbol]
            units_available = updated_holdings[symbol]
            
            self.logger.debug(f"   {symbol}: price=₹{price:.2f}, available={units_available}")
            
            # CONTINUOUS SELLING LOOP - Keep selling until target is reached
            while total_raised < target_capital and updated_holdings.get(symbol, 0) > 0:
                units_available = updated_holdings[symbol]
                remaining_needed = target_capital - total_raised
                
                # Calculate units to sell - only what's needed to reach target
                units_needed = int(remaining_needed / price)
                units_to_sell = min(units_available, max(1, units_needed))
                
                self.logger.debug(f"   {symbol}: price=₹{price:.2f}, available={units_available}, to_sell={units_to_sell}")
                
                if units_to_sell > 0:
                    # Execute sell
                    self.logger.debug(f"   Executing sell transaction for {symbol}...")
                    sell_result = self.execute_sell_transaction(
                        week_num, execution_date, symbol, units_to_sell, 
                        price, brokerage_percent
                    )
                    
                    self.logger.debug(f"   Sell result: {sell_result}")
                    
                    if sell_result.get('success'):
                        net_proceeds = sell_result['net_proceeds']
                        total_raised += net_proceeds
                        updated_holdings[symbol] -= units_to_sell
                        
                        # Track this sell transaction
                        sell_transactions.append({
                            'ticker': symbol,
                            'units': units_to_sell,
                            'price': price,
                            'amount': sell_result['amount'],
                            'costs': sell_result['costs'],
                            'capital_gains_tax': sell_result['capital_gains_tax'],
                            'net_proceeds': net_proceeds
                        })
                        
                        # NEW: Decrement purchase counter when selling
                        if hasattr(self, 'etf_purchase_counts') and symbol in self.etf_purchase_counts:
                            self.etf_purchase_counts[symbol] = max(0, self.etf_purchase_counts[symbol] - 1)
                            new_count = self.etf_purchase_counts[symbol]
                            limit = self.etf_purchase_limits.get(symbol, 0) if hasattr(self, 'etf_purchase_limits') else 0
                            if hasattr(self, '_log'): # Safety check
                                self._log("limit", f"📉 Purchase count decremented: {symbol} = {new_count}/{limit}")
                                
                                # Print to terminal directly as requested
                                print(f"📉 Purchase Count Decremented (Payout): {symbol} {new_count}/{limit}")
                        
                        self.logger.trade(f"🔴 Sold {units_to_sell} units of {symbol} @ ₹{price:.2f}")
                        self.logger.trade(f"   Net proceeds: ₹{net_proceeds:,.0f}")
                        self.logger.trade(f"   Total raised: ₹{total_raised:,.0f}")
                        
                        # Check if target reached - STOP selling
                        if total_raised >= target_capital:
                            self.logger.progress(f"✅ Target capital reached: ₹{total_raised:,.0f} >= ₹{target_capital:,.0f}")
                            break
                    else:
                        self.logger.error(f"   ❌ Sell failed for {symbol}: {sell_result.get('error', 'Unknown error')}")
                        break  # Exit the while loop if sell fails
            
            # Check if target reached - exit outer loop
            if total_raised >= target_capital:
                break
        
        # Check if we raised enough
        if total_raised < target_capital * 0.95:  # Allow 5% tolerance
            self.logger.warning(f"⚠️ Could not raise full target. Raised: ₹{total_raised:,.0f}")
            # Adjust withdrawal based on what we actually raised (if payout is active)
            if self.current_week >= self.payout_start_week:
                actual_withdrawal = max(0, total_raised - base_capital)
            else:
                actual_withdrawal = 0.0
        else:
            actual_withdrawal = withdrawal_portion if self.current_week >= self.payout_start_week else 0.0
        
        # STEP 2: Reinvest only base_capital (not the full amount raised)
        reinvestment_amount = min(base_capital, total_raised - actual_withdrawal)
        
        self.logger.progress(f"🟢 REALLOCATION PROCESS (Reinvest: ₹{reinvestment_amount:,.0f})")
        
        # Get the best ETF to buy (closest to 52-week low)
        sorted_for_buy = high_low_data.sort_values('distance_from_low', ascending=True)
        
        # NEW: Check purchase limits
        target_etf = None
        
        for idx, row in sorted_for_buy.iterrows():
            etf_symbol = row['symbol']
            
            # Check limits if they exist
            if hasattr(self, 'etf_purchase_limits') and hasattr(self, 'etf_purchase_counts'):
                current_count = self.etf_purchase_counts.get(etf_symbol, 0)
                limit = self.etf_purchase_limits.get(etf_symbol, float('inf'))
                
                if current_count < limit:
                    target_etf = etf_symbol
                    distance_from_low = row['distance_from_low']
                    if hasattr(self, '_log'):
                        self._log("limit", f"🎯 Reinvestment target: {target_etf} ({distance_from_low:.2f}% from low)")
                        self._log("limit", f"   Purchase Status: {current_count}/{limit} purchases")
                    break
                else:
                    if hasattr(self, '_log'):
                        self._log("limit", f"⚠️ Skipped {etf_symbol} for reinvestment: Limit reached ({current_count}/{limit})")
            else:
                # Fallback if limits not initialized (should not happen if base class updated)
                target_etf = etf_symbol
                break
        
        if target_etf is None:
             self.logger.error("❌ All ETFs at purchase limit - cannot reinvest")
             # Proceed with empty target, checks below will handle it
             target_etf = sorted_for_buy.iloc[0]['symbol'] if not sorted_for_buy.empty else None
        
        buy_transaction = {}  # Track buy transaction
        
        # Buy the target ETF with reinvestment amount
        if target_etf in open_prices and not pd.isna(open_prices[target_etf]):
            price = open_prices[target_etf]
            
            # Calculate units to buy directly from reinvestment amount - NEW LOGIC
            units_to_buy = int(reinvestment_amount / price) if price > 0 else 0
            
            # BUG FIX: Verify total cost and adjust if needed
            # Calculate expected total cost
            actual_amount = units_to_buy * price
            actual_costs = self.calculate_transaction_costs('buy', actual_amount, brokerage_percent)
            total_cost = actual_amount + actual_costs['total_costs']
            
            # Decrement units if cost exceeds reinvestment amount
            while total_cost > reinvestment_amount and units_to_buy > 0:
                units_to_buy -= 1
                actual_amount = units_to_buy * price
                actual_costs = self.calculate_transaction_costs('buy', actual_amount, brokerage_percent)
                total_cost = actual_amount + actual_costs['total_costs']
            
            if units_to_buy > 0:
                buy_result = self.execute_buy_transaction(
                    week_num, execution_date, target_etf, units_to_buy,
                    price, brokerage_percent
                )
                
                if buy_result.get('success'):
                    updated_holdings[target_etf] = updated_holdings.get(target_etf, 0) + units_to_buy
                    
                    # Track this buy transaction
                    buy_transaction = {
                        'ticker': target_etf,
                        'units': units_to_buy,
                        'price': price,
                        'amount': buy_result['amount'],
                        'costs': buy_result['costs'],
                        'total_cost': buy_result['total_cost']
                    }
                    
                    self.logger.trade(f"🟢 Bought {units_to_buy} units of {target_etf} @ ₹{price:.2f}")
                    self.logger.trade(f"   Total cost: ₹{buy_result['total_cost']:,.0f}")
        
        # STEP 3: Track the actual withdrawal
        self.total_withdrawn_amount += actual_withdrawal
        
        # Log withdrawal
        self.withdrawal_log.append({
            'week': week_num,
            'date': execution_date,
            'withdrawal_amount': actual_withdrawal,
            'cumulative_withdrawn': self.total_withdrawn_amount
        })
        
        self.logger.trade(f"💸 Withdrawal: ₹{actual_withdrawal:,.0f}")
        self.logger.trade(f"   Total withdrawn so far: ₹{self.total_withdrawn_amount:,.0f}")
        
        # STEP 4: Calculate final state for return
        # NOTE: We do NOT append to portfolio_log here because the parent class's
        # run_backtest method already handles that (line 1696 in parent class).
        # Appending here would create duplicate entries (e.g., week 79 appearing twice).
        
        # Calculate new cash balance
        new_cash = cash + total_raised - reinvestment_amount - actual_withdrawal
        
        # Return result in parent's expected format
        # The parent class will append this to portfolio_log
        return {
            'action': 'churn',
            'ticker': 'N/A',  # Churning involves multiple tickers
            'total_raised': total_raised,
            'reinvestment_amount': reinvestment_amount,
            'withdrawal_amount': actual_withdrawal,
            'holdings': updated_holdings,
            'cash_after': new_cash,
            'nav': sum(updated_holdings.get(sym, 0) * open_prices.get(sym, 0) 
                      for sym in updated_holdings) + new_cash,
            'costs': {},  # Costs are in individual transactions
            'capital_gains_tax': sum(tx.get('capital_gains_tax', 0) for tx in sell_transactions),
            # Include detailed transaction info for API
            'sell_transactions': sell_transactions,
            'buy_transaction': buy_transaction,
            'withdrawal_info': {
                'withdrawal_amount': actual_withdrawal,
                'cumulative_withdrawn': self.total_withdrawn_amount
            }
        }
    
    def execute_sell_transaction(self, week_num: int, execution_date: datetime,
                                 ticker: str, units: int, price: float,
                                 brokerage_percent: float) -> Dict:
        """Execute a sell transaction with all costs"""
        try:
            sell_amount = units * price
            sell_costs = self.calculate_transaction_costs('sell', sell_amount, brokerage_percent)
            
            # Calculate capital gains tax (if applicable)
            capital_gains_result = self.calculate_capital_gains_tax(ticker, units, price, execution_date)
            
            # Extract tax amount - handle both dict and float returns
            if isinstance(capital_gains_result, dict):
                capital_gains_tax = capital_gains_result.get('total_tax', 0)
            else:
                capital_gains_tax = capital_gains_result if capital_gains_result else 0
            
            # Net proceeds = sell amount - costs - tax
            net_proceeds = sell_amount - sell_costs['total_costs'] - capital_gains_tax
            
            # Note: Parent class handles FIFO tracking internally
            # No need to manually remove purchase records
            
            # Log transaction
            self.log_transaction_costs(week_num, execution_date, 'sell', ticker,
                                      units, price, sell_costs, capital_gains_tax)
            
            return {
                'success': True,
                'ticker': ticker,
                'units': units,
                'price': price,
                'amount': sell_amount,
                'costs': sell_costs,
                'capital_gains_tax': capital_gains_tax,
                'net_proceeds': net_proceeds
            }
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.logger.error(f"Sell transaction error: {error_details}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def execute_buy_transaction(self, week_num: int, execution_date: datetime,
                                ticker: str, units: int, price: float,
                                brokerage_percent: float) -> Dict:
        """Execute a buy transaction with all costs"""
        try:
            buy_amount = units * price
            buy_costs = self.calculate_transaction_costs('buy', buy_amount, brokerage_percent)
            
            # Total cost = buy amount + transaction costs
            total_cost = buy_amount + buy_costs['total_costs']
            
            # Note: Parent class handles FIFO tracking internally
            # No need to manually add purchase records
            
            # Log transaction costs
            self.log_transaction_costs(week_num, execution_date, 'buy', ticker,
                                      units, price, buy_costs, 0)
            
            # NEW: Increment purchase counter
            if hasattr(self, 'etf_purchase_counts'):
                self.etf_purchase_counts[ticker] = self.etf_purchase_counts.get(ticker, 0) + 1
                current_count = self.etf_purchase_counts[ticker]
                limit = self.etf_purchase_limits.get(ticker, 0) if hasattr(self, 'etf_purchase_limits') else 0
                if hasattr(self, '_log'):
                    self._log("limit", f"📈 Purchase count updated: {ticker} = {current_count}/{limit}")
                
                # Print to terminal directly as requested
                print(f"📊 Purchase Count (Payout): {ticker} {current_count}/{limit}")
            
            return {
                'success': True,
                'ticker': ticker,
                'units': units,
                'price': price,
                'amount': buy_amount,
                'costs': buy_costs,
                'total_cost': total_cost
            }
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.logger.error(f"Buy transaction error: {error_details}")
            return {
                'success': False,
                'error': str(e)
            }


# Example usage
if __name__ == "__main__":
    # Create backtester instance
    backtester = RotationETFPayoutBacktester()
    
    # Run backtest with example parameters
    results = backtester.run_backtest(
        tickers=['NIFTYBEES', 'JUNIORBEES', 'BANKBEES', 'ITBEES', 'GOLDBEES'],
        start_date='2020-01-01',
        end_date='2024-12-31',
        capital_per_week=50000,
        accumulation_weeks=13,
        brokerage_percent=0.03,
        compounding_enabled=True
    )
    
    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)
    print(f"Success: {results.get('success', False)}")
    if not results.get('error'):
        print(f"Total Withdrawn: ₹{backtester.total_withdrawn_amount:,.0f}")
        print(f"Final NAV: ₹{results.get('final_nav', 0):,.0f}")

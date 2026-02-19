import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
import json
from Segments.EquitySegment import EquitySegment
from Strategies.utilities.logging_config import StrategyLogger

class ETFSwingStrategy(EquitySegment):
    """
    ETF-Based Equity Swing Trading Strategy
    
    Rules:
    - Entry: Close > SMA(50)
    - Ranking: (Close - SMA) / SMA (Lower positive distance preferred)
    - Exit (Profit): Close < SMA AND Profit >= Threshold
    - Stop Loss: Fixed % from Entry
    - Portfolio: Slot-based with dynamic capital reallocation
    """
    
    def __init__(self, config_path: str = None):
        super().__init__()
        self.strategy_name = "ETF_Swing_Strategy"
        self.logger = StrategyLogger(self.strategy_name)
        
        # Default Config
        self.config = {
            "logging": {"level": 3, "categories": {}},
            "parameters": {
                "sma_lookback": 50,
                "stop_loss_pct": 5.0,
                "profit_threshold_pct": 10.0,
                "number_of_slots": 5
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config.update(json.load(f))
        
        # Strategy Parameters
        params = self.config.get("parameters", {})
        self.sma_lookback = params.get("sma_lookback", 50)
        self.stop_loss_pct = params.get("stop_loss_pct", 5.0)
        self.profit_threshold_pct = params.get("profit_threshold_pct", 10.0)
        self.num_slots = params.get("number_of_slots", 5)
        
        # Logging Setup
        self.log_level = self.config.get("logging", {}).get("level", 3)
        self.log_categories = self.config.get("logging", {}).get("categories", {})
        
        # Portfolio State
        self.slots: List[Dict[str, Any]] = [{"id": i, "status": "FREE", "data": {}} for i in range(self.num_slots)]
        self.total_capital = 0.0
        self.available_cash = 0.0
        self.slot_capital = 0.0
        
        self._log(1, "system", f"Strategy Initialized: {self.strategy_name}")
        self._log(1, "system", f"Strategy Initialized: {self.strategy_name}")
        self.brokerage_percent = 0.0
        self._log(2, "system", f"Parameters: SMA={self.sma_lookback}, SL={self.stop_loss_pct}%, ProfitThr={self.profit_threshold_pct}%, Slots={self.num_slots}, Brokerage={self.brokerage_percent}%")

    def update_config(self, params: Dict[str, Any]):
        """Update strategy configuration dynamically and log changes"""
        updated = []
        
        if 'sma_lookback' in params:
            self.sma_lookback = int(params['sma_lookback'])
            updated.append(f"SMA={self.sma_lookback}")
            
        if 'stop_loss_pct' in params:
            self.stop_loss_pct = float(params['stop_loss_pct'])
            updated.append(f"SL={self.stop_loss_pct}%")
            
        if 'profit_threshold_pct' in params:
            self.profit_threshold_pct = float(params['profit_threshold_pct'])
            updated.append(f"ProfitThr={self.profit_threshold_pct}%")
            
        if 'number_of_slots' in params:
            new_slots = int(params['number_of_slots'])
            if new_slots != self.num_slots:
                self.num_slots = new_slots
                # Re-initialize slots
                self.slots = [{"id": i, "status": "FREE", "data": {}} for i in range(self.num_slots)]
                self._recalculate_slot_capital() # Update slot capital based on new count
                updated.append(f"Slots={self.num_slots} (Re-initialized)")
        
        if updated:
                self._recalculate_slot_capital() # Update slot capital based on new count
                updated.append(f"Slots={self.num_slots} (Re-initialized)")
        
        if 'brokerage_percent' in params:
            self.brokerage_percent = float(params['brokerage_percent'])
            updated.append(f"Brokerage={self.brokerage_percent}%")
        
        if updated:
            self._log(1, "system", f"Strategy Parameters Updated: {', '.join(updated)}")

    def _log(self, level: int, category: str, message: str):
        """Custom logging with levels 1-5 and categories"""
        if level <= self.log_level:
            if self.log_categories.get(category, True):
                # Map to StrategyLogger methods
                log_msg = f"[L{level}][{category}] {message}"
                if level >= 4: # Detailed/Debug
                    self.logger.debug(log_msg)
                elif level == 1: # Essential info (Not an error)
                    # Using info with a star for visibility instead of error
                    self.logger.info(f"[*] {log_msg}")
                elif category == "execution":
                    self.logger.trade(log_msg)
                elif category == "performance":
                    self.logger.performance(log_msg)
                else:
                    self.logger.info(log_msg)

    def load_data(self, start_date: datetime, end_date: datetime) -> Any:
        """Required abstract method implementation. Data loading is handled by the backtester."""
        pass

    def initialize_portfolio(self, initial_capital: float):
        """Initialize portfolio with starting capital divided into slots"""
        self.total_capital = initial_capital
        self.available_cash = initial_capital
        self.slot_capital = initial_capital / self.num_slots
        self._log(1, "execution", f"Portfolio Initialized with {initial_capital:,.2f}. Slot Capital: {self.slot_capital:,.2f}")

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate SMA for entry/exit signals"""
        self._log(4, "calculation", "Calculating SMA indicators")
        df = data.copy()
        df['sma'] = df['close'].rolling(window=self.sma_lookback).mean()
        df['distance_pct'] = (df['close'] - df['sma']) / df['sma'] * 100
        return df

    def evaluate_signals(self, symbol: str, df: pd.DataFrame, current_date: datetime) -> Dict[str, Any]:
        """Evaluate entry and ranking metrics for a given ETF with extreme verbose logging"""
        date_str = current_date.strftime('%Y-%m-%d')
        
        if df.empty or len(df) < self.sma_lookback:
            self._log(4, "signal", f"[{date_str}] {symbol}: Insufficient data for SMA (Need {self.sma_lookback}, Got {len(df)})")
            return {"eligible": False}
        
        # Get lookback window for SMA 50
        lookback_df = df.iloc[-self.sma_lookback:]
        lookback_start = lookback_df.index[0].strftime('%Y-%m-%d')
        lookback_end = lookback_df.index[-1].strftime('%Y-%m-%d')
        
        latest = df.iloc[-1]
        close = latest['close']
        sma = latest['sma']
        distance = latest['distance_pct']
        
        # LOGGING: Data Fetch Breakdown
        self._log(5, "data_fetch", "="*80)
        self._log(5, "data_fetch", f"[{date_str}] STEP 1: DATABASE DATA FETCH FOR {symbol}")
        self._log(5, "data_fetch", f"Ticker: {symbol} | Current Trade Date: {date_str}")
        self._log(5, "data_fetch", f"SMA Lookback Period: {self.sma_lookback} trading days")
        self._log(5, "data_fetch", f"Lookback Window: From {lookback_start} to {lookback_end}")
        self._log(5, "data_fetch", f"Sample Values from Database used for SMA:")
        # Show first 3 and last 3 closing prices in the window
        sample_prices = lookback_df['close'].tolist()
        self._log(5, "data_fetch", f"  First 3 Prices: {sample_prices[:3]}")
        self._log(5, "data_fetch", f"  Last 3 Prices: {sample_prices[-3:]}")
        
        # LOGGING: SMA Calculation Breakdown
        self._log(5, "calculation", f"[{date_str}] STEP 2: SMA {self.sma_lookback} CALCULATION FOR {symbol}")
        self._log(5, "calculation", f"Formula: SMA = (Sum of last {self.sma_lookback} Close Prices) / {self.sma_lookback}")
        self._log(5, "calculation", f"Calculation: Sum({lookback_df['close'].sum():.2f}) / {self.sma_lookback} = ₹{sma:.2f}")
        
        # LOGGING: Distance / Ranking Calculation Breakdown
        self._log(5, "calculation", f"[{date_str}] STEP 3: RANKING METRIC (DISTANCE %) FOR {symbol}")
        self._log(5, "calculation", f"Formula: Distance % = ((Current Close - SMA) / SMA) * 100")
        self._log(5, "calculation", f"Calculation: ((₹{close:.2f} - ₹{sma:.2f}) / ₹{sma:.2f}) * 100 = {distance:.4f}%")
        
        eligible = close > sma
        
        if eligible:
            self._log(1, "signal", f"[{date_str}] {symbol}: ELIGIBLE (Price {close:.2f} > SMA {sma:.2f}). Distance: {distance:.2f}% | Threshold: > 0%")
        else:
            self._log(4, "signal", f"[{date_str}] {symbol}: NOT ELIGIBLE (Price {close:.2f} < SMA {sma:.2f}). Distance: {distance:.2f}%")
        
        self._log(5, "calculation", "="*80)
        
        return {
            "symbol": symbol,
            "eligible": eligible,
            "close": close,
            "sma": sma,
            "distance": distance,
            "date": current_date
        }

    def process_exits(self, prices: Dict[str, float], current_date: datetime) -> List[Dict]:
        """Process stop loss and profit-based exits"""
        exits = []
        realized_something = False
        
        for slot in self.slots:
            if slot["status"] == "OCCUPIED":
                data = slot["data"]
                symbol = data["symbol"]
                entry_price = data["entry_price"]
                current_price = prices.get(symbol)
                
                if current_price is None:
                    continue
                
                # Check Stop Loss
                sl_price = entry_price * (1 - self.stop_loss_pct / 100)
                if current_price <= sl_price:
                    loss_pct = (current_price - entry_price) / entry_price * 100
                    reason_with_pct = f"STOP_LOSS ({loss_pct:.2f}%)"
                    self._log(1, "execution", f"STOP LOSS TRIGGERED: {symbol} at {current_price:.2f} (Entry: {entry_price:.2f}, SL: {sl_price:.2f})")
                    exits.append(self._execute_exit(slot, current_price, current_date, reason_with_pct))
                    realized_something = True
                    continue
                
                # Check Trend-Based Profit Exit
                sma_val = data.get("sma") # This might need updating with current SMA
                profit_pct = (current_price - entry_price) / entry_price * 100
                
                # Note: SMA should be recalculated for exit evaluation. 
                # For simplicity in this method, we assume SMA is passed or available.
                # In a real backtest, we'd have the latest SMA for each holding.
                
                # If we have current SMA for this symbol
                if "current_sma" in data and current_price < data["current_sma"]:
                    if profit_pct >= self.profit_threshold_pct:
                        reason_with_pct = f"PROFIT_EXIT ({profit_pct:.2f}%)"
                        self._log(1, "execution", f"PROFIT EXIT TRIGGERED: {symbol} at {current_price:.2f} (Profit: {profit_pct:.2f}%)")
                        exits.append(self._execute_exit(slot, current_price, current_date, reason_with_pct))
                        realized_something = True
                    else:
                        self._log(3, "signal", f"{symbol}: Trend violated (Price < SMA) but Profit ({profit_pct:.2f}%) below threshold. Holding.")
        
        if realized_something:
            self._recalculate_slot_capital()
            
        return exits

    def _execute_exit(self, slot: Dict, price: float, date: datetime, reason: str) -> Dict:
        """Execute exit and free slot"""
        data = slot["data"]
        symbol = data["symbol"]
        qty = data["qty"]
        amount = qty * price
        
        # Calculate costs
        costs = self.calculate_etf_delivery_costs('sell', amount, self.brokerage_percent)
        
        # Update FIFO inventory but ignore Capital Gains Tax as per request
        tax_info = self.calculate_capital_gains(symbol, qty, price, date)
        
        # Net Proceeds = (Amount - Transaction Costs) -> No Capital Gains Tax deduction
        net_proceeds = costs['net_amount'] 
        self.available_cash += net_proceeds
        
        log_entry = {
            "symbol": symbol,
            "action": "SELL",
            "reason": reason,
            "qty": qty,
            "price": price,
            "amount": amount,
            "costs": costs, # Store full costs dictionary
            "tax": 0.0, # Capital Gains Tax ignored
            "net_proceeds": net_proceeds,
            "execution_date": date, # Renamed for cache compatibility
            "date": date
        }
        
        # Enhanced Logging for Profit/Loss and Costs
        buy_price = data["entry_price"]
        pnl = amount - (qty * buy_price)
        pnl_pct = (pnl / (qty * buy_price)) * 100
        
        pnl_label = "PROFIT" if pnl >= 0 else "LOSS"
        
        self._log(1, "execution", f"[{date.strftime('%Y-%m-%d')}] [EXIT - SELL] Sold {symbol}. "
                                  f"Qty: {qty}, Price: {price:.2f}, "
                                  f"Buy Price: {buy_price:.2f}, PnL: {pnl:,.2f} ({pnl_pct:.2f}%)")
                                  
        # Calculate taxes (Total - Brokerage) since 'taxes' key doesn't exist in costs dict
        taxes = costs['total_costs'] - costs['brokerage']
        
        self._log(1, "execution", f"Transaction Costs for {symbol} (SELL): "
                                  f"Brokerage={costs['brokerage']:.2f}, "
                                  f"Taxes={taxes:.2f}, "
                                  f"Total Costs={costs['total_costs']:.2f}. "
                                  f"Net Proceeds: {net_proceeds:,.2f}")
        
        
        slot["status"] = "PENDING_FREE" # Mark as pending free to prevent same-day re-entry
        slot["data"] = {}
        
        return log_entry

    def _recalculate_slot_capital(self):
        """Update slot capital based on current available cash and free slots"""
        free_slots_count = sum(1 for s in self.slots if s["status"] == "FREE")
        if free_slots_count > 0:
            self.slot_capital = self.available_cash / free_slots_count
            self._log(2, "execution", f"RECALCULATING: Available Cash: {self.available_cash:,.2f}. New Slot Capital: {self.slot_capital:,.2f}")

    def process_entries(self, eligible_etfs: List[Dict], current_date: datetime) -> List[Dict]:
        """Rank and allocate free slots to eligible ETFs"""
        entries = []
        free_slots = [s for s in self.slots if s["status"] == "FREE"]
        
        if not free_slots or not eligible_etfs:
            return entries
            
        # Filter out ETFs already held
        held_symbols = [s["data"]["symbol"] for s in self.slots if s["status"] == "OCCUPIED"]
        eligible_new = [e for e in eligible_etfs if e["symbol"] not in held_symbols]
        
        if not eligible_new:
            return entries
            
        # Ranking Rule: ETFs with lower positive distance preferred
        eligible_new.sort(key=lambda x: x["distance"])
        
        # Allocate up to the number of free slots
        num_to_buy = min(len(free_slots), len(eligible_new))
        to_buy = eligible_new[:num_to_buy]
        
        for i, etf in enumerate(to_buy):
            slot = free_slots[i]
            symbol = etf["symbol"]
            price = etf["close"] # Execution at Day T+1 Open would be handled by backtester
            
            # Cost-Aware Quantity Calculation (Iterative)
            # 1. Start with max possible quantity based on price
            qty = int(self.slot_capital // price)
            
            # 2. Iteratively reduce quantity until Total Cost <= Slot Capital AND available cash
            # This loop typically runs 0-2 times.
            while qty > 0:
                amount = qty * price
                costs = self.calculate_etf_delivery_costs('buy', amount, self.brokerage_percent)
                total_outflow = costs['net_amount']
                
                # Check if we can afford it (both within slot allocation AND actual cash)
                # Note: self.slot_capital is a theoretical limit per slot. 
                # self.available_cash is the hard limit.
                if total_outflow <= self.slot_capital and total_outflow <= self.available_cash:
                    break # Affordability confirmed
                
                qty -= 1 # Reduce by 1 unit and re-check costs
            
            if qty > 0:
                # Recalculate final costs for the confirmed quantity
                amount = qty * price
                costs = self.calculate_etf_delivery_costs('buy', amount, self.brokerage_percent)
                
                self.available_cash -= costs['net_amount']
                
                slot["status"] = "OCCUPIED"
                slot["data"] = {
                    "symbol": symbol,
                    "qty": qty,
                    "entry_price": price,
                    "entry_date": current_date,
                    "sma": etf["sma"], # Original SMA
                    "current_sma": etf["sma"] # To be updated daily
                }
                
                self.manage_fifo_inventory(symbol, qty, price, current_date)
                
                entries.append({
                    "symbol": symbol,
                    "action": "BUY",
                    "qty": qty,
                    "price": price,
                    "amount": amount,
                    "costs": costs, # Store full costs dictionary
                    "execution_date": current_date, # Renamed for cache compatibility
                    "date": current_date
                })
                # Calculate taxes explicitly
                taxes = costs['total_costs'] - costs['brokerage']

                self._log(1, "execution", f"[{current_date.strftime('%Y-%m-%d')}] [ENTRY - BUY] Bought {symbol}. "
                                          f"Qty: {qty}, Price: {price:.2f}, Amount: {amount:,.2f}")
                self._log(1, "execution", f"Transaction Costs for {symbol} (BUY): "
                                          f"Brokerage={costs['brokerage']:.2f}, "
                                          f"Taxes={taxes:.2f}, "
                                          f"Total Costs={costs['total_costs']:.2f}. "
                                          f"Net Outflow: {costs['net_amount']:,.2f}")

        return entries

    def update_holding_sma(self, symbol: str, current_sma: float):
        """Update the latest SMA for an occupied slot"""
        for slot in self.slots:
            if slot["status"] == "OCCUPIED" and slot["data"]["symbol"] == symbol:
                slot["data"]["current_sma"] = current_sma
                break

    def finalize_daily_updates(self):
        """Finalize pending slot status changes at the end of the day"""
        for slot in self.slots:
            if slot["status"] == "PENDING_FREE":
                slot["status"] = "FREE"
                self._recalculate_slot_capital()

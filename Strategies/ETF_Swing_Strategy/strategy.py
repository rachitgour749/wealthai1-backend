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
    Generalized Swing Trading Strategy
    
    Now supports both Indian and US markets (Stocks & ETFs).
    """
    
    def __init__(self, market: str = "INDIA", asset_type: str = "ETF", config_path: str = None):
        # Initialize properly with injected policy
        if market.upper() == "US":
            from Exchange.USExchangePolicy import USExchangePolicy
            policy = USExchangePolicy()
        else:
            from Exchange.IndianExchangePolicy import IndianExchangePolicy
            policy = IndianExchangePolicy()
            
        super().__init__(policy=policy)
        self.market = market.upper()
        self.asset_type = asset_type.upper()
        self.strategy_name = f"{self.market}_{self.asset_type}_Swing_Strategy"
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
        self.brokerage_percent = float(params.get("brokerage_percent", 0.0))
        
        # Logging Setup
        self.log_level = self.config.get("logging", {}).get("level", 3)
        self.log_categories = self.config.get("logging", {}).get("categories", {})
        
        # Portfolio State
        self.total_capital = 0.0
        self._initialize_slots()

    def _initialize_slots(self):
        """Initialize slot structures based on num_slots"""
        self.slots: List[Dict[str, Any]] = [{"id": i, "status": "FREE", "data": {}} for i in range(self.num_slots)]

    def update_config(self, params: Dict[str, Any]):
        """Update strategy parameters dynamically from API request"""
        if "sma_lookback" in params: self.sma_lookback = int(params["sma_lookback"])
        if "stop_loss_pct" in params: self.stop_loss_pct = float(params["stop_loss_pct"])
        if "profit_threshold_pct" in params: self.profit_threshold_pct = float(params["profit_threshold_pct"])
        
        if "number_of_slots" in params or "num_slots" in params:
            new_slots = params.get("number_of_slots") or params.get("num_slots")
            self.num_slots = int(new_slots)
            self._initialize_slots()
            
        if "brokerage_percent" in params:
            self.brokerage_percent = float(params["brokerage_percent"])
            self._log(2, "system", f"Brokerage updated to {self.brokerage_percent}%")

        self._log(2, "system", f"Parameters updated: SMA={self.sma_lookback}, SL={self.stop_loss_pct}%, Profit={self.profit_threshold_pct}%, Slots={self.num_slots}, Brokerage={self.brokerage_percent}%")
        self.available_cash = 0.0
        self.slot_capital = 0.0
        
        self._log(1, "system", f"Strategy Initialized: {self.strategy_name} for {self.market} {self.asset_type}")
        # Removed hardcoded self.brokerage_percent = 0.0
        self._log(2, "system", f"Parameters: SMA={self.sma_lookback}, SL={self.stop_loss_pct}%, ProfitThr={self.profit_threshold_pct}%, Slots={self.num_slots}")

    def _log(self, level: int, category: str, message: str):
        """Standardized logger integration"""
        if level <= self.log_level:
            log_msg = f"[L{level}][{category}] {message}"
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
        self._log(5, "calculation", f"Calculation: Sum({lookback_df['close'].sum():.2f}) / {self.sma_lookback} = {self.policy.format_currency(sma)}")
        
        # LOGGING: Distance / Ranking Calculation Breakdown
        self._log(5, "calculation", f"[{date_str}] STEP 3: RANKING METRIC (DISTANCE %) FOR {symbol}")
        self._log(5, "calculation", f"Formula: Distance % = ((Current Close - SMA) / SMA) * 100")
        self._log(5, "calculation", f"Calculation: (({self.policy.format_currency(close)} - {self.policy.format_currency(sma)}) / {self.policy.format_currency(sma)}) * 100 = {distance:.4f}%")
        
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

    def process_exits(self, eval_prices: Dict[str, float], exec_prices: Dict[str, float], eval_date: datetime, exec_date: datetime) -> List[Dict]:
        """Process stop loss and profit-based exits"""
        exits = []
        realized_something = False
        
        for slot in self.slots:
            if slot["status"] == "OCCUPIED":
                data = slot["data"]
                symbol = data["symbol"]
                entry_price = data["entry_price"]
                eval_price = eval_prices.get(symbol)
                exec_price = exec_prices.get(symbol)
                
                if eval_price is None or exec_price is None:
                    continue
                
                # Check Stop Loss on eval_price (Today's Close)
                sl_price = entry_price * (1 - self.stop_loss_pct / 100)
                if eval_price <= sl_price:
                    # Loss pct calculated on execution price for actual result
                    loss_pct = (exec_price - entry_price) / entry_price * 100
                    reason_with_pct = f"STOP_LOSS ({loss_pct:.2f}%)"
                    self._log(1, "execution", f"STOP LOSS TRIGGERED: {symbol} at {eval_price:.2f} (Entry: {entry_price:.2f}, SL: {sl_price:.2f}). Executing at {exec_price:.2f} on {exec_date.strftime('%Y-%m-%d')}")
                    exits.append(self._execute_exit(slot, exec_price, exec_date, reason_with_pct))
                    realized_something = True
                    continue
                
                # Check Trend-Based Profit Exit on eval_price
                profit_pct = (eval_price - entry_price) / entry_price * 100
                
                if "current_sma" in data and eval_price < data["current_sma"]:
                    if profit_pct >= self.profit_threshold_pct:
                        real_profit_pct = (exec_price - entry_price) / entry_price * 100
                        reason_with_pct = f"PROFIT_EXIT ({real_profit_pct:.2f}%)"
                        self._log(1, "execution", f"PROFIT EXIT TRIGGERED: {symbol} at {eval_price:.2f} (Profit: {profit_pct:.2f}%). Executing at {exec_price:.2f} on {exec_date.strftime('%Y-%m-%d')}")
                        exits.append(self._execute_exit(slot, exec_price, exec_date, reason_with_pct))
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
                                  f"Qty: {qty}, Price: {self.policy.format_currency(price)}, "
                                  f"Buy Price: {self.policy.format_currency(buy_price)}, PnL: {self.policy.format_currency(pnl)} ({pnl_pct:.2f}%)")
                                  
        # Calculate taxes (Total - Brokerage) since 'taxes' key doesn't exist in costs dict
        taxes = costs['total_costs'] - costs['brokerage']
        
        self._log(1, "execution", f"Transaction Costs for {symbol} (SELL): "
                                  f"Brokerage={self.policy.format_currency(costs['brokerage'])}, "
                                  f"Taxes={self.policy.format_currency(taxes)}, "
                                  f"Total Costs={self.policy.format_currency(costs['total_costs'])}. "
                                  f"Net Proceeds: {self.policy.format_currency(net_proceeds)}")
        
        prev_cash = self.available_cash - net_proceeds
        self._log(1, "execution", f"--- CASH BREAKDOWN (SELL) ---")
        self._log(1, "execution", f"  Previous Cash: {self.policy.format_currency(prev_cash)}")
        self._log(1, "execution", f"  Net Proceeds: +{self.policy.format_currency(net_proceeds)}")
        self._log(1, "execution", f"  Calculation: {self.policy.format_currency(prev_cash)} + {self.policy.format_currency(net_proceeds)}")
        self._log(1, "execution", f"  Remaining Cash: {self.policy.format_currency(self.available_cash)}")
        self._log(1, "execution", f"-----------------------------")
        
        
        slot["status"] = "PENDING_FREE" # Mark as pending free to prevent same-day re-entry
        slot["last_symbol_info"] = {"symbol": symbol} # Track symbol for same-day re-entry prevention
        slot["data"] = {}
        
        return log_entry

    def _recalculate_slot_capital(self):
        """Update slot capital based on current available cash and free slots (including pending)"""
        # Include PENDING_FREE in the count because they represent capital that will be available
        free_slots_count = sum(1 for s in self.slots if s["status"] in ["FREE", "PENDING_FREE"])
        
        if free_slots_count > 0:
            self.slot_capital = self.available_cash / free_slots_count
            self._log(2, "execution", f"--- CAPITAL RECALCULATION ---")
            self._log(2, "execution", f"Available Cash: {self.policy.format_currency(self.available_cash)}")
            self._log(2, "execution", f"Free/Pending Slots: {free_slots_count}")
            self._log(2, "execution", f"Calculation: {self.policy.format_currency(self.available_cash)} / {free_slots_count} slots")
            self._log(2, "execution", f"New Slot Capital: {self.policy.format_currency(self.slot_capital)}")
            self._log(2, "execution", f"-----------------------------")

    def process_entries(self, eligible_etfs: List[Dict], current_date: datetime) -> List[Dict]:
        """Rank and allocate free slots to eligible ETFs"""
        entries = []
        free_slots = [s for s in self.slots if s["status"] == "FREE"]
        
        if not free_slots or not eligible_etfs:
            return entries
            
        # Filter out ETFs already held or just sold today (PENDING_FREE)
        # This prevents same-day re-entry after a Stop Loss exit
        held_symbols = []
        for s in self.slots:
            if s["status"] == "OCCUPIED":
                held_symbols.append(s["data"]["symbol"])
            elif s["status"] == "PENDING_FREE" and "symbol" in s.get("last_symbol_info", {}):
                # We need to track what was in the slot before it was marked PENDING_FREE
                held_symbols.append(s["last_symbol_info"]["symbol"])
        
        eligible_new = [e for e in eligible_etfs if e["symbol"] not in held_symbols]
        
        if not eligible_new:
            return entries
            
        # Ranking Rule: ETFs with lower positive distance preferred
        eligible_new.sort(key=lambda x: x["distance"])
        
        # Allocate up to the number of free slots
        num_to_buy = min(len(free_slots), len(eligible_new))
        to_buy = eligible_new[:num_to_buy]
        
        self._log(2, "execution", f"--- ENTRY PROCESSING ---")
        self._log(2, "execution", f"Available Cash: {self.policy.format_currency(self.available_cash)}")
        self._log(2, "execution", f"Total Free Slots: {len(free_slots)}")
        self._log(2, "execution", f"Slot Capital: {self.policy.format_currency(self.slot_capital)}")

        for i, etf in enumerate(to_buy):
            slot = free_slots[i]
            symbol = etf["symbol"]
            price = etf["close"] # Execution at Day T+1 Open would be handled by backtester
            
            self._log(2, "execution", f"Calculating Qty for {symbol}:")
            self._log(2, "execution", f"  Target Purchase Price: {self.policy.format_currency(price)}")
            self._log(2, "execution", f"  Budget (Min of Slot Cap or Avail Cash): {self.policy.format_currency(min(self.slot_capital, self.available_cash))}")

            # Cost-Aware Quantity Calculation
            # US market supports fractional shares; India uses whole units only.
            if self.market == "US":
                # Fractional: allocate full slot capital, account for costs
                # For US (zero costs), qty = slot_capital / price
                qty = round(min(self.slot_capital, self.available_cash) / price, 4)
                self._log(2, "execution", f"  US Market (Fractional): {min(self.slot_capital, self.available_cash):.2f} / {price:.2f} = {qty}")
            else:
                # India: whole units only — iterative floor approach
                qty_estimate = int(min(self.slot_capital, self.available_cash) // price)
                self._log(2, "execution", f"  India Market (Whole Units): Initial Estimate = {qty_estimate} units")
                
                qty = qty_estimate
                while qty > 0:
                    amount = qty * price
                    costs = self.calculate_etf_delivery_costs('buy', amount, self.brokerage_percent)
                    total_outflow = costs['net_amount']
                    if total_outflow <= self.slot_capital and total_outflow <= self.available_cash:
                        self._log(2, "execution", f"  Final Qty: {qty} (Total Outflow: {self.policy.format_currency(total_outflow)} fits budget)")
                        break
                    self._log(3, "execution", f"  Reducing Qty: {qty} too expensive (Outflow {self.policy.format_currency(total_outflow)} > Budget)")
                    qty -= 1
            
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
                                          f"Qty: {qty}, Price: {self.policy.format_currency(price)}, Amount: {self.policy.format_currency(amount)}")
                self._log(1, "execution", f"Transaction Costs for {symbol} (BUY): "
                                          f"Brokerage={self.policy.format_currency(costs['brokerage'])}, "
                                          f"Taxes={self.policy.format_currency(taxes)}, "
                                          f"Total Costs={self.policy.format_currency(costs['total_costs'])}. "
                                          f"Net Outflow: {self.policy.format_currency(costs['net_amount'])}")
                
                prev_cash = self.available_cash + costs['net_amount']
                self._log(1, "execution", f"--- CASH BREAKDOWN (BUY) ---")
                self._log(1, "execution", f"  Previous Cash: {self.policy.format_currency(prev_cash)}")
                self._log(1, "execution", f"  Net Outflow: -{self.policy.format_currency(costs['net_amount'])}")
                self._log(1, "execution", f"  Calculation: {self.policy.format_currency(prev_cash)} - {self.policy.format_currency(costs['net_amount'])}")
                self._log(1, "execution", f"  Remaining Cash: {self.policy.format_currency(self.available_cash)}")
                self._log(1, "execution", f"----------------------------")
            else:
                 self._log(1, "execution", f"[{current_date.strftime('%Y-%m-%d')}] [SKIPPED] Could not afford even 1 unit of {symbol}")

        self._log(2, "execution", f"-------------------------")
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
                if "last_symbol_info" in slot:
                    del slot["last_symbol_info"]
                self._recalculate_slot_capital()

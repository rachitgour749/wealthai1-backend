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
        self._log(2, "system", f"Parameters: SMA={self.sma_lookback}, SL={self.stop_loss_pct}%, ProfitThr={self.profit_threshold_pct}%, Slots={self.num_slots}")

    def _log(self, level: int, category: str, message: str):
        """Custom logging with levels 1-5 and categories"""
        if level <= self.log_level:
            if self.log_categories.get(category, True):
                if level >= 4: # Debug levels
                    self.logger.debug(f"[L{level}][{category}] {message}")
                elif level == 1: # Essential
                    self.logger.critical(f"[L{level}][{category}] {message}")
                else:
                    self.logger.info(f"[L{level}][{category}] {message}")

    def load_data(self, start_date: datetime, end_date: datetime) -> Any:
        """Required abstract method implementation. Data loading is handled by the backtester."""
        pass

    def initialize_portfolio(self, initial_capital: float):
        """Initialize portfolio with starting capital divided into slots"""
        self.total_capital = initial_capital
        self.available_cash = initial_capital
        self.slot_capital = initial_capital / self.num_slots
        self._log(1, "execution", f"Portfolio Initialized with ₹{initial_capital:,.2f}. Slot Capital: ₹{self.slot_capital:,.2f}")

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate SMA for entry/exit signals"""
        self._log(4, "calculation", "Calculating SMA indicators")
        df = data.copy()
        df['sma'] = df['close'].rolling(window=self.sma_lookback).mean()
        df['distance_pct'] = (df['close'] - df['sma']) / df['sma'] * 100
        return df

    def evaluate_signals(self, symbol: str, df: pd.DataFrame, current_date: datetime) -> Dict[str, Any]:
        """Evaluate entry and ranking metrics for a given ETF"""
        if df.empty or len(df) < self.sma_lookback:
            return {"eligible": False}
        
        latest = df.iloc[-1]
        close = latest['close']
        sma = latest['sma']
        distance = latest['distance_pct']
        
        eligible = close > sma
        
        if eligible:
            self._log(5, "signal", f"{symbol}: Eligible (Close={close:.2f} > SMA={sma:.2f}, Dist={distance:.2f}%)")
        
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
                    self._log(1, "execution", f"STOP LOSS TRIGGERED: {symbol} at ₹{current_price:.2f} (Entry: ₹{entry_price:.2f}, SL: ₹{sl_price:.2f})")
                    exits.append(self._execute_exit(slot, current_price, current_date, "STOP_LOSS"))
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
                        self._log(1, "execution", f"PROFIT EXIT TRIGGERED: {symbol} at ₹{current_price:.2f} (Profit: {profit_pct:.2f}%)")
                        exits.append(self._execute_exit(slot, current_price, current_date, "PROFIT_EXIT"))
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
        costs = self.calculate_etf_delivery_costs('sell', amount)
        tax_info = self.calculate_capital_gains(symbol, qty, price, date)
        
        net_proceeds = costs['net_amount'] - tax_info['capital_gains_tax']
        self.available_cash += net_proceeds
        
        log_entry = {
            "symbol": symbol,
            "action": "SELL",
            "reason": reason,
            "qty": qty,
            "price": price,
            "amount": amount,
            "costs": costs['total_costs'],
            "tax": tax_info['capital_gains_tax'],
            "net_proceeds": net_proceeds,
            "date": date
        }
        
        slot["status"] = "FREE"
        slot["data"] = {}
        
        return log_entry

    def _recalculate_slot_capital(self):
        """Update slot capital based on current available cash and free slots"""
        free_slots_count = sum(1 for s in self.slots if s["status"] == "FREE")
        if free_slots_count > 0:
            self.slot_capital = self.available_cash / free_slots_count
            self._log(2, "execution", f"RECALCULATING: Available Cash: ₹{self.available_cash:,.2f}. New Slot Capital: ₹{self.slot_capital:,.2f}")

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
            
            # Simple assumption: price is the execution price
            qty = int(self.slot_capital // price)
            
            if qty > 0:
                amount = qty * price
                costs = self.calculate_etf_delivery_costs('buy', amount)
                
                if self.available_cash >= costs['net_amount']:
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
                        "costs": costs['total_costs'],
                        "date": current_date
                    })
                    self._log(1, "execution", f"ENTRY: {symbol} - {qty} units at ₹{price:.2f}. Total: ₹{amount:,.2f}")

        return entries

    def update_holding_sma(self, symbol: str, current_sma: float):
        """Update the latest SMA for an occupied slot"""
        for slot in self.slots:
            if slot["status"] == "OCCUPIED" and slot["data"]["symbol"] == symbol:
                slot["data"]["current_sma"] = current_sma
                break

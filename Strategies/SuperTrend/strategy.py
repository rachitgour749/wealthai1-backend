import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os
import json
from Segments.EquitySegment import EquitySegment
from Strategies.utilities.logging_config import StrategyLogger

class SuperTrendStrategy(EquitySegment):
    """
    Weekly SuperTrend Strategy
    
    Compatible with US and Indian markets.
    """
    
    def __init__(self, market: str = "INDIA", asset_type: str = "STOCK", config_path: str = None):
        if market.upper() == "US":
            from Exchange.USExchangePolicy import USExchangePolicy
            policy = USExchangePolicy()
        else:
            from Exchange.IndianExchangePolicy import IndianExchangePolicy
            policy = IndianExchangePolicy()
            
        super().__init__(policy=policy)
        self.market = market.upper()
        self.asset_type = asset_type.upper()
        self.strategy_name = f"{self.market}_{self.asset_type}_SuperTrend"
        self.logger = StrategyLogger(self.strategy_name)
        
        # Default Config
        self.config = {
            "logging": {"level": 3, "categories": {}},
            "parameters": {
                "atr_multiplier": 3.0,
                "atr_period": 10,
                "stop_loss_pct": 5.0,
                "number_of_slots": 5,
                "brokerage_percent": 0.05
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config.update(json.load(f))
        
        # Strategy Parameters
        params = self.config.get("parameters", {})
        self.atr_multiplier = float(params.get("atr_multiplier", 3.0))
        self.atr_period = int(params.get("atr_period", 10))
        self.stop_loss_pct = float(params.get("stop_loss_pct", 5.0))
        self.num_slots = int(params.get("number_of_slots", 5))
        self.brokerage_percent = float(params.get("brokerage_percent", 0.05))
        
        # Logging Setup
        self.log_level = self.config.get("logging", {}).get("level", 3)
        
        # Portfolio State
        self.total_capital = 0.0
        self._initialize_slots()
        
        # Input Initialization Logs
        self._log(1, "system", f"=== Inputs Initialized for {self.strategy_name} ===")
        self._log(1, "system", f"ATR Multiplier: {self.atr_multiplier}")
        self._log(1, "system", f"ATR Period: {self.atr_period}")
        self._log(1, "system", f"Stop Loss %: {self.stop_loss_pct}")
        self._log(1, "system", f"Positions/Slots: {self.num_slots}")
        self._log(1, "system", f"Brokerage %: {self.brokerage_percent}")
        self._log(1, "system", f"================================================")

    def _initialize_slots(self):
        """Initialize slot structures based on num_slots"""
        self.slots: List[Dict[str, Any]] = [{"id": i, "status": "FREE", "data": {}} for i in range(self.num_slots)]

    def _log(self, level: int, category: str, message: str):
        if level <= self.log_level:
            log_msg = f"[{category.upper()}] {message}"
            self.logger.info(log_msg)

    def update_config(self, params: Dict[str, Any]):
        if "atr_multiplier" in params: self.atr_multiplier = float(params["atr_multiplier"])
        if "atr_period" in params: self.atr_period = int(params["atr_period"])
        if "stop_loss_pct" in params: self.stop_loss_pct = float(params["stop_loss_pct"])
        if "number_of_slots" in params: 
            self.num_slots = int(params["number_of_slots"])
            self._initialize_slots()
        if "brokerage_percent" in params: self.brokerage_percent = float(params["brokerage_percent"])

    def initialize_portfolio(self, initial_capital: float):
        """Capital calculation breakdown with formula"""
        self.total_capital = initial_capital
        self.available_cash = initial_capital
        self.slot_capital = initial_capital / self.num_slots
        
        self._log(1, "capital", f"=== Portfolio Capital Init ===")
        self._log(1, "capital", f"Total Capital: {self.policy.format_currency(initial_capital)}")
        self._log(1, "capital", f"Positions/Slots: {self.num_slots}")
        self._log(1, "capital", f"Formula: Slot Capital = Total Capital / Slots")
        self._log(1, "capital", f"Calculation: {initial_capital} / {self.num_slots} = {self.slot_capital}")
        self._log(1, "capital", f"Allocated Slot Capital: {self.policy.format_currency(self.slot_capital)}")
        self._log(1, "capital", f"==============================")

    def load_data(self, start_date: datetime, end_date: datetime) -> Any:
        """Required abstract method implementation. Data loading is handled by the backtester."""
        pass

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Weekly SuperTrend."""
        if data.empty:
            return data
            
        df = data.copy()
        
        # We need a week column to group by. Resample to weekly ('W-FRI')
        df.index = pd.to_datetime(df.index)
        
        # Need weekly high, low, close.
        # Ensure we sort by index
        df = df.sort_index()
        weekly_df = df.resample('W-FRI').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        if len(weekly_df) < self.atr_period:
            return df
            
        # Calculate Supertrend
        # pandas_ta supertrend returns DataFrame with columns like:
        # SUPERT_10_3.0, SUPERTd_10_3.0, SUPERTl_10_3.0, SUPERTs_10_3.0
        st_res = ta.supertrend(high=weekly_df['high'], low=weekly_df['low'], close=weekly_df['close'], period=self.atr_period, multiplier=self.atr_multiplier)
        
        if st_res is not None and not st_res.empty:
            # Get the exact column names. Format is usually SUPERT_period_multiplier
            st_col = next(col for col in st_res.columns if col.startswith('SUPERT_'))
            dir_col = next(col for col in st_res.columns if col.startswith('SUPERTd_'))
            weekly_df['supertrend'] = st_res[st_col]
            weekly_df['st_direction'] = st_res[dir_col] # 1 is uptrend, -1 is downtrend
        else:
            weekly_df['supertrend'] = np.nan
            weekly_df['st_direction'] = np.nan
        
        # Merge back to daily using forward fill, so we can use daily prices with weekly ST
        # But signals are ONLY based on Last Trading Day of Week. We will mark them.
        weekly_df['is_last_day_of_week'] = True
        
        # Create mapping of weekly ST to daily DataFrame index
        # We'll reindex or merge
        merged_df = df.join(weekly_df[['supertrend', 'st_direction', 'is_last_day_of_week']], how='left')
        
        # Since not every week ends exactly on Friday (holidays), let's find the actual last trading day per week
        df['year_week'] = df.index.isocalendar().year.astype(str) + '_' + df.index.isocalendar().week.astype(str)
        # Group by week, get the last date
        last_days = df.groupby('year_week').apply(lambda x: x.index[-1]).values
        
        df['is_last_day_of_week'] = False
        df.loc[df.index.isin(last_days), 'is_last_day_of_week'] = True
        
        # Re-calc Weekly Data strictly based on exact trading days available to be precise
        # This approach ensures holiday logic works
        
        weekly_precise = df.groupby('year_week').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
        
        st_precise = ta.supertrend(high=weekly_precise['high'], low=weekly_precise['low'], close=weekly_precise['close'], period=self.atr_period, multiplier=self.atr_multiplier)
        
        if st_precise is not None and not st_precise.empty:
            st_col = next(col for col in st_precise.columns if col.startswith('SUPERT_'))
            dir_col = next(col for col in st_precise.columns if col.startswith('SUPERTd_'))
            weekly_precise['supertrend'] = st_precise[st_col]
            weekly_precise['st_direction'] = st_precise[dir_col]
        else:
            weekly_precise['supertrend'] = np.nan
            weekly_precise['st_direction'] = np.nan
            
        # Map back to daily dataframe based on year_week
        df = df.join(weekly_precise[['supertrend', 'st_direction']], on='year_week')
        
        self._log(4, "calculation", f"SuperTrend calculated. ATR Period: {self.atr_period}, Multiplier: {self.atr_multiplier}")
        
        return df

    def evaluate_signals(self, symbol: str, df: pd.DataFrame, current_date: datetime) -> Dict[str, Any]:
        """Evaluate if today is the exact end of week and there is a breakout"""
        date_str = current_date.strftime('%Y-%m-%d')
        
        if current_date not in df.index:
            return {"eligible": False}
            
        row = df.loc[current_date]
        
        if not row['is_last_day_of_week']:
            return {"eligible": False} # Only end of week gets signals
            
        if pd.isna(row['supertrend']):
            return {"eligible": False}
            
        close = row['close']
        st_value = row['supertrend']
        
        # Breakout: Close > Supertrend
        eligible = row['st_direction'] == 1 and close > st_value
        
        if eligible:
            self._log(1, "signal", f"[{date_str}] SIGNAL GENERATED: {symbol} Buy Signal Triggered!")
            self._log(1, "signal", f"  Weekly Supertrend Calculation used: ATR({self.atr_period}) * {self.atr_multiplier}")
            self._log(1, "signal", f"  Formula: Breakout = Close > SuperTrend")
            self._log(1, "signal", f"  Breakout Values: Close ({close:.2f}) > ST ({st_value:.2f})")
            
        return {
            "symbol": symbol,
            "eligible": eligible,
            "close": close,
            "supertrend": st_value,
            "date": current_date # Signal generation date
        }

    def process_exits(self, eval_prices: Dict[str, float], exec_prices: Dict[str, float], dfs: Dict[str, pd.DataFrame], current_date: datetime) -> List[Dict]:
        """Process weekly stoploss based on open price AND trend breakdown"""
        exits = []
        realized_something = False
        
        # First check if we have data for today
        if not dfs: return exits
        
        date_str = current_date.strftime('%Y-%m-%d')
        
        for slot in self.slots:
            if slot["status"] == "OCCUPIED":
                data = slot["data"]
                symbol = data["symbol"]
                
                if symbol not in dfs or current_date not in dfs[symbol].index:
                    continue
                    
                row = dfs[symbol].loc[current_date]
                entry_price = data["entry_price"]
                
                # Check for Weekly Stop Loss based on OPEN Price
                # "stoploss also check on weekly basic on open price" -> If it's Monday Open
                # Let's see if this is the first day of the week
                year_week = dfs[symbol].loc[current_date, 'year_week']
                first_day_of_week = dfs[symbol][dfs[symbol]['year_week'] == year_week].index[0]
                
                exec_price = exec_prices.get(symbol, row['close'])
                
                if current_date == first_day_of_week:
                    open_price = row['open']
                    sl_price = entry_price * (1 - self.stop_loss_pct / 100)
                    if open_price <= sl_price:
                        loss_pct = (open_price - entry_price) / entry_price * 100
                        reason = f"STOP_LOSS_OPEN (-{abs(loss_pct):.2f}%)"
                        self._log(1, "execution", f"[{date_str}] Weekly Stop Loss on Open triggered for {symbol}: Open {open_price:.2f} <= SL {sl_price:.2f}")
                        
                        # Execute immediately at open price
                        exits.append(self._execute_exit(slot, open_price, current_date, reason))
                        realized_something = True
                        continue
                
                # Check Trend Breakdown: last trading day of week, Check Close < Supertrend
                if row['is_last_day_of_week']:
                    close_price = row['close']
                    st_value = row['supertrend']
                    
                    if not pd.isna(st_value) and row['st_direction'] == -1 and close_price < st_value:
                        # Breakdown confirmed, exit on NEXT open (or execution price provided)
                        self._log(1, "signal", f"[{date_str}] SIGNAL GENERATED: {symbol} Sell Signal Triggered!")
                        self._log(1, "signal", f"  Formula: Breakdown = Close < SuperTrend")
                        self._log(1, "signal", f"  Breakdown Values: Close ({close_price:.2f}) < ST ({st_value:.2f})")
                        
                        # Note: Execution is typically next day open for these signals. We'll simulate execution at exec_price.
                        
                        loss_pct = (exec_price - entry_price) / entry_price * 100
                        reason = f"TREND_BREAKDOWN ({loss_pct:.2f}%)"
                        exits.append(self._execute_exit(slot, exec_price, current_date + timedelta(days=1), reason))
                        realized_something = True
                        continue

        if realized_something:
            self._recalculate_slot_capital()
            
        return exits

    def _execute_exit(self, slot: Dict, price: float, date: datetime, reason: str) -> Dict:
        data = slot["data"]
        symbol = data["symbol"]
        qty = data["qty"]
        amount = qty * price
        
        # Calculate costs: assuming ETF/Stock delivery
        costs = self.calculate_etf_delivery_costs('sell', amount, self.brokerage_percent)
        
        # Tax ignored as per rules: NO CAPITAL GAIN
        net_proceeds = costs['net_amount']
        self.available_cash += net_proceeds
        
        log_entry = {
            "symbol": symbol, "action": "SELL", "reason": reason,
            "qty": qty, "price": price, "amount": amount,
            "costs": costs, "tax": 0.0, "net_proceeds": net_proceeds,
            "execution_date": date, "date": date
        }
        
        buy_price = data["entry_price"]
        pnl = amount - (qty * buy_price)
        
        self._log(1, "execution", f"[{date.strftime('%Y-%m-%d')}] EXECUTION: SELL {symbol} | Reason: {reason}")
        self._log(1, "capital", f"--- CASH BREAKDOWN (SELL) ---")
        prev_cash = self.available_cash - net_proceeds
        self._log(1, "capital", f"  Previous Cash: {self.policy.format_currency(prev_cash)}")
        self._log(1, "capital", f"  Gross Amount: {self.policy.format_currency(amount)}")
        self._log(1, "capital", f"  Transaction Costs: -{self.policy.format_currency(costs['total_costs'])}")
        self._log(1, "capital", f"  Net Proceeds: {self.policy.format_currency(net_proceeds)}")
        self._log(1, "capital", f"  Calculation: {self.policy.format_currency(prev_cash)} + {self.policy.format_currency(net_proceeds)}")
        self._log(1, "capital", f"  Remaining Cash: {self.policy.format_currency(self.available_cash)}")
        self._log(1, "capital", f"-----------------------------")
        
        slot["status"] = "PENDING_FREE"
        slot["last_symbol_info"] = {"symbol": symbol}
        slot["data"] = {}
        
        return log_entry

    def _recalculate_slot_capital(self):
        free_slots = sum(1 for s in self.slots if s["status"] in ["FREE", "PENDING_FREE"])
        if free_slots > 0:
            self.slot_capital = self.available_cash / free_slots

    def process_entries(self, eligible_stocks: List[Dict], execution_date: datetime, exec_prices: Dict[str, float]) -> List[Dict]:
        entries = []
        free_slots = [s for s in self.slots if s["status"] == "FREE"]
        
        if not free_slots or not eligible_stocks: return entries
        
        held_symbols = []
        for s in self.slots:
            if s["status"] == "OCCUPIED": held_symbols.append(s["data"]["symbol"])
            elif s["status"] == "PENDING_FREE" and "symbol" in s.get("last_symbol_info", {}):
                held_symbols.append(s["last_symbol_info"]["symbol"])
                
        eligible_new = [e for e in eligible_stocks if e["symbol"] not in held_symbols]
        if not eligible_new: return entries
        
        to_buy = eligible_new[:len(free_slots)]
        
        for i, stock in enumerate(to_buy):
            slot = free_slots[i]
            symbol = stock["symbol"]
            signal_date = stock["date"]
            
            # Exec price should be Monday open or similar, derived from backtester
            price = exec_prices.get(symbol, stock["close"])
            
            qty = int(min(self.slot_capital, self.available_cash) // price)
            
            while qty > 0:
                amount = qty * price
                costs = self.calculate_etf_delivery_costs('buy', amount, self.brokerage_percent)
                if costs['net_amount'] <= self.slot_capital and costs['net_amount'] <= self.available_cash:
                    break
                qty -= 1
                
            if qty > 0:
                amount = qty * price
                costs = self.calculate_etf_delivery_costs('buy', amount, self.brokerage_percent)
                self.available_cash -= costs['net_amount']
                
                slot["status"] = "OCCUPIED"
                slot["data"] = {
                    "symbol": symbol, "qty": qty, "entry_price": price,
                    "entry_date": execution_date, "signal_date": signal_date
                }
                
                self.manage_fifo_inventory(symbol, qty, price, execution_date)
                
                entries.append({
                    "symbol": symbol, "action": "BUY", "qty": qty,
                    "price": price, "amount": amount, "costs": costs,
                    "execution_date": execution_date, "date": execution_date
                })
                
                date_str = execution_date.strftime('%Y-%m-%d')
                sig_date_str = signal_date.strftime('%Y-%m-%d')
                
                self._log(1, "execution", f"[{date_str}] EXECUTION: BUY {symbol}")
                self._log(1, "execution", f"  Signal Generated: {sig_date_str}")
                self._log(1, "execution", f"  Execution Completed: {date_str} at Price {price:.2f}")
                
                self._log(1, "capital", f"--- CASH BREAKDOWN (BUY) ---")
                prev_cash = self.available_cash + costs['net_amount']
                self._log(1, "capital", f"  Previous Cash: {self.policy.format_currency(prev_cash)}")
                self._log(1, "capital", f"  Gross Amount: {self.policy.format_currency(amount)}")
                self._log(1, "capital", f"  Transaction Costs: {self.policy.format_currency(costs['total_costs'])}")
                self._log(1, "capital", f"  Net Outflow: -{self.policy.format_currency(costs['net_amount'])}")
                self._log(1, "capital", f"  Calculation: {self.policy.format_currency(prev_cash)} - {self.policy.format_currency(costs['net_amount'])}")
                self._log(1, "capital", f"  Remaining Cash: {self.policy.format_currency(self.available_cash)}")
                self._log(1, "capital", f"----------------------------")

        return entries

    def finalize_daily_updates(self):
        for slot in self.slots:
            if slot["status"] == "PENDING_FREE":
                slot["status"] = "FREE"
                if "last_symbol_info" in slot: del slot["last_symbol_info"]
                self._recalculate_slot_capital()

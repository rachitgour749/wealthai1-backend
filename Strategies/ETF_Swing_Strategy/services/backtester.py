import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os
import json
from sqlalchemy import text
from Databases.app_data_db_connection import get_session
from Strategies.ETF_Swing_Strategy.strategy import ETFSwingStrategy

class ETFSwingBacktester:
    def __init__(self, config_path: str = None):
        self.strategy = ETFSwingStrategy(config_path)
        self._data_cache = {}
        self.portfolio_history = []
        self.transaction_log = []
        self.daily_nav = pd.DataFrame()

    def load_data(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Load data from etf_market table"""
        session = get_session()
        data_dict = {}
        
        # Add buffer for SMA
        buffer_days = self.strategy.sma_lookback * 2
        adj_start = (pd.to_datetime(start_date) - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
        
        self.strategy._log(2, "data_fetch", f"Loading data for {len(tickers)} ETFs from {adj_start} to {end_date}")
        
        try:
            for ticker in tickers:
                query = text(f"""
                    SELECT date, open, high, low, close, volume 
                    FROM etf_market 
                    WHERE symbol = :symbol 
                    AND date >= :start AND date <= :end
                    ORDER BY date ASC
                """)
                result = session.execute(query, {"symbol": ticker, "start": adj_start, "end": end_date})
                rows = result.fetchall()
                
                if rows:
                    df = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    data_dict[ticker] = df
                    self.strategy._log(5, "data_fetch", f"Loaded {len(df)} records for {ticker}")
                else:
                    self.strategy._log(3, "data_fetch", f"No data found for {ticker}")
                    
            return data_dict
        finally:
            session.close()

    def run_backtest(self, tickers: List[str], start_date: str, end_date: str, initial_capital: float):
        """Run the backtest loop"""
        self.strategy._log(1, "execution", f"Starting Backtest: {start_date} to {end_date}")
        
        # Load Data
        all_data = self.load_data(tickers, start_date, end_date)
        if not all_data:
            return {"error": "No data found for the selected ETFs"}

        # Calculate Indicators for all ETFs
        processed_data = {}
        for ticker, df in all_data.items():
            processed_data[ticker] = self.strategy.calculate_indicators(df)

        # Get Common Trading Dates
        trading_dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        self.strategy.initialize_portfolio(initial_capital)
        self.portfolio_history = []
        self.transaction_log = []

        for current_date in trading_dates:
            # 1. Update Holding SMAs for exit evaluation
            prices_at_close = {}
            for slot in self.strategy.slots:
                if slot["status"] == "OCCUPIED":
                    symbol = slot["data"]["symbol"]
                    if symbol in processed_data and current_date in processed_data[symbol].index:
                        current_row = processed_data[symbol].loc[current_date]
                        self.strategy.update_holding_sma(symbol, current_row['sma'])
                        prices_at_close[symbol] = current_row['close']

            # 2. Process Exits (Evaluated at Close of Day T, executed at Open of Day T+1)
            # For backtest simplification, we execute at current day's close or next day's open
            # BRD: Signal Day (T) Close evaluations. Execution Day (T+1) Open.
            
            # Find next trading day for execution
            next_date = current_date + timedelta(days=1)
            # Find actual next available data point
            
            # Simplified for now: use T+1 open if available, else T close
            execution_prices = {}
            for ticker, df in processed_data.items():
                if next_date in df.index:
                    execution_prices[ticker] = df.loc[next_date]['open']
                elif current_date in df.index:
                    execution_prices[ticker] = df.loc[current_date]['close']

            exits = self.strategy.process_exits(execution_prices, current_date)
            self.transaction_log.extend(exits)

            # 3. Evaluate Entry Signals (Close of Day T)
            eligible_etfs = []
            for ticker, df in processed_data.items():
                if current_date in df.index:
                    signal = self.strategy.evaluate_signals(ticker, df.loc[:current_date], current_date)
                    if signal.get("eligible"):
                        # Use T+1 Open as the candidate execution price if available
                        if next_date in df.index:
                            signal["close"] = df.loc[next_date]['open'] 
                        eligible_etfs.append(signal)

            # 4. Process Entries
            entries = self.strategy.process_entries(eligible_etfs, current_date)
            self.transaction_log.extend(entries)

            # 5. Record NAV
            current_value = self.strategy.available_cash
            for slot in self.strategy.slots:
                if slot["status"] == "OCCUPIED":
                    symbol = slot["data"]["symbol"]
                    qty = slot["data"]["qty"]
                    if symbol in processed_data and current_date in processed_data[symbol].index:
                        current_value += qty * processed_data[symbol].loc[current_date]['close']
            
            self.portfolio_history.append({
                "date": current_date,
                "nav": current_value,
                "cash": self.strategy.available_cash,
                "holdings": sum(1 for s in self.slots if s["status"] == "OCCUPIED")
            })

        self.daily_nav = pd.DataFrame(self.portfolio_history)
        self.strategy._log(1, "performance", "Backtest Completed.")
        
        return self.calculate_results()

    def calculate_results(self):
        """Calculate final metrics"""
        if self.daily_nav.empty:
            return {"error": "No data recorded"}
            
        initial_val = self.daily_nav['nav'].iloc[0]
        final_val = self.daily_nav['nav'].iloc[-1]
        total_return = (final_val - initial_val) / initial_val * 100
        
        # CAGR
        days = (self.daily_nav['date'].iloc[-1] - self.daily_nav['date'].iloc[0]).days
        years = days / 365.25
        cagr = ((final_val / initial_val) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Max Drawdown
        self.daily_nav['peak'] = self.daily_nav['nav'].cummax()
        self.daily_nav['drawdown'] = (self.daily_nav['nav'] - self.daily_nav['peak']) / self.daily_nav['peak'] * 100
        max_dd = self.daily_nav['drawdown'].min()
        
        metrics = {
            "Total Return (%)": round(total_return, 2),
            "CAGR (%)": round(cagr, 2),
            "Max Drawdown (%)": round(max_dd, 2),
            "Final Capital": round(final_val, 2),
            "Total Trades": len(self.transaction_log)
        }
        
        self.strategy._log(1, "performance", f"Metrics: {metrics}")
        
        return {
            "metrics": metrics,
            "performance_data": self.daily_nav.to_dict('records'),
            "transaction_log": self.transaction_log
        }

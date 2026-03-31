import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os
import json
from sqlalchemy import text
from Databases.app_data_db_connection import get_session
<<<<<<< HEAD
from Services.market_data_service import MarketDataService
from Strategies.ETF_Swing_Strategy.strategy import ETFSwingStrategy

class ETFSwingBacktester:
    def __init__(self, market: str = "INDIA", asset_type: str = "ETF", config_path: str = None):
        self.market = market.upper()
        self.asset_type = asset_type.upper()
        self.strategy = ETFSwingStrategy(market=self.market, asset_type=self.asset_type, config_path=config_path)
=======
from Strategies.ETF_Swing_Strategy.strategy import ETFSwingStrategy

class ETFSwingBacktester:
    def __init__(self, config_path: str = None):
        self.strategy = ETFSwingStrategy(config_path)
>>>>>>> feature/chatai
        self._data_cache = {}
        self.portfolio_history = []
        self.transaction_log = []
        self.daily_nav = pd.DataFrame()

    def load_data(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
<<<<<<< HEAD
        """Load data using MarketDataService"""
=======
        """Load data from etf_market table"""
        session = get_session()
>>>>>>> feature/chatai
        data_dict = {}
        
        # Add buffer for SMA
        buffer_days = self.strategy.sma_lookback * 2
<<<<<<< HEAD
        adj_start = pd.to_datetime(start_date) - timedelta(days=buffer_days)
        adj_end = pd.to_datetime(end_date)
        
        self.strategy._log(2, "data_fetch", f"Loading {self.market} {self.asset_type} data for {len(tickers)} assets and Benchmark from {adj_start.date()} to {adj_end.date()}")
        
        try:
            # 1. Load Asset Data
            assets_df = MarketDataService.fetch_close_prices(
                tickers=tickers, market=self.market, asset_type=self.asset_type,
                start_date=adj_start, end_date=adj_end
            )
            
            if not assets_df.empty:
                for ticker in tickers:
                    if ticker in assets_df.columns:
                        ticker_df = assets_df[[ticker]].rename(columns={ticker: 'close'})
                        for col in ['open', 'high', 'low']: ticker_df[col] = ticker_df['close']
                        ticker_df['volume'] = 0
                        data_dict[ticker] = ticker_df
                        self.strategy._log(2, "data_fetch", f"Fetched {len(ticker_df)} records for {ticker}.")

            # 2. Load Benchmark Data
            benchmark_candidates = ["NIFTY_50", "NIFTYBEES", "NIFTY50"] if self.market == "INDIA" else ["^GSPC", "S&P_500", "SPY", "SPX"]
            
            bench_df = MarketDataService.fetch_close_prices(
                tickers=benchmark_candidates, market=self.market, asset_type="INDEX",
                start_date=adj_start, end_date=adj_end
            )
            
            if not bench_df.empty:
                # Find which candidate actually returned data
                benchmark_symbol = None
                for candidate in benchmark_candidates:
                    if candidate in bench_df.columns and not bench_df[candidate].dropna().empty:
                        benchmark_symbol = candidate
                        break
                
                if benchmark_symbol:
                    bench_data = bench_df[[benchmark_symbol]].rename(columns={benchmark_symbol: 'close'})
                    for col in ['open', 'high', 'low']: bench_data[col] = bench_data['close']
                    bench_data['volume'] = 0
                    data_dict["BENCHMARK"] = bench_data
                    self.strategy._log(2, "data_fetch", f"Fetched benchmark {benchmark_symbol}.")
                else:
                    self.strategy._log(1, "data_fetch", f"WARNING: None of the benchmark candidates {benchmark_candidates} found in data!")
            else:
                self.strategy._log(1, "data_fetch", f"WARNING: Benchmark candidates {benchmark_candidates} not found!")
                    
            return data_dict
        finally:
            pass

    def run_backtest(self, tickers: List[str], start_date: str, end_date: str, initial_capital: float, risk_free_rate: float = 8.0):
        """Run the backtest loop with benchmark comparison"""
=======
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
>>>>>>> feature/chatai
        self.strategy._log(1, "execution", f"Starting Backtest: {start_date} to {end_date}")
        
        # Load Data
        all_data = self.load_data(tickers, start_date, end_date)
        if not all_data:
<<<<<<< HEAD
            return {"error": f"No data found for any of the requested tickers: {tickers}"}
            
        if "BENCHMARK" not in all_data:
            market_bench = "NIFTY 50" if self.market == "INDIA" else "S&P 500"
            return {"error": f"Missing benchmark data for {market_bench}. Check index market tables."}

        benchmark_df = all_data.pop("BENCHMARK")

        # Normalize all tickers to the benchmark trading dates 
        # This prevents "spikes" to zero when an ETF has a missing data point
        benchmark_dates = benchmark_df.index
        
        processed_data = {}
        insufficient_data_tickers = []
        for ticker, df in all_data.items():
            # Reindex to benchmark dates and forward-fill missing values
            df = df.reindex(benchmark_dates).ffill().bfill()
            
            # Check if enough data exists for SMA calculation for this specific ticker
            if df.empty or len(df.dropna()) < self.strategy.sma_lookback:
                available = len(df.dropna()) if not df.empty else 0
                insufficient_data_tickers.append(f"{ticker} ({available} days)")
            
            processed_data[ticker] = self.strategy.calculate_indicators(df)
        
        if insufficient_data_tickers:
            tickers_str = ", ".join(insufficient_data_tickers)
            error_msg = f"Insufficient data for SMA. Your SMA Lookback is set to {self.strategy.sma_lookback}, but the following tickers have less data: {tickers_str}. Please select a shorter SMA lookback or exclude these tickers."
            self.strategy._log(1, "system", f"ERROR: {error_msg}")
            return {"error": error_msg}

        self.strategy._log(1, "data_fetch", f"Total ETFs in Universe for Backtest: {len(processed_data)}")

        # Get Common Trading Dates (Intersects with Benchmark)
        trading_dates = benchmark_dates[benchmark_dates.slice_indexer(start_date, end_date)]
=======
            return {"error": "No data found for the selected ETFs"}

        # Calculate Indicators for all ETFs
        processed_data = {}
        for ticker, df in all_data.items():
            processed_data[ticker] = self.strategy.calculate_indicators(df)

        # Get Common Trading Dates
        trading_dates = pd.date_range(start=start_date, end=end_date, freq='B')
>>>>>>> feature/chatai
        
        self.strategy.initialize_portfolio(initial_capital)
        self.portfolio_history = []
        self.transaction_log = []

<<<<<<< HEAD
        # Benchmark Initial State (Buy & Hold Nifty 50)
        nifty_initial_price = benchmark_df.loc[trading_dates[0], 'open']
        nifty_qty = initial_capital / nifty_initial_price

        for current_date in trading_dates:
            # Log Slot Usage
            occupied_slots = sum(1 for s in self.strategy.slots if s["status"] == "OCCUPIED")
            self.strategy._log(1, "execution", f"[{current_date.strftime('%Y-%m-%d')}] SLOTS: {occupied_slots} Occupied / {self.strategy.num_slots} Total")

=======
        for current_date in trading_dates:
>>>>>>> feature/chatai
            # 1. Update Holding SMAs for exit evaluation
            prices_at_close = {}
            for slot in self.strategy.slots:
                if slot["status"] == "OCCUPIED":
                    symbol = slot["data"]["symbol"]
                    if symbol in processed_data and current_date in processed_data[symbol].index:
                        current_row = processed_data[symbol].loc[current_date]
                        self.strategy.update_holding_sma(symbol, current_row['sma'])
                        prices_at_close[symbol] = current_row['close']

<<<<<<< HEAD
            # 2. Process Exits (Signal T -> Execution T+1 Close)
            # We look at today's signals, executed at next day's close
            next_date_idx = benchmark_df.index.get_loc(current_date) + 1
            execution_date = current_date
            execution_prices = {}
            
            if next_date_idx < len(benchmark_df):
                next_date = benchmark_df.index[next_date_idx]
                execution_date = next_date # Use Next Day for Execution Date
                for ticker, df in processed_data.items():
                    if next_date in df.index:
                        execution_prices[ticker] = df.loc[next_date]['close']
            else:
                # Last day fallback
                for ticker, df in processed_data.items():
                    if current_date in df.index:
                        execution_prices[ticker] = df.loc[current_date]['close']

            exits = self.strategy.process_exits(
                eval_prices=prices_at_close,
                exec_prices=execution_prices,
                eval_date=current_date,
                exec_date=execution_date
            )
=======
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
>>>>>>> feature/chatai
            self.transaction_log.extend(exits)

            # 3. Evaluate Entry Signals (Close of Day T)
            eligible_etfs = []
            for ticker, df in processed_data.items():
                if current_date in df.index:
                    signal = self.strategy.evaluate_signals(ticker, df.loc[:current_date], current_date)
                    if signal.get("eligible"):
<<<<<<< HEAD
                        # If signal today, execution price is T+1 close (already calculated if possible)
                        if ticker in execution_prices:
                            signal["close"] = execution_prices[ticker]
                        eligible_etfs.append(signal)

            # 4. Process Entries - Execute on Next Date (T+1)
            # PENDING_FREE slots are ignored here by strategy logic
            entries = self.strategy.process_entries(eligible_etfs, execution_date)
            self.transaction_log.extend(entries)

            # 5. Finalize Daily Updates (Convert PENDING_FREE -> FREE)
            self.strategy.finalize_daily_updates()

            # 6. Record NAV (Strategy vs Benchmark)
            # Strategy NAV
            # Strategy NAV
=======
                        # Use T+1 Open as the candidate execution price if available
                        if next_date in df.index:
                            signal["close"] = df.loc[next_date]['open'] 
                        eligible_etfs.append(signal)

            # 4. Process Entries
            entries = self.strategy.process_entries(eligible_etfs, current_date)
            self.transaction_log.extend(entries)

            # 5. Record NAV
>>>>>>> feature/chatai
            current_value = self.strategy.available_cash
            for slot in self.strategy.slots:
                if slot["status"] == "OCCUPIED":
                    symbol = slot["data"]["symbol"]
                    qty = slot["data"]["qty"]
<<<<<<< HEAD
                    
                    if symbol in processed_data:
                        df = processed_data[symbol]
                        price = df.loc[current_date]['close'] if current_date in df.index else 0
                        current_value += qty * price
            
            # Benchmark NAV (Buy & Hold Nifty 50)
            nifty_current_value = nifty_qty * benchmark_df.loc[current_date, 'close']
            
            self.portfolio_history.append({
                "date": current_date,
                "strategy": current_value,
                "benchmark_buyhold": nifty_current_value,
                "cumulative_investment": initial_capital,
                "cash": self.strategy.available_cash,
                "holdings": sum(1 for s in self.strategy.slots if s["status"] == "OCCUPIED")
            })

        self.daily_nav = pd.DataFrame(self.portfolio_history)
        if not self.daily_nav.empty:
            self.daily_nav['nav'] = self.daily_nav['strategy']  # Alias for handler compatibility
        self.strategy._log(1, "performance", "Backtest Completed.")
        
        return self.calculate_results(risk_free_rate)

    def calculate_results(self, risk_free_rate: float = 8.0):
        """Calculate final metrics and benchmark comparison"""
        if self.daily_nav.empty:
            return {"error": "No data recorded"}
            
        def get_metrics(series, dates):
            if len(series) < 2:
                return {}
                
            initial_val = series.iloc[0]
            final_val = series.iloc[-1]
            total_return = (final_val - initial_val) / initial_val * 100
            
            # CAGR
            days = (dates.iloc[-1] - dates.iloc[0]).days
            years = days / 365.25
            cagr = ((final_val / initial_val) ** (1 / years) - 1) * 100 if years > 0 else 0
            
            # Daily Returns for Sharpe
            daily_returns = series.pct_change().dropna()
            mean_daily_return = daily_returns.mean()
            std_daily_return = daily_returns.std()
            
            # Annualized Metrics
            trading_days_per_year = 252
            annualized_return = mean_daily_return * trading_days_per_year
            annualized_volatility = std_daily_return * np.sqrt(trading_days_per_year)
            
            # Sharpe Ratio
            rf_daily = (risk_free_rate / 100) / trading_days_per_year
            sharpe_ratio = 0
            if std_daily_return > 0:
                sharpe_ratio = (mean_daily_return - rf_daily) / std_daily_return * np.sqrt(trading_days_per_year)
            
            # Max Drawdown
            peak = series.cummax()
            drawdown = (series - peak) / peak * 100
            max_dd = drawdown.min()
            
            # Calmar Ratio
            calmar_ratio = 0
            if max_dd < 0:
                calmar_ratio = abs(cagr / max_dd) # Using CAGR / MaxDD
            
            return {
                "total_investment": round(float(initial_val), 2),
                "final_capital": round(float(final_val), 2),
                "total_return_pct": round(total_return, 2),
                "cagr": round(cagr, 2),
                "sharpe_ratio": round(sharpe_ratio, 2),
                "calmar_ratio": round(calmar_ratio, 2),
                "max_drawdown_pct": round(max_dd, 2)
            }

        strategy_metrics = get_metrics(self.daily_nav['strategy'], self.daily_nav['date'])
        benchmark_metrics = get_metrics(self.daily_nav['benchmark_buyhold'], self.daily_nav['date'])
        
        # Add Total Trades to Strategy Metrics
        strategy_metrics["total_trades"] = len([t for t in self.transaction_log if t['action'] in ['BUY', 'SELL']])
        
        # Nest benchmark metrics inside strategy metrics as requested
        strategy_metrics["benchmark_metrics"] = benchmark_metrics
        
        self.strategy._log(1, "performance", f"Metrics: {strategy_metrics}")
        
        return {
            "metrics": strategy_metrics,
            "performance_data": self.daily_nav.to_dict('records'),
            "transaction_log": self.transaction_log
        }

    @property
    def portfolio_log(self) -> List[Dict]:
        """Alias for transaction_log to support caching system"""
        return self.transaction_log

    @property
    def weekly_nav_df(self) -> pd.DataFrame:
        """Alias for compatibility with caching system"""
        return self.daily_nav

    def get_transaction_costs_summary(self) -> Dict:
        """Get summary of transaction costs grouped by year"""
        if not self.transaction_log:
            return {}

        df = pd.DataFrame(self.transaction_log)
        if 'date' not in df.columns:
            return {}

        df['year'] = pd.to_datetime(df['date']).dt.year.astype(str)
        
        summary = {}
        for year, group in df.groupby('year'):
            # Calculate metrics for the year
            yearly_brokerage = 0.0
            yearly_taxes = 0.0
            yearly_total = 0.0
            
            for _, row in group.iterrows():
                trade_costs = row.get('costs', {})
                if isinstance(trade_costs, dict):
                    # Sum all individual tax/fee components
                    b = float(trade_costs.get('brokerage', 0))
                    tc = float(
                        trade_costs.get('stt', 0) + 
                        trade_costs.get('stamp_duty', 0) + 
                        trade_costs.get('exchange_charges', 0) + 
                        trade_costs.get('sebi_charges', 0) + 
                        trade_costs.get('gst', 0)
                    )
                    yearly_brokerage += b
                    yearly_taxes += tc
                    yearly_total += (b + tc)

            summary[str(year)] = {
                'transaction_costs': round(yearly_taxes, 2),
                'capital_gains_tax': 0.0,
                'total_brokerage': round(yearly_brokerage, 2),
                'total_costs': round(yearly_total, 2),
                'transactions': len(group)
            }
            
        return summary
=======
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
>>>>>>> feature/chatai

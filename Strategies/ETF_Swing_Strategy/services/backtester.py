import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os
import json
from sqlalchemy import text
from Databases.app_data_db_connection import get_session
from Services.market_data_service import MarketDataService
from Strategies.ETF_Swing_Strategy.strategy import ETFSwingStrategy

class ETFSwingBacktester:
    def __init__(self, market: str = "INDIA", asset_type: str = "ETF", config_path: str = None):
        self.market = market.upper()
        self.asset_type = asset_type.upper()
        self.strategy = ETFSwingStrategy(market=self.market, asset_type=self.asset_type, config_path=config_path)
        self._data_cache = {}
        self.portfolio_history = []
        self.transaction_log = []
        self.daily_nav = pd.DataFrame()

    def load_data(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Load data using MarketDataService"""
        data_dict = {}
        
        # Add buffer for SMA
        buffer_days = self.strategy.sma_lookback * 2
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
        self.strategy._log(1, "execution", f"Starting Backtest: {start_date} to {end_date}")
        
        # Load Data
        all_data = self.load_data(tickers, start_date, end_date)
        if not all_data:
            return {"error": f"No data found for any of the requested tickers: {tickers}"}
            
        if "BENCHMARK" not in all_data:
            market_bench = "NIFTY 50" if self.market == "INDIA" else "S&P 500"
            return {"error": f"Missing benchmark data for {market_bench}. Check index market tables."}

        benchmark_df = all_data.pop("BENCHMARK")

        # Calculate Indicators for all ETFs
        processed_data = {}
        insufficient_data_tickers = []
        for ticker, df in all_data.items():
            # Check if enough data exists for SMA calculation for this specific ticker
            if df.empty or len(df) < self.strategy.sma_lookback:
                available = len(df) if not df.empty else 0
                insufficient_data_tickers.append(f"{ticker} ({available} days)")
            
            processed_data[ticker] = self.strategy.calculate_indicators(df)
        
        if insufficient_data_tickers:
            tickers_str = ", ".join(insufficient_data_tickers)
            error_msg = f"Insufficient data for SMA. Your SMA Lookback is set to {self.strategy.sma_lookback}, but the following tickers have less data: {tickers_str}. Please select a shorter SMA lookback or exclude these tickers."
            self.strategy._log(1, "system", f"ERROR: {error_msg}")
            return {"error": error_msg}

        self.strategy._log(1, "data_fetch", f"Total ETFs in Universe for Backtest: {len(processed_data)}")

        # Get Common Trading Dates (Intersects with Benchmark)
        trading_dates = benchmark_df.loc[start_date:end_date].index
        
        self.strategy.initialize_portfolio(initial_capital)
        self.portfolio_history = []
        self.transaction_log = []

        # Benchmark Initial State (Buy & Hold Nifty 50)
        nifty_initial_price = benchmark_df.loc[trading_dates[0], 'open']
        nifty_qty = initial_capital / nifty_initial_price

        for current_date in trading_dates:
            # Log Slot Usage
            occupied_slots = sum(1 for s in self.strategy.slots if s["status"] == "OCCUPIED")
            self.strategy._log(1, "execution", f"[{current_date.strftime('%Y-%m-%d')}] SLOTS: {occupied_slots} Occupied / {self.strategy.num_slots} Total")

            # 1. Update Holding SMAs for exit evaluation
            prices_at_close = {}
            for slot in self.strategy.slots:
                if slot["status"] == "OCCUPIED":
                    symbol = slot["data"]["symbol"]
                    if symbol in processed_data and current_date in processed_data[symbol].index:
                        current_row = processed_data[symbol].loc[current_date]
                        self.strategy.update_holding_sma(symbol, current_row['sma'])
                        prices_at_close[symbol] = current_row['close']

            # 2. Process Exits (Signal T -> Execution T+1 Open)
            # Simplified for backtest: We look at today's signals, executed later
            next_date_idx = benchmark_df.index.get_loc(current_date) + 1
            execution_date = current_date
            execution_prices = {}
            
            if next_date_idx < len(benchmark_df):
                next_date = benchmark_df.index[next_date_idx]
                execution_date = next_date # Use Next Day for Execution Date
                for ticker, df in processed_data.items():
                    if next_date in df.index:
                        execution_prices[ticker] = df.loc[next_date]['open']
            else:
                # Last day fallback
                for ticker, df in processed_data.items():
                    if current_date in df.index:
                        execution_prices[ticker] = df.loc[current_date]['close']

            exits = self.strategy.process_exits(execution_prices, execution_date)
            self.transaction_log.extend(exits)

            # 3. Evaluate Entry Signals (Close of Day T)
            eligible_etfs = []
            for ticker, df in processed_data.items():
                if current_date in df.index:
                    signal = self.strategy.evaluate_signals(ticker, df.loc[:current_date], current_date)
                    if signal.get("eligible"):
                        # If signal today, execution price is T+1 open (already calculated if possible)
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
            current_value = self.strategy.available_cash
            for slot in self.strategy.slots:
                if slot["status"] == "OCCUPIED":
                    symbol = slot["data"]["symbol"]
                    qty = slot["data"]["qty"]
                    
                    if symbol in processed_data:
                        df = processed_data[symbol]
                        price = 0
                        if current_date in df.index:
                            price = df.loc[current_date]['close']
                        else:
                            # Forward fill: get last available price before current_date
                            # This handles missing data (e.g., holidays where benchmark open but ETF not)
                            # to prevent -100% drawdown spikes
                            last_available = df.loc[:current_date]
                            if not last_available.empty:
                                price = last_available.iloc[-1]['close']
                        
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

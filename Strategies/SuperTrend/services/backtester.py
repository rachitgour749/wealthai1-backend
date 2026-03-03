import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os
from Services.market_data_service import MarketDataService
from Strategies.SuperTrend.strategy import SuperTrendStrategy

class SuperTrendBacktester:
    def __init__(self, market: str = "INDIA", asset_type: str = "STOCK", config_path: str = None):
        self.market = market.upper()
        self.asset_type = asset_type.upper()
        self.strategy = SuperTrendStrategy(market=self.market, asset_type=self.asset_type, config_path=config_path)
        self.portfolio_history = []
        self.transaction_log = []
        self.daily_nav = pd.DataFrame()

    def load_data(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Load data using MarketDataService"""
        data_dict = {}
        
        # Buffer for ATR calculation (need weekly bars). e.g., 10 weeks * 7 = 70 days. Let's buffer 150 days.
        buffer_days = self.strategy.atr_period * 15
        adj_start = pd.to_datetime(start_date) - timedelta(days=buffer_days)
        adj_end = pd.to_datetime(end_date)
        
        self.strategy._log(2, "data_fetch", f"Loading {self.market} data from {adj_start.date()} to {adj_end.date()}")
        
        try:
            assets_df = MarketDataService.fetch_close_prices(
                tickers=tickers, market=self.market, asset_type=self.asset_type,
                start_date=adj_start, end_date=adj_end
            )
            
            if not assets_df.empty:
                for ticker in tickers:
                    if ticker in assets_df.columns:
                        ticker_df = assets_df[[ticker]].rename(columns={ticker: 'close'})
                        
                        # MarketDataService right now fetches only close prices!
                        # The user has "buy signal generate on close post supertrend breakout"
                        # "stoploss also check on weekly basic on open price"
                        # We need full OHLCV data to correctly simulate SuperTrend calculation and open price stoploss!
                        
                        full_ticker_df = MarketDataService.fetch_stock_data(
                            ticker=ticker, market=self.market, asset_type=self.asset_type, 
                            start_date=adj_start, end_date=adj_end
                        )
                        
                        if not full_ticker_df.empty:
                            data_dict[ticker] = full_ticker_df
                            self.strategy._log(2, "data_fetch", f"Fetched {len(full_ticker_df)} OHLVC records for {ticker}.")
                        else:
                            # Fallback to close prices if full OHLCV not natively supported
                            for col in ['open', 'high', 'low']: ticker_df[col] = ticker_df['close']
                            ticker_df['volume'] = 0
                            data_dict[ticker] = ticker_df

            # Benchmark
            benchmark_candidates = ["NIFTY_50", "NIFTYBEES", "NIFTY50"] if self.market == "INDIA" else ["^GSPC", "S&P_500", "SPY", "SPX"]
            bench_df = MarketDataService.fetch_close_prices(
                tickers=benchmark_candidates, market=self.market, asset_type="INDEX",
                start_date=adj_start, end_date=adj_end
            )
            
            if not bench_df.empty:
                benchmark_symbol = next((c for c in benchmark_candidates if c in bench_df.columns), None)
                if benchmark_symbol:
                    bench_data = bench_df[[benchmark_symbol]].rename(columns={benchmark_symbol: 'close'})
                    for col in ['open', 'high', 'low']: bench_data[col] = bench_data['close']
                    bench_data['volume'] = 0
                    data_dict["BENCHMARK"] = bench_data
            return data_dict
        finally:
            pass

    def run_backtest(self, tickers: List[str], start_date: str, end_date: str, initial_capital: float, risk_free_rate: float = 8.0, config_params: dict = None):
        """Run the backtest loop"""
        if config_params:
            self.strategy.update_config(config_params)
            
        self.strategy._log(1, "system", f"Starting SuperTrend Backtest: {start_date} to {end_date}")
        
        all_data = self.load_data(tickers, start_date, end_date)
        if not all_data or "BENCHMARK" not in all_data:
            return {"error": "Insufficient data"}

        benchmark_df = all_data.pop("BENCHMARK")

        processed_data = {}
        for ticker, df in all_data.items():
            processed_data[ticker] = self.strategy.calculate_indicators(df)

        trading_dates = benchmark_df.loc[start_date:end_date].index
        self.strategy.initialize_portfolio(initial_capital)
        self.portfolio_history = []
        self.transaction_log = []

        nifty_initial_price = benchmark_df.loc[trading_dates[0], 'open']
        nifty_qty = initial_capital / nifty_initial_price

        for i, current_date in enumerate(trading_dates):
            # 1. Update Holding SMAs for exit evaluation
            prices_at_close = {}
            for slot in self.strategy.slots:
                if slot["status"] == "OCCUPIED":
                    symbol = slot["data"]["symbol"]
                    if symbol in processed_data and current_date in processed_data[symbol].index:
                        prices_at_close[symbol] = processed_data[symbol].loc[current_date]['close']

            # Determine next trading day open price for execution
            exec_prices = {}
            next_date = None
            if i + 1 < len(trading_dates):
                next_date = trading_dates[i + 1]
                for ticker, df in processed_data.items():
                    if next_date in df.index:
                        exec_prices[ticker] = df.loc[next_date]['open']
            
            if not next_date: next_date = current_date

            # 2. Process Exits (Evaluates based on today's Close, or Today's Open for weekly start)
            # Strategy checks `dfs` natively to see if it's the start of the week for stoploss, or end for breakdown
            exits = self.strategy.process_exits(
                eval_prices=prices_at_close,
                exec_prices=exec_prices,
                dfs=processed_data,
                current_date=current_date
            )
            self.transaction_log.extend(exits)

            # 3. Evaluate Entry Signals (Breakout on Friday)
            eligible_stocks = []
            for ticker, df in processed_data.items():
                signal = self.strategy.evaluate_signals(ticker, df.loc[:current_date], current_date)
                if signal.get("eligible"): eligible_stocks.append(signal)

            # 4. Process Entries - Execute on Next Date T+1 (Monday)
            if eligible_stocks and next_date:
                # Execution happens on next_date using exec_prices (next day open)
                self.strategy._log(1, "execution", f"--- SIGNALS RECEIVED FOR {next_date.strftime('%Y-%m-%d')} ---")
                entries = self.strategy.process_entries(eligible_stocks, next_date, exec_prices)
                self.transaction_log.extend(entries)

            # 5. Finalize Daily Updates (Convert PENDING_FREE -> FREE)
            self.strategy.finalize_daily_updates()

            # 6. Record NAV
            current_value = self.strategy.available_cash
            for slot in self.strategy.slots:
                if slot["status"] == "OCCUPIED":
                    symbol = slot["data"]["symbol"]
                    qty = slot["data"]["qty"]
                    if symbol in processed_data:
                        df = processed_data[symbol]
                        price = df.loc[current_date]['close'] if current_date in df.index else 0
                        current_value += qty * price
            
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
        if not self.daily_nav.empty: self.daily_nav['nav'] = self.daily_nav['strategy']
        self.strategy._log(1, "performance", "Backtest Completed.")
        
        return self.calculate_results(risk_free_rate)

    def calculate_results(self, risk_free_rate):
        if self.daily_nav.empty: return {"error": "No data recorded"}
        
        initial_val = self.daily_nav['strategy'].iloc[0]
        final_val = self.daily_nav['strategy'].iloc[-1]
        
        strategy_metrics = {
            "total_investment": round(float(initial_val), 2),
            "final_capital": round(float(final_val), 2),
            "total_return_pct": round((final_val - initial_val) / initial_val * 100, 2),
            "total_trades": len([t for t in self.transaction_log if t['action'] in ['BUY', 'SELL']])
        }
        
        self.strategy._log(1, "performance", f"Metrics: {strategy_metrics}")
        
        return {
            "metrics": strategy_metrics,
            "performance_data": self.daily_nav.to_dict('records'),
            "transaction_log": self.transaction_log
        }

    @property
    def portfolio_log(self): return self.transaction_log

    @property
    def weekly_nav_df(self): return self.daily_nav

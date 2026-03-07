import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import os
from sqlalchemy import func
from Databases.app_data_db_connection import get_session
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
        """Load OHLCV data using MarketDataService"""
        data_dict = {}

        # Buffer enough history for ATR warm-up: atr_period weeks * ~7 days/week + margin
        buffer_days = self.strategy.atr_period * 15
        adj_start = pd.to_datetime(start_date) - timedelta(days=buffer_days)
        adj_end = pd.to_datetime(end_date)

        self.strategy._log(2, "data_fetch", f"Loading {self.market} data from {adj_start.date()} to {adj_end.date()}")

        for ticker in tickers:
            df = MarketDataService.fetch_ohlcv(
                ticker=ticker,
                market=self.market,
                asset_type=self.asset_type,
                start_date=adj_start,
                end_date=adj_end
            )
            if not df.empty:
                data_dict[ticker] = df
                self.strategy._log(2, "data_fetch", f"Fetched {len(df)} OHLCV records for {ticker}.")
            else:
                self.strategy._log(1, "data_fetch", f"[WARNING] No data found for {ticker} — skipping.")

        # Benchmark
        if self.market == "INDIA":
            benchmark_candidates = ["NIFTY_50", "NIFTYBEES", "NIFTY50"]
            bench_asset_type = "INDEX"
        else:
            benchmark_candidates = ["^GSPC", "S&P_500", "SPY"]
            bench_asset_type = "INDEX"

        for candidate in benchmark_candidates:
            bench_df = MarketDataService.fetch_ohlcv(
                ticker=candidate,
                market=self.market,
                asset_type=bench_asset_type,
                start_date=adj_start,
                end_date=adj_end
            )
            if not bench_df.empty:
                data_dict["BENCHMARK"] = bench_df
                self.strategy._log(2, "data_fetch", f"Benchmark loaded: {candidate} ({len(bench_df)} records)")
                break

        if "BENCHMARK" not in data_dict:
            # Use first available ticker as benchmark fallback
            if data_dict:
                first_ticker = list(data_dict.keys())[0]
                data_dict["BENCHMARK"] = data_dict[first_ticker].copy()
                self.strategy._log(1, "data_fetch", f"[WARNING] No index data found — using {first_ticker} as benchmark fallback.")

        return data_dict

    def run_backtest(self, tickers: List[str], start_date: str, end_date: str, initial_capital: float, risk_free_rate: float = 8.0, config_params: dict = None):
        """Run the backtest loop"""
        if config_params:
            self.strategy.update_config(config_params)
            
        self.strategy._log(1, "system", f"Starting SuperTrend Backtest: {start_date} to {end_date}")
        
        all_data = self.load_data(tickers, start_date, end_date)
        if not all_data or "BENCHMARK" not in all_data:
            return {"error": "Insufficient data"}

        benchmark_df = all_data.pop("BENCHMARK")
        
        # ── Step 1: Normalize all tickers to the benchmark trading dates ──────
        # This prevents "spikes" to zero when an ETF has a missing data point
        benchmark_dates = benchmark_df.index
        
        processed_data = {}
        for ticker, df in all_data.items():
            # Reindex to benchmark dates and forward-fill missing values
            df = df.reindex(benchmark_dates).ffill().bfill()
            processed_data[ticker] = self.strategy.calculate_indicators(df)

        trading_dates = benchmark_dates[benchmark_dates.slice_indexer(start_date, end_date)]
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
                calmar_ratio = abs(cagr / max_dd)
            
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
        
        # Benchmark comparison
        if 'benchmark_buyhold' in self.daily_nav.columns:
            benchmark_metrics = get_metrics(self.daily_nav['benchmark_buyhold'], self.daily_nav['date'])
            strategy_metrics["benchmark_metrics"] = benchmark_metrics
        
        # Add Total Trades
        strategy_metrics["total_trades"] = len([t for t in self.transaction_log if t['action'] in ['BUY', 'SELL']])
        
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

    def calculate_common_date_range(self, tickers: List[str]) -> Tuple[Optional[str], Optional[str], float]:
        """Calculate common date range for tickers with SuperTrend buffer"""
        # Buffer enough history for ATR warm-up: atr_period weeks * 7 days/week + margin
        # 10 weeks * 15 (buffer multiplier from load_data) = 150 days
        buffer_days = self.strategy.atr_period * 15
        return MarketDataService.calculate_date_range(
            tickers=tickers,
            market=self.market,
            asset_type=self.asset_type,
            buffer_days=buffer_days
        )

    def load_metadata(self) -> Dict[str, Any]:
        """Load metadata for all assets in this market/asset_type"""
        db = get_session()
        try:
            model = MarketDataService.get_model(self.market, self.asset_type)
            # Fetch summary stats for all tickers
            query = db.query(
                model.symbol,
                func.min(model.date).label('start_date'),
                func.max(model.date).label('end_date'),
                func.count(model.date).label('total_records')
            ).group_by(model.symbol).all()
            
            metadata = {}
            for row in query:
                start_dt = pd.to_datetime(row.start_date)
                end_dt = pd.to_datetime(row.end_date)
                years = (end_dt - start_dt).days / 365.25
                metadata[row.symbol] = {
                    "start_date": row.start_date.strftime('%Y-%m-%d') if hasattr(row.start_date, 'strftime') else str(row.start_date),
                    "end_date": row.end_date.strftime('%Y-%m-%d') if hasattr(row.end_date, 'strftime') else str(row.end_date),
                    "years_available": years,
                    "total_records": row.total_records,
                    "name": self.generate_asset_description(row.symbol),
                    "category": self.get_asset_sector_classification(row.symbol)
                }
            return metadata
        finally:
            db.close()

    def generate_asset_description(self, symbol: str) -> str:
        """Generate human-readable name for symbol"""
        # Simple implementation, can be enhanced with a proper lookup table
        return symbol.replace(".NS", "").replace("_", " ")

    def get_asset_sector_classification(self, symbol: str) -> str:
        """Get sector classification for symbol"""
        # Placeholder, can be enhanced
        if self.asset_type == "ETF":
            return "Equity ETF"
        return "Equity Stock"

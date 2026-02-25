from Strategies.RS.RSBaseStrategy import RSBaseStrategy
import pandas as pd
from typing import Dict
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from Strategies.base.marketConfig import getMarketConfig, getBenchmarkSymbolsClause

class RSETFStrategy(RSBaseStrategy):
    """
    RS Level 5: ETF Strategy Implementation
    
    Handles specific data loading for ETFs.
    """
    
    def __init__(self, db_session: Session, config: Dict):
        config['strategy_name'] = 'RS_ETF_Level4'
        super().__init__(db_session, config)
        self.market = config.get('market', 'INDIA').upper()
        
    def load_data(self, start_date: datetime, end_date: datetime):
        """
        Optimized Data Loading for ETFs with Lookback Buffer.
        """
        # Calculate max lookback in TRADING DAYS
        max_lookback_trading_days = max(
            self.config.get('lookback_weeks', 5),
            self.config.get('lookback_months', 20),
            self.config.get('lookback_quarters', 60)
        )
        
        # Convert trading days to calendar days (approx 1.4x for weekends/holidays)
        buffer_days = int(max_lookback_trading_days * 1.5)
        
        buffer_date = start_date - timedelta(days=buffer_days)
        
        self.logger.progress(f"📥 Loading ETF Data (Market: {self.market}, Buffer: {max_lookback_trading_days} trading days = ~{buffer_days} calendar days)")
        
        # Format dates for SQL
        start_str = buffer_date.strftime('%Y-%m-%d') # Use buffered start
        end_str = end_date.strftime('%Y-%m-%d')
        
        # 1. Fetch Index Data — resolved from centralized marketConfig
        _cfg   = getMarketConfig(self.market, 'ETF')
        index_table   = _cfg['indexTable']
        index_symbols = _cfg['benchmarkSymbols']
        
        # Build the symbol IN clause dynamically
        symbol_placeholders = ", ".join([f"'{s}'" for s in index_symbols])
        index_query = f"""
        SELECT date, COALESCE(adj_close, close) as adjusted_close 
        FROM {index_table}
        WHERE symbol IN ({symbol_placeholders})
        AND date >= '{start_str}' AND date <= '{end_str}'
        ORDER BY date
        """
        # Ensure we get a single time series if multiple symbols exist
        index_df = pd.read_sql(index_query, self.db_session.bind, index_col='date', parse_dates=['date'])
        if not index_df.empty:
            # If multiple symbols matched, group by date and take first price
            index_df = index_df.groupby('date').first()
        
        # 2. Fetch ETF Data based on market
        etf_list = self.config.get('custom_etfs', [])
        etf_table = "us_etf_market" if self.market == 'US' else "etf_market"
        
        universe_filter = ""
        if etf_list and len(etf_list) > 0:
            formatted_list = "', '".join(etf_list)
            universe_filter = f"AND symbol IN ('{formatted_list}')"
        
        etf_query = f"""
        SELECT date, symbol, adj_close as adjusted_close
        FROM {etf_table}
        WHERE date >= '{start_str}' AND date <= '{end_str}' {universe_filter}
        ORDER BY date, symbol
        """
        
        etf_df = pd.read_sql(etf_query, self.db_session.bind, parse_dates=['date'])
            
        self.logger.info(f"Loaded {len(etf_df)} ETF records from {etf_table} and {len(index_df)} Index records from {index_table}.")

        return etf_df, index_df

    def run_backtest(self, start_date: datetime, end_date: datetime):
        """
        Main Execution Loop
        """
        self.logger.info(f"🏁 Starting RS ETF Backtest ({self.market}) ({start_date} to {end_date})")
        
        # 1. Load Data
        etf_df, index_df = self.load_data(start_date, end_date)
        
        if etf_df.empty or index_df.empty:
            self.logger.error("No data found for backtest period.")
            return {}
            
        # 2. Pre-calculate Signals (Vectorized)
        self.precalculate_signals(etf_df, index_df)
        
        # 3. Pivot Prices for Execution Lookups
        prices_df = etf_df.pivot_table(index='date', columns='symbol', values='adjusted_close').ffill()
        
        # 4. Get all trading days and rebalance dates
        all_trading_days = self.rs_scores_df.index.tolist()
        rebalance_dates = self.get_weekly_rebalance_dates(start_date, end_date)
        
        self.log_stop_loss_mode()
        self.logger.info(f"📅 Identified {len(rebalance_dates)} rebalance weeks and {len(all_trading_days)} total trading days.")
        
        # 5. Event Loop
        for current_date in all_trading_days:
            # Skip dates before backtest start
            if current_date < pd.to_datetime(start_date):
                continue
                
            # A. Daily Stop Loss Check (Only if enabled and NOT a rebalance day)
            # (Rebalance logic has its own SL check)
            if self.daily_stop_loss_check and current_date not in rebalance_dates:
                if current_date in prices_df.index:
                    self.check_stop_loss(prices_df.loc[current_date], current_date, mode="Daily")
            
            # B. Weekly Rebalancing (Fridays/Signal Days)
            if current_date in rebalance_dates:
                self.logger.progress(f"  > Processing Rebalance Week: {current_date.strftime('%Y-%m-%d')}")
                # Update Portfolio Value (MTM) and mark to market logic happens inside execute_rebalance
                self.execute_rebalance(current_date, prices_df)
                
        self.logger.info("✅ Backtest Completed.")
        
        # Calculate Benchmark Performance (Buy & Hold Market Index)
        # Filter index to match backtest period
        mask = (index_df.index >= pd.to_datetime(start_date)) & (index_df.index <= pd.to_datetime(end_date))
        period_index = index_df.loc[mask].copy()
        
        benchmark_buyhold = []
        if not period_index.empty:
            start_price = period_index['adjusted_close'].iloc[0]
            # Normalize to total capital
            period_index['equity_curve'] = (period_index['adjusted_close'] / start_price) * self.total_capital
            benchmark_buyhold = period_index['equity_curve'].tolist()
            
        return self.get_results(benchmark_buyhold)

    def get_results(self, benchmark_data=None):
        """Format results for API with comprehensive metrics"""
        if not self.portfolio_log: 
            return {}
        
        # Extract portfolio values
        dates = [log['date'] for log in self.portfolio_log]
        portfolio_values = [log['total_value'] for log in self.portfolio_log]
        
        # Calculate Strategy Metrics
        start_val = self.total_capital
        final_val = portfolio_values[-1]
        total_return_pct = ((final_val - start_val) / start_val) * 100
        
        # Calculate CAGR
        days = (dates[-1] - dates[0]).days
        years = days / 365.25
        cagr = 0.0
        if years > 0 and final_val > 0:
            cagr = (((final_val / start_val) ** (1 / years)) - 1) * 100
        
        # Calculate Max Drawdown
        peak = portfolio_values[0]
        max_dd = 0.0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = ((value - peak) / peak) * 100
            if dd < max_dd:
                max_dd = dd
        
        # Calculate Daily Returns for Sharpe
        daily_returns = []
        for i in range(1, len(portfolio_values)):
            ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            daily_returns.append(ret)
        
        sharpe_ratio = 0.0
        if len(daily_returns) > 1:
            import numpy as np
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            if std_return > 0:
                # Annualized Sharpe (assuming weekly data)
                risk_free_daily = (self.config.get('risk_free_rate', 6.0) / 100) / 52
                sharpe_ratio = ((mean_return - risk_free_daily) / std_return) * np.sqrt(52)
        
        # Calculate Calmar Ratio
        calmar_ratio = 0.0
        if max_dd < 0:
            calmar_ratio = cagr / abs(max_dd)
        
        # Strategy Metrics
        strategy_metrics = {
            'total_investment': round(start_val, 2),
            'final_capital': round(final_val, 2),
            'total_return_pct': round(total_return_pct, 2),
            'cagr': round(cagr, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'calmar_ratio': round(calmar_ratio, 2),
            'max_drawdown_pct': round(max_dd, 2)
        }
        
        # Benchmark Metrics (if available)
        benchmark_metrics = {}
        if benchmark_data and len(benchmark_data) > 0:
            bench_start = benchmark_data[0]
            bench_final = benchmark_data[-1]
            bench_return_pct = ((bench_final - bench_start) / bench_start) * 100
            
            # Benchmark CAGR
            bench_cagr = 0.0
            if years > 0 and bench_final > 0:
                bench_cagr = (((bench_final / bench_start) ** (1 / years)) - 1) * 100
            
            # Benchmark Max Drawdown
            bench_peak = benchmark_data[0]
            bench_max_dd = 0.0
            for val in benchmark_data:
                if val > bench_peak:
                    bench_peak = val
                dd = ((val - bench_peak) / bench_peak) * 100
                if dd < bench_max_dd:
                    bench_max_dd = dd
            
            # Benchmark Daily Returns
            bench_returns = []
            for i in range(1, len(benchmark_data)):
                ret = (benchmark_data[i] - benchmark_data[i-1]) / benchmark_data[i-1]
                bench_returns.append(ret)
            
            bench_sharpe = 0.0
            if len(bench_returns) > 1:
                import numpy as np
                mean_ret = np.mean(bench_returns)
                std_ret = np.std(bench_returns)
                if std_ret > 0:
                    risk_free_daily = (self.config.get('risk_free_rate', 6.0) / 100) / 52
                    bench_sharpe = ((mean_ret - risk_free_daily) / std_ret) * np.sqrt(52)
            
            bench_calmar = 0.0
            if bench_max_dd < 0:
                bench_calmar = bench_cagr / abs(bench_max_dd)
            
            benchmark_metrics = {
                'total_investment': round(bench_start, 2),
                'final_capital': round(bench_final, 2),
                'total_return_pct': round(bench_return_pct, 2),
                'cagr': round(bench_cagr, 2),
                'sharpe_ratio': round(bench_sharpe, 2),
                'calmar_ratio': round(bench_calmar, 2),
                'max_drawdown_pct': round(bench_max_dd, 2)
            }
        
        results = {
            'total_return_pct': total_return_pct,
            'final_capital': final_val,
            'trades': self.trades,
            'portfolio_snapshots': self.portfolio_log,
            'benchmark_buyhold': benchmark_data or [],
            'strategy_metrics': strategy_metrics,
            'benchmark_metrics': benchmark_metrics
        }
        
        return results

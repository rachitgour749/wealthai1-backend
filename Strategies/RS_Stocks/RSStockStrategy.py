from Strategies.RS.RSBaseStrategy import RSBaseStrategy
import pandas as pd
from typing import Dict
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

class RSStockStrategy(RSBaseStrategy):
    """
    RS Level 5: Stock Strategy Implementation
    
    Handles specific data loading for Stocks (Nifty 500 etc).
    """
    
    def __init__(self, db_session: Session, config: Dict):
        config['strategy_name'] = 'RS_Stocks_Level4'
        super().__init__(db_session, config)
        
    def load_data(self, start_date: datetime, end_date: datetime):
        """
        Optimized Data Loading for Stocks.
        """
        self.logger.progress("📥 Loading Stock Data...")
        
        # Format dates for SQL
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # 1. Fetch Index Data (Nifty 50)
        index_query = f"""
        SELECT date, close as adjusted_close 
        FROM nifty_50_index_market 
        WHERE date >= '{start_str}' AND date <= '{end_str}'
        ORDER BY date
        """
        index_df = pd.read_sql(index_query, self.db_session.bind, index_col='date', parse_dates=['date'])
        
        # 2. Fetch Stock Data
        # Filter Universe if specified
        universe_filter = ""
        stock_list = self.config.get('custom_stocks', [])
        if stock_list and len(stock_list) > 0:
            formatted_list = "', '".join(stock_list)
            universe_filter = f"AND symbol IN ('{formatted_list}')"
            
        # Assuming 'equity_market' or similar table for stocks
        # Standardizing on 'stock_market_data' or similar if known, but using generic assumption based on previous context
        # Or 'nifty_500_market_data'
        
        # Let's use 'nifty_500_market_data' as a safe bet for "RS Stocks" usually implying Nifty 500
        stock_query = f"""
        SELECT date, symbol, close as adjusted_close
        FROM nifty_500_market_data
        WHERE date >= '{start_str}' AND date <= '{end_str}' {universe_filter}
        ORDER BY date, symbol
        """
        
        try:
            stock_df = pd.read_sql(stock_query, self.db_session.bind, parse_dates=['date'])
        except Exception:
             # Fallback if table doesn't exist (Strategy might fail but better than crashing import)
             self.logger.error("Could not load Stock Data. Check Table Name.")
             stock_df = pd.DataFrame()
        
        self.logger.info(f"Loaded {len(stock_df)} Stock records and {len(index_df)} Index records.")
        return stock_df, index_df

    def run_backtest(self, start_date: datetime, end_date: datetime):
        """
        Main Execution Loop (Identical to ETF but uses own load_data)
        """
        self.logger.info(f"🏁 Starting RS Stock Backtest ({start_date} to {end_date})")
        
        # 1. Load Data
        stock_df, index_df = self.load_data(start_date, end_date)
        
        if stock_df.empty or index_df.empty:
            self.logger.error("No data found for backtest period.")
            return {}
            
        # 2. Pre-calculate Signals (Vectorized)
        self.precalculate_signals(stock_df, index_df)
        
        # 3. Pivot Prices
        prices_df = stock_df.pivot_table(index='date', columns='symbol', values='adjusted_close').ffill()
        
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
            if self.daily_stop_loss_check and current_date not in rebalance_dates:
                if current_date in prices_df.index:
                    self.check_stop_loss(prices_df.loc[current_date], current_date, mode="Daily")
            
            # B. Weekly Rebalancing
            if current_date in rebalance_dates:
                self.logger.progress(f"  > Processing Rebalance Week: {current_date.strftime('%Y-%m-%d')}")
                self.execute_rebalance(current_date, prices_df)
                
        self.logger.info("✅ Backtest Completed.")

    def get_results(self):
        if not self.portfolio_log: return {}
        
        results = {
            'total_return_pct': 0, 
            'final_capital': self.cash_balance, 
            'trades': self.trades,
            'portfolio_snapshots': self.portfolio_log 
        }
        
        if self.portfolio_log:
            final_val = self.portfolio_log[-1]['total_value']
            start_val = self.total_capital
            results['total_return_pct'] = ((final_val - start_val) / start_val) * 100
            results['final_capital'] = final_val
            
        return results

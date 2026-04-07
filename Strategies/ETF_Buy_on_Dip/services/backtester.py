"""
ETF Monthly Buy-on-Dip Backtester

Strategy Logic:
- Frequency: Monthly (Last trading day)
- Trigger: % Drop from 52-Week High
- Allocation Rule:
  - Dip < 5%: 1x Base Amount
  - Dip 5-10%: 2x Base Amount
  - Dip 10-20%: 4x Base Amount
  - Dip > 20%: 8x Base Amount
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json
import os
import sys

# Import base backtester to reuse data loading and cost calculation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Strategies.Rotation_ETF.services.backtester import ETFRotationBacktester
from Strategies.utilities.logging_config import StrategyLogger

class ETFBuyOnDipBacktester(ETFRotationBacktester):
    def __init__(self, db_path: str = None):
        super().__init__(db_path)
        self.logger = StrategyLogger('ETF_Buy_on_Dip')
        self.strategy_config = {}
        self._load_strategy_config()

    def _load_strategy_config(self):
        """Load strategy specific configuration"""
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logging_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.strategy_config = json.load(f)
            else:
                self.strategy_config = {
                    "dip_thresholds": {"tier_1": 5.0, "tier_2": 10.0, "tier_3": 20.0},
                    "allocation_multipliers": {"tier_1": 1.0, "tier_2": 2.0, "tier_3": 4.0, "tier_4": 8.0}
                }
        except Exception as e:
            self.logger.error(f"Error loading strategy config: {e}")

    def get_last_trading_day_of_month(self, dates: pd.DatetimeIndex) -> List[datetime]:
        """Identify execution dates (last trading day of each month)"""
        # Group by Year-Month and take the last date
        execution_dates = []
        df = pd.DataFrame({'date': dates})
        df['year_month'] = df['date'].dt.to_period('M')
        
        last_days = df.groupby('year_month')['date'].max()
        return last_days.tolist()

    def run_backtest(self, tickers: List[str], start_date: str, end_date: str,
                     capital_per_month: float, brokerage_percent: float) -> Dict:
        """
        Run Buy-on-Dip Backtest
        
        Args:
            tickers: List of ETFs (Strategy supports portfolio of ETFs, or single)
            capital_per_month: Base amount for 1x multiplier
        """
        self.logger.progress(f"Starting ETF Buy-on-Dip Backtest")
        self.logger.info(f"   Period: {start_date} to {end_date}")
        base_per_ticker = capital_per_month / len(tickers) if tickers else 0
        self.logger.info(f"   Base Monthly Capital (Total): INR {capital_per_month:,.0f}")
        self.logger.info(f"   Base Capital Per Ticker: INR {base_per_ticker:,.0f} (Split among {len(tickers)} ETFs)")
        self.logger.info(f"   Tickers: {tickers}")

        # Load data
        data_dict = self.load_data_from_database(tickers, start_date, end_date)
        if not data_dict:
            return {"error": "Failed to load data"}

        close_df = data_dict['close']
        
        # Identify execution dates (Last trading day of each month)
        valid_dates = close_df.index
        execution_dates = self.get_last_trading_day_of_month(valid_dates)
        execution_dates = [d for d in execution_dates if d >= pd.to_datetime(start_date)]

        portfolio = {ticker: {'units': 0, 'avg_price': 0.0, 'invested_amount': 0.0} for ticker in tickers}
        cash_flows = []  # For XIRR [(date, -amount), ..., (end_date, final_value)]
        transaction_log = []
        portfolio_snapshots = []
        
        total_invested = 0.0
        
        thresholds = self.strategy_config.get('dip_thresholds', {"tier_1": 5.0, "tier_2": 10.0, "tier_3": 20.0})
        multipliers = self.strategy_config.get('allocation_multipliers', {"tier_1": 1.0, "tier_2": 2.0, "tier_3": 4.0, "tier_4": 8.0})

        for date in execution_dates:
            month_str = date.strftime('%Y-%m')
            self.logger.progress(f"Processing Month: {month_str} (Execution: {date.strftime('%Y-%m-%d')})")
            
            # For each ticker, calculate dip and invest
            for ticker in tickers:
                if ticker not in close_df.columns or pd.isna(close_df.loc[date, ticker]):
                    self.logger.warning(f"   No data for {ticker} on {date.strftime('%Y-%m-%d')}")
                    continue

                current_price = close_df.loc[date, ticker]
                
                # Calculate 52-week high up to this date
                # Use 365 calendar days lookback
                lookback_start = date - timedelta(days=365)
                # Slice historical data
                history = close_df.loc[lookback_start:date, ticker]
                
                if history.empty:
                    self.logger.warning(f"   Insufficient history for {ticker}")
                    continue
                    
                high_52w = history.max()
                
                # Calculate Dip %
                if high_52w > 0:
                    dip_pct = ((high_52w - current_price) / high_52w) * 100
                else:
                    dip_pct = 0.0
                
                # Determine Multiplier
                multiplier = multipliers['tier_1'] # Default 1x (Dip < 5%)
                tier_mechanic = "Tier 1 (<5%)"
                
                if dip_pct > thresholds['tier_3']: # > 20%
                    multiplier = multipliers['tier_4'] # 8x
                    tier_mechanic = "Tier 4 (>20%)"
                elif dip_pct > thresholds['tier_2']: # 10-20%
                    multiplier = multipliers['tier_3'] # 4x
                    tier_mechanic = "Tier 3 (10-20%)"
                elif dip_pct > thresholds['tier_1']: # 5-10%
                    multiplier = multipliers['tier_2'] # 2x
                    tier_mechanic = "Tier 2 (5-10%)"
                
                # Base capital allocation per ticker
                base_per_ticker = capital_per_month / len(tickers)
                investment_amount = base_per_ticker * multiplier
                
                # Execute Buy
                # Calculate units
                # Estimate cost to fit within investment_amount
                # cost = amount + brokerage
                # amount = units * price
                # units * price * (1 + brokerage_pct) <= investment_amount
                
                cost_factor = 1 + (brokerage_percent if brokerage_percent else 0.0)
                units = int(investment_amount / (current_price * cost_factor))
                
                if units > 0:
                    buy_val = units * current_price
                    costs = self.calculate_transaction_costs('buy', buy_val, brokerage_percent)
                    total_outflow = buy_val + costs['total_costs']
                    
                    # Update Portfolio
                    current_units = portfolio[ticker]['units']
                    current_invested = portfolio[ticker]['invested_amount']
                    
                    new_units = current_units + units
                    new_invested = current_invested + total_outflow
                    
                    # Avg Price calculation (simple average of costs)
                    new_avg_price = new_invested / new_units if new_units > 0 else 0
                    
                    portfolio[ticker]['units'] = new_units
                    portfolio[ticker]['invested_amount'] = new_invested
                    portfolio[ticker]['avg_price'] = new_avg_price
                    
                    total_invested += total_outflow
                    cash_flows.append((date, -total_outflow))
                    
                    log_entry = {
                        "date": date.strftime('%Y-%m-%d'),
                        "ticker": ticker,
                        "action": "BUY",
                        "price": current_price,
                        "52w_high": high_52w,
                        "dip_pct": round(dip_pct, 2),
                        "multiplier": multiplier,
                        "tier": tier_mechanic,
                        "units": units,
                        "amount": total_outflow,
                        "transaction_costs": costs['total_costs'],
                        "nav": self._calculate_portfolio_nav(portfolio, close_df.loc[date])
                    }
                    transaction_log.append(log_entry)
                    
                    self.logger.trade(f"   {ticker}: Dip {dip_pct:.1f}% | High {high_52w:.1f} | Price {current_price:.1f}")
                    self.logger.trade(f"      Invest: INR {total_outflow:,.0f} ({multiplier}x Base) -> +{units} units")
                else:
                    self.logger.warning(f"   Funds insufficient for 1 unit of {ticker}")

            # Calculate and Log Monthly Summary
            monthly_invested = sum(t['amount'] for t in transaction_log if t['date'] == month_str or t['date'] == date.strftime('%Y-%m-%d'))
            # Note: transaction_log has date string, but we are in the loop for 'date' (datetime)
            # Filter properly using the current date string
            current_date_str = date.strftime('%Y-%m-%d')
            monthly_invested = sum(t['amount'] for t in transaction_log if t['date'] == current_date_str)
            
            total_base_capital = capital_per_month 
            pct_usage = (monthly_invested / total_base_capital * 100) if total_base_capital > 0 else 0
            
            self.logger.info(f"   --------------------------------------------------")
            self.logger.info(f"   Total Invested this Month: INR {monthly_invested:,.0f}")
            self.logger.info(f"   Capital Utilization: {pct_usage:.1f}% of Total Base (INR {capital_per_month:,.0f})")
            self.logger.info(f"   --------------------------------------------------")

            # Snapshot
            current_nav = self._calculate_portfolio_nav(portfolio, close_df.loc[date])
            portfolio_snapshots.append({
                "date": date.strftime('%Y-%m-%d'),
                "total_invested": total_invested,
                "current_value": current_nav,
                "holdings": {k: v['units'] for k, v in portfolio.items()}
            })

        # Final Metrics
        final_nav = self._calculate_portfolio_nav(portfolio, close_df.iloc[-1])
        
        # Populate weekly_nav_df for benchmark comparison (Parent logic)
        self.weekly_nav_df = pd.DataFrame([
            {
                'date': pd.to_datetime(s['date']),
                'week': i + 1,
                'nav': s['current_value'],
                'cumulative_investment': s['total_invested'],
                'base_capital_per_week': capital_per_month
            }
            for i, s in enumerate(portfolio_snapshots)
        ])
        
        # Calculate benchmark (typically Nifty 50 for Indian ETFs)
        benchmark_df = self.calculate_benchmark_buyhold(start_date, end_date, total_invested, brokerage_percent)
        benchmark_results = []
        benchmark_metrics = {}
        
        if not benchmark_df.empty:
            # Align benchmark with strategy dates
            strategy_dates = pd.to_datetime([s['date'] for s in portfolio_snapshots])
            benchmark_aligned = benchmark_df.reindex(strategy_dates, method='ffill')
            benchmark_results = benchmark_aligned['nav'].fillna(0).tolist()
            
            # Calculate benchmark metrics
            benchmark_metrics = {
                "total_return_pct": ((benchmark_df['nav'].iloc[-1] - total_invested) / total_invested * 100) if total_invested > 0 else 0,
                "final_value": benchmark_df['nav'].iloc[-1],
                "xirr": self.calculate_benchmark_xirr(total_invested)
            }

        cash_flows.append((pd.to_datetime(end_date), final_nav))
        
        # Calculate XIRR
        try:
             xirr = self.xirr_calculation(pd.Series([cf[1] for cf in cash_flows], index=[cf[0] for cf in cash_flows])) * 100
        except:
             xirr = 0.0
             
        # Generate final response structure
        metrics = {
            "total_invested": total_invested,
            "final_value": final_nav,
            "absolute_return": final_nav - total_invested,
            "absolute_return_pct": ((final_nav - total_invested) / total_invested * 100) if total_invested > 0 else 0,
            "xirr": xirr,
            "months_invested": len(execution_dates)
        }
        
        return {
            "success": True,
            "strategy_type": "ETF_Buy_on_Dip",
            "metrics": metrics,
            "benchmark_data": benchmark_results,
            "benchmark_metrics": benchmark_metrics,
            "transaction_log": transaction_log,
            "portfolio_snapshots": portfolio_snapshots,
            "final_nav": final_nav
        }

    def _calculate_portfolio_nav(self, portfolio, current_prices):
        nav = 0.0
        for ticker, data in portfolio.items():
            if ticker in current_prices:
                nav += data['units'] * current_prices[ticker]
        return nav

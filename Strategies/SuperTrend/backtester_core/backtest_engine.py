"""
Main Backtest Engine
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

from .indicators import add_indicators
from .rs_calculator import calculate_rs_for_universe, rank_by_rs
from .signal_logic import add_all_signals
from .portfolio_manager import PortfolioManager
from Strategies.utilities.logging_config import StrategyLogger


class BacktestEngine:
    """
    Main backtest engine implementing the strategy
    """
    
    def __init__(self, stock_data: pd.DataFrame, index_data: pd.DataFrame, config: Dict):
        """
        # Initialize centralized logger
        self.logger = StrategyLogger('SuperTrend')

        Initialize backtest engine
        
        Args:
            stock_data: Stock price data
            index_data: Benchmark index data
            config: Strategy configuration
        """
        self.stock_data = stock_data.copy()
        self.index_data = index_data.copy()
        self.config = config
        
        # Extract config parameters
        self.ema_short = config.get('ema_short', 15)
        self.ema_long = config.get('ema_long', 21)
        self.max_holdings = config.get('max_holdings', 5)
        self.price_floor = config.get('price_floor', 50.0)
        self.liquidity_cr = config.get('liquidity_cr', 10.0)
        self.rs_windows = [
            config.get('rs_window_1', 5),
            config.get('rs_window_2', 21),
            config.get('rs_window_3', 63)
        ]
        self.supertrend_period = config.get('supertrend_period', 10)
        self.supertrend_stop_pct = config.get('supertrend_stop_pct', 10.0)
        
        # Debug parameters
        self.debug_rs = config.get('debug_rs', True)  # Default enabled for debugging
        self.debug_rs_symbol = config.get('debug_rs_symbol', 'TRENT')  # Default to TRENT
        
        # Results
        self.results = None
        self.portfolio_manager = None
    
    def prepare_data(self) -> pd.DataFrame:
        """
        Prepare data with all indicators
        
        Returns:
            DataFrame with indicators and signals
        """
        # ============================================================
        # STEP 1: CALCULATING INDICATORS
        # ============================================================
        print("=" * 60)
        self.logger.progress("STEP 1: Calculating indicators...")
        print("=" * 60)
        
        # Calculate indicators for each symbol
        symbols = self.stock_data['symbol'].unique()
        self.logger.progress(f"  → Processing {len(symbols)} symbols")
        all_data = []
        
        for idx, symbol in enumerate(symbols, 1):
            stock_df = self.stock_data[self.stock_data['symbol'] == symbol].copy()
            stock_df = stock_df.sort_values('date').reset_index(drop=True)
            
            # Add indicators
            stock_df = add_indicators(
                stock_df,
                self.ema_short,
                self.ema_long,
                self.supertrend_period,
                self.supertrend_stop_pct
            )
            
            stock_df['symbol'] = symbol
            all_data.append(stock_df)
            
            # Show ALL processed symbols
            latest = stock_df.iloc[-1]
            print(f"     {symbol}: Price={latest['adj_close']:.2f}, "
                  f"EMA15={latest['ema10']:.2f}, EMA21={latest['ema20']:.2f}, ST={latest['supertrend_color']} , Date={latest['date']}")
            
            # Progress update every 20 symbols
            if idx % 20 == 0 or idx == len(symbols):
                self.logger.info(f"  → Processed {idx}/{len(symbols)} symbols")
        
        # Combine all stocks
        all_stocks = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"✓ Indicators calculated: {len(all_stocks):,} total records")
        print()
        
        # ============================================================
        # STEP 2: CALCULATING RS SCORES
        # ============================================================
        print("=" * 60)
        self.logger.progress("STEP 2: Calculating RS scores...")
        print("=" * 60)
        self.logger.info(f"  → RS Windows: {self.rs_windows} days")
        self.logger.info(f"  → Index benchmark records: {len(self.index_data)}")
        
        # Calculate RS scores
        all_stocks = calculate_rs_for_universe(all_stocks, self.index_data, self.rs_windows,
                                               debug=self.debug_rs, debug_symbol=self.debug_rs_symbol)
        
        # RS Statistics
        rs_stats = all_stocks['rs_score'].describe()
        rs_null_count = all_stocks['rs_score'].isna().sum()
        rs_valid_count = all_stocks['rs_score'].notna().sum()
        
        self.logger.info(f"✓ RS Calculation Complete:")
        self.logger.info(f"  → Valid RS scores: {rs_valid_count:,}")
        self.logger.info(f"  → Null RS scores: {rs_null_count:,} (insufficient history)")
        self.logger.info(f"  → RS Score Range: [{rs_stats['min']:.2f}, {rs_stats['max']:.2f}]")
        self.logger.info(f"  → RS Score Mean: {rs_stats['mean']:.2f}")
        self.logger.info(f"  → RS Score Median: {rs_stats['50%']:.2f}")
        
        # Show top 5 stocks by latest RS
        latest_date = all_stocks['date'].max()
        latest_rs = all_stocks[all_stocks['date'] == latest_date].nlargest(5, 'rs_score')
        self.logger.info(f"\n  Top 5 stocks by RS on {latest_date}:")
        for idx, row in latest_rs.iterrows():
            self.logger.info(f"    {idx+1}. {row['symbol']}: RS={row['rs_score']:.2f}")
        print()
        
        # ============================================================
        # STEP 3: CALCULATING SIGNALS
        # ============================================================
        print("=" * 60)
        self.logger.progress("STEP 3: Calculating signals...")
        print("=" * 60)
        self.logger.info(f"  → Price Floor: ≥₹{self.price_floor}")
        self.logger.info(f"  → Liquidity: ≥₹{self.liquidity_cr} Cr median turnover")
        
        # Count before filters
        before_count = len(all_stocks)
        
        # Add signals
        all_stocks = add_all_signals(all_stocks, self.price_floor, self.liquidity_cr)
        
        # Count after filters
        after_count = len(all_stocks)
        filtered_out = before_count - after_count
        
        self.logger.info(f"✓ Hygiene Filters Applied:")
        self.logger.info(f"  → Records before filters: {before_count:,}")
        self.logger.info(f"  → Records after filters: {after_count:,}")
        self.logger.info(f"  → Filtered out: {filtered_out:,} ({(filtered_out/before_count*100):.1f}%)")
        
        # Signal statistics for latest date
        latest_data = all_stocks[all_stocks['date'] == latest_date]
        eligible_count = latest_data['eligible'].sum()
        entry_signal_count = latest_data['entry_signal'].sum()
        exit_signal_count = latest_data['exit_signal'].sum()
        
        self.logger.info(f"\n  Signal Statistics (Latest Date: {latest_date}):")
        self.logger.info(f"  → Eligible stocks (Price>EMA20 & EMA10>EMA20): {eligible_count}")
        self.logger.info(f"  → Entry signals (Green ST & Eligible): {entry_signal_count}")
        self.logger.info(f"  → Exit signals (Red ST or bearish EMAs): {exit_signal_count}")
        
        # Show entry candidates
        if entry_signal_count > 0:
            entry_candidates = latest_data[latest_data['entry_signal'] == 1].nlargest(5, 'rs_score')
            self.logger.info(f"\n  Top Entry Candidates:")
            for idx, row in entry_candidates.iterrows():
                print(f"    • {row['symbol']}: RS={row['rs_score']:.2f}, "
                      f"Price={row['adj_close']:.2f}, ST={row['supertrend_color']}")
        
        print("=" * 60)
        print()
        
        return all_stocks
    
    def run_backtest(self, start_date: str, end_date: str, initial_capital: float = 1000000.0,
                     brokerage_pct: float = 0.1, buffer_pct: float = 10.0) -> Dict:
        """
        Run full backtest with REALISTIC EXECUTION (Next-Day Open Prices)
        
        EXECUTION LOGIC:
        ----------------
        Day T EOD:
            1. Generate signals based on Day T close data
            2. Identify entry candidates (Close > EMA15 & EMA21, EMA15 > EMA21)
            3. Identify exit signals (Close < EMA15 & EMA21)
            4. Store signals for next day
        
        Day T+1 Open:
            1. Execute exits from Day T signals at Day T+1 OPEN price
            2. Execute entries from Day T signals at Day T+1 OPEN price
        
        This eliminates forward-looking bias by ensuring:
        - Signals are generated at close (when you make decisions)
        - Trades execute at next day's open (realistic execution)
        
        Args:
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            initial_capital: Starting capital
            brokerage_pct: Brokerage percentage (e.g., 0.03 for 0.03%)
            buffer_pct: Buffer percentage for P&L absorption (e.g., 10.0 for 10%)
        
        Returns:
            Backtest results dictionary
        """
        self.logger.info(f"Running backtest from {start_date} to {end_date}")
        self.logger.trade(f"⚠️  REALISTIC EXECUTION: Signals generated at close → Executed at next day's open")
        self.logger.info(f"💰 Brokerage: {brokerage_pct}% | Transaction costs: ENABLED")
        self.logger.info(f"🔒 Buffer: {buffer_pct}% (P&L absorption system)")
        
        # Prepare data
        data = self.prepare_data()
        
        # Filter date range
        data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        
        # Initialize portfolio manager with brokerage and buffer
        self.portfolio_manager = PortfolioManager(
            initial_capital, 
            self.max_holdings,
            brokerage_pct=brokerage_pct,
            buffer_pct=buffer_pct
        )
        
        # Get unique trading dates
        trading_dates = sorted(data['date'].unique())
        
        self.logger.info(f"Simulating {len(trading_dates)} trading days...")
        
        # Variables to store signals from previous day
        prev_exit_symbols = []
        prev_candidates = pd.DataFrame()
        
        # Daily simulation loop
        for i, date in enumerate(trading_dates):
            if i % 50 == 0:
                self.logger.progress(f"Processing day {i+1}/{len(trading_dates)}: {date}")
            
            # DETAILED OUTPUT - Show first 5 days in detail
            show_details = i < 5 or i % 100 == 0
            
            if show_details:
                # Convert date to datetime to get day name
                date_obj = pd.to_datetime(date)
                day_name = date_obj.strftime('%A')  # Monday, Tuesday, etc.
                self.logger.info(f"\n{'='*70}")
                self.logger.info(f"📅 TRADING DAY {i+1}: {day_name}, {date}")
                self.logger.info(f"{'='*70}")
            
            # Get data for current date
            date_data = data[data['date'] == date].copy()
            
            # Get current prices for portfolio valuation (using close)
            current_prices_close = dict(zip(date_data['symbol'], date_data['adj_close']))
            
            # Get OPEN prices for trade execution (next-day execution)
            current_prices_open = dict(zip(date_data['symbol'], date_data['open']))
            
            if show_details:
                self.logger.info(f"\n💰 CURRENT PORTFOLIO STATE:")
                self.logger.info(f"  Cash: ₹{self.portfolio_manager.cash:,.0f}")
                self.logger.info(f"  Positions: {len(self.portfolio_manager.positions)}")
                if self.portfolio_manager.positions:
                    for sym, pos in self.portfolio_manager.positions.items():
                        curr_price = current_prices_close.get(sym, pos['entry_price'])
                        curr_val = pos['quantity'] * curr_price
                        pnl = curr_val - pos['cost']
                        pnl_pct = (pnl / pos['cost']) * 100
                        print(f"    {sym}: Qty={pos['quantity']}, Entry=₹{pos['entry_price']:.2f}, "
                              f"Current=₹{curr_price:.2f}, P&L=₹{pnl:,.0f} ({pnl_pct:.1f}%)")
            
            # ================================================================
            # STEP 1: EXECUTE TRADES FROM PREVIOUS DAY'S SIGNALS
            # Using TODAY's OPEN prices (realistic execution)
            # ================================================================
            if i > 0:  # Skip first day (no previous signals)
                if show_details and (prev_exit_symbols or not prev_candidates.empty):
                    self.logger.progress(f"\n🔄 EXECUTING PREVIOUS DAY'S SIGNALS AT TODAY'S OPEN:")
                
                # Execute trades using TODAY's OPEN prices
                trades_executed = self.portfolio_manager.rebalance(
                    prev_candidates, 
                    current_prices_open,  # Use OPEN prices for execution
                    date, 
                    prev_exit_symbols
                )
                
                if show_details and trades_executed:
                    self.logger.trade(f"  💼 TRADES EXECUTED: {len(trades_executed)}")
                    for trade in trades_executed:
                        if trade['action'] == 'BUY':
                            print(f"    ✅ BUY {trade['symbol']}: Qty={trade['quantity']}, "
                                  f"Open Price=₹{trade['price']:.2f}, Value=₹{trade['value']:,.0f}")
                        else:
                            print(f"    ❌ SELL {trade['symbol']}: Qty={trade['quantity']}, "
                                  f"Open Price=₹{trade['price']:.2f}, "
                                  f"P&L=₹{trade.get('pnl', 0):,.0f} ({trade.get('pnl_pct', 0):.1f}%)")
            
            # ================================================================
            # STEP 2: GENERATE SIGNALS FOR NEXT DAY
            # These will be executed tomorrow at open
            # ================================================================
            
            # Check exit signals for current positions (for NEXT day execution)
            exit_symbols = []
            for symbol in self.portfolio_manager.positions.keys():
                symbol_data = date_data[date_data['symbol'] == symbol]
                if not symbol_data.empty and symbol_data.iloc[0]['exit_signal'] == 1:
                    exit_symbols.append(symbol)
                    if show_details:
                        self.logger.info(f"\n🚪 EXIT SIGNAL GENERATED: {symbol} (will execute tomorrow at open)")
                        row = symbol_data.iloc[0]
                        print(f"  Reason: Close < EMA15 & EMA21, "
                              f"Price=₹{row['adj_close']:.2f}, "
                              f"EMA15=₹{row['ema10']:.2f}, EMA21=₹{row['ema20']:.2f}")
            
            # Get eligible candidates with entry signals
            candidates = date_data[
                (date_data['entry_signal'] == 1) & 
                (date_data['eligible'] == 1)
            ].copy()
            
            if show_details and not candidates.empty:
                self.logger.progress(f"\n✅ ENTRY CANDIDATES GENERATED: {len(candidates)} stocks (will execute tomorrow at open)")
                self.logger.info(f"  Showing top 10 by RS:")
                top = candidates.nlargest(10, 'rs_score')
                for idx, (_, row) in enumerate(top.iterrows(), 1):
                    print(f"    {idx}. {row['symbol']}: RS={row['rs_score']:.2f}, "
                          f"Close Price=₹{row['adj_close']:.2f}, "
                          f"EMA15=₹{row['ema10']:.2f}, EMA21=₹{row['ema20']:.2f}")
                
                # ================================================================
                # DETAILED RS CALCULATION DEBUG FOR TOP 10 ENTRY CANDIDATES
                # ================================================================
                if self.debug_rs:
                    self.logger.info(f"\n{'='*80}")
                    self.logger.info(f"DETAILED RS CALCULATION FOR TOP 10 ENTRY CANDIDATES")
                    self.logger.info(f"{'='*80}")
                    
                    top_10 = candidates.nlargest(10, 'rs_score')
                    
                    for rank, (_, stock) in enumerate(top_10.iterrows(), 1):
                        symbol = stock['symbol']
                        rs_score = stock['rs_score']
                        
                        # Get full history for this stock from the prepared data
                        stock_history = data[data['symbol'] == symbol].copy()
                        stock_history = stock_history.sort_values('date').reset_index(drop=True)
                        
                        # Find index for current date
                        current_idx = stock_history[stock_history['date'] == date].index
                        if len(current_idx) == 0:
                            continue
                        current_idx = current_idx[0]
                        
                        # Only show debug if we have enough historical data
                        if current_idx >= max(self.rs_windows):
                            self.logger.info(f"\n{'='*80}")
                            self.logger.info(f"DEBUG RS Calculation for {symbol} (Rank #{rank}, RS={rs_score:.2f})")
                            self.logger.info(f"{'='*80}")
                            
                            current_price = stock_history.loc[current_idx, 'adj_close']
                            
                            # Get index data for this date
                            index_row = self.index_data[self.index_data['date'] == date]
                            current_index = index_row['adj_close'].iloc[0] if len(index_row) > 0 else 0
                            
                            self.logger.info(f"    Current Date: {date}")
                            self.logger.info(f"    Current stock index: {current_idx}")
                            self.logger.info(f"    Max lookback required: {max(self.rs_windows)}")
                            self.logger.info(f"    Lookback periods: week={self.rs_windows[0]}, month={self.rs_windows[1]}, quarter={self.rs_windows[2]}")
                            self.logger.info(f"    Current prices:")
                            self.logger.info(f"      Stock: ₹{current_price:.2f}, Index: ₹{current_index:.2f}")
                            self.logger.info(f"    Historical dates and prices:")
                            
                            # Calculate historical data for each window
                            window_labels = ['Week', 'Month', 'Quarter']
                            rs_values = []
                            
                            for window, label in zip(self.rs_windows, window_labels):
                                lookback_idx = current_idx - window
                                if lookback_idx >= 0:
                                    hist_date = stock_history.loc[lookback_idx, 'date']
                                    hist_price = stock_history.loc[lookback_idx, 'adj_close']
                                    
                                    # Get index price for historical date
                                    hist_index_row = self.index_data[self.index_data['date'] == hist_date]
                                    hist_index = hist_index_row['adj_close'].iloc[0] if len(hist_index_row) > 0 else 0
                                    
                                    # Calculate RS for this window
                                    stock_return = ((current_price - hist_price) / hist_price) if hist_price > 0 else 0
                                    index_return = ((current_index - hist_index) / hist_index) if hist_index > 0 else 0
                                    rs_val = stock_return - index_return
                                    rs_values.append(rs_val)
                                    
                                    self.logger.info(f"      {label} ago ({window}d): {hist_date} - Stock: ₹{hist_price:.2f}, Index: ₹{hist_index:.2f}")
                            
                            # Show RS values
                            print(f"    RS values: ", end="")
                            rs_parts = [f"{label.lower()}={val:.3f}" for label, val in zip(window_labels, rs_values)]
                            print(", ".join(rs_parts))
                            
                            self.logger.info(f"    Final RS score: {rs_score:.3f}")
                            self.logger.info(f"    {symbol}: RS Score = {rs_score:.3f}")
                            self.logger.info(f"{'='*80}")
            
            # Rank by RS score (handle None values properly)
            if not candidates.empty:
                # Sort by RS score, putting None values at the end
                candidates = candidates.sort_values('rs_score', ascending=False, na_position='last')
                candidates['rank'] = range(1, len(candidates) + 1)
                
                # Check if current positions have dropped out of top N
                for symbol in list(self.portfolio_manager.positions.keys()):
                    symbol_rank = candidates[candidates['symbol'] == symbol]
                    if symbol_rank.empty or symbol_rank.iloc[0]['rank'] > self.max_holdings:
                        if symbol not in exit_symbols:
                            exit_symbols.append(symbol)
                            if show_details:
                                rank = symbol_rank.iloc[0]['rank'] if not symbol_rank.empty else 'N/A'
                                self.logger.info(f"\n🚪 RANK EXIT: {symbol} (Rank: {rank}, Max: {self.max_holdings})")
                
                # Select top N
                top_candidates = candidates.head(self.max_holdings)
                
                if show_details:
                    self.logger.info(f"\n🎯 SELECTED TOP {self.max_holdings} CANDIDATES (for tomorrow):")
                    for _, row in top_candidates.iterrows():
                        print(f"  {row['rank']}. {row['symbol']}: RS={row['rs_score']:.2f}, "
                              f"Close Price=₹{row['adj_close']:.2f}")
            else:
                top_candidates = pd.DataFrame()
                if show_details:
                    self.logger.info(f"\n⚠️  NO ENTRY CANDIDATES TODAY")
            
            # ================================================================
            # STEP 3: STORE SIGNALS FOR NEXT DAY EXECUTION
            # ================================================================
            prev_exit_symbols = exit_symbols
            prev_candidates = top_candidates
            
            # ================================================================
            # STEP 4: UPDATE PORTFOLIO VALUE USING CLOSE PRICES
            # ================================================================
            self.portfolio_manager.update_portfolio_value(current_prices_close)
            
            if show_details:
                self.logger.progress(f"\n📊 END OF DAY PORTFOLIO:")
                self.logger.info(f"  Portfolio Value: ₹{self.portfolio_manager.portfolio_value:,.0f}")
                self.logger.info(f"  Cash: ₹{self.portfolio_manager.cash:,.0f}")
                self.logger.info(f"  Open Positions: {len(self.portfolio_manager.positions)}")
            
            # Record equity curve
            self.portfolio_manager.record_equity(date)
        
        # Calculate performance metrics
        metrics = self.calculate_metrics()
        
        # BACKTEST COMPLETION SUMMARY
        print("\n" + "=" * 70)
        self.logger.info("✓ BACKTEST COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
        self.logger.performance(f"\n📊 PERFORMANCE SUMMARY")
        print("-" * 70)
        self.logger.info(f"  Period:          {start_date} to {end_date}")
        self.logger.info(f"  Trading Days:    {len(trading_dates)}")
        self.logger.info(f"  Initial Capital: ₹{initial_capital:,.0f}")
        self.logger.info(f"  Final Value:     ₹{self.portfolio_manager.portfolio_value:,.0f}")
        self.logger.performance(f"  Total Return:    {metrics['total_return']:.2f}%")
        self.logger.performance(f"  CAGR:            {metrics['cagr']:.2f}%")
        self.logger.trade(f"  Total Trades:    {len(self.portfolio_manager.trades)}")
        print("-" * 70)
        
        # Store results (with NaN handling for JSON serialization)
        equity_df = self.portfolio_manager.get_equity_curve_df()
        trades_df = self.portfolio_manager.get_trades_df()
        
        # Replace NaN values with 0 to prevent JSON serialization errors
        equity_df = equity_df.fillna(0)
        trades_df = trades_df.fillna(0)
        
        # Calculate cost analysis
        cost_analysis = self.calculate_cost_analysis(
            trades_df.to_dict('records'), 
            self.portfolio_manager.total_costs
        )
        
        self.results = {
            'metrics': metrics,
            'equity_curve': equity_df.to_dict('records'),
            'trades': trades_df.to_dict('records'),
            'cost_analysis': cost_analysis,
            'final_portfolio_value': self.portfolio_manager.portfolio_value,
            'total_trades': len(self.portfolio_manager.trades),
            'total_costs': self.portfolio_manager.total_costs
        }
        
        return self.results
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate performance metrics
        
        Returns:
            Dictionary of metrics
        """
        equity_df = self.portfolio_manager.get_equity_curve_df()
        
        if equity_df.empty:
            return {}
        
        # Calculate returns
        equity_df['returns'] = equity_df['portfolio_value'].pct_change()
        
        # Total return
        total_return = ((equity_df.iloc[-1]['portfolio_value'] / 
                        self.portfolio_manager.initial_capital) - 1) * 100
        
        # CAGR
        days = len(equity_df)
        years = days / 252  # Trading days
        cagr = (((equity_df.iloc[-1]['portfolio_value'] / 
                 self.portfolio_manager.initial_capital) ** (1/years)) - 1) * 100 if years > 0 else 0
        
        # Drawdown
        equity_df['cummax'] = equity_df['portfolio_value'].cummax()
        equity_df['drawdown'] = ((equity_df['portfolio_value'] - equity_df['cummax']) / 
                                 equity_df['cummax']) * 100
        max_drawdown = equity_df['drawdown'].min()
        
        # Volatility (annualized)
        volatility = equity_df['returns'].std() * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = (cagr / volatility) if volatility > 0 else 0
        
        # Win rate
        trades_df = self.portfolio_manager.get_trades_df()
        if not trades_df.empty:
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            if not sell_trades.empty:
                winning_trades = len(sell_trades[sell_trades['pnl'] > 0])
                total_trades = len(sell_trades)
                win_rate = (winning_trades / total_trades) * 100
                avg_win = sell_trades[sell_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
                avg_loss = sell_trades[sell_trades['pnl'] <= 0]['pnl'].mean() if total_trades - winning_trades > 0 else 0
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
        
        # Calculate metrics with proper NaN handling
        def safe_round(value, decimals=2):
            """Round value safely, handling NaN and None"""
            if pd.isna(value) or value is None:
                return 0.0
            try:
                return round(float(value), decimals)
            except (ValueError, TypeError):
                return 0.0
        
        metrics = {
            'total_return': safe_round(total_return),
            'cagr': safe_round(cagr),
            'max_drawdown': safe_round(max_drawdown),
            'volatility': safe_round(volatility),
            'sharpe_ratio': safe_round(sharpe),
            'win_rate': safe_round(win_rate),
            'avg_win': safe_round(avg_win),
            'avg_loss': safe_round(avg_loss),
            'total_trades': trades_df['date'].nunique() if not trades_df.empty else 0,
            'final_value': safe_round(self.portfolio_manager.portfolio_value),
            'initial_capital': self.portfolio_manager.initial_capital
        }
        
        # Final check: ensure no NaN values in metrics
        for key, value in metrics.items():
            if pd.isna(value):
                metrics[key] = 0.0
                self.logger.info(f"Warning: {key} was NaN, set to 0.0")
        
        return metrics
    
    def calculate_cost_analysis(self, trades, total_costs):
        """
        Calculate comprehensive cost analysis for the cost tab
        Standardized format to match Rotation strategies (Year-wise breakdown)
        """
        if not trades:
            return {}
            
        df = pd.DataFrame(trades)
        
        # Robust date column
        if 'date' in df.columns:
            df['year'] = pd.to_datetime(df['date']).dt.year.astype(str)
        else:
             df['year'] = "Total"

        summary = {}

        def calculate_metrics(d):
            # Calculate raw values
            transaction_costs = d['total_costs'].sum()
            # Capital gains tax is not explicitly tracked in SuperTrend trades yet, assume 0 or derived
            capital_gains_tax = d['capital_gains_tax'].sum() if 'capital_gains_tax' in d.columns else 0
            total_costs = transaction_costs + capital_gains_tax
            brokerage = d['brokerage'].sum() if 'brokerage' in d.columns else 0
            # Use unique dates as "Transactions" count (Trading Days) to match Rotation logic
            transactions = d['date'].nunique()
            
            # Return simplified dictionary with numeric values
            return {
                'transaction_costs': round(transaction_costs, 2),
                'capital_gains_tax': round(capital_gains_tax, 2),
                'total_costs': round(total_costs, 2),
                'total_brokerage': round(brokerage, 2),
                'transactions': transactions
            }
        
        if 'year' in df.columns:
             for year, group in df.groupby('year'):
                 summary[str(year)] = calculate_metrics(group)
                 
        # Ensure Total is present (or calculated by frontend? Rotation backend provides Total?)
        # Rotation code: 'year' column has "Total" if date missing.
        # But if date present, Rotation code usually iterates groupby.
        # The user's JSON had "Total" key.
        # Wait, Rotation `get_transaction_costs_summary` ONLY returns the grouped dict.
        # Does it include "Total"?
        # My previous code in Rotation:
        # summary[str(year)] = ...
        # It does NOT explicitly add "Total" key if groups exist.
        # But the USER showed "Total" key in their output.
        # Maybe the frontend sums it up?
        # Or maybe I missed where "Total" is added?
        # Re-check Rotation code.
        # Step 1600 (Rotation ETF):
        # for year, group in costs_df.groupby('year'): summary[str(year)] = ...
        # return summary
        # It does NOT add "Total".
        # So "Total" in user screenshot/json MUST come from Frontend sum or somewhere else.
        # OR "Total" comes when `date` is missing.
        # But user showed "2022", "2023"... AND "Total" row in table.
        # Usually table computes Total row.
        # So I just need to return Yearly keys.
        
        return summary


if __name__ == "__main__":
    # Test with sample data
    from datetime import datetime, timedelta
    
    # Sample stock data
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    symbols = ['STOCK1', 'STOCK2', 'STOCK3']
    
    stock_data = []
    for symbol in symbols:
        for date in dates:
            stock_data.append({
                'symbol': symbol,
                'date': date.strftime('%Y-%m-%d'),
                'open': 100 + np.random.randn(),
                'high': 102 + np.random.randn(),
                'low': 98 + np.random.randn(),
                'close': 100 + np.random.randn(),
                'adj_close': 100 + np.random.randn(),
                'volume': np.random.randint(1000000, 10000000)
            })
    
    stock_df = pd.DataFrame(stock_data)
    
    # Sample index data
    index_data = pd.DataFrame({
        'date': [d.strftime('%Y-%m-%d') for d in dates],
        'adj_close': 100 + np.random.randn(len(dates)).cumsum()
    })
    
    # Config
    config = {
        'ema_short': 15,
        'ema_long': 21,
        'max_holdings': 3,
        'price_floor': 50.0,
        'liquidity_cr': 1.0,
        'rs_window_1': 5,
        'rs_window_2': 21,
        'rs_window_3': 63
    }
    
    # Run backtest
    engine = BacktestEngine(stock_df, index_data, config)
    results = engine.run_backtest('2023-06-01', '2023-12-31', 1000000)
    
    self.logger.info("\nBacktest Results:")
    print(results['metrics'])


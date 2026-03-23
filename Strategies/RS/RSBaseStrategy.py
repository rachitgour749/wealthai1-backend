import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy.orm import Session
from Strategies.utilities.logging_config import StrategyLogger
from Segments.EquitySegment import EquitySegment
from Strategies.utilities.indicator_utils import IndicatorHelper

class RSBaseStrategy(EquitySegment):
    """
    RS Level 4: Base Strategy Implementation
    
    Implements optimized ranking and execution logic for Relative Strength strategies.
    Now reverted to pandas for standardized PRS calculations.
    """
    
    def __init__(self, db_session: Session, config: Dict):
        super().__init__(db_session)
        self.config = config
        
        # Initialize Logger
        strategy_name = config.get('strategy_name', 'RS_Strategy')
        self.logger = StrategyLogger(strategy_name)
        
        # Strategy Parameters
        self.lookback_weeks = config.get('lookback_weeks', 5)
        self.lookback_months = config.get('lookback_months', 20)
        self.lookback_quarters = config.get('lookback_quarters', 60)
        
        # Max Positions (Required, no default)
        self.max_positions = config.get('max_positions')
        if self.max_positions is None:
            raise ValueError("Missing required parameter: 'max_positions'. Please specify the maximum number of positions (e.g., 6) in your request.")
            
        self.total_capital = config.get('total_capital', 1000000.0)
        self.risk_free_rate = config.get('risk_free_rate', 6.0)
        
        # Stop Loss Mode (from rs_config.json)
        from Strategies.RS.rs_config_loader import get_rs_config
        rs_global_config = get_rs_config()
        
        self.daily_stop_loss_check = config.get('daily_stop_loss_check', rs_global_config.get_daily_stop_loss_check())
        self.stop_loss_pct = config.get('stop_loss_pct')
        if self.stop_loss_pct is None:
            self.stop_loss_pct = rs_global_config.get_stop_loss_pct()
        
        # State
        self.current_capital = self.total_capital
        self.cash_balance = self.total_capital
        self.positions: Dict[str, Dict] = {}  # Symbol -> {qty, buy_price, date, stop_loss}
        self.trades: List[Dict] = []
        self.portfolio_log: List[Dict] = []
        
        # Pre-calculated DataFrames
        self.ranks_df: pd.DataFrame = pd.DataFrame()
        self.rs_scores_df: pd.DataFrame = pd.DataFrame()

        # Fixed Allocation Logic (Calculated ONCE based on Initial Capital)
        buffer_pct = config.get('buffer_capital_pct', 0.0)
        self.initial_buffer = self.total_capital * (buffer_pct / 100)
        deployable_initial = self.total_capital - self.initial_buffer
        self.fixed_allocation_amount = deployable_initial / self.max_positions
        
        self.logger.info(f"💰 FIXED ALLOCATION CONFIGURED:")
        self.logger.info(f"   Initial Capital: ₹{self.total_capital:,.2f}")
        self.logger.info(f"   Buffer Reserve ({buffer_pct}%): ₹{self.initial_buffer:,.2f}")
        self.logger.info(f"   Fixed Allocation/Pos: ₹{self.fixed_allocation_amount:,.2f}")
        
    def precalculate_signals(self, univ_df: pd.DataFrame, index_df: pd.DataFrame):
        """
        Calculation of RS Scores and Ranks using the stock-indicators PRS logic.
        """
        self.logger.progress("🚀 Starting signal pre-calculation (via stock-indicators library)...")
        
        # 1. Pivot data to wide format (Dates x Symbols)
        closes = univ_df.pivot_table(index='date', columns='symbol', values='adjusted_close').ffill()
        # index_df may already have 'date' as its index (e.g. loaded via pd.read_sql with index_col='date')
        if 'date' in index_df.columns:
            index_df = index_df.set_index('date').sort_index()
        else:
            index_df = index_df.sort_index()
        index_closes = index_df['adjusted_close'].reindex(closes.index).ffill()
        
        # Store for logging
        self.closes_df = closes
        self.index_closes_df = index_closes
        
        # 2. Conversion to Quotes
        index_quotes = IndicatorHelper.df_to_quotes(pd.DataFrame({'open': index_closes, 'high': index_closes, 'low': index_closes, 'close': index_closes}))
        
        # Dictionary to store results per symbol
        all_rs_w = {}
        all_rs_m = {}
        all_rs_q = {}
        
        symbols = closes.columns.tolist()
        total = len(symbols)
        
        for i, symbol in enumerate(symbols):
            if i % 10 == 0:
                self.logger.info(f"  Calculating PRS for {i}/{total} symbols...")
                
            # Manual PRS calculation (Old method)
            # PRS = Ticker / Index
            prs_series = (closes[symbol] / index_closes)
            
            # (PRS_t / PRS_t-n) - 1
            all_rs_w[symbol] = prs_series / prs_series.shift(self.lookback_weeks) - 1
            all_rs_m[symbol] = prs_series / prs_series.shift(self.lookback_months) - 1
            all_rs_q[symbol] = prs_series / prs_series.shift(self.lookback_quarters) - 1
            
        # 3. Combine into Scores
        self.rs_w_df = pd.DataFrame(all_rs_w)
        self.rs_m_df = pd.DataFrame(all_rs_m)
        self.rs_q_df = pd.DataFrame(all_rs_q)
        
        # Average RS Score
        self.rs_scores_df = (self.rs_w_df + self.rs_m_df + self.rs_q_df) / 3
        
        # 4. Filter Negative RS (Must be > 0)
        positive_rs_scores = self.rs_scores_df.where(self.rs_scores_df > 0)
        
        # 5. Generate Ranks (Higher Score = Rank 1)
        self.ranks_df = positive_rs_scores.rank(axis=1, ascending=False, method='min')
        
        self.logger.progress("✅ Indicator-based calculation complete!")
        self.logger.info(f"  [DEBUG] RS Scores Shape: {self.rs_scores_df.shape}")
        
    def get_weekly_rebalance_dates(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Get list of Fridays (Signal Days)"""
        # Get all dates from data
        all_dates = self.rs_scores_df.index
        mask = (all_dates >= pd.to_datetime(start_date)) & (all_dates <= pd.to_datetime(end_date))
        period_dates = all_dates[mask]
        
        # Filter for Fridays (weekday == 4)
        is_friday = period_dates.weekday == 4
        fridays = period_dates[is_friday].tolist()
        
        return fridays

    def get_next_trading_day(self, current_date: datetime) -> Optional[datetime]:
        """Find the next available trading day after current_date"""
        # Efficient search in sorted index
        try:
            loc = self.rs_scores_df.index.get_indexer([current_date], method='bfill')[0]
            if loc == -1: return None
            
            # If current_date is Saturday/Sunday, bfill gives Monday.
            # If current_date is Friday, rank generation happens Friday Close.
            # Execution happens Next Trading Day Open/Close.
            # So if we are AT Friday, we need the day AFTER Friday.
            
            # Check if the found date is actually > current_date
            next_date = self.rs_scores_df.index[loc]
            
            if next_date <= current_date:
                # We need the next one
                if loc + 1 < len(self.rs_scores_df.index):
                    next_date = self.rs_scores_df.index[loc + 1]
                else:
                    return None
                    
            return next_date
        except:
            return None

    def execute_rebalance(self, signal_date: datetime, prices_df: pd.DataFrame):
        """
        Execute Portfolio Rebalance based on Signals from signal_date.
        Actual execution happens on next_trading_day.
        """
        # 1. Get Ranks for Signal Date
        if signal_date not in self.ranks_df.index:
            return
            
        daily_ranks = self.ranks_df.loc[signal_date]
        daily_rs_scores = self.rs_scores_df.loc[signal_date]
        
        # Log Signal Generation
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"📊 SIGNAL GENERATION - {signal_date.strftime('%Y-%m-%d (%A)')}")
        self.logger.info(f"{'='*80}")
        
        # -- DIAGNOSTIC LOGGING --
        # active_universe_count should be the number of assets with positive RS scores
        active_universe_count = daily_rs_scores.notna().sum()
        total_universe_count = len(daily_rs_scores)
        
        if active_universe_count == 0:
            self.logger.warning(f"⚠️  SEARCH ALERT: 0 out of {total_universe_count} ETFs have a positive Relative Strength score today.")
            self.logger.warning("    Check if recent market data is available or if all assets are in a downtrend.")
        elif self.max_positions > active_universe_count:
             self.logger.warning(f"⚠️  CONFIGURATION ALERT: Max Positions ({self.max_positions}) > Available Positive RS ETFs ({active_universe_count}).")
             self.logger.warning(f"    Result: Only {active_universe_count} positions will be taken. Total allocation will be {(active_universe_count/self.max_positions*100):.1f}% of capital.")
             self.logger.warning(f"    To fix 'Idle Cash', reduce 'max_positions' or expand your ETF universe.")
        # ------------------------
        
        # Display RS Scores and Rankings for all assets
        self.logger.info(f"\n{'Symbol':<15} {'RS Score':<12} {'Rank':<8} {'Status'}")
        self.logger.info(f"{'-'*60}")
        
        # Get all symbols (both valid and NaN)
        all_symbols = daily_rs_scores.index.tolist()
        
        # Sort by rank (NaN ranks go to end)
        sorted_symbols = sorted(all_symbols, key=lambda x: (
            pd.isna(daily_ranks.get(x, np.nan)),  # NaN ranks last
            daily_ranks.get(x, np.inf)  # Then by rank value
        ))
        
        for symbol in sorted_symbols:
            rs_score = daily_rs_scores.get(symbol, np.nan)
            rank = daily_ranks.get(symbol, np.nan)
            
            if pd.isna(rs_score):
                self.logger.info(f"{symbol:<15} {'N/A':<12} {'N/A':<8} ⏳ Insufficient Data")
                continue # Skip breakdown if no data
            elif pd.isna(rank):
                status = "✗ Negative RS"
            elif rank <= self.max_positions:
                status = f"✓ TOP {self.max_positions}"
            else:
                status = "✗ Excluded"
                
            self.logger.info(f"{symbol:<15} {rs_score:>10.4f}   {('N/A' if pd.isna(rank) else int(rank)):<8} {status}")
            
            # --- DETAILED BREAKDOWN ---
            try:
                # 1. Lookback Prices
                idx_loc = self.closes_df.index.get_loc(signal_date)
                
                def get_comp_info(period):
                    if idx_loc < period: return "Insufficient Data", np.nan, "N/A"
                    prev_date = self.closes_df.index[idx_loc - period]
                    p_curr = self.closes_df.loc[signal_date, symbol]
                    p_prev = self.closes_df.loc[prev_date, symbol]
                    idx_curr = self.index_closes_df.loc[signal_date]
                    idx_prev = self.index_closes_df.loc[prev_date]
                    
                    s_ratio = (p_curr / p_prev)
                    i_ratio = (idx_curr / idx_prev)
                    excess = (s_ratio / i_ratio) - 1
                    
                    line = (
                        f"ETF[P: {p_curr:>8.2f}, Prev: {p_prev:>8.2f}, Ratio: {s_ratio:>7.4f}] | "
                        f"IDX[P: {idx_curr:>8.2f}, Prev: {idx_prev:>8.2f}, Ratio: {i_ratio:>7.4f}]"
                    )
                    return line, excess, prev_date
                
                info_w, ex_w, date_w = get_comp_info(self.lookback_weeks)
                info_m, ex_m, date_m = get_comp_info(self.lookback_months)
                info_q, ex_q, date_q = get_comp_info(self.lookback_quarters)
                
                self.logger.info(f"    ├─ {self.lookback_weeks:2d}d Lookback ({date_w.strftime('%Y-%m-%d') if hasattr(date_w, 'strftime') else 'N/A'}):")
                self.logger.info(f"    │  └─ {info_w} | RS: {ex_w:>8.4f}")
                self.logger.info(f"    ├─ {self.lookback_months:2d}d Lookback ({date_m.strftime('%Y-%m-%d') if hasattr(date_m, 'strftime') else 'N/A'}):")
                self.logger.info(f"    │  └─ {info_m} | RS: {ex_m:>8.4f}")
                self.logger.info(f"    ├─ {self.lookback_quarters:2d}d Lookback ({date_q.strftime('%Y-%m-%d') if hasattr(date_q, 'strftime') else 'N/A'}):")
                self.logger.info(f"    │  └─ {info_q} | RS: {ex_q:>8.4f}")
                
                # Formula matches decimal RS score used for ranking
                calc_score = (ex_w + ex_m + ex_q) / 3
                self.logger.info(f"    └─ Total RS Score = ({ex_w:.4f} + {ex_m:.4f} + {ex_q:.4f}) / 3 = {calc_score:.4f}")
                self.logger.info(f"    {'-'*110}")
            except Exception as e:
                self.logger.debug(f"      [DEBUG] Error generating breakdown for {symbol}: {e}")
        
        # 2. Identify Targets (Top 20)
        targets = daily_ranks[daily_ranks <= self.max_positions].index.tolist()
        
        if targets:
            self.logger.info(f"\n🎯 Selected Targets ({len(targets)}): {', '.join(targets)}")
        else:
            self.logger.info(f"\n⚠️  No valid targets (insufficient historical data or all RS scores negative)")
        
        # 3. Determine Execution Date (Monday)
        exec_date = self.get_next_trading_day(signal_date)
        if not exec_date:
            return

        # 0. Initialize State if missing (for existing backtesters)
        if not hasattr(self, 'peak_capital'): self.peak_capital = self.total_capital
        if not hasattr(self, 'defensive_mode'): self.defensive_mode = False
        
        # 1. Update Peak and Check Drawdown (Defensive Logic)
        self.peak_capital = max(self.peak_capital, self.current_capital)
        current_dd_pct = ((self.peak_capital - self.current_capital) / self.peak_capital * 100) if self.peak_capital > 0 else 0
        
        reset_threshold = self.config.get('capital_reset_threshold_pct', 25.0)
        
        # Check Mode Switch
        if not self.defensive_mode:
            if current_dd_pct > reset_threshold:
                self.defensive_mode = True
                self.logger.warning(f"\n🛡️ DEFENSIVE MODE ACTIVATED (Drawdown: {current_dd_pct:.2f}% > {reset_threshold}%)")
                self.logger.warning("   - Reducing new position size by 50%")
                self.logger.warning("   - Trimming portfolio to Top 5 holdings")
        else:
            # Recovery Check: Recover to 80% of peak (i.e. DD < 20%? No, 'Portfolio Value recovers to 80% of peak')
            # Meaning Current >= 0.8 * Peak. Equivalently DD <= 20%.
            # Let's interpret "recovers to at least 80% of its peak" literally.
            if self.current_capital >= (0.80 * self.peak_capital):
                self.defensive_mode = False
                self.logger.info(f"\n✅ DEFENSIVE MODE DEACTIVATED (Recovery: Market Value > 80% of Peak)")
                self.logger.info("   - Resuming normal position sizing")

        if exec_date not in prices_df.index:
            self.logger.warning(f"Execution Date {exec_date.date()} missing in price data.")
            return
            
        prices = prices_df.loc[exec_date]
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"⚡ EXECUTION - {exec_date.strftime('%Y-%m-%d (%A)')}")
        self.logger.info(f"{'='*80}")
        
        if self.defensive_mode:
             self.logger.info(f"🛡️  [DEFENSIVE MODE ACTIVE] Peak: ₹{self.peak_capital:,.2f} | Current: ₹{self.current_capital:,.2f} | DD: {current_dd_pct:.2f}%")
        
        # 3.5 Stop Loss Logic
        self.check_stop_loss(prices, exec_date, mode="Weekly (Rebalance Day)")
        
        # 4. Exit Logic (Rank Based + Defensive Trim)
        # If Defensive: Limit targets to Top 5
        if self.defensive_mode and len(targets) > 5:
            original_target_count = len(targets)
            targets = targets[:5]
            self.logger.info(f"  ✂️  Defensive Trim: Restricted targets from {original_target_count} to Top 5.")

        # Refresh current symbols after SL exits
        current_symbols = list(self.positions.keys())
        if current_symbols:
            self.logger.info(f"\n📤 EXIT ANALYSIS:")
            for symbol in current_symbols:
                if symbol not in targets:
                    price = prices.get(symbol)
                    pos = self.positions[symbol]
                    pnl = ((price - pos['buy_price']) / pos['buy_price'] * 100) if price else 0
                    reason = "Defensive Trim" if self.defensive_mode and symbol not in targets else "Rank Drop"
                    self.logger.info(f"  SELL {symbol} @ {price:.2f} (Bought @ {pos['buy_price']:.2f}, P&L: {pnl:+.2f}%) [{reason}]")
                    self.sell_position(symbol, exec_date, price, reason)
        
        # 5. Entry Logic
        self.logger.info(f"\n📥 ENTRY ANALYSIS:")
        
        # Apply Buffer Capital
        # FIXED ALLOCATION: We use the pre-calculated fixed amount.
        # Buffer is implicitly maintained because we never re-invest profits into sizing.
        # Profits naturally accumulate in cash/buffer.
        
        allocation_per_pos = self.fixed_allocation_amount
        
        # Defensive Mode Reduced Allocation
        if self.defensive_mode:
            allocation_per_pos *= 0.5
            self.logger.info("  🛡️  Defensive Mode: New entry allocation reduced by 50%.")
        
        # Calculate current effective buffer (Cash - (Fixed Reserve? No, just Cash is Cash))
        # But for logging, we can show how much "Buffer" logic was intended.
        buffer_pct = self.config.get('buffer_capital_pct', 0.0)

        self.logger.info(f"  Available Cash: ₹{self.cash_balance:,.2f}")
        
        # Log the Fixed Strategy details
        self.logger.info(f"  Allocation Strategy: FIXED (Based on Initial Capital)")
        self.logger.info(f"  Standard Allocation per Position: ₹{self.fixed_allocation_amount:,.2f}")
        if self.defensive_mode:
             self.logger.info(f"  Actual Allocation (Defensive): ₹{allocation_per_pos:,.2f}")
        else:
             self.logger.info(f"  Actual Allocation: ₹{allocation_per_pos:,.2f}")
        
        for symbol in targets:
            if symbol not in self.positions:
                # Check if we have cash (respecting buffer implicitly by allocation size, but also need absolute cash check?)
                # Actually, cash_balance is absolute. If we buy, we reduce cash.
                # If we have 1M, buffer 10% (100k). Deployable 900k.
                # Allocation 900k/20 = 45k.
                # If we buy 20 positions: 20*45k = 900k. Cash left = 100k. Correct.
                
                if self.cash_balance < allocation_per_pos: 
                    self.logger.info(f"  ⚠️  Insufficient Cash for full allocation. Remaining: ₹{self.cash_balance:,.2f}")
                    # Try to buy with remaining cash? Or skip?
                    # Usually skip if too low, or partial fill.
                    if self.cash_balance < 1000: # Min threshold
                         self.logger.info(f"      Skipping {symbol}.")
                         break
                
                # Use calculated allocation, but limited by actual cash
                allocation = min(allocation_per_pos, self.cash_balance)
                price = prices.get(symbol)
                
                if allocation > 1000:
                    qty = int((allocation / (1 + 0.001)) // price) if price else 0
                    if qty > 0:
                        # self.logger.info(f"  BUY {symbol} @ {price:.2f} | Qty: {qty} | Amount: ₹{(qty * price):,.2f}")
                        self.buy_position(symbol, exec_date, price, allocation, "Top Rank")
                    else:
                        self.logger.info(f"  ⚠️  Cannot buy {symbol} - Insufficient allocation for 1 unit")

        # 6. Update Portfolio Value (Mark to Market)
        equity_value = 0.0
        for sym, pos in self.positions.items():
            current_price = prices.get(sym)
            if current_price and not np.isnan(current_price):
                equity_value += pos['qty'] * current_price
            else:
                equity_value += pos['qty'] * pos['buy_price']
        
        self.current_capital = self.cash_balance + equity_value

        # 7. Log Portfolio State
        self.logger.info(f"\n💼 PORTFOLIO SUMMARY:")
        self.logger.info(f"  Total Value: ₹{self.current_capital:,.2f}")
        self.logger.info(f"  Cash: ₹{self.cash_balance:,.2f}")
        self.logger.info(f"  Equity: ₹{equity_value:,.2f}")
        self.logger.info(f"  Positions: {len(self.positions)}")
        if self.positions:
            self.logger.info(f"  Holdings: {', '.join(self.positions.keys())}")
        
        self.log_portfolio(exec_date)
        self.logger.info(f"{'='*80}\n")
        
    def sell_position(self, symbol: str, date: datetime, price: float, reason: str):
        if not price or np.isnan(price): return
        
        pos = self.positions.pop(symbol)
        qty = pos['qty']
        
        transaction_value = qty * price
        
        # 1. Brokerage
        brokerage = transaction_value * (self.config.get('transaction_cost_pct', 0.1) / 100)
        
        # 2. Statutory / Regulatory Charges (ETF SELL Specific)
        # STT: 0.001% on Sell (ETF)
        stt = transaction_value * 0.00001
        # Exchange Txn Charges (NSE): 0.00297%
        exch_txn = transaction_value * 0.0000297
        # SEBI Turnover Fee: 0.0001%
        sebi_fee = transaction_value * 0.000001
        # Stamp Duty: 0 (Sell)
        stamp_duty = 0
        # GST: 18% on (Brokerage + Exchange + SEBI)
        gst = (brokerage + exch_txn + sebi_fee) * 0.18
        
        other_charges = stt + exch_txn + sebi_fee + stamp_duty + gst
        total_costs = brokerage + other_charges
        
        net_amount = transaction_value - total_costs
        
        # Update Balance
        self.cash_balance += net_amount
        self.current_capital = self.cash_balance + self.get_portfolio_value(date, {}) 
        
        # Log Trade
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'action': 'SELL',
            'qty': qty,
            'price': price,
            'amount': net_amount,
            'reason': reason
        })

        # Explicitly log the Sell with separate details
        self.logger.info(f"  SELL {symbol} @ {price:.2f} | Qty: {qty} | Brokerage: ₹{brokerage:.2f} | Txn Charges: ₹{other_charges:.2f} | Net: ₹{net_amount:,.2f}")
        
    def buy_position(self, symbol: str, date: datetime, price: float, allocation: float, reason: str):
        if not price or np.isnan(price): return
        
        # Estimate Costs for Qty Calculation (Approx 0.1% + 0.15% overhead safety)
        est_cost_factor = (self.config.get('transaction_cost_pct', 0.1) / 100) + 0.0015
        
        invest_amount = allocation / (1 + est_cost_factor)
        qty = int(invest_amount // price)
        
        if qty <= 0: return
        
        transaction_value = qty * price
        
        # 1. Brokerage
        brokerage = transaction_value * (self.config.get('transaction_cost_pct', 0.1) / 100)
        
        # 2. Statutory / Regulatory Charges (ETF BUY Specific)
        # STT: 0 on ETF Buy
        stt = 0
        # Exchange Txn Charges: 0.00297%
        exch_txn = transaction_value * 0.0000297
        # SEBI Turnover Fee: 0.0001%
        sebi_fee = transaction_value * 0.000001
        # Stamp Duty: 0.015% on Buy
        stamp_duty = transaction_value * 0.00015
        # GST: 18% on (Brokerage + Exchange + SEBI)
        gst = (brokerage + exch_txn + sebi_fee) * 0.18
        
        other_charges = stt + exch_txn + sebi_fee + stamp_duty + gst
        total_costs = brokerage + other_charges
        
        total_deduction = transaction_value + total_costs
        
        # Update Balance
        self.cash_balance -= total_deduction
        
        if self.stop_loss_pct > 0:
            sl_price = price * (1 - self.stop_loss_pct / 100)
            self.logger.info(f"  🛑 Stoploss set to ₹{sl_price:.2f} (Formula: ₹{price:.2f} * (1 - {self.stop_loss_pct}/100))")
        else:
            sl_price = None
            self.logger.info(f"  ℹ️  Stoploss disabled (0%)")

        # Explicitly log the Buy with separate details
        self.logger.info(f"  BUY {symbol} @ {price:.2f} | Qty: {qty} | Brokerage: ₹{brokerage:.2f} | Txn Charges: ₹{other_charges:.2f} | Total: ₹{total_deduction:,.2f}")

        # Store Position
        self.positions[symbol] = {
            'qty': qty,
            'buy_price': price,
            'date': date,
            'stop_loss': sl_price
        }
        
        # Log Trade
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'action': 'BUY',
            'qty': qty,
            'price': price,
            'amount': total_deduction,
            'reason': reason
        })

    def get_portfolio_value(self, date: datetime, prices: Any) -> float:
        # If prices not passed (optimization), we rely on cash only roughly or need lookup
        # For simplicity in loop, we update current_capital carefully
        # But better to calculate properly if prices available
        if isinstance(prices, dict) and not prices: return self.cash_balance # Fallback
        
        holdings_val = 0
        for sym, pos in self.positions.items():
            # Need current price
            # In backtest loop we usually have daily prices. 
            # If optimizing, we might skip daily MTM calculation and only do it on rebalance days.
            pass
        return self.cash_balance # Placeholder structure

    def log_portfolio(self, date: datetime):
        # Optimized: Only log essentials
        self.portfolio_log.append({
            'date': date,
            'total_value': self.current_capital,
            'cash': self.cash_balance,
            'positions_count': len(self.positions)
        })

    def log_stop_loss_mode(self):
        """Explicitly log the stop loss configuration"""
        mode = "DAILY Check" if self.daily_stop_loss_check else "WEEKLY Check (on Friday/Signal Day)"
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🛡️  STOP LOSS CONFIGURATION")
        self.logger.info(f"   Mode: {mode}")
        self.logger.info(f"   Threshold: {self.stop_loss_pct}%")
        self.logger.info(f"   Formula: Stoploss Price = Buy Price * (1 - {self.stop_loss_pct}/100)")
        self.logger.info(f"{'='*80}\n")

    def check_stop_loss(self, prices: pd.Series, current_date: datetime, mode: str = "Daily"):
        """
        Evaluate and execute Stop Loss for all open positions.
        Used by both daily check and weekly rebalance loops.
        """
        sl_hits = []
        current_symbols_check = list(self.positions.keys())
        
        for symbol in current_symbols_check:
            pos = self.positions[symbol]
            price = prices.get(symbol)
            
            if price and not np.isnan(price) and pos.get('stop_loss'):
                # Calculate current P&L (Drakedown)
                pnl = ((price - pos['buy_price']) / pos['buy_price'] * 100)
                
                # Check if price dropped below SL
                if price < pos['stop_loss']:
                    self.logger.info(f"  ⚠️  STOP LOSS HIT: {symbol} @ ₹{price:.2f} <= SL: ₹{pos['stop_loss']:.2f} (Drakedown: {pnl:+.2f}%)")
                    self.sell_position(symbol, current_date, price, "Stop Loss")
                    sl_hits.append(symbol)
                else:
                    # Explicitly log safe status
                    self.logger.info(f"  ✓  Stoploss safe: {symbol} @ ₹{price:.2f} > SL: ₹{pos['stop_loss']:.2f} (Drakedown: {pnl:+.2f}%)")
        
        return sl_hits

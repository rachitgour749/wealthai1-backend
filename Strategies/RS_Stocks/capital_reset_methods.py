    def check_capital_reset_threshold(self, current_portfolio_value: float, current_date: datetime) -> bool:
        """Check if capital reset threshold is triggered and manage the reset process"""
        try:
            # Update peak value
            if current_portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = current_portfolio_value
                # If we're in reset mode and recovered, exit reset mode
                if self.is_capital_reset_active:
                    recovery_threshold = self.peak_portfolio_value * 0.8  # 20% recovery from peak
                    if current_portfolio_value >= recovery_threshold:
                        print(f"  Capital reset deactivated - portfolio recovered to {current_portfolio_value:.0f}")
                        self.is_capital_reset_active = False
                        self.capital_reset_start_date = None
                        return False
            
            # Check if we need to trigger capital reset
            if not self.is_capital_reset_active:
                drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
                if drawdown >= self.capital_reset_threshold_pct:
                    print(f"  CAPITAL RESET TRIGGERED: Drawdown {drawdown:.1%} >= {self.capital_reset_threshold_pct:.1%}")
                    self.is_capital_reset_active = True
                    self.capital_reset_start_date = current_date
                    return True
            
            return self.is_capital_reset_active
            
        except Exception as e:
            print(f"Error in capital reset check: {e}")
            return False
    
    def apply_capital_reset_logic(self, entries: List[str], exits: List[str]) -> Tuple[List[str], List[str]]:
        """Apply capital reset logic by reducing positions and being more conservative"""
        if not self.is_capital_reset_active:
            return entries, exits
        
        print(f"  Applying capital reset logic - reducing risk")
        
        # Reduce entries by 50%
        reduced_entries = entries[:len(entries)//2] if len(entries) > 1 else []
        
        # Add more exits to reduce portfolio size
        additional_exits = []
        if len(self.positions) > 5:  # If we have more than 5 positions
            # Exit positions with lowest RS scores
            current_positions = list(self.positions.keys())
            additional_exits = current_positions[5:]  # Keep only top 5 positions
        
        all_exits = exits + additional_exits
        
        print(f"  Capital reset: {len(entries)} -> {len(reduced_entries)} entries, {len(exits)} -> {len(all_exits)} exits")
        
        return reduced_entries, all_exits
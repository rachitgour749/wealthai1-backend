"""
Dynamic Churn Calculator

Handles dynamic capital allocation calculations for rotation strategies.
"""


class DynamicChurnCalculator:
    """
    Calculate dynamic churn amount based on compounding logic.
    
    This calculator determines how much capital should be churned (rotated)
    each week based on the strategy's compounding settings.
    """
    
    @staticmethod
    def calculate(current_nav: float, cash: float, capital_per_week: float,
                 accumulation_weeks: int, compounding_enabled: bool) -> float:
        """
        Calculate dynamic churn amount based on compounding logic.
        
        Args:
            current_nav: Current Net Asset Value
            cash: Available cash (not invested)
            capital_per_week: Weekly capital allocation
            accumulation_weeks: Number of accumulation weeks
            compounding_enabled: Whether compounding is enabled
            
        Returns:
            Churn amount for this week
            
        Logic:
            - With compounding: Churn proportional to NAV (reinvest profits)
            - Without compounding: Fixed weekly capital (original capital only)
        """
        if compounding_enabled:
            # With compounding: churn proportional to NAV
            target_capital = capital_per_week * accumulation_weeks
            churn_amount = (current_nav / accumulation_weeks) if current_nav > 0 else capital_per_week
        else:
            # Without compounding: fixed weekly capital
            churn_amount = capital_per_week
        
        # Ensure we don't churn more than available NAV
        max_churn = current_nav - cash
        churn_amount = min(churn_amount, max_churn) if max_churn > 0 else 0
        
        return churn_amount

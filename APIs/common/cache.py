"""
Shared Cache for Backtest Results
Stores results in memory to be accessed by other components (like transaction log API)
"""
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# In-memory cache for latest backtest results per strategy type
_backtest_results_cache: Dict[str, Dict[str, Any]] = {}

def cache_backtest_results(strategy_type: str, backtester_instance: Any) -> None:
    """Cache backtest results for later retrieval"""
    print(f"[CACHE] cache_backtest_results called for {strategy_type}")
    
    try:
        portfolio_log = getattr(backtester_instance, 'portfolio_log', [])
        
        # Calculate cost breakdown from portfolio_log
        cost_breakdown = {}
        cost_summary = {}
        cost_analysis = {}
        
        # Calculate costs from portfolio_log - grouped by year
        if portfolio_log and len(portfolio_log) > 0:
            yearly_breakdown = {}
            
            for log in portfolio_log:
                # Get the year from execution_date
                execution_date = log.get('execution_date')
                if hasattr(execution_date, 'year'):
                    year = str(execution_date.year)
                else:
                    # Try to parse string date
                    try:
                        from datetime import datetime
                        if isinstance(execution_date, str):
                            year = str(datetime.strptime(execution_date, '%Y-%m-%d').year)
                        else:
                            year = "Unknown"
                    except:
                        year = "Unknown"
                
                # Initialize year if not exists
                if year not in yearly_breakdown:
                    yearly_breakdown[year] = {
                        'transaction_costs': 0,
                        'capital_gains_tax': 0,
                        'total_costs': 0,
                        'transactions': 0
                    }
                
                # Extract costs - handle multiple formats
                transaction_cost = 0
                capital_gains_tax = 0
                
                # Try to get from 'costs' dictionary first
                costs = log.get('costs', {})
                if costs and isinstance(costs, dict):
                    transaction_cost = costs.get('total_costs', 0)
                
                # If still 0, try direct fields
                if transaction_cost == 0:
                    transaction_cost = log.get('transaction_costs', 0) or log.get('transaction_cost', 0) or log.get('brokerage', 0)
                
                # Get capital gains tax
                capital_gains_tax = log.get('capital_gains_tax', 0)
                
                # Add to yearly totals
                yearly_breakdown[year]['transaction_costs'] += transaction_cost
                yearly_breakdown[year]['capital_gains_tax'] += capital_gains_tax
                yearly_breakdown[year]['total_costs'] += (transaction_cost + capital_gains_tax)
                yearly_breakdown[year]['transactions'] += 1
            
            # Round all values
            for year in yearly_breakdown:
                yearly_breakdown[year]['transaction_costs'] = round(yearly_breakdown[year]['transaction_costs'], 2)
                yearly_breakdown[year]['capital_gains_tax'] = round(yearly_breakdown[year]['capital_gains_tax'], 2)
                yearly_breakdown[year]['total_costs'] = round(yearly_breakdown[year]['total_costs'], 2)
            
            cost_breakdown = {'breakdown': yearly_breakdown}
        
        # Try to get additional cost data from backtester methods
        try:
            if hasattr(backtester_instance, 'get_cost_summary'):
                cost_summary = backtester_instance.get_cost_summary()
        except Exception:
            pass
        
        try:
            if hasattr(backtester_instance, 'get_cost_analysis'):
                cost_analysis = backtester_instance.get_cost_analysis()
        except Exception:
            pass
        
        _backtest_results_cache[strategy_type] = {
            'portfolio_log': portfolio_log.copy() if isinstance(portfolio_log, list) else [],
            'weekly_nav_df': backtester_instance.weekly_nav_df.copy() if hasattr(backtester_instance, 'weekly_nav_df') and hasattr(backtester_instance.weekly_nav_df, 'copy') else None,
            'trading_summary': getattr(backtester_instance, 'trading_summary', {}).copy() if isinstance(getattr(backtester_instance, 'trading_summary', {}), dict) else {},
            'withdrawal_log': getattr(backtester_instance, 'withdrawal_log', []).copy() if isinstance(getattr(backtester_instance, 'withdrawal_log', []), list) else [],
            'total_withdrawn_amount': getattr(backtester_instance, 'total_withdrawn_amount', 0),
            'cost_breakdown': cost_breakdown,
            'cost_summary': cost_summary,
            'cost_analysis': cost_analysis
        }
        
        cached_count = len(_backtest_results_cache[strategy_type]['portfolio_log'])
        print(f"[CACHE] ✅ Successfully cached {cached_count} transactions for {strategy_type}")
        logger.info(f"✅ Cached backtest results for {strategy_type}: {cached_count} transactions")
    except Exception as e:
        print(f"[CACHE] ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"❌ Failed to cache backtest results for {strategy_type}: {e}")

def get_cached_backtest_results(strategy_type: str) -> Dict[str, Any]:
    """Retrieve cached backtest results"""
    result = _backtest_results_cache.get(strategy_type, {})
    return result

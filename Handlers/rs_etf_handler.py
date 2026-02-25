"""
RS ETF Strategy Handler
Handles backtest execution for RS ETF Rotation strategy
"""
import sys
import os
import logging
from datetime import datetime
from sqlalchemy.orm import Session

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Strategies'))

from Handlers.base_handler import BaseStrategyHandler
from APIs.unified_schemas import UnifiedBacktestRequest, UnifiedBacktestResponse
from Strategies.RS_ETF.rs_etf_backtester_core import RSETFStrategyBacktester


logger = logging.getLogger(__name__)

class RSETFHandler(BaseStrategyHandler):
    """Handler for RS ETF Rotation strategy"""
    
    def validate_request(self, request: UnifiedBacktestRequest) -> None:
        """Validate RS ETF specific parameters"""
        if not request.total_capital:
            raise ValueError("RS_ETF_Rotation requires 'total_capital' parameter")
    
    async def run_backtest(self, request: UnifiedBacktestRequest) -> UnifiedBacktestResponse:
        """Run RS ETF Backtest using Level 4 Optimized Strategy"""
        db_created = False
        db_session = self.db
        
        try:
            # Ensure we have a database session if not provided
            if db_session is None:
                from Strategies.RS_ETF.database import get_db
                db_session = next(get_db())
                db_created = True

            # Import new Level 4 Strategy and Config Loader
            from Strategies.RS_ETF.RSETFStrategy import RSETFStrategy
            from Strategies.RS.rs_config_loader import get_rs_config
            rs_global_config = get_rs_config()
            
            # Extract market
            market = getattr(request, 'market', 'INDIA') or 'INDIA'
            if hasattr(request, 'parameters') and isinstance(request.parameters, dict):
                market = request.parameters.get('market', market)
            
            # Prepare config
            # Priority: API Parameter (stop_loss_pct or stop_loss) > Global Config Default
            sl_val = request.stop_loss_pct if request.stop_loss_pct is not None else getattr(request, 'stop_loss', None)
            if sl_val is None:
                sl_val = rs_global_config.get_stop_loss_pct()
            
            config_dict = {
                'market': market,
                'total_capital': request.total_capital,
                'lookback_weeks': request.lookback_weeks or 5,
                'lookback_months': request.lookback_months or 20,
                'lookback_quarters': request.lookback_quarters or 60,
                # Map alternate/legacy parameter names
                'max_positions': request.max_positions or getattr(request, 'no_of_positions', 4), # Default to 4 if missing
                'risk_free_rate': request.risk_free_rate or 8.0,
                'custom_etfs': self._clean_tickers(request.custom_etfs or request.tickers),
                'transaction_cost_pct': request.transaction_cost_pct or 0.1,
                'stop_loss_pct': sl_val,
                'daily_stop_loss_check': getattr(request, 'daily_stop_loss_check', rs_global_config.get_daily_stop_loss_check()),
                'buffer_capital_pct': request.buffer_capital_pct or getattr(request, 'buffer_capital', 10.0),
                'capital_reset_threshold_pct': request.capital_reset_threshold_pct or getattr(request, 'compounding_threshold', 25.0)
            }
            
            logger.info(f"Initialized RSETFStrategy with config: {config_dict}")
            
            # Initialize Strategy
            strategy = RSETFStrategy(db_session, config_dict)
            
            # Run Backtest
            start_dt = datetime.fromisoformat(request.start_date.replace('Z', '+00:00')).replace(tzinfo=None)
            end_dt = datetime.fromisoformat(request.end_date.replace('Z', '+00:00')).replace(tzinfo=None)
            
            results = strategy.run_backtest(start_dt, end_dt)
            
            # Sanitize data (convert datetime objects to strings)
            portfolio_snapshots = results.get('portfolio_snapshots', [])
            trades = results.get('trades', [])
            benchmark = results.get('benchmark_buyhold', [])
            
            # Convert datetime objects in snapshots
            sanitized_snapshots = []
            for snap in portfolio_snapshots:
                sanitized_snap = snap.copy()
                if 'date' in sanitized_snap and hasattr(sanitized_snap['date'], 'isoformat'):
                    sanitized_snap['date'] = sanitized_snap['date'].isoformat()
                sanitized_snapshots.append(sanitized_snap)
            
            # Convert datetime objects in trades
            sanitized_trades = []
            for trade in trades:
                sanitized_trade = trade.copy()
                if 'date' in sanitized_trade and hasattr(sanitized_trade['date'], 'isoformat'):
                    sanitized_trade['date'] = sanitized_trade['date'].isoformat()
                sanitized_trades.append(sanitized_trade)
            
            # Build performance data
            performance_data = {
                'dates': [],
                'strategy': [],
                'cumulative_investment': [],
                'benchmark_buyhold': benchmark
            }
            
            if sanitized_snapshots:
                performance_data['dates'] = [s['date'] for s in sanitized_snapshots]
                performance_data['strategy'] = [s['total_value'] for s in sanitized_snapshots]
                performance_data['cumulative_investment'] = [request.total_capital] * len(sanitized_snapshots)
            
            # Build metrics from strategy results
            strategy_metrics = results.get('strategy_metrics', {})
            benchmark_metrics = results.get('benchmark_metrics', {})
            
            # Combine into single metrics dict for response
            # Combine into single metrics dict for response
            metrics = {
                "xirr": strategy_metrics.get('xirr', 0),
                "volatility": strategy_metrics.get('volatility', 0),
                "win_rate_pct": strategy_metrics.get('win_rate_pct', 0),
                "total_trades": len(results.get('trades', [])),
                **strategy_metrics,
                "benchmark_metrics": benchmark_metrics
            }
            
            response = UnifiedBacktestResponse(
                success=True,
                strategy_type="RS_ETF_Rotation",
                metrics=metrics,
                performance_data=performance_data,
                transaction_log=sanitized_trades,
                portfolio_snapshots=sanitized_snapshots
            )
            
            return response
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return UnifiedBacktestResponse(
                success=False, 
                error=str(e), 
                strategy_type="RS_ETF_Rotation",
                metrics={}
            )
        finally:
            if db_created and db_session:
                db_session.close()

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
from .database import get_db, StrategyConfig, BacktestResult, TradeLog, PortfolioSnapshot, execute_with_retry, check_database_health, save_backtest_result_safely, save_additional_data_safely, reset_database_connections, force_unlock_database, BacktestSessionLocal, SessionLocal, get_backtest_session, get_main_session
from .rs_etf_backtester_core import RSETFStrategyBacktester
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import json
import numpy as np
import time
import logging
import os
import sys

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# RS EOD Data Fetcher not available for RS_ETF - removed import
RSEODDataFetcher = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_all_to_native_types(obj):
    """Aggressively convert all data types to JSON-safe native Python types"""
    if obj is None:
        return None
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        return float(obj)
    elif isinstance(obj, str):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {str(key): convert_all_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_all_to_native_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return convert_all_to_native_types(obj.__dict__)
    else:
        # For any other type, try to convert to string as last resort
        try:
            return str(obj)
        except:
            return None

router = APIRouter()

# Dependency to get backtest database session
def get_backtest_db():
    """Generator function for backtest database sessions with proper exception handling"""
    with get_backtest_session() as db:
        yield db

# Pydantic models for API requests/responses
class StrategyConfigCreate(BaseModel):
    config_name: str
    main_index: str = "^NSEI"
    etf_universe: Optional[str] = None  # ETF universe field
    lookback_weeks: int = 5
    lookback_months: int = 20
    lookback_quarters: int = 60
    max_positions: int = 20
    position_size_pct: float = 5.0
    buffer_capital_pct: float = 10.0
    total_capital: float = 1000000.0
    stop_loss_pct: float = 15.0
    capital_reset_threshold_pct: float = 25.0
    max_holding_period: int = 52
    min_turnover: float = 1000000.0
    min_price: float = 10.0
    transaction_cost_pct: float = 0.1

class StrategyConfigUpdate(BaseModel):
    config_name: Optional[str] = None
    main_index: Optional[str] = None
    etf_universe: Optional[str] = None  # ETF universe field
    lookback_weeks: Optional[int] = None
    lookback_months: Optional[int] = None
    lookback_quarters: Optional[int] = None
    max_positions: Optional[int] = None
    position_size_pct: Optional[float] = None
    buffer_capital_pct: Optional[float] = None
    total_capital: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    capital_reset_threshold_pct: Optional[float] = None
    max_holding_period: Optional[int] = None
    min_turnover: Optional[float] = None
    min_price: Optional[float] = None
    transaction_cost_pct: Optional[float] = None
    is_active: Optional[bool] = None

class StrategyConfigResponse(BaseModel):
    id: int
    config_name: str
    main_index: str
    etf_universe: Optional[str] = None  # ETF universe field
    lookback_weeks: int
    lookback_months: int
    lookback_quarters: int
    max_positions: int
    position_size_pct: float
    buffer_capital_pct: float
    total_capital: float
    stop_loss_pct: float
    capital_reset_threshold_pct: float
    max_holding_period: int
    min_turnover: float
    min_price: float
    transaction_cost_pct: float
    is_active: bool
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    class Config:
        from_attributes = True  # This allows SQLAlchemy models to be converted automatically

class BacktestRequest(BaseModel):
    # Date range (accept as string and convert to datetime)
    start_date: str
    end_date: str
    
    # Strategy configuration parameters (no config_id dependency)
    main_index: str = "^NSEI"
    etf_universe: str = "ALL_ETFS"
    custom_etfs: Optional[List[str]] = None  # Custom ETF selection
    max_positions: int = 20
    position_size_pct: Optional[float] = None  # Optional - will be auto-calculated if not provided
    total_capital: float = 1000000.0
    stop_loss_pct: float = 15.0
    buffer_capital_pct: float = 10.0
    capital_reset_threshold_pct: float = 25.0
    max_holding_period: Optional[int] = None  # Optional - will use default 52 if not provided
    transaction_cost_pct: float = 0.1
    min_price: Optional[float] = None  # Optional - will use default 10.0 if not provided
    min_turnover: Optional[float] = None  # Optional - will use default 1000000.0 if not provided
    lookback_weeks: int = 5
    risk_free_rate: float = 8.0  # Risk-free rate for Treynor ratio calculation
    lookback_months: int = 20
    lookback_quarters: int = 60

class RuleOf72Metrics(BaseModel):
    years_to_double: float
    expected_doublings: float
    rule_of_72_return_pct: float
    compounding_factor: float

class BacktestResponse(BaseModel):
    id: int
    config_id: Optional[int] = None  # Allow None for custom config backtests
    start_date: datetime
    end_date: datetime
    total_return: float
    annualized_return_pct: float
    cagr_pct: float
    xirr_pct: float
    max_drawdown: float
    sharpe_ratio: float
    beta: Optional[float] = 1.0
    treynor_ratio: Optional[float] = 0.0
    calmar_ratio: Optional[float] = 0.0
    win_rate_pct: float
    total_trades: int
    avg_holding_period: float
    final_capital: float
    rule_of_72: Optional[RuleOf72Metrics] = None
    created_at: datetime

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Configuration endpoints
# StrategyConfig CRUD endpoints removed - using only custom config backtests now
# Backtest endpoints
@router.get("/backtests", response_model=List[BacktestResponse])
async def get_backtest_results(user_id: str = Query(..., description="User ID to filter backtest results"), db: Session = Depends(get_backtest_db), limit: int = 50):
    """Get backtest results for a specific user"""
    try:
        results = db.query(BacktestResult).filter(BacktestResult.user_id == user_id).order_by(BacktestResult.created_at.desc()).limit(limit).all()
        print(f"Found {len(results)} backtest results")
        
        # Convert to response format with Rule of 72 calculation for existing results
        response_results = []
        for i, result in enumerate(results):
            try:
                print(f"Processing result {i+1}: ID={result.id}, CAGR={result.cagr_pct}")
                
                # Calculate Rule of 72 metrics for existing results
                years = (result.end_date - result.start_date).days / 365.25
                rule_72_metrics = None
                
                if result.cagr_pct and result.cagr_pct > 0 and years > 0:
                    try:
                        years_to_double = 72 / result.cagr_pct
                        expected_doublings = years / years_to_double
                        compounding_factor = 2 ** expected_doublings
                        rule_of_72_return = (compounding_factor - 1) * 100
                        
                        rule_72_metrics = RuleOf72Metrics(
                            years_to_double=years_to_double,
                            expected_doublings=expected_doublings,
                            rule_of_72_return_pct=rule_of_72_return,
                            compounding_factor=compounding_factor
                        )
                    except (ZeroDivisionError, ValueError, OverflowError) as e:
                        print(f"Error calculating Rule of 72 for result {result.id}: {e}")
                        rule_72_metrics = None
                
                response_results.append(BacktestResponse(
                    id=result.id,
                    config_id=result.config_id,
                    start_date=result.start_date,
                    end_date=result.end_date,
                    total_return=result.total_return_pct or 0,
                    annualized_return_pct=result.annualized_return_pct or 0,
                    cagr_pct=result.cagr_pct or 0,
                    xirr_pct=result.xirr_pct or 0,
                    max_drawdown=result.max_drawdown_pct or 0,
                    sharpe_ratio=result.sharpe_ratio or 0,
                    beta=getattr(result, 'beta', None) or 1.0,
                    treynor_ratio=getattr(result, 'treynor_ratio', None) or 0.0,
                    calmar_ratio=getattr(result, 'calmar_ratio', None) or 0.0,
                    win_rate_pct=result.win_rate_pct or 0,
                    total_trades=result.total_trades or 0,
                    avg_holding_period=result.avg_holding_period or 0,
                    final_capital=result.final_capital or 0,
                    rule_of_72=rule_72_metrics,
                    created_at=result.created_at
                ))
            except Exception as e:
                print(f"Error processing result {i+1}: {e}")
                continue
        
        print(f"Successfully processed {len(response_results)} results")
        return response_results
        
    except Exception as e:
        print(f"Error in get_backtest_results: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/backtests/{backtest_id}", response_model=BacktestResponse)
async def get_backtest_result(backtest_id: int, db: Session = Depends(get_backtest_db)):
    """Get a specific backtest result"""
    result = db.query(BacktestResult).filter(BacktestResult.id == backtest_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Backtest result not found")
    
    # Calculate Rule of 72 metrics for existing results
    years = (result.end_date - result.start_date).days / 365.25
    rule_72_metrics = None
    
    if result.cagr_pct and result.cagr_pct > 0 and years > 0:
        try:
            years_to_double = 72 / result.cagr_pct
            expected_doublings = years / years_to_double
            compounding_factor = 2 ** expected_doublings
            rule_of_72_return = (compounding_factor - 1) * 100
            
            rule_72_metrics = RuleOf72Metrics(
                years_to_double=years_to_double,
                expected_doublings=expected_doublings,
                rule_of_72_return_pct=rule_of_72_return,
                compounding_factor=compounding_factor
            )
        except (ZeroDivisionError, ValueError, OverflowError):
            rule_72_metrics = None
    
    return BacktestResponse(
        id=result.id,
        config_id=result.config_id,
        start_date=result.start_date,
        end_date=result.end_date,
        total_return=result.total_return_pct or 0,
        annualized_return_pct=result.annualized_return_pct or 0,
        cagr_pct=result.cagr_pct or 0,
        xirr_pct=result.xirr_pct or 0,
        max_drawdown=result.max_drawdown_pct or 0,
        sharpe_ratio=result.sharpe_ratio or 0,
        beta=getattr(result, 'beta', None) or 1.0,
        treynor_ratio=getattr(result, 'treynor_ratio', None) or 0.0,
        calmar_ratio=getattr(result, 'calmar_ratio', None) or 0.0,
        win_rate_pct=result.win_rate_pct or 0,
        total_trades=result.total_trades or 0,
        avg_holding_period=result.avg_holding_period or 0,
        final_capital=result.final_capital or 0,
        rule_of_72=rule_72_metrics,
        created_at=result.created_at
    )

@router.post("/backtests/run")
async def run_backtest(backtest_request: BacktestRequest, user_id: str = Query(..., description="User ID for the backtest"), db: Session = Depends(get_db)):
    """Run a new backtest with custom configuration parameters"""
    import traceback
    
    # Convert string dates to datetime objects
    try:
        start_date = datetime.fromisoformat(backtest_request.start_date.replace('Z', '+00:00')).replace(tzinfo=None)
        end_date = datetime.fromisoformat(backtest_request.end_date.replace('Z', '+00:00')).replace(tzinfo=None)
    except (ValueError, AttributeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
    
    # Validate date range
    if start_date >= end_date:
        raise HTTPException(status_code=400, detail="Start date must be before end date")
    
    # Create config dict from request parameters
    # Use provided values or None (backend will use defaults)
    config_dict = {
        'main_index': backtest_request.main_index,
        'etf_universe': backtest_request.etf_universe,
        'custom_etfs': backtest_request.custom_etfs,  # Add custom ETF selection
        'max_positions': backtest_request.max_positions,
        'position_size_pct': backtest_request.position_size_pct,  # None = auto-calculate
        'total_capital': backtest_request.total_capital,
        'stop_loss_pct': backtest_request.stop_loss_pct,
        'buffer_capital_pct': backtest_request.buffer_capital_pct,
        'capital_reset_threshold_pct': backtest_request.capital_reset_threshold_pct,
        'max_holding_period': backtest_request.max_holding_period,  # None = use default 52
        'transaction_cost_pct': backtest_request.transaction_cost_pct,
        'min_price': backtest_request.min_price,  # None = use default 10.0
        'min_turnover': backtest_request.min_turnover,  # None = use default 1000000.0
        'lookback_weeks': backtest_request.lookback_weeks,
        'lookback_months': backtest_request.lookback_months,
        'lookback_quarters': backtest_request.lookback_quarters
    }
    
    # Check if backtest already exists (using config hash or parameters)
    # For now, we'll skip this check since config_id is no longer used
    # You can implement a hash-based check if needed
    
    try:
        print(f"=== API BACKTEST REQUEST ===")
        print(f"User ID: {user_id}")
        print(f"Start Date: {start_date}")
        print(f"End Date: {end_date}")
        print(f"Date Range: {(end_date - start_date).days} days")
        print(f"Configuration: {config_dict}")
        
        # Initialize backtester with config dict instead of config_id
        print("Initializing backtester with custom config...")
        backtester = RSETFStrategyBacktester.from_config_dict(db, config_dict)
        
        # Run backtest
        print("Starting backtest execution...")
        backtester_results = backtester.run_backtest(start_date, end_date)
        
        # Recalculate metrics with risk_free_rate from request
        print(f"Recalculating metrics with risk_free_rate: {backtest_request.risk_free_rate}%")
        results = backtester.calculate_metrics(risk_free_rate=backtest_request.risk_free_rate)
        print(f"Backtest completed successfully. Total return: {results.get('total_return_pct', 0):.2f}%")
        
        # Additional JSON safety check at API level
        try:
            import json
            json.dumps(results)  # Test if results can be serialized
            print("Results passed JSON serialization test")
        except Exception as json_error:
            print(f"JSON serialization failed: {json_error}")
            # Apply aggressive conversion
            results = convert_all_to_native_types(results)
        
        # In-memory only: Return all data in response (no database save)
        # Structure response similar to ETF Strategy
        
        # Prepare performance data for charts
        performance_data = {
            "dates": [],
            "rs_strategy": [],
            "cumulative_investment": []
        }
        
        # Extract portfolio snapshots data
        portfolio_snapshots = results.get('portfolio_snapshots', [])
        if portfolio_snapshots:
            # Convert dates to string format for JSON serialization
            performance_data["dates"] = [
                snapshot.get('date') if isinstance(snapshot.get('date'), str) 
                else snapshot.get('date').isoformat() if hasattr(snapshot.get('date'), 'isoformat')
                else str(snapshot.get('date'))
                for snapshot in portfolio_snapshots
            ]
            performance_data["rs_strategy"] = [snapshot.get('total_value', 0) for snapshot in portfolio_snapshots]
            performance_data["cumulative_investment"] = [backtest_request.total_capital] * len(portfolio_snapshots)
        
        # Transform results to match frontend expectations (ETF-like structure)
        response_data = {
            "success": True,
            "rs_metrics": {
                'total_return': results.get('total_return_pct', 0),
                'annualized_return_pct': results.get('annualized_return_pct', 0),
                'cagr_pct': results.get('cagr_pct', 0),
                'xirr_pct': results.get('xirr_pct', 0),
                'max_drawdown': results.get('max_drawdown_pct', 0),
                'sharpe_ratio': results.get('sharpe_ratio', 0),
                'beta': results.get('beta', 1.0),
                'treynor_ratio': results.get('treynor_ratio', 0.0),
                'calmar_ratio': results.get('calmar_ratio', 0.0),
                'win_rate_pct': results.get('win_rate_pct', 0),
                'total_trades': results.get('total_trades', 0),
                'final_capital': results.get('final_capital', 0),
            },
            "backtest_result": {
                'total_investment': backtest_request.total_capital,
                'final_portfolio_value': results.get('final_capital', 0),
                'total_return': results.get('total_return_pct', 0),
                'total_brokerage': sum(trade.get('total_costs', 0) for trade in results.get('trades', [])),
            },
            "performance_data": performance_data,
            "trades": results.get('trades', []),
            "portfolio_snapshots": portfolio_snapshots,
            "transaction_log": results.get('trades', []),  # For costs calculation
            "rule_of_72": results.get('rule_of_72', {})
        }
        
        # Ensure the entire response is JSON-safe
        try:
            json.dumps(response_data)
        except Exception as final_error:
            print(f"Final JSON check failed: {final_error}")
            response_data = convert_all_to_native_types(response_data)
        
        return response_data
        
    except Exception as e:
        error_msg = f"Backtest failed: {str(e)}"
        full_traceback = traceback.format_exc()
        print(f"=== BACKTEST ERROR ===")
        print(f"ERROR: {error_msg}")
        print(f"ERROR TYPE: {type(e).__name__}")
        print(f"FULL TRACEBACK:\n{full_traceback}")
        print("=== END ERROR ===")
        
        # Provide more specific error messages
        if "No data available" in str(e):
            error_msg = "No stock data available for the specified date range"
        elif "timeout" in str(e).lower():
            error_msg = "Backtest timed out. Please try a shorter date range"
        elif "memory" in str(e).lower():
            error_msg = "Insufficient memory for backtest. Please try a shorter date range"
        elif "numpy" in str(e).lower() or "json" in str(e).lower():
            error_msg = f"Data serialization error: {str(e)}"
        else:
            # Include the actual error for debugging
            error_msg = f"Backtest execution error: {str(e)}"
        
        raise HTTPException(status_code=500, detail=error_msg)

@router.get("/backtests/{backtest_id}/trades")
async def get_backtest_trades(backtest_id: int, user_id: str = Query(..., description="User ID for the trades"), db: Session = Depends(get_backtest_db)):
    """Get trades for a specific backtest with transaction cost"""
    try:
        trades = db.query(TradeLog).filter(
            TradeLog.backtest_id == backtest_id,
            TradeLog.user_id == user_id
        ).order_by(TradeLog.date).all()
        
        # Convert to dict format with transaction cost
        result = []
        for trade in trades:
            trade_dict = {
                'id': trade.id,
                'date': trade.date,
                'symbol': trade.symbol,
                'action': trade.action,
                'quantity': trade.quantity,
                'price': trade.price,
                'amount': trade.amount,  # This already includes transaction costs
                'reason': trade.reason,
                'rs_score': trade.rs_score,
                'rs_rank': trade.rs_rank,
                # Add transaction cost for display
                'transaction_cost': trade.total_costs or 0
            }
            result.append(trade_dict)
        
        return result
        
    except Exception as e:
        if "no such column" in str(e):
            # Fallback for old backtests
            try:
                from sqlalchemy import text
                trades = db.execute(text("""
                    SELECT id, user_id, backtest_id, date, symbol, action, quantity, price, amount, reason, rs_score, rs_rank
                    FROM trade_logs 
                    WHERE backtest_id = %s AND user_id = %s
                    ORDER BY date
                """), (backtest_id, user_id)).fetchall()
                
                result = []
                for trade in trades:
                    trade_dict = {
                        'id': trade[0],
                        'date': trade[3],
                        'symbol': trade[4],
                        'action': trade[5],
                        'quantity': trade[6],
                        'price': trade[7],
                        'amount': trade[8],  # Original amount
                        'reason': trade[9],
                        'rs_score': trade[10],
                        'rs_rank': trade[11],
                        'transaction_cost': 0  # No cost data for old backtests
                    }
                    result.append(trade_dict)
                return result
            except Exception as fallback_error:
                logger.error(f"Fallback query also failed: {fallback_error}")
                return []
        else:
            raise e

@router.get("/backtests/{backtest_id}/portfolio")
async def get_backtest_portfolio(backtest_id: int, user_id: str = Query(..., description="User ID for the portfolio"), db: Session = Depends(get_backtest_db)):
    """Get portfolio snapshots for a specific backtest"""
    snapshots = db.query(PortfolioSnapshot).filter(
        PortfolioSnapshot.backtest_id == backtest_id,
        PortfolioSnapshot.user_id == user_id
    ).order_by(PortfolioSnapshot.date).all()
    
    # Parse positions JSON
    result = []
    for snapshot in snapshots:
        snapshot_dict = {
            'date': snapshot.date,
            'total_value': snapshot.total_value,
            'cash_balance': snapshot.cash_balance,
            'positions': json.loads(snapshot.positions_json) if snapshot.positions_json else {},
            'daily_pnl': snapshot.daily_pnl,
            'cumulative_pnl': snapshot.cumulative_pnl,
            'drawdown_pct': snapshot.drawdown_pct
        }
        result.append(snapshot_dict)
    
    return result

@router.get("/backtests/{backtest_id}/costs")
async def get_backtest_costs(backtest_id: int, user_id: str = Query(..., description="User ID for the cost analysis"), db: Session = Depends(get_backtest_db)):
    """Get detailed cost analysis for a specific backtest"""
    try:
        trades = db.query(TradeLog).filter(
            TradeLog.backtest_id == backtest_id,
            TradeLog.user_id == user_id
        ).order_by(TradeLog.date).all()
    except Exception as e:
        if "no such column" in str(e):
            # For old backtests without cost data, return empty cost analysis
            return {
                "total_brokerage": 0,
                "total_stt": 0,
                "total_stamp_duty": 0,
                "total_exchange_charges": 0,
                "total_sebi_charges": 0,
                "total_gst": 0,
                "total_costs": 0,
                "total_turnover": 0,
                "cost_percentage": 0,
                "avg_cost_per_trade": 0,
                "return_impact_pct": 0,
                "cost_breakdown": [],
                "message": "Cost analysis not available for this backtest. Run a new backtest to see detailed cost breakdown."
            }
        else:
            raise e
    
    if not trades:
        return {
            "total_brokerage": 0,
            "total_stt": 0,
            "total_stamp_duty": 0,
            "total_exchange_charges": 0,
            "total_sebi_charges": 0,
            "total_gst": 0,
            "total_costs": 0,
            "total_turnover": 0,
            "cost_percentage": 0,
            "avg_cost_per_trade": 0,
            "return_impact_pct": 0,
            "cost_breakdown": []
        }
    
    # Calculate cost totals
    total_brokerage = sum(trade.brokerage or 0 for trade in trades)
    total_stt = sum(trade.stt or 0 for trade in trades)
    total_stamp_duty = sum(trade.stamp_duty or 0 for trade in trades)
    total_exchange_charges = sum(trade.exchange_charges or 0 for trade in trades)
    total_sebi_charges = sum(trade.sebi_charges or 0 for trade in trades)
    total_gst = sum(trade.gst or 0 for trade in trades)
    total_costs = sum(trade.total_costs or 0 for trade in trades)
    total_turnover = sum(trade.transaction_value or 0 for trade in trades)
    
    # Calculate metrics
    cost_percentage = (total_costs / total_turnover * 100) if total_turnover > 0 else 0
    avg_cost_per_trade = total_costs / len(trades) if trades else 0
    
    # Get backtest result for return impact calculation
    backtest_result = db.query(BacktestResult).filter(
        BacktestResult.id == backtest_id,
        BacktestResult.user_id == user_id
    ).first()
    
    return_impact_pct = 0
    if backtest_result and backtest_result.final_capital:
        # Calculate what the return would be without costs
        gross_final_capital = backtest_result.final_capital + total_costs
        gross_return = ((gross_final_capital - backtest_result.final_capital) / backtest_result.final_capital * 100) if backtest_result.final_capital else 0
        actual_return = backtest_result.total_return_pct or 0
        return_impact_pct = gross_return - actual_return
    
    # Cost breakdown by trade
    cost_breakdown = []
    for trade in trades:
        cost_breakdown.append({
            "date": trade.date,
            "symbol": trade.symbol,
            "action": trade.action,
            "transaction_value": trade.transaction_value or 0,
            "brokerage": trade.brokerage or 0,
            "stt": trade.stt or 0,
            "stamp_duty": trade.stamp_duty or 0,
            "exchange_charges": trade.exchange_charges or 0,
            "sebi_charges": trade.sebi_charges or 0,
            "gst": trade.gst or 0,
            "total_costs": trade.total_costs or 0,
            "net_amount": trade.net_amount or 0
        })
    
    return {
        "total_brokerage": total_brokerage,
        "total_stt": total_stt,
        "total_stamp_duty": total_stamp_duty,
        "total_exchange_charges": total_exchange_charges,
        "total_sebi_charges": total_sebi_charges,
        "total_gst": total_gst,
        "total_costs": total_costs,
        "total_turnover": total_turnover,
        "cost_percentage": cost_percentage,
        "avg_cost_per_trade": avg_cost_per_trade,
        "return_impact_pct": return_impact_pct,
        "cost_breakdown": cost_breakdown
    }

# Market data endpoints
@router.get("/etfs/universe/{universe}")
async def get_etfs_by_universe(universe: str, db: Session = Depends(get_db)):
    """Get ETFs for a specific universe"""
    try:
        # For ETFs, accept ALL_ETFS or any universe value
        # The backtester will query the database for available ETFs
        universe = universe.upper()
        
        # Create a temporary backtester instance to access ETF methods
        temp_config = {
            'main_index': '^NSEI',
            'etf_universe': universe,
            'max_positions': 20,
            'position_size_pct': 5.0,
            'total_capital': 1000000.0,
            'stop_loss_pct': 15.0,
            'buffer_capital_pct': 10.0,
            'capital_reset_threshold_pct': 25.0,
            'max_holding_period': 52,
            'transaction_cost_pct': 0.1,
            'min_price': 10.0,
            'min_turnover': 1000000.0,
            'lookback_weeks': 5,
            'lookback_months': 20,
            'lookback_quarters': 60
        }
        backtester = RSETFStrategyBacktester.from_config_dict(db, temp_config)
        
        # Get ETFs for the universe
        etfs = backtester.get_custom_etf_universe()
        
        # Format response with ETF details
        etf_list = []
        for symbol in etfs:
            etf_list.append({
                "symbol": symbol,
                "value": symbol,
                "label": symbol
            })
        
        return {
            "success": True,
            "universe": universe,
            "etfs": etf_list,
            "count": len(etf_list)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching ETFs for universe {universe}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching ETFs: {str(e)}")

@router.post("/date-range")
async def calculate_rs_strategy_date_range(request: Dict[str, Any], db: Session = Depends(get_db)):
    """Calculate common date range for selected ETFs for RS ETF Strategy"""
    try:
        tickers = request.get("tickers", [])
        if not tickers:
            raise HTTPException(status_code=400, detail="No tickers provided")
        
        print(f"Calculating date range for RS strategy tickers: {tickers}")
        
        # Create a temporary backtester instance to access the method
        temp_config = {
            'main_index': '^NSEI',
            'etf_universe': 'ALL_ETFS',
            'max_positions': 20,
            'position_size_pct': 5.0,
            'total_capital': 1000000.0,
            'stop_loss_pct': 15.0,
            'buffer_capital_pct': 10.0,
            'capital_reset_threshold_pct': 25.0,
            'max_holding_period': 52,
            'transaction_cost_pct': 0.1,
            'min_price': 10.0,
            'min_turnover': 1000000.0,
            'lookback_weeks': 5,
            'lookback_months': 20,
            'lookback_quarters': 60
        }
        
        backtester = RSETFStrategyBacktester.from_config_dict(db, temp_config)
        start_date, end_date, years = backtester.calculate_common_date_range(tickers)
        
        if start_date and end_date:
            return {
                "start_date": start_date,
                "end_date": end_date,
                "years": years
            }
        else:
            raise HTTPException(status_code=400, detail="Could not calculate date range. Please check if stocks have data available.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating date range: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error calculating date range: {str(e)}")

@router.get("/market-data/symbols")
async def get_nifty500_symbols(db: Session = Depends(get_db)):
    """Get Nifty 500 constituent symbols"""
    from .database import Nifty500Constituents
    symbols = db.query(Nifty500Constituents).all()
    result = [{"symbol": s.symbol, "start_date": s.start_date, "end_date": s.end_date, "total_records": s.total_records, "data_source": s.data_source} for s in symbols]
    return result

@router.get("/market-data/etf/{symbol}")
async def get_stock_data(symbol: str, start_date: datetime, end_date: datetime, db: Session = Depends(get_db)):
    """Get ETF data for a specific symbol"""
    from .database import ETFData
    data = db.query(ETFData).filter(
        ETFData.symbol == symbol,
        ETFData.date >= start_date,
        ETFData.date <= end_date
    ).order_by(ETFData.date).all()
    
    return [{
        "date": d.date,
        "open": d.open,
        "high": d.high,
        "low": d.low,
        "close": d.close,
        "adjusted_close": d.adjusted_close,
        "volume": d.volume
    } for d in data]

@router.get("/market-data/index/{index_symbol}")
async def get_index_data(index_symbol: str, start_date: datetime, end_date: datetime, db: Session = Depends(get_db)):
    """Get index data for a specific index"""
    from .database import IndexData
    data = db.query(IndexData).filter(
        IndexData.symbol == index_symbol,
        IndexData.date >= start_date,
        IndexData.date <= end_date
    ).order_by(IndexData.date).all()
    
    return [{
        "date": d.date,
        "open": d.open,
        "high": d.high,
        "low": d.low,
        "close": d.close,
        "adjusted_close": d.adjusted_close,
        "volume": d.volume
    } for d in data]

# Utility endpoints
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

@router.get("/health/database")
async def database_health_check():
    """Database health check endpoint"""
    return check_database_health()

@router.post("/health/database/reset")
async def reset_database():
    """Reset database connections endpoint"""
    try:
        reset_database_connections()
        return {"message": "Database connections reset successfully", "timestamp": datetime.now()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset database connections: {str(e)}")

@router.post("/health/database/force-unlock")
async def force_unlock_database_endpoint():
    """Force unlock database endpoint - removes lock files and resets connections"""
    try:
        success = force_unlock_database()
        if success:
            return {"message": "Database force unlocked successfully", "timestamp": datetime.now()}
        else:
            raise HTTPException(status_code=500, detail="Failed to force unlock database")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to force unlock database: {str(e)}")

@router.get("/status")
async def get_system_status(user_id: str = Query(..., description="User ID to filter statistics"), db: Session = Depends(get_db), backtest_db: Session = Depends(get_backtest_db)):
    """Get system status and statistics for a specific user"""
    config_count = db.query(StrategyConfig).filter(StrategyConfig.user_id == user_id).count()
    backtest_count = backtest_db.query(BacktestResult).filter(BacktestResult.user_id == user_id).count()
    
    return {
        "user_id": user_id,
        "configurations": config_count,
        "backtests": backtest_count,
        "timestamp": datetime.now()
    }

# ============================================================================
# RS STRATEGY EOD DATA FETCHING ENDPOINTS
# ============================================================================

@router.post("/eod-data/fetch-daily")
async def fetch_daily_eod_data(days_back: int = 1):
    """Fetch daily EOD data for Nifty 500 stocks and Nifty 50 index"""
    if not RSEODDataFetcher:
        raise HTTPException(status_code=500, detail="EOD data fetcher not available")
    
    try:
        fetcher = RSEODDataFetcher()
        result = fetcher.fetch_daily_data(days_back=days_back)
        
        return {
            "status": "success",
            "message": "Daily EOD data fetch completed",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error fetching daily EOD data: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching daily EOD data: {str(e)}")

@router.post("/eod-data/fetch-historical")
async def fetch_historical_eod_data(days_back: int = 365):
    """Fetch historical EOD data for Nifty 500 stocks and Nifty 50 index"""
    if not RSEODDataFetcher:
        raise HTTPException(status_code=500, detail="EOD data fetcher not available")
    
    try:
        fetcher = RSEODDataFetcher()
        result = fetcher.fetch_historical_data(days_back=days_back)
        
        return {
            "status": "success",
            "message": "Historical EOD data fetch completed",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error fetching historical EOD data: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching historical EOD data: {str(e)}")

@router.get("/eod-data/summary")
async def get_eod_data_summary():
    """Get summary of EOD data in the database"""
    if not RSEODDataFetcher:
        raise HTTPException(status_code=500, detail="EOD data fetcher not available")
    
    try:
        fetcher = RSEODDataFetcher()
        summary = fetcher.get_data_summary()
        
        return {
            "status": "success",
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error getting EOD data summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting EOD data summary: {str(e)}")

@router.post("/eod-data/cleanup")
async def cleanup_old_eod_data(days_to_keep: int = 365):
    """Clean up old EOD data from the database"""
    if not RSEODDataFetcher:
        raise HTTPException(status_code=500, detail="EOD data fetcher not available")
    
    try:
        fetcher = RSEODDataFetcher()
        fetcher.cleanup_old_data(days_to_keep=days_to_keep)
        
        return {
            "status": "success",
            "message": f"Cleaned up EOD data older than {days_to_keep} days"
        }
    except Exception as e:
        logger.error(f"Error cleaning up EOD data: {e}")
        raise HTTPException(status_code=500, detail=f"Error cleaning up EOD data: {str(e)}")

# ============================================================================
# RS STRATEGY SCHEDULER ENDPOINTS
# ============================================================================

@router.post("/scheduler/trigger-eod")
async def trigger_eod_fetch():
    """Manually trigger EOD data fetch"""
    if not RSEODDataFetcher:
        raise HTTPException(status_code=500, detail="EOD data fetcher not available")
    
    try:
        fetcher = RSEODDataFetcher()
        result = fetcher.fetch_daily_data(days_back=1)
        
        return {
            "status": "success",
            "message": "EOD data fetch triggered successfully",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error triggering EOD fetch: {e}")
        raise HTTPException(status_code=500, detail=f"Error triggering EOD fetch: {str(e)}")

@router.post("/scheduler/trigger-all")
async def trigger_all_rs_tasks():
    """Manually trigger EOD data fetch"""
    if not RSEODDataFetcher:
        raise HTTPException(status_code=500, detail="RS strategy modules not available")
    
    try:
        results = {}
        
        # Fetch EOD data
        fetcher = RSEODDataFetcher()
        eod_result = fetcher.fetch_daily_data(days_back=1)
        results["eod_fetch"] = eod_result
        
        return {
            "status": "success",
            "message": "All RS strategy tasks completed successfully",
            "results": results
        }
    except Exception as e:
        logger.error(f"Error triggering all RS tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Error triggering all RS tasks: {str(e)}")

@router.get("/get-run-status/{run_id}/{user_id}")
async def get_run_status(run_id: str, user_id: str):
    """
    Get the status of a specific run
    
    Args:
        run_id: Run ID from saved_rs_strategies table
        user_id: User ID
        
    Returns:
        Run status information
    """
    try:
        from .database import SavedETFStrategy, SessionLocal
        from sqlalchemy import text
        
        db = SessionLocal()
        try:
            # Query using SQLAlchemy
            query = text("""
                SELECT strategy_name, status, rs_etf_universe, webhook_url, client_information_json
                FROM rs_etf_saved_strategies
                WHERE run_id = :run_id AND user_id = :user_id
            """)
            
            result = db.execute(query, {"run_id": run_id, "user_id": user_id})
            row = result.fetchone()
            
            if not row:
                return {
                    "success": False,
                    "run_id": run_id,
                    "user_id": user_id,
                    "status": "not_found",
                    "message": "Run not found"
                }
            
            # Get run status from saved_rs_strategies table
            strategy_data = {
                'strategy_name': row[0],
                'status': row[1],
                'stock_universe': row[2] or 'ALL_ETFS',
                'webhook_url': row[3],
                'client_information': json.loads(row[4]) if row[4] else {}
            }
        finally:
            db.close()
        
        return {
            "success": True,
            "run_id": run_id,
            "user_id": user_id,
            "status": strategy_data['status'],
            "strategy_name": strategy_data['strategy_name'],
            "stock_universe": strategy_data.get('stock_universe', 'NIFTY500'),
            "webhook_url": strategy_data.get('webhook_url'),
            "client_information": strategy_data.get('client_information', {}),
            "message": f"Run status: {strategy_data['status']}"
        }
        
    except Exception as e:
        logger.error(f"Error getting run status for {run_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting run status: {str(e)}")

# Delete backtest result endpoint
@router.delete("/backtests/{backtest_id}")
async def delete_backtest_result(backtest_id: int, user_id: str = Query(..., description="User ID"), db: Session = Depends(get_backtest_db)):
    """Delete a specific backtest result and all related data"""
    try:
        # Find the backtest result
        result = db.query(BacktestResult).filter(
            BacktestResult.id == backtest_id,
            BacktestResult.user_id == user_id
        ).first()
        
        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")
        
        # Delete related trade logs first (foreign key constraint)
        trade_logs_deleted = db.query(TradeLog).filter(TradeLog.backtest_id == backtest_id).delete()
        print(f"Deleted {trade_logs_deleted} trade logs for backtest {backtest_id}")
        
        # Delete related portfolio snapshots
        portfolio_snapshots_deleted = db.query(PortfolioSnapshot).filter(PortfolioSnapshot.backtest_id == backtest_id).delete()
        print(f"Deleted {portfolio_snapshots_deleted} portfolio snapshots for backtest {backtest_id}")
        
        # Delete the backtest result
        db.delete(result)
        db.commit()
        
        print(f"Successfully deleted backtest {backtest_id} for user {user_id}")
        return {"success": True, "message": "Backtest result deleted successfully"}
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        db.rollback()
        print(f"Error deleting backtest {backtest_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete backtest: {str(e)}")

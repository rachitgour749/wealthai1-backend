"""
Helper functions for Strategy and Signal database operations
Provides convenient functions for common database operations using PostgreSQL
"""

from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from Databases.app_data_db_connection import get_session
from Databases.strategy_models import (
    ETFSavedStrategy, StockSavedStrategy, RSEtFSavedStrategy, CustomStrategy,
    LiveSignal, LiveRun, LiveStockSignal, LiveStockRun,
    ExecutedDetail, Strategy, SaveJson, DeployDetail
)
import json
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ETF STRATEGY HELPERS
# ============================================================================

def get_etf_strategy_by_id(strategy_id: int) -> Optional[Dict[str, Any]]:
    """Get ETF strategy by ID"""
    session = get_session()
    try:
        strategy = session.query(ETFSavedStrategy).filter(ETFSavedStrategy.id == strategy_id).first()
        if strategy:
            return _etf_strategy_to_dict(strategy)
        return None
    finally:
        session.close()


def get_etf_strategies_by_user(user_id: str) -> List[Dict[str, Any]]:
    """Get all ETF strategies for a user"""
    session = get_session()
    try:
        strategies = session.query(ETFSavedStrategy).filter(
            ETFSavedStrategy.user_id == user_id
        ).order_by(ETFSavedStrategy.created_timestamp.desc()).all()
        return [_etf_strategy_to_dict(s) for s in strategies]
    finally:
        session.close()


def get_etf_strategy_by_run_id(run_id: str) -> Optional[Dict[str, Any]]:
    """Get ETF strategy by run_id"""
    session = get_session()
    try:
        strategy = session.query(ETFSavedStrategy).filter(ETFSavedStrategy.run_id == run_id).first()
        if strategy:
            return _etf_strategy_to_dict(strategy)
        return None
    finally:
        session.close()


def save_etf_strategy(strategy_data: Dict[str, Any]) -> int:
    """Save or update ETF strategy"""
    session = get_session()
    try:
        if 'id' in strategy_data and strategy_data['id']:
            # Update existing
            strategy = session.query(ETFSavedStrategy).filter(
                ETFSavedStrategy.id == strategy_data['id']
            ).first()
            if not strategy:
                raise ValueError(f"ETF strategy with id {strategy_data['id']} not found")
        else:
            # Create new
            strategy = ETFSavedStrategy()
        
        # Update fields
        for key, value in strategy_data.items():
            if hasattr(strategy, key) and key != 'id':
                setattr(strategy, key, value)
        
        session.add(strategy)
        session.commit()
        session.refresh(strategy)
        return strategy.id
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving ETF strategy: {e}")
        raise
    finally:
        session.close()


def _etf_strategy_to_dict(strategy: ETFSavedStrategy) -> Dict[str, Any]:
    """Convert ETF strategy model to dictionary"""
    return {
        'id': strategy.id,
        'strategy_name': strategy.strategy_name,
        'strategy_type': strategy.strategy_type,
        'user_id': strategy.user_id,
        'tickers': strategy.tickers,
        'start_date': strategy.start_date,
        'end_date': strategy.end_date,
        'capital_per_week': strategy.capital_per_week,
        'accumulation_weeks': strategy.accumulation_weeks,
        'brokerage_percent': strategy.brokerage_percent,
        'compounding_enabled': strategy.compounding_enabled,
        'risk_free_rate': strategy.risk_free_rate,
        'use_custom_dates': strategy.use_custom_dates,
        'backtest_results': strategy.backtest_results,
        'created_at': strategy.created_at,
        'created_timestamp': strategy.created_timestamp.isoformat() if strategy.created_timestamp else None,
        'config_id': strategy.config_id,
        'backtest_id': strategy.backtest_id,
        'etf_universe': strategy.etf_universe,
        'strategy_config': strategy.strategy_config,
        'run_id': strategy.run_id,
        'client_information_json': strategy.client_information_json,
        'webhook_url': strategy.webhook_url,
        'status': strategy.status,
        'execution_date': strategy.execution_date,
        'ltp': strategy.ltp,
        'reference_capital': strategy.reference_capital,
        'last_execution_date': strategy.last_execution_date,
        'next_execution_date': strategy.next_execution_date,
        'deployment_data': strategy.deployment_data,
        'etf_count': strategy.etf_count,
        'etf_names': strategy.etf_names,
        'updated_at': strategy.updated_at.isoformat() if strategy.updated_at else None,
    }


# ============================================================================
# STOCK STRATEGY HELPERS
# ============================================================================

def get_stock_strategy_by_id(strategy_id: int) -> Optional[Dict[str, Any]]:
    """Get Stock strategy by ID"""
    session = get_session()
    try:
        strategy = session.query(StockSavedStrategy).filter(StockSavedStrategy.id == strategy_id).first()
        if strategy:
            return _stock_strategy_to_dict(strategy)
        return None
    finally:
        session.close()


def get_stock_strategies_by_user(user_id: str) -> List[Dict[str, Any]]:
    """Get all Stock strategies for a user"""
    session = get_session()
    try:
        strategies = session.query(StockSavedStrategy).filter(
            StockSavedStrategy.user_id == user_id
        ).order_by(StockSavedStrategy.created_timestamp.desc()).all()
        return [_stock_strategy_to_dict(s) for s in strategies]
    finally:
        session.close()


def get_stock_strategy_by_run_id(run_id: str) -> Optional[Dict[str, Any]]:
    """Get Stock strategy by run_id"""
    session = get_session()
    try:
        strategy = session.query(StockSavedStrategy).filter(StockSavedStrategy.run_id == run_id).first()
        if strategy:
            return _stock_strategy_to_dict(strategy)
        return None
    finally:
        session.close()


def save_stock_strategy(strategy_data: Dict[str, Any]) -> int:
    """Save or update Stock strategy"""
    session = get_session()
    try:
        if 'id' in strategy_data and strategy_data['id']:
            strategy = session.query(StockSavedStrategy).filter(
                StockSavedStrategy.id == strategy_data['id']
            ).first()
            if not strategy:
                raise ValueError(f"Stock strategy with id {strategy_data['id']} not found")
        else:
            strategy = StockSavedStrategy()
        
        for key, value in strategy_data.items():
            if hasattr(strategy, key) and key != 'id':
                setattr(strategy, key, value)
        
        session.add(strategy)
        session.commit()
        session.refresh(strategy)
        return strategy.id
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving Stock strategy: {e}")
        raise
    finally:
        session.close()


def _stock_strategy_to_dict(strategy: StockSavedStrategy) -> Dict[str, Any]:
    """Convert Stock strategy model to dictionary"""
    return {
        'id': strategy.id,
        'strategy_name': strategy.strategy_name,
        'strategy_type': strategy.strategy_type,
        'user_id': strategy.user_id,
        'tickers': strategy.tickers,
        'start_date': strategy.start_date,
        'end_date': strategy.end_date,
        'capital_per_week': strategy.capital_per_week,
        'accumulation_weeks': strategy.accumulation_weeks,
        'brokerage_percent': strategy.brokerage_percent,
        'compounding_enabled': strategy.compounding_enabled,
        'risk_free_rate': strategy.risk_free_rate,
        'use_custom_dates': strategy.use_custom_dates,
        'backtest_results': strategy.backtest_results,
        'created_at': strategy.created_at,
        'created_timestamp': strategy.created_timestamp.isoformat() if strategy.created_timestamp else None,
        'config_id': strategy.config_id,
        'backtest_id': strategy.backtest_id,
        'stock_universe': strategy.stock_universe,
        'strategy_config': strategy.strategy_config,
        'run_id': strategy.run_id,
        'client_information_json': strategy.client_information_json,
        'webhook_url': strategy.webhook_url,
        'status': strategy.status,
        'execution_date': strategy.execution_date,
        'reference_capital': strategy.reference_capital,
        'last_execution_date': strategy.last_execution_date,
        'next_execution_date': strategy.next_execution_date,
        'updated_at': strategy.updated_at.isoformat() if strategy.updated_at else None,
    }


# ============================================================================
# LIVE SIGNAL HELPERS
# ============================================================================

def save_live_signal(signal_data: Dict[str, Any]) -> int:
    """Save a live ETF signal"""
    session = get_session()
    try:
        signal = LiveSignal(**signal_data)
        session.add(signal)
        session.commit()
        session.refresh(signal)
        return signal.id
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving live signal: {e}")
        raise
    finally:
        session.close()


def get_live_signals_by_run_id(run_id: str) -> List[Dict[str, Any]]:
    """Get all live signals for a run_id"""
    session = get_session()
    try:
        signals = session.query(LiveSignal).filter(LiveSignal.run_id == run_id).all()
        return [_live_signal_to_dict(s) for s in signals]
    finally:
        session.close()


def save_live_run(run_data: Dict[str, Any]) -> int:
    """Save a live run"""
    session = get_session()
    try:
        run = LiveRun(**run_data)
        session.add(run)
        session.commit()
        session.refresh(run)
        return run.id
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving live run: {e}")
        raise
    finally:
        session.close()


def save_live_stock_signal(signal_data: Dict[str, Any]) -> int:
    """Save a live stock signal"""
    session = get_session()
    try:
        signal = LiveStockSignal(**signal_data)
        session.add(signal)
        session.commit()
        session.refresh(signal)
        return signal.id
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving live stock signal: {e}")
        raise
    finally:
        session.close()


def save_live_stock_run(run_data: Dict[str, Any]) -> int:
    """Save a live stock run"""
    session = get_session()
    try:
        run = LiveStockRun(**run_data)
        session.add(run)
        session.commit()
        session.refresh(run)
        return run.id
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving live stock run: {e}")
        raise
    finally:
        session.close()


def _live_signal_to_dict(signal: LiveSignal) -> Dict[str, Any]:
    """Convert live signal model to dictionary"""
    return {
        'id': signal.id,
        'run_id': signal.run_id,
        'user_id': signal.user_id,
        'strategy_version': signal.strategy_version,
        'signal_date': signal.signal_date,
        'signal_symbol': signal.signal_symbol,
        'etf_name': signal.etf_name,
        'side': signal.side,
        'score': signal.score,
        'reason': signal.reason,
        'payload_json': signal.payload_json,
        'created_at': signal.created_at.isoformat() if signal.created_at else None,
    }


# ============================================================================
# EXECUTION HELPERS
# ============================================================================

def save_execution_record(execution_data: Dict[str, Any]) -> int:
    """Save an execution record"""
    session = get_session()
    try:
        record = ExecutedDetail(**execution_data)
        session.add(record)
        session.commit()
        session.refresh(record)
        return record.id
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving execution record: {e}")
        raise
    finally:
        session.close()


def get_execution_history(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Get execution history with filters"""
    session = get_session()
    try:
        query = session.query(ExecutedDetail)
        
        if user_id:
            query = query.filter(ExecutedDetail.user_id == user_id)
        if run_id:
            query = query.filter(ExecutedDetail.run_id == run_id)
        
        query = query.order_by(ExecutedDetail.created_at.desc())
        
        if limit:
            query = query.limit(limit)
        
        records = query.all()
        return [_execution_record_to_dict(r) for r in records]
    finally:
        session.close()


def _execution_record_to_dict(record: ExecutedDetail) -> Dict[str, Any]:
    """Convert execution record to dictionary"""
    result = {
        'id': record.id,
        'user_id': record.user_id,
        'run_id': record.run_id,
        'signal_symbol': record.signal_symbol,
        'side': record.side,
        'success': record.success,
        'error_message': record.error_message,
        'webhook_url': record.webhook_url,
        'execution_date': record.execution_date,
        'created_at': record.created_at.isoformat() if record.created_at else None,
    }
    
    # Parse JSON fields
    if record.response_data:
        try:
            result['response_data'] = json.loads(record.response_data)
        except:
            result['response_data'] = record.response_data
    
    if record.payload_sent:
        try:
            result['payload_sent'] = json.loads(record.payload_sent)
        except:
            result['payload_sent'] = record.payload_sent
    
    return result


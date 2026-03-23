"""
Deployment Helper API for RS Strategies and ETF/Stock Deployments
Migrated from SQLite to PostgreSQL (Neon)
"""

from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from datetime import datetime, timedelta
import json
import re
import time
import logging
import holidays

from Databases.app_data_db_connection import get_session, create_connection, init_database

logger = logging.getLogger(__name__)

# Create router
deployment_router = APIRouter(prefix="/api", tags=["Deployment"])


# ============================================================================
# RS STRATEGY FUNCTIONS (PostgreSQL)
# ============================================================================

# ============================================================================
# RS STRATEGY FUNCTIONS (PostgreSQL - Unified SavedInstance)
# ============================================================================

def init_saved_instances_table():
    """Ensure the centralized saved_instances table is initialized"""
    try:
        if not create_connection():
            logger.error("Failed to connect to PostgreSQL database")
            return False
        
        if not init_database():
            logger.error("Failed to initialize database tables")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error initializing saved_instances table: {e}")
        return False


@deployment_router.post("/save-rs-etf-strategy")
async def save_rs_etf_strategy(request: dict):
    """Save RS ETF Strategy configuration to unified saved_instances table"""
    session = None
    try:
        # Initialize database if needed
        if not init_saved_instances_table():
            return {
                "success": False,
                "message": "Failed to initialize database"
            }

        session = get_session()
        
        # Check if strategy already exists
        result = session.execute(text("""
            SELECT id FROM saved_instances 
            WHERE strategy_name = :strategy_name AND user_id = :user_id
        """), {
            "strategy_name": request.get('strategy_name'),
            "user_id": request.get('user_id')
        })
        
        if result.fetchone():
            return {
                "success": False,
                "message": "Strategy with this name already exists",
                "strategy_exists": True
            }
        
        # Prepare parameters for strategies_parameters column
        strategies_params = {
            "rs_etf_universe": request.get('rs_etf_universe') or request.get('stock_universe') or 'ALL_ETFS',
            "backtest_results": request.get('backtest_results', {}),
            "strategy_config": request.get('strategy_config', {})
        }

        # Insert new strategy
        result = session.execute(text("""
            INSERT INTO saved_instances (
                strategy_name, strategy_type, user_id, 
                start_date, end_date, strategies_parameters, created_at, status
            ) VALUES (
                :strategy_name, :strategy_type, :user_id, 
                :start_date, :end_date, :strategies_params, :created_at, :status
            )
            RETURNING id
        """), {
            "strategy_name": request.get('strategy_name'),
            "strategy_type": request.get('strategy_type', 'RS ETF Strategy'),
            "user_id": request.get('user_id'),
            "start_date": request.get('start_date'),
            "end_date": request.get('end_date'),
            "strategies_params": json.dumps(strategies_params),
            "created_at": request.get('created_at', datetime.now()),
            "status": request.get('status', 'not deploy')
        })
        
        strategy_id = result.scalar_one()
        session.commit()
        
        return {
            "success": True,
            "message": "RS ETF Strategy saved successfully",
            "strategy_id": strategy_id,
            "strategy_exists": False
        }
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Error saving RS ETF strategy: {e}")
        return {
            "success": False,
            "message": f"Error saving RS ETF strategy: {str(e)}"
        }
    finally:
        if session:
            session.close()


@deployment_router.get("/get-saved-rs-etf-strategies/{user_id}")
async def get_saved_rs_etf_strategies(user_id: str):
    """Get all saved RS ETF strategies for a specific user from unified saved_instances table"""
    session = None
    try:
        session = get_session()
        
        result = session.execute(text("""
            SELECT * FROM saved_instances 
            WHERE user_id = :user_id AND strategy_type LIKE '%ETF%'
            ORDER BY created_at DESC
        """), {"user_id": user_id})
        
        rows = result.fetchall()
        columns = result.keys()
        
        strategies = []
        for row in rows:
            strategy_dict = dict(zip(columns, row))
            
            # Map centralized fields back to expected format if necessary
            if strategy_dict.get('strategies_parameters'):
                params = strategy_dict['strategies_parameters']
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except:
                        params = {}
                
                # Flatten some params if frontend expects them at top level
                strategy_dict.update(params)
            
            # Ensure client_info is parsed
            if strategy_dict.get('client_info'):
                if isinstance(strategy_dict['client_info'], str):
                    try:
                        strategy_dict['client_info'] = json.loads(strategy_dict['client_info'])
                    except:
                        strategy_dict['client_info'] = {}
            
            strategies.append(strategy_dict)
        
        return {
            "success": True,
            "strategies": strategies
        }
    except Exception as e:
        logger.error(f"Error fetching RS ETF strategies: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching RS ETF strategies: {str(e)}")
    finally:
        if session:
            session.close()


@deployment_router.post("/save-rs-stock-strategy")
async def save_rs_stock_strategy(request: dict):
    """Save RS Stock Strategy configuration to unified saved_instances table"""
    session = None
    try:
        if not init_saved_instances_table():
            return {
                "success": False,
                "message": "Failed to initialize database"
            }

        session = get_session()
        
        # Check if strategy already exists
        result = session.execute(text("""
            SELECT id FROM saved_instances 
            WHERE strategy_name = :strategy_name AND user_id = :user_id
        """), {
            "strategy_name": request.get('strategy_name'),
            "user_id": request.get('user_id')
        })
        
        if result.fetchone():
            return {
                "success": False,
                "message": "Strategy with this name already exists",
                "strategy_exists": True
            }
        
        # Prepare parameters
        strategies_params = {
            "stock_universe": request.get('stock_universe') or 'NIFTY50',
            "backtest_results": request.get('backtest_results', {}),
            "strategy_config": request.get('strategy_config', {})
        }

        # Insert new strategy
        result = session.execute(text("""
            INSERT INTO saved_instances (
                strategy_name, strategy_type, user_id, 
                start_date, end_date, strategies_parameters, created_at, status
            ) VALUES (
                :strategy_name, :strategy_type, :user_id, 
                :start_date, :end_date, :strategies_params, :created_at, :status
            )
            RETURNING id
        """), {
            "strategy_name": request.get('strategy_name'),
            "strategy_type": request.get('strategy_type', 'RS Stock Strategy'),
            "user_id": request.get('user_id'),
            "start_date": request.get('start_date'),
            "end_date": request.get('end_date'),
            "strategies_params": json.dumps(strategies_params),
            "created_at": request.get('created_at', datetime.now()),
            "status": request.get('status', 'not deploy')
        })
        
        strategy_id = result.scalar_one()
        session.commit()
        
        return {
            "success": True,
            "message": "RS Stock Strategy saved successfully",
            "strategy_id": strategy_id,
            "strategy_exists": False
        }
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Error saving RS Stock strategy: {e}")
        return {
            "success": False,
            "message": f"Error saving RS Stock strategy: {str(e)}"
        }
    finally:
        if session:
            session.close()





@deployment_router.get("/get-saved-rs-strategies/{user_id}")
async def get_saved_rs_strategies(user_id: str):
    """Get all saved RS Stock strategies for a specific user from unified saved_instances table"""
    session = None
    try:
        session = get_session()
        
        result = session.execute(text("""
            SELECT * FROM saved_instances 
            WHERE user_id = :user_id AND strategy_type LIKE '%Stock%'
            ORDER BY created_at DESC
        """), {"user_id": user_id})
        
        rows = result.fetchall()
        columns = result.keys()
        
        strategies = []
        for row in rows:
            strategy_dict = dict(zip(columns, row))
            
            if strategy_dict.get('strategies_parameters'):
                params = strategy_dict['strategies_parameters']
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except:
                        params = {}
                strategy_dict.update(params)
                
            if strategy_dict.get('client_info'):
                if isinstance(strategy_dict['client_info'], str):
                    try:
                        strategy_dict['client_info'] = json.loads(strategy_dict['client_info'])
                    except:
                        strategy_dict['client_info'] = {}
            
            strategies.append(strategy_dict)
        
        return {
            "success": True,
            "strategies": strategies
        }
    except Exception as e:
        logger.error(f"Error fetching RS Stock strategies: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching RS Stock strategies: {str(e)}")
    finally:
        if session:
            session.close()


def calculate_next_execution_date(current_date: datetime = None) -> str:
    """
    Calculate the next execution date (next Monday or Tuesday if holiday).
    Uses dynamic holiday detection for India (NSE).
    """
    if current_date is None:
        current_date = datetime.now()
    
    logger.info(f"📅 Calculating next execution date from: {current_date}")
    
    # Get India holidays for current and next year
    year = current_date.year
    in_holidays = holidays.India(years=[year, year + 1])
    
    # Calculate days until next Monday (0 = Monday, 6 = Sunday)
    days_ahead = 0 - current_date.weekday()
    if days_ahead <= 0: # Target day already happened this week
        days_ahead += 7
        
    next_execution = current_date + timedelta(days=days_ahead)
    
    # Check if Monday is a holiday
    # If holiday, move to next day until a non-holiday is found
    # Limit to 5 days to avoid infinite loops (though unlikely)
    for _ in range(5):
        if next_execution in in_holidays:
            logger.info(f"Skipping holiday: {next_execution.strftime('%Y-%m-%d')} - {in_holidays.get(next_execution)}")
            next_execution += timedelta(days=1)
        else:
            break
    
    result = next_execution.strftime('%Y-%m-%d')
    logger.info(f"📅 Final next execution date: {result}")
    return result





@deployment_router.post("/save-rs-deployment")
async def save_deployment(request: dict):
    """Save strategy deployment to unified saved_instances table"""
    session = None
    try:
        user_email = request.get('user_email')
        strategy_name = request.get('strategy_name')
        client_info = request.get('client_information_json', '{}')
        webhook_url = request.get('webhook_url', '')
        email_notification = request.get('email_notification', False)
        telegram_notification = request.get('telegram_notification', False)
        
        if not user_email or not strategy_name:
            return {"success": False, "message": "user_email and strategy_name are required"}
        
        session = get_session()
        result = session.execute(text("""
            SELECT id, strategy_type FROM saved_instances 
            WHERE strategy_name = :strategy_name AND user_id = :user_id
        """), {"strategy_name": strategy_name, "user_id": user_email})
        
        row = result.fetchone()
        if not row:
            return {"success": False, "message": f"Strategy '{strategy_name}' not found"}
        
        strategy_id, strategy_type = row
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', strategy_name.replace(' ', '_'))
        run_id = f"RUN_{strategy_type.replace(' ', '_').upper()}_{clean_name}_{int(time.time())}"
        next_exec_date = calculate_next_execution_date()
        
        session.execute(text("""
            UPDATE saved_instances 
            SET run_id = :run_id, client_info = :client_info, 
                webhook_url = :webhook_url, status = 'running',
                next_execution_date = :next_date,
                email_notification = :email_notif,
                telegram_notification = :telegram_notif
            WHERE id = :strategy_id
        """), {
            "run_id": run_id,
            "client_info": json.dumps(client_info) if isinstance(client_info, dict) else client_info,
            "webhook_url": webhook_url,
            "next_date": next_exec_date,
            "email_notif": email_notification,
            "telegram_notif": telegram_notification,
            "strategy_id": strategy_id
        })
        session.commit()
        return {"success": True, "message": "Deployment saved successfully", "run_id": run_id}
    except Exception as e:
        if session: session.rollback()
        logger.error(f"Error in save_deployment: {e}")
        return {"success": False, "message": str(e)}
    finally:
        if session: session.close()


@deployment_router.post("/live-signals/save-deployment")
async def save_live_signals_deployment(request: dict):
    """Refactored save_live_signals_deployment for Rotation strategies using saved_instances"""
    session = None
    try:
        user_email = request.get('user_email')
        strategy_name = request.get('strategy_name')
        client_info = request.get('client_information_json', '{}')
        webhook_url = request.get('webhook_url', '')
        
        if not user_email or not strategy_name:
            return {"success": False, "message": "user_email and strategy_name are required"}
            
        session = get_session()
        result = session.execute(text("""
            SELECT id, strategy_type, strategies_parameters FROM saved_instances 
            WHERE strategy_name = :name AND user_id = :user
        """), {"name": strategy_name, "user": user_email})
        
        row = result.fetchone()
        if not row:
            return {"success": False, "message": f"Strategy '{strategy_name}' not found"}
            
        strategy_id, strategy_type, existing_params = row
        params = json.loads(existing_params) if isinstance(existing_params, str) else (existing_params or {})
        
        # Update params with new deployment fields
        params.update({
            "ltp": request.get('ltp', 0),
            "deployment_data": request.get('deployment_data'),
            "etf_count": request.get('etf_count'),
            "etf_names": request.get('etf_names')
        })
        
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', strategy_name.replace(' ', '_'))
        run_id = f"RUN_ROTATION_{clean_name}_{int(time.time())}"
        next_date = calculate_next_execution_date()
        
        session.execute(text("""
            UPDATE saved_instances 
            SET run_id = :run_id, client_info = :client_info, 
                webhook_url = :webhook_url, status = 'running',
                next_execution_date = :next_date,
                reference_capital = :ref_capital,
                email_notification = :email_notif,
                telegram_notification = :telegram_notif,
                strategies_parameters = :params
            WHERE id = :id
        """), {
            "run_id": run_id,
            "client_info": json.dumps(client_info) if isinstance(client_info, dict) else client_info,
            "webhook_url": webhook_url,
            "next_date": next_date,
            "ref_capital": request.get('reference_capital'),
            "email_notif": request.get('email_notification', False),
            "telegram_notif": request.get('telegram_notification', False),
            "params": json.dumps(params),
            "id": strategy_id
        })
        session.commit()
        return {"success": True, "message": "Rotation deployment saved successfully", "run_id": run_id}
    except Exception as e:
        if session: session.rollback()
        logger.error(f"Error in save_live_signals_deployment: {e}")
        return {"success": False, "message": str(e)}
    finally:
        if session: session.close()


@deployment_router.post("/live-signals/deployment-status-by-strategy")
async def get_deployment_status_by_strategy(request: dict):
    """Unified status check for all strategy types via saved_instances"""
    session = None
    try:
        strategy_name = request.get('strategy_name')
        if not strategy_name:
            return {"success": False, "message": "strategy_name is required", "data": {"exists": False}}
            
        session = get_session()
        result = session.execute(text("""
            SELECT id, run_id, status, client_info, webhook_url, next_execution_date, created_at
            FROM saved_instances 
            WHERE strategy_name = :name
        """), {"name": strategy_name})
        
        row = result.fetchone()
        if row:
            columns = result.keys()
            data = dict(zip(columns, row))
            return {
                "success": True,
                "data": {
                    "exists": True,
                    "status": data['status'] or 'not deploy',
                    "run_id": data['run_id'],
                    "client_information_json": data['client_info'],
                    "webhook_url": data['webhook_url'],
                    "execution_date": str(data['next_execution_date']) if data['next_execution_date'] else None,
                    "created_at": str(data['created_at']) if data['created_at'] else None,
                    "strategy_id": data['id'],
                    "table": "saved_instances"
                }
            }
        return {"success": True, "data": {"exists": False, "status": "deploy"}}
    except Exception as e:
        logger.error(f"Error in get_deployment_status_by_strategy: {e}")
        return {"success": False, "message": str(e), "data": {"exists": False}}
    finally:
        if session: session.close()


@deployment_router.post("/live-signals/update-client-information")
async def update_client_information(request: dict):
    """Update client info via run_id in saved_instances"""
    session = None
    try:
        run_id = request.get('run_id')
        client_info = request.get('client_information_json', '{}')
        if not run_id: return {"success": False, "message": "run_id is required"}
            
        session = get_session()
        result = session.execute(text("""
            UPDATE saved_instances SET client_info = :info WHERE run_id = :run_id
        """), {"info": json.dumps(client_info) if isinstance(client_info, dict) else client_info, "run_id": run_id})
        
        if result.rowcount == 0:
            return {"success": False, "message": f"Strategy with run_id '{run_id}' not found"}
            
        session.commit()
        return {"success": True, "message": "Client information updated successfully"}
    except Exception as e:
        if session: session.rollback()
        logger.error(f"Error in update_client_information: {e}")
        return {"success": False, "message": str(e)}
    finally:
        if session: session.close()


@deployment_router.post("/live-signals/update-deployment-status")
async def update_deployment_status(request: dict):
    """Update status via strategy_name in saved_instances"""
    session = None
    try:
        strategy_name = request.get('strategy_name')
        status = request.get('new_status')
        if not strategy_name or not status:
            return {"success": False, "message": "strategy_name and new_status required"}
            
        session = get_session()
        result = session.execute(text("""
            UPDATE saved_instances SET status = :status WHERE strategy_name = :name
        """), {"status": status, "name": strategy_name})
        
        if result.rowcount == 0:
            return {"success": False, "message": f"Strategy '{strategy_name}' not found"}
            
        session.commit()
        return {"success": True, "message": f"Deployment status updated to '{status}'"}
    except Exception as e:
        if session: session.rollback()
        logger.error(f"Error in update_deployment_status: {e}")
        return {"success": False, "message": str(e)}
    finally:
        if session: session.close()


@deployment_router.post("/update-rs-client-information")
async def update_rs_client_information(request: dict):
    """Same as update_client_information but uses id + user_id"""
    session = None
    try:
        strategy_id = request.get('strategy_id')
        user_id = request.get('user_id')
        client_info = request.get('client_information_json', '{}')
        
        if not strategy_id or not user_id:
            return {"success": False, "message": "strategy_id and user_id required"}
            
        session = get_session()
        result = session.execute(text("""
            UPDATE saved_instances SET client_info = :info 
            WHERE id = :id AND user_id = :user
        """), {
            "info": json.dumps(client_info) if isinstance(client_info, dict) else client_info,
            "id": strategy_id,
            "user": user_id
        })
        
        if result.rowcount == 0:
            return {"success": False, "message": "Strategy not found or user mismatch"}
            
        session.commit()
        return {"success": True, "message": "RS Strategy client information updated"}
    except Exception as e:
        if session: session.rollback()
        return {"success": False, "message": str(e)}
    finally:
        if session: session.close()


@deployment_router.post("/stop-rs-strategy")
@deployment_router.post("/stop-rs-etf-strategy")
async def stop_rs_strategy_unified(request: dict):
    """Stop RS strategy (unified endpoint)"""
    session = None
    try:
        strategy_id = request.get('strategy_id')
        user_id = request.get('user_id')
        if not strategy_id or not user_id:
            return {"success": False, "message": "strategy_id and user_id required"}
            
        session = get_session()
        result = session.execute(text("""
            UPDATE saved_instances SET status = 'stop' WHERE id = :id AND user_id = :user
        """), {"id": strategy_id, "user": user_id})
        
        if result.rowcount == 0:
            return {"success": False, "message": "Strategy not found or already stopped"}
            
        session.commit()
        return {"success": True, "message": "RS Strategy stopped successfully"}
    except Exception as e:
        if session: session.rollback()
        return {"success": False, "message": str(e)}
    finally:
        if session: session.close()


@deployment_router.post("/restart-rs-strategy")
@deployment_router.post("/restart-rs-etf-strategy")
async def restart_rs_strategy_unified(request: dict):
    """Restart RS strategy (unified endpoint)"""
    session = None
    try:
        strategy_id = request.get('strategy_id')
        user_id = request.get('user_id')
        if not strategy_id: return {"success": False, "message": "strategy_id required"}
            
        session = get_session()
        query = "UPDATE saved_instances SET status = 'running' WHERE id = :id"
        params = {"id": strategy_id}
        if user_id:
            query += " AND user_id = :user"
            params["user"] = user_id
            
        result = session.execute(text(query), params)
        if result.rowcount == 0:
            return {"success": False, "message": "Strategy not found"}
            
        session.commit()
        return {"success": True, "message": "RS Strategy restarted successfully"}
    except Exception as e:
        if session: session.rollback()
        return {"success": False, "message": str(e)}
    finally:
        if session: session.close()


@deployment_router.delete("/delete-rs-strategy/{strategy_id}")
@deployment_router.delete("/delete-rs-etf-strategy/{strategy_id}")
async def delete_rs_strategy_unified(strategy_id: int):
    """Delete RS strategy (unified endpoint)"""
    session = None
    try:
        session = get_session()
        session.execute(text("DELETE FROM saved_instances WHERE id = :id"), {"id": strategy_id})
        session.commit()
        return {"success": True, "message": "RS Strategy deleted successfully"}
    except Exception as e:
        if session: session.rollback()
        return {"success": False, "message": str(e)}
    finally:
        if session: session.close()


# ============================================================================
# EXECUTION ENDPOINTS (Calling Execution Service)
# ============================================================================

@deployment_router.post("/execute/etf-signals")
async def execute_etf_signals(request: dict = None):
    """Execute ETF trading signals"""
    try:
        from Services.execution.execution_service import ExecutionService
        execution_service = ExecutionService()
        signal_date = request.get("signal_date") if request else None
        side = request.get("side") if request else None
        return execution_service.execute_all_signals(signal_date=signal_date, side=side, signal_type='etf')
    except Exception as e:
        logger.error(f"Error executing ETF signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@deployment_router.post("/execute/stock-signals")
async def execute_stock_signals(request: dict = None):
    """Execute Stock trading signals"""
    try:
        from Services.execution.execution_service import ExecutionService
        execution_service = ExecutionService()
        signal_date = request.get("signal_date") if request else None
        side = request.get("side") if request else None
        return execution_service.execute_all_signals(signal_date=signal_date, side=side, signal_type='stock')
    except Exception as e:
        logger.error(f"Error executing Stock signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# UNIFIED STRATEGY ENDPOINTS
# ============================================================================

async def save_rs_strategy(request: dict):
    """Internal helper to save an RS strategy to saved_instances"""
    # This maps back to save_rs_stock_strategy or save_rs_etf_strategy logic
    # but for simplicity, we'll just handle it directly here if it's called as 'rs_strategy'
    strategy_type = request.get('strategy_type', 'RS Strategy')
    if 'etf' in strategy_type.lower():
        return await save_rs_etf_strategy(request)
    else:
        return await save_rs_stock_strategy(request)

@deployment_router.post("/save-strategy")
async def save_strategy_unified(request: dict):
    """Unified endpoint to save strategies - routes based on strategy_type"""
    try:
        strategy_type = request.get("strategy_type", "").lower()
        if strategy_type in ["stock", "stock_rotation"]:
            from Strategies.Rotation_Stocks.stock_api import save_stock_strategy
            return await save_stock_strategy(request)
        elif strategy_type in ["etf", "etf_rotation"]:
            from Strategies.Rotation_ETF.etf_api import save_etf_strategy
            return await save_etf_strategy(request)
        elif strategy_type in ["rs_strategy", "rs"]:
            return await save_rs_strategy(request)
        else:
            # Fallback to saving directly to saved_instances if type is unknown but we want to store it
            return await save_rs_strategy(request)
    except Exception as e:
        logger.error(f"Error in save_strategy_unified: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@deployment_router.get("/get-saved-strategies/{user_id}")
async def get_saved_strategies_unified(user_id: str):
    """Unified endpoint to get all saved strategies for a user"""
    try:
        all_strategies = []
        # RS ETF and Stock are now in the same table
        rs_etf = await get_saved_rs_etf_strategies(user_id)
        if rs_etf.get("strategies"): all_strategies.extend(rs_etf["strategies"])
        
        rs_stock = await get_saved_rs_strategies(user_id)
        if rs_stock.get("strategies"): all_strategies.extend(rs_stock["strategies"])
        
        # rotation_stocks/etfs might still have their own endpoints for now, 
        # but the goal is to have everything in saved_instances
        
        # Sort by creation date
        all_strategies.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
        return {"strategies": all_strategies}
    except Exception as e:
        logger.error(f"Error retrieving saved strategies: {e}")
        return {"strategies": []}

@deployment_router.post("/debug-request")
async def debug_request(request: dict):
    """Debug endpoint to inspect request format"""
    return {
        "received_keys": list(request.keys()),
        "strategy_type": request.get("strategy_type", "NOT_PROVIDED"),
        "request_sample": {k: str(v)[:100] + "..." if len(str(v)) > 100 else v for k, v in request.items()}
    }


@deployment_router.post("/rs-strategy/deploy")
async def deploy_rs_strategy(request: dict):
    """Deploy RS strategy to live signals"""
    try:
        # This endpoint will handle RS strategy deployment
        # For now, return success - you can implement the actual deployment logic here
        return {
            "success": True,
            "message": "RS Strategy deployed successfully",
            "run_id": request.get("run_id", f"RS_{int(time.time())}_{request.get('user_email', 'unknown')}")
        }
    except Exception as e:
        return {"success": False, "message": f"Error deploying RS strategy: {str(e)}"}


@deployment_router.get("/rs-strategy/signals/latest")
async def get_rs_strategy_signals():
    """Get latest RS strategy signals"""
    try:
        # This endpoint will return latest RS strategy signals
        # For now, return empty signals - you can implement the actual signal logic here
        return {
            "success": True,
            "signals": [],
            "message": "No RS strategy signals available"
        }
    except Exception as e:
        return {"success": False, "message": f"Error fetching RS strategy signals: {str(e)}"}

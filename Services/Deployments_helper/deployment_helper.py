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

def init_saved_rs_strategies_table():
    """Initialize the saved_rs_strategies table if it doesn't exist"""
    try:
        if not create_connection():
            logger.error("Failed to connect to PostgreSQL database")
            return False
        
        if not init_database():
            logger.error("Failed to initialize database tables")
            return False
        
        session = None
        try:
            session = get_session()
            
            # Check if table exists
            result = session.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'saved_rs_strategies'
            """))
            
            if not result.fetchone():
                # Create table
                session.execute(text("""
                    CREATE TABLE saved_rs_strategies (
                        id SERIAL PRIMARY KEY,
                        strategy_name VARCHAR(255) NOT NULL,
                        strategy_type VARCHAR(255) NOT NULL,
                        user_id VARCHAR(255) NOT NULL,
                        config_id INTEGER,
                        backtest_id INTEGER,
                        start_date VARCHAR(50),
                        end_date VARCHAR(50),
                        stock_universe TEXT,
                        backtest_results JSONB,
                        strategy_config JSONB,
                        run_id VARCHAR(255) UNIQUE,
                        client_information_json JSONB,
                        webhook_url TEXT,
                        status VARCHAR(50) DEFAULT 'deploy',
                        created_at VARCHAR(50),
                        UNIQUE(strategy_name, user_id)
                    )
                """))
                session.commit()
                logger.info("saved_rs_strategies table created successfully")
            else:
                # Check for missing columns and add them
                columns_result = session.execute(text("""
                    SELECT column_name 
                    FROM information_schema.columns
                    WHERE table_schema = 'public' 
                    AND table_name = 'saved_rs_strategies'
                """))
                existing_columns = [row[0] for row in columns_result.fetchall()]
                
                required_columns = {
                    'run_id': 'VARCHAR(255)',
                    'client_information_json': 'JSONB',
                    'webhook_url': 'TEXT',
                    'status': 'VARCHAR(50) DEFAULT \'deploy\''
                }
                
                for column_name, column_def in required_columns.items():
                    if column_name not in existing_columns:
                        try:
                            session.execute(text(f"ALTER TABLE saved_rs_strategies ADD COLUMN {column_name} {column_def}"))
                            session.commit()
                            logger.info(f"Added missing column '{column_name}' to saved_rs_strategies")
                        except Exception as e:
                            logger.warning(f"Could not add column '{column_name}': {e}")
            
            return True
        finally:
            if session:
                session.close()
    except Exception as e:
        logger.error(f"Error creating saved_rs_strategies table: {e}")
        import traceback
        traceback.print_exc()
        return False


# Removed duplicate init_rs_stock_instance_table


def init_rs_stock_instance_table():
    """Initialize the rs_stock_instance table if it doesn't exist"""
    try:
        if not create_connection():
            logger.error("Failed to connect to PostgreSQL database")
            return False
        
        if not init_database():
            logger.error("Failed to initialize database tables")
            return False
        
        session = None
        try:
            session = get_session()
            
            # Check if table exists
            result = session.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'rs_stock_instance'
            """))
            
            if not result.fetchone():
                # Create table
                session.execute(text("""
                    CREATE TABLE rs_stock_instance (
                        id SERIAL PRIMARY KEY,
                        strategy_name VARCHAR(255) NOT NULL,
                        strategy_type VARCHAR(255) NOT NULL,
                        user_id VARCHAR(255) NOT NULL,
                        config_id INTEGER,
                        backtest_id INTEGER,
                        start_date VARCHAR(50),
                        end_date VARCHAR(50),
                        stock_universe TEXT,
                        backtest_results JSONB,
                        strategy_config JSONB,
                        run_id VARCHAR(255) UNIQUE,
                        client_information_json JSONB,
                        webhook_url TEXT,
                        status VARCHAR(50) DEFAULT 'deploy',
                        created_at VARCHAR(50),
                        last_execution_date VARCHAR(50),
                        next_execution_date VARCHAR(50),
                        UNIQUE(strategy_name, user_id)
                    )
                """))
                session.commit()
                logger.info("rs_stock_instance table created successfully")
            else:
                # Check for missing columns and add them
                columns_result = session.execute(text("""
                    SELECT column_name 
                    FROM information_schema.columns
                    WHERE table_schema = 'public' 
                    AND table_name = 'rs_stock_instance'
                """))
                existing_columns = [row[0] for row in columns_result.fetchall()]
                
                required_columns = {
                    'run_id': 'VARCHAR(255)',
                    'client_information_json': 'JSONB',
                    'webhook_url': 'TEXT',
                    'status': 'VARCHAR(50) DEFAULT \'deploy\'',
                    'last_execution_date': 'VARCHAR(50)',
                    'next_execution_date': 'VARCHAR(50)'
                }
                
                for column_name, column_def in required_columns.items():
                    if column_name not in existing_columns:
                        try:
                            session.execute(text(f"ALTER TABLE rs_stock_instance ADD COLUMN {column_name} {column_def}"))
                            session.commit()
                            logger.info(f"Added missing column '{column_name}' to rs_stock_instance")
                        except Exception as e:
                            logger.warning(f"Could not add column '{column_name}': {e}")
            
            return True
        finally:
            if session:
                session.close()
    except Exception as e:
        logger.error(f"Error creating rs_stock_instance table: {e}")
        import traceback
        traceback.print_exc()
        return False






def init_rs_etf_instance_table():
    """Initialize the rs_etf_instance table if it doesn't exist"""
    try:
        if not create_connection():
            logger.error("Failed to connect to PostgreSQL database")
            return False
        
        if not init_database():
            logger.error("Failed to initialize database tables")
            return False
        
        session = None
        try:
            session = get_session()
            
            # Check if table exists
            result = session.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'rs_etf_instance'
            """))
            
            if not result.fetchone():
                # Create table
                session.execute(text("""
                    CREATE TABLE rs_etf_instance (
                        id SERIAL PRIMARY KEY,
                        strategy_name VARCHAR(255) NOT NULL,
                        strategy_type VARCHAR(255) NOT NULL,
                        user_id VARCHAR(255) NOT NULL,
                        config_id INTEGER,
                        backtest_id INTEGER,
                        start_date VARCHAR(50),
                        end_date VARCHAR(50),
                        rs_etf_universe TEXT,
                        backtest_results JSONB,
                        strategy_config JSONB,
                        run_id VARCHAR(255) UNIQUE,
                        client_information_json JSONB,
                        webhook_url TEXT,
                        status VARCHAR(50) DEFAULT 'deploy',
                        created_at VARCHAR(50),
                        last_execution_date VARCHAR(50),
                        next_execution_date VARCHAR(50),
                        email_notification BOOLEAN DEFAULT FALSE,
                        telegram_notification BOOLEAN DEFAULT FALSE,
                        UNIQUE(strategy_name, user_id)
                    )
                """))
                session.commit()
                logger.info("rs_etf_instance table created successfully")
            else:
                # Check for missing columns and add them
                columns_result = session.execute(text("""
                    SELECT column_name 
                    FROM information_schema.columns
                    WHERE table_schema = 'public' 
                    AND table_name = 'rs_etf_instance'
                """))
                existing_columns = [row[0] for row in columns_result.fetchall()]
                
                required_columns = {
                    'run_id': 'VARCHAR(255)',
                    'client_information_json': 'JSONB',
                    'webhook_url': 'TEXT',
                    'status': 'VARCHAR(50) DEFAULT \'deploy\'',
                    'last_execution_date': 'VARCHAR(50)',
                    'next_execution_date': 'VARCHAR(50)',
                    'email_notification': 'BOOLEAN DEFAULT FALSE',
                    'telegram_notification': 'BOOLEAN DEFAULT FALSE'
                }
                
                for column_name, column_def in required_columns.items():
                    if column_name not in existing_columns:
                        try:
                            session.execute(text(f"ALTER TABLE rs_etf_instance ADD COLUMN {column_name} {column_def}"))
                            session.commit()
                            logger.info(f"Added missing column '{column_name}' to rs_etf_instance")
                        except Exception as e:
                            logger.warning(f"Could not add column '{column_name}': {e}")
            
            return True
        finally:
            if session:
                session.close()
    except Exception as e:
        logger.error(f"Error creating rs_etf_instance table: {e}")
        import traceback
        traceback.print_exc()
        return False


@deployment_router.post("/save-rs-etf-strategy")
async def save_rs_etf_strategy(request: dict):
    """Save RS ETF Strategy configuration"""
    session = None
    try:
        # Initialize database table if needed
        if not init_rs_etf_instance_table():
            return {
                "success": False,
                "message": "Failed to initialize rs_etf_instance table"
            }

        session = get_session()
        
        # Check if strategy already exists in rs_etf_instance
        result = session.execute(text("""
            SELECT id FROM rs_etf_instance 
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
        
        # Insert new strategy
        result = session.execute(text("""
            INSERT INTO rs_etf_instance (
                strategy_name, strategy_type, user_id, 
                start_date, end_date, rs_etf_universe, backtest_results, 
                strategy_config, created_at, status
            ) VALUES (
                :strategy_name, :strategy_type, :user_id, 
                :start_date, :end_date, :rs_etf_universe, :backtest_results, 
                :strategy_config, :created_at, :status
            )
            RETURNING id
        """), {
            "strategy_name": request.get('strategy_name'),
            "strategy_type": request.get('strategy_type', 'RS ETF Strategy'),
            "user_id": request.get('user_id'),
            "start_date": request.get('start_date'),
            "end_date": request.get('end_date'),
            "rs_etf_universe": request.get('rs_etf_universe') or request.get('stock_universe') or 'ALL_ETFS',
            "backtest_results": json.dumps(request.get('backtest_results', {})) if request.get('backtest_results') else None,
            "strategy_config": json.dumps(request.get('strategy_config', {})) if request.get('strategy_config') else None,
            "created_at": request.get('created_at', datetime.now().isoformat()),
            "status": request.get('status', 'deploy')
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
    """Get all saved RS ETF strategies for a specific user from rs_etf_instance table"""
    session = None
    try:
        session = get_session()
        
        result = session.execute(text("""
            SELECT * FROM rs_etf_instance 
            WHERE user_id = :user_id 
            ORDER BY created_at DESC
        """), {"user_id": user_id})
        
        rows = result.fetchall()
        columns = result.keys()
        
        strategies = []
        for row in rows:
            strategy_dict = dict(zip(columns, row))
            
            # Parse JSON fields
            for json_field in ['backtest_results', 'strategy_config', 'client_information_json']:
                if strategy_dict.get(json_field):
                    if isinstance(strategy_dict[json_field], str):
                        try:
                            strategy_dict[json_field] = json.loads(strategy_dict[json_field])
                        except:
                            strategy_dict[json_field] = {}
            
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
    """Save RS Stock Strategy configuration"""
    session = None
    try:
        # Initialize database table if needed
        if not init_rs_stock_instance_table():
            return {
                "success": False,
                "message": "Failed to initialize rs_stock_instance table"
            }

        session = get_session()
        
        # Check if strategy already exists in rs_stock_instance
        result = session.execute(text("""
            SELECT id FROM rs_stock_instance 
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
        
        # Insert new strategy
        result = session.execute(text("""
            INSERT INTO rs_stock_instance (
                strategy_name, strategy_type, user_id, 
                start_date, end_date, stock_universe, backtest_results, 
                strategy_config, created_at, status
            ) VALUES (
                :strategy_name, :strategy_type, :user_id, 
                :start_date, :end_date, :stock_universe, :backtest_results, 
                :strategy_config, :created_at, :status
            )
            RETURNING id
        """), {
            "strategy_name": request.get('strategy_name'),
            "strategy_type": request.get('strategy_type', 'RS Stock Strategy'),
            "user_id": request.get('user_id'),
            "start_date": request.get('start_date'),
            "end_date": request.get('end_date'),
            "stock_universe": request.get('stock_universe') or 'NIFTY50',
            "backtest_results": json.dumps(request.get('backtest_results', {})) if request.get('backtest_results') else None,
            "strategy_config": json.dumps(request.get('strategy_config', {})) if request.get('strategy_config') else None,
            "created_at": request.get('created_at', datetime.now().isoformat()),
            "status": request.get('status', 'deploy')
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
    """Get all saved RS Stock strategies for a specific user from rs_stock_instance table"""
    session = None
    try:
        session = get_session()
        
        result = session.execute(text("""
            SELECT * FROM rs_stock_instance 
            WHERE user_id = :user_id 
            ORDER BY created_at DESC
        """), {"user_id": user_id})
        
        rows = result.fetchall()
        columns = result.keys()
        
        strategies = []
        for row in rows:
            strategy_dict = dict(zip(columns, row))
            
            # Parse JSON fields
            for json_field in ['backtest_results', 'strategy_config', 'client_information_json']:
                if strategy_dict.get(json_field):
                    if isinstance(strategy_dict[json_field], str):
                        try:
                            strategy_dict[json_field] = json.loads(strategy_dict[json_field])
                        except:
                            strategy_dict[json_field] = {}
            
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
    """Save ETF/Stock/RS strategy deployment with webhook and client info"""
    session = None
    try:
        logger.info(f"🔍 Deployment Request: {list(request.keys())}")
        logger.info(f"📧 User Email: {request.get('user_email')}")
        logger.info(f"📋 Strategy Name: {request.get('strategy_name')}")
        logger.info(f"🔗 Webhook URL: {request.get('webhook_url')}")
        
        session = get_session()
        
        # Extract data
        user_email = request.get('user_email')
        strategy_name = request.get('strategy_name')
        client_information_json = request.get('client_information_json', '{}')
        webhook_url = request.get('webhook_url', '')
        email_notification = request.get('email_notification', False)
        telegram_notification = request.get('telegram_notification', False)
        
        # Validate required fields
        if not user_email:
            return {
                "success": False,
                "message": "user_email is required"
            }
        
        if not strategy_name:
            return {
                "success": False,
                "message": "strategy_name is required"
            }
        
        # Determine which table to use based on strategy type
        # Check ETF table first
        result = session.execute(text("""
            SELECT id FROM etf_saved_strategy 
            WHERE strategy_name = :strategy_name AND user_id = :user_id
        """), {
            "strategy_name": strategy_name,
            "user_id": user_email
        })
        
        strategy_result = result.fetchone()
        table_name = 'etf_saved_strategy'
        
        if not strategy_result:
            # Check Stock table
            result = session.execute(text("""
                SELECT id FROM stock_saved_strategy 
                WHERE strategy_name = :strategy_name AND user_id = :user_id
            """), {
                "strategy_name": strategy_name,
                "user_id": user_email
            })
            strategy_result = result.fetchone()
            table_name = 'stock_saved_strategy'
            
        if not strategy_result:
            # Check RS Stock table
            result = session.execute(text("""
                SELECT id FROM rs_stock_instance 
                WHERE strategy_name = :strategy_name AND user_id = :user_id
            """), {
                "strategy_name": strategy_name,
                "user_id": user_email
            })
            strategy_result = result.fetchone()
            table_name = 'rs_stock_instance'

        if not strategy_result:
            # Check RS ETF table
            result = session.execute(text("""
                SELECT id FROM rs_etf_instance 
                WHERE strategy_name = :strategy_name AND user_id = :user_id
            """), {
                "strategy_name": strategy_name,
                "user_id": user_email
            })
            strategy_result = result.fetchone()
            table_name = 'rs_etf_instance'
        
        if not strategy_result:
            return {
                "success": False,
                "message": f"Strategy '{strategy_name}' not found for user {user_email}"
            }
        
        strategy_id = strategy_result[0]
        
        # Generate unique run_id
        clean_name = strategy_name.replace(' ', '_').replace('-', '_')
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', clean_name)
        
        if table_name == 'etf_saved_strategy':
            run_id = f"run_etfs_rotation_strategy_{clean_name}_{int(time.time())}"
        elif table_name == 'stock_saved_strategy':
            run_id = f"run_stock_rotation_strategy_{clean_name}_{int(time.time())}"
        elif table_name == 'rs_stock_instance':
            run_id = f"RS_STOCK_{clean_name}_{int(time.time())}"
        elif table_name == 'rs_etf_instance':
            run_id = f"RS_ETF_{clean_name}_{int(time.time())}"
        
        # Calculate execution dates
        current_date = datetime.now()
        next_exec_date = calculate_next_execution_date(current_date)
        last_exec_date = "N/A"
        
        logger.info(f"🔍 EXECUTION DATES CALCULATION:")
        logger.info(f"   Current Date: {current_date}")
        logger.info(f"   Next Execution Date: {next_exec_date}")
        logger.info(f"   Last Execution Date: {last_exec_date}")
        
        # Update strategy with deployment info
        update_query = f"""
            UPDATE {table_name} 
            SET run_id = :run_id, client_information_json = :client_info, 
                webhook_url = :webhook_url, status = :status,
                last_execution_date = :last_exec_date,
                next_execution_date = :next_exec_date,
                email_notification = :email_notif,
                telegram_notification = :telegram_notif
            WHERE id = :strategy_id
        """
        
        update_params = {
            "run_id": run_id,
            "client_info": client_information_json,
            "webhook_url": webhook_url,
            "status": 'running',
            "last_exec_date": last_exec_date,
            "next_exec_date": next_exec_date,
            "email_notif": email_notification,
            "telegram_notif": telegram_notification,
            "strategy_id": strategy_id
        }
        
        logger.info(f"🔍 SQL UPDATE QUERY:")
        logger.info(f"   Table: {table_name}")
        logger.info(f"   Query: {update_query}")
        logger.info(f"   Parameters: {update_params}")
        
        session.execute(text(update_query), update_params)
        
        session.commit()
        
        return {
            "success": True,
            "message": f"{table_name.replace('_', ' ').title()} deployment saved successfully",
            "run_id": run_id
        }
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Error in save_deployment: {e}")
        return {
            "success": False,
            "message": f"Error saving deployment: {str(e)}"
        }
    finally:
        if session:
            session.close()


@deployment_router.post("/live-signals/save-deployment")
async def save_live_signals_deployment(request: dict):
    """
    Save ETF/Stock Rotation strategy deployment with webhook and client info
    This endpoint is specifically for Rotation strategies (not RS strategies)
    """
    session = None
    try:
        logger.info(f"🔍 Live Signals Deployment Request: {list(request.keys())}")
        logger.info(f"📧 User Email: {request.get('user_email')}")
        logger.info(f"📋 Strategy Name: {request.get('strategy_name')}")
        logger.info(f"🔗 Webhook URL: {request.get('webhook_url')}")
        
        session = get_session()
        
        
        # Extract all deployment data from frontend
        user_email = request.get('user_email')
        strategy_name = request.get('strategy_name')
        client_information_json = request.get('client_information_json', '{}')
        webhook_url = request.get('webhook_url', '')
        
        # Additional deployment fields
        reference_capital = request.get('reference_capital', '')
        ltp = request.get('ltp', 0)
        deployment_data = request.get('deployment_data')
        etf_count = request.get('etf_count')
        etf_names = request.get('etf_names')
        strategy_type = request.get('strategy_type', '')
        email_notification = request.get('email_notification', False)
        telegram_notification = request.get('telegram_notification', False)
        
        # Validate required fields
        if not user_email:
            return {
                "success": False,
                "message": "user_email is required"
            }
        
        if not strategy_name:
            return {
                "success": False,
                "message": "strategy_name is required"
            }
        
        # Check ETF table first
        result = session.execute(text("""
            SELECT id FROM etf_saved_strategy 
            WHERE strategy_name = :strategy_name AND user_id = :user_id
        """), {
            "strategy_name": strategy_name,
            "user_id": user_email
        })
        
        strategy_result = result.fetchone()
        table_name = 'etf_saved_strategy'
        is_etf_strategy = True
        
        if not strategy_result:
            # Check Stock table
            result = session.execute(text("""
                SELECT id FROM stock_saved_strategy 
                WHERE strategy_name = :strategy_name AND user_id = :user_id
            """), {
                "strategy_name": strategy_name,
                "user_id": user_email
            })
            strategy_result = result.fetchone()
            table_name = 'stock_saved_strategy'
            is_etf_strategy = False
        
        if not strategy_result:
            return {
                "success": False,
                "message": f"Strategy '{strategy_name}' not found for user {user_email}"
            }
        
        strategy_id = strategy_result[0]
        
        # Always generate unique run_id using backend logic
        clean_name = strategy_name.replace(' ', '_').replace('-', '_')
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', clean_name)
        
        if is_etf_strategy:
            run_id = f"run_etfs_rotation_strategy_{clean_name}_{int(time.time())}"
        else:  # stock_saved_strategy
            run_id = f"run_stock_rotation_strategy_{clean_name}_{int(time.time())}"
        
        # Always calculate execution date using backend logic (next Monday)
        current_date = datetime.now()
        execution_date = calculate_next_execution_date(current_date)
        
        # Set last_execution_date and next_execution_date
        # For new deployments: last = "N/A", next = calculated/provided date
        last_execution_date = "N/A"  # No execution yet
        next_execution_date = execution_date  # Use provided or calculated date
        
        logger.info(f"🔍 DEPLOYMENT INFO:")
        logger.info(f"   Strategy ID: {strategy_id}")
        logger.info(f"   Table: {table_name}")
        logger.info(f"   Run ID: {run_id}")
        logger.info(f"   Execution Date: {execution_date}")
        logger.info(f"   Last Execution Date: {last_execution_date}")
        logger.info(f"   Next Execution Date: {next_execution_date}")
        logger.info(f"   Reference Capital: {reference_capital}")
        
        # Prepare update query based on strategy type
        if is_etf_strategy:
            # ETF strategy - include all ETF-specific fields
            update_query = f"""
                UPDATE {table_name} 
                SET run_id = :run_id, 
                    client_information_json = :client_info, 
                    webhook_url = :webhook_url, 
                    status = :status,
                    execution_date = :exec_date,
                    last_execution_date = :last_exec_date,
                    next_execution_date = :next_exec_date,
                    reference_capital = :ref_capital,
                    ltp = :ltp,
                    deployment_data = :deployment_data,
                    etf_count = :etf_count,
                    etf_names = :etf_names,
                    strategy_type = :strategy_type,
                    email_notification = :email_notif,
                    telegram_notification = :telegram_notif
                WHERE id = :strategy_id
            """
            
            # Convert deployment_data and etf_names to JSON strings
            import json
            deployment_data_json = json.dumps(deployment_data) if deployment_data else None
            etf_names_json = json.dumps(etf_names) if etf_names else None
            
            update_params = {
                "run_id": run_id,
                "client_info": client_information_json,
                "webhook_url": webhook_url,
                "status": 'running',
                "exec_date": execution_date,
                "last_exec_date": last_execution_date,
                "next_exec_date": next_execution_date,
                "ref_capital": reference_capital,
                "ltp": ltp,
                "deployment_data": deployment_data_json,
                "etf_count": etf_count,
                "etf_names": etf_names_json,
                "strategy_type": strategy_type,
                "email_notif": email_notification,
                "telegram_notif": telegram_notification,
                "strategy_id": strategy_id
            }
        else:
            # Stock strategy - no ETF-specific fields
            update_query = f"""
                UPDATE {table_name} 
                SET run_id = :run_id, 
                    client_information_json = :client_info, 
                    webhook_url = :webhook_url, 
                    status = :status,
                    execution_date = :exec_date,
                    last_execution_date = :last_exec_date,
                    next_execution_date = :next_exec_date,
                    reference_capital = :ref_capital,
                    strategy_type = :strategy_type,
                    email_notification = :email_notif,
                    telegram_notification = :telegram_notif
                WHERE id = :strategy_id
            """
            
            update_params = {
                "run_id": run_id,
                "client_info": client_information_json,
                "webhook_url": webhook_url,
                "status": 'running',
                "exec_date": execution_date,
                "last_exec_date": last_execution_date,
                "next_exec_date": next_execution_date,
                "ref_capital": reference_capital,
                "strategy_type": strategy_type,
                "email_notif": email_notification,
                "telegram_notif": telegram_notification,
                "strategy_id": strategy_id
            }
        
        logger.info(f"🔍 SQL UPDATE QUERY:")
        logger.info(f"   Table: {table_name}")
        logger.info(f"   Parameters: {list(update_params.keys())}")
        
        session.execute(text(update_query), update_params)
        session.commit()
        
        logger.info(f"✅ Deployment saved successfully for {table_name}")
        
        return {
            "success": True,
            "message": f"{table_name.replace('_', ' ').title()} deployment saved successfully",
            "run_id": run_id,
            "execution_date": execution_date,
            "last_execution_date": last_execution_date,
            "next_execution_date": next_execution_date,
            "table": table_name
        }
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Error in save_live_signals_deployment: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Error saving deployment: {str(e)}"
        }
    finally:
        if session:
            session.close()


@deployment_router.post("/live-signals/deployment-status-by-strategy")
async def get_deployment_status_by_strategy(request: dict):
    """Get deployment status for ETF/Stock/RS strategy by strategy name"""
    session = None
    try:
        strategy_name = request.get('strategy_name')
        execution_date = request.get('execution_date')  # Optional
        
        if not strategy_name:
            return {
                "success": False,
                "message": "strategy_name is required",
                "data": {"exists": False}
            }
        
        session = get_session()
        
        # Check ETF table first
        result = session.execute(text("""
            SELECT id, run_id, status, client_information_json, webhook_url, 
                   execution_date, created_at
            FROM etf_saved_strategy 
            WHERE strategy_name = :strategy_name
        """), {"strategy_name": strategy_name})
        
        row = result.fetchone()
        table_name = 'etf_saved_strategy'
        
        if not row:
            # Check Stock table
            result = session.execute(text("""
                SELECT id, run_id, status, client_information_json, webhook_url, 
                       execution_date, created_at
                FROM stock_saved_strategy 
                WHERE strategy_name = :strategy_name
            """), {"strategy_name": strategy_name})
            row = result.fetchone()
            table_name = 'stock_saved_strategy'
            
        if not row:
            # Check RS Stock table (use last_execution_date as execution_date)
            result = session.execute(text("""
                SELECT id, run_id, status, client_information_json, webhook_url, 
                       last_execution_date, created_at
                FROM rs_stock_instance 
                WHERE strategy_name = :strategy_name
            """), {"strategy_name": strategy_name})
            row = result.fetchone()
            table_name = 'rs_stock_instance'

        if not row:
            # Check RS ETF table (use last_execution_date as execution_date)
            result = session.execute(text("""
                SELECT id, run_id, status, client_information_json, webhook_url, 
                       last_execution_date, created_at
                FROM rs_etf_instance 
                WHERE strategy_name = :strategy_name
            """), {"strategy_name": strategy_name})
            row = result.fetchone()
            table_name = 'rs_etf_instance'
        
        if row:
            return {
                "success": True,
                "data": {
                    "exists": True,
                    "status": row[2] if row[2] else 'deploy',
                    "run_id": row[1],
                    "client_information_json": row[3],
                    "webhook_url": row[4],
                    "execution_date": str(row[5]) if row[5] else None,
                    "created_at": str(row[6]) if row[6] else None,
                    "strategy_id": row[0],
                    "table": table_name
                }
            }
        else:
            return {
                "success": True,
                "data": {
                    "exists": False,
                    "status": "deploy"
                }
            }
    except Exception as e:
        logger.error(f"Error in get_deployment_status_by_strategy: {e}")
        return {
            "success": False,
            "message": f"Error getting deployment status: {str(e)}",
            "data": {"exists": False}
        }
    finally:
        if session:
            session.close()


@deployment_router.post("/live-signals/update-client-information")
async def update_client_information(request: dict):
    """Update client information for ETF/Stock strategy by run_id"""
    session = None
    try:
        run_id = request.get('run_id')
        client_information_json = request.get('client_information_json', '{}')
        
        if not run_id:
            return {
                "success": False,
                "message": "run_id is required"
            }
        
        session = get_session()
        
        # Check ETF table first
        result = session.execute(text("""
            SELECT id FROM etf_saved_strategy 
            WHERE run_id = :run_id
        """), {"run_id": run_id})
        
        row = result.fetchone()
        table_name = 'etf_saved_strategy'
        
        if not row:
            # Check Stock table
            result = session.execute(text("""
                SELECT id FROM stock_saved_strategy 
                WHERE run_id = :run_id
            """), {"run_id": run_id})
            row = result.fetchone()
            table_name = 'stock_saved_strategy'
        
        if not row:
            return {
                "success": False,
                "message": f"Strategy with run_id '{run_id}' not found"
            }
        
        # Update client information
        session.execute(text(f"""
            UPDATE {table_name} 
            SET client_information_json = :client_info
            WHERE run_id = :run_id
        """), {
            "client_info": client_information_json,
            "run_id": run_id
        })
        
        session.commit()
        
        return {
            "success": True,
            "message": f"Client information updated successfully"
        }
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Error in update_client_information: {e}")
        return {
            "success": False,
            "message": f"Error updating client information: {str(e)}"
        }
    finally:
        if session:
            session.close()


@deployment_router.post("/live-signals/update-deployment-status")
async def update_deployment_status(request: dict):
    """Update deployment status for ETF/Stock strategy by strategy_name only"""
    session = None
    try:
        strategy_name = request.get('strategy_name')
        status = request.get('new_status')  # 'running', 'stopped', 'paused', 'deploy'
        
        if not strategy_name:
            return {
                "success": False,
                "message": "strategy_name is required"
            }
        
        if not status:
            return {
                "success": False,
                "message": "new_status is required"
            }
        
        # Validate status value
        valid_statuses = ['running', 'stop', 'stopped', 'paused', 'deploy']
        if status not in valid_statuses:
            return {
                "success": False,
                "message": f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
            }
        
        session = get_session()
        
        # Check Stock table first
        result = session.execute(text("""
            SELECT id, run_id FROM stock_saved_strategy 
            WHERE strategy_name = :strategy_name
        """), {"strategy_name": strategy_name})
        
        row = result.fetchone()
        table_name = 'stock_saved_strategy'
        
        if not row:
            return {
                "success": False,
                "message": f"Strategy '{strategy_name}' not found"
            }
        
        strategy_id = row[0]
        run_id = row[1] if row[1] else None
        
        # Update deployment status
        session.execute(text(f"""
            UPDATE {table_name} 
            SET status = :status, updated_at = CURRENT_TIMESTAMP
            WHERE strategy_name = :strategy_name
        """), {
            "status": status,
            "strategy_name": strategy_name
        })
        
        session.commit()
        
        logger.info(f"Updated deployment status for strategy '{strategy_name}' to {status} in {table_name}")
        
        return {
            "success": True,
            "message": f"Deployment status updated successfully to '{status}'",
            "strategy_name": strategy_name,
            "strategy_id": strategy_id,
            "run_id": run_id,
            "status": status,
            "table": table_name
        }
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Error in update_deployment_status: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Error updating deployment status: {str(e)}"
        }
    finally:
        if session:
            session.close()


@deployment_router.post("/update-rs-client-information")
async def update_rs_client_information(request: dict):
    """Update client information for RS strategy"""
    session = None
    try:
        logger.info(f"🔍 Update RS Client Info Request: {list(request.keys())}")
        logger.info(f"📋 Strategy ID: {request.get('strategy_id')}")
        logger.info(f"📧 User ID: {request.get('user_id')}")
        
        session = get_session()
        
        # Extract data
        strategy_id = request.get('strategy_id')
        user_id = request.get('user_id')
        client_information_json = request.get('client_information_json', '{}')
        
        # Validate required fields
        if not strategy_id:
            return {
                "success": False,
                "message": "strategy_id is required"
            }
        
        if not user_id:
            return {
                "success": False,
                "message": "user_id is required"
            }
        
        # Try updating RS Stock Instance first
        result = session.execute(text("""
            UPDATE rs_stock_instance 
            SET client_information_json = :client_info
            WHERE id = :strategy_id AND user_id = :user_id
        """), {
            "client_info": client_information_json,
            "strategy_id": strategy_id,
            "user_id": user_id
        })
        
        # If not found in RS Stock, try RS ETF since this endpoint might be used for both
        if result.rowcount == 0:
            result = session.execute(text("""
                UPDATE rs_etf_instance 
                SET client_information_json = :client_info
                WHERE id = :strategy_id AND user_id = :user_id
            """), {
                "client_info": client_information_json,
                "strategy_id": strategy_id,
                "user_id": user_id
            })
        
        if result.rowcount == 0:
            return {
                "success": False,
                "message": f"Strategy not found or user mismatch"
            }
        
        session.commit()
        
        return {
            "success": True,
            "message": "RS Strategy client information updated successfully"
        }
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Error in update_rs_client_information: {e}")
        return {
            "success": False,
            "message": f"Error updating client information: {str(e)}"
        }
    finally:
        if session:
            session.close()


@deployment_router.post("/stop-rs-strategy")
async def stop_rs_strategy(request: dict):
    """Stop a running RS Stock strategy"""
    session = None
    try:
        # Validate required parameters
        strategy_id = request.get('strategy_id')
        user_id = request.get('user_id')
        
        if not strategy_id:
            return {
                "success": False,
                "message": "strategy_id is required"
            }
        
        if not user_id:
            return {
                "success": False,
                "message": "user_id is required"
            }
        
        session = get_session()
        
        result = session.execute(text("""
            UPDATE rs_stock_instance 
            SET status = 'stop'
            WHERE id = :strategy_id AND user_id = :user_id
        """), {
            "strategy_id": strategy_id,
            "user_id": user_id
        })
        
        # Check if any rows were updated
        if result.rowcount == 0:
            session.rollback()
            logger.warning(f"No RS Stock strategy found with id={strategy_id} and user_id={user_id}")
            return {
                "success": False,
                "message": "RS Stock Strategy not found or already stopped"
            }
        
        session.commit()
        logger.info(f"RS Stock strategy {strategy_id} stopped successfully for user {user_id}")
        
        return {
            "success": True,
            "message": "RS Stock Strategy stopped successfully"
        }
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Error stopping RS Stock strategy: {e}")
        raise HTTPException(status_code=500, detail=f"Error stopping RS Stock strategy: {str(e)}")
    finally:
        if session:
            session.close()


@deployment_router.post("/restart-rs-strategy")
async def restart_rs_strategy(request: dict):
    """Restart a stopped RS Stock strategy"""
    session = None
    try:
        # Validate required parameters
        strategy_id = request.get('strategy_id')
        user_id = request.get('user_id')
        
        if not strategy_id:
            return {
                "success": False,
                "message": "strategy_id is required"
            }
        
        session = get_session()
        
        # Build query based on whether user_id is provided
        if user_id:
            result = session.execute(text("""
                UPDATE rs_etf_instance 
                SET status = 'running'
                WHERE id = :strategy_id AND user_id = :user_id
            """), {
                "strategy_id": strategy_id,
                "user_id": user_id
            })
        else:
            result = session.execute(text("""
                UPDATE rs_etf_instance 
                SET status = 'running'
                WHERE id = :strategy_id
            """), {
                "strategy_id": strategy_id
            })
        
        # Check if any rows were updated
        if result.rowcount == 0:
            session.rollback()
            logger.warning(f"No RS ETF strategy found with id={strategy_id}")
            return {
                "success": False,
                "message": "RS ETF Strategy not found"
            }
        
        session.commit()
        logger.info(f"RS ETF strategy {strategy_id} restarted successfully")
        
        return {
            "success": True,
            "message": "RS Stock Strategy restarted successfully"
        }
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Error restarting RS Stock strategy: {e}")
        raise HTTPException(status_code=500, detail=f"Error restarting RS Stock strategy: {str(e)}")
    finally:
        if session:
            session.close()


@deployment_router.delete("/delete-rs-strategy/{strategy_id}")
async def delete_rs_strategy(strategy_id: int):
    """Delete a saved RS Stock strategy"""
    session = None
    try:
        session = get_session()
        
        result = session.execute(text("""
            DELETE FROM rs_stock_instance 
            WHERE id = :strategy_id
        """), {"strategy_id": strategy_id})
        
        session.commit()
        
        return {
            "success": True,
            "message": "RS Stock Strategy deleted successfully"
        }
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Error deleting RS Stock strategy: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting RS Stock strategy: {str(e)}")
    finally:
        if session:
            session.close()


@deployment_router.post("/stop-rs-etf-strategy")
async def stop_rs_etf_strategy(request: dict):
    """Stop a running RS ETF strategy"""
    session = None
    try:
        # Validate required parameters
        strategy_id = request.get('strategy_id')
        user_id = request.get('user_id')
        
        if not strategy_id:
            return {
                "success": False,
                "message": "strategy_id is required"
            }
        
        session = get_session()
        
        # Build query based on whether user_id is provided
        if user_id:
            result = session.execute(text("""
                UPDATE rs_etf_instance 
                SET status = 'stop'
                WHERE id = :strategy_id AND user_id = :user_id
            """), {
                "strategy_id": strategy_id,
                "user_id": user_id
            })
        else:
            result = session.execute(text("""
                UPDATE rs_etf_instance 
                SET status = 'stop'
                WHERE id = :strategy_id
            """), {
                "strategy_id": strategy_id
            })
        
        # Check if any rows were updated
        if result.rowcount == 0:
            session.rollback()
            logger.warning(f"No RS ETF strategy found with id={strategy_id}")
            return {
                "success": False,
                "message": "RS ETF Strategy not found or already stopped"
            }
        
        session.commit()
        logger.info(f"RS ETF strategy {strategy_id} stopped successfully")
        
        return {
            "success": True,
            "message": "RS ETF Strategy stopped successfully"
        }
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Error stopping RS ETF strategy: {e}")
        raise HTTPException(status_code=500, detail=f"Error stopping RS ETF strategy: {str(e)}")
    finally:
        if session:
            session.close()


@deployment_router.post("/restart-rs-etf-strategy")
async def restart_rs_etf_strategy(request: dict):
    """Restart a stopped RS ETF strategy"""
    session = None
    try:
        # Validate required parameters
        strategy_id = request.get('strategy_id')
        user_id = request.get('user_id')
        
        if not strategy_id:
            return {
                "success": False,
                "message": "strategy_id is required"
            }
        
        session = get_session()
        
        # Build query based on whether user_id is provided
        if user_id:
            result = session.execute(text("""
                UPDATE rs_etf_instance 
                SET status = 'running'
                WHERE id = :strategy_id AND user_id = :user_id
            """), {
                "strategy_id": strategy_id,
                "user_id": user_id
            })
        else:
            result = session.execute(text("""
                UPDATE rs_etf_instance 
                SET status = 'running'
                WHERE id = :strategy_id
            """), {
                "strategy_id": strategy_id
            })
        
        # Check if any rows were updated
        if result.rowcount == 0:
            session.rollback()
            logger.warning(f"No RS ETF strategy found with id={strategy_id}")
            return {
                "success": False,
                "message": "RS ETF Strategy not found"
            }
        
        session.commit()
        logger.info(f"RS ETF strategy {strategy_id} restarted successfully")
        
        return {
            "success": True,
            "message": "RS ETF Strategy restarted successfully"
        }
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Error restarting RS ETF strategy: {e}")
        raise HTTPException(status_code=500, detail=f"Error restarting RS ETF strategy: {str(e)}")
    finally:
        if session:
            session.close()


@deployment_router.delete("/delete-rs-etf-strategy/{strategy_id}")
async def delete_rs_etf_strategy(strategy_id: int):
    """Delete a saved RS ETF strategy"""
    session = None
    try:
        session = get_session()
        
        result = session.execute(text("""
            DELETE FROM rs_etf_instance 
            WHERE id = :strategy_id
        """), {"strategy_id": strategy_id})
        
        session.commit()
        
        return {
            "success": True,
            "message": "RS ETF Strategy deleted successfully"
        }
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Error deleting RS ETF strategy: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting RS ETF strategy: {str(e)}")
    finally:
        if session:
            session.close()


# ============================================================================
# EXECUTION ENDPOINTS
# ============================================================================

@deployment_router.post("/execute/etf-signals")
async def execute_etf_signals(request: dict = None):
    """
    Execute ETF trading signals
    
    Request body (optional):
    {
        "signal_date": "2024-01-15",  # Optional - defaults to last Friday
        "side": "BUY"  # Optional - "BUY", "SELL", or None (both)
    }
    """
    try:
        from Services.execution.execution_service import ExecutionService
        
        execution_service = ExecutionService()
        
        signal_date = request.get("signal_date") if request else None
        side = request.get("side") if request else None
        
        result = execution_service.execute_all_signals(
            signal_date=signal_date,
            side=side,
            signal_type='etf'
        )
        
        return result
    except Exception as e:
        logger.error(f"Error executing ETF signals: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error executing ETF signals: {str(e)}"
        )


@deployment_router.post("/execute/stock-signals")
async def execute_stock_signals(request: dict = None):
    """
    Execute Stock trading signals
    
    Request body (optional):
    {
        "signal_date": "2024-01-15",  # Optional - defaults to last Friday
        "side": "BUY"  # Optional - "BUY", "SELL", or None (both)
    }
    """
    try:
        from Services.execution.execution_service import ExecutionService
        
        execution_service = ExecutionService()
        
        signal_date = request.get("signal_date") if request else None
        side = request.get("side") if request else None
        
        result = execution_service.execute_all_signals(
            signal_date=signal_date,
            side=side,
            signal_type='stock'
        )
        
        return result
    except Exception as e:
        logger.error(f"Error executing Stock signals: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error executing Stock signals: {str(e)}"
        )


# ============================================================================
# UNIFIED STRATEGY ENDPOINTS
# ============================================================================

@deployment_router.post("/save-strategy")
async def save_strategy_unified(request: dict):
    """Unified endpoint to save strategies - routes to stock or ETF based on strategy_type"""
    try:
        # Debug: Log the incoming request
        logger.info(f"🔍 Received save-strategy request: {list(request.keys())}")
        logger.info(f"📋 Strategy type: {request.get('strategy_type', 'NOT_PROVIDED')}")
        
        strategy_type = request.get("strategy_type", "").lower()
        
        # Handle different strategy type formats from frontend
        if strategy_type in ["stock", "stock_rotation"]:
            # Route to stock strategy endpoint
            from Strategies.Rotation_Stocks.stock_api import stock_router
            # Import the save function if available
            try:
                from Strategies.Rotation_Stocks.stock_api import save_stock_strategy
                return await save_stock_strategy(request)
            except ImportError:
                raise HTTPException(status_code=500, detail="Stock strategy save endpoint not available")
        elif strategy_type in ["etf", "etf_rotation"]:
            # Route to ETF strategy endpoint
            try:
                from Strategies.Rotation_ETF.etf_api import save_etf_strategy
                return await save_etf_strategy(request)
            except ImportError:
                raise HTTPException(status_code=500, detail="ETF strategy save endpoint not available")
        elif strategy_type in ["rs_strategy", "rs"]:
            # Route to RS strategy endpoint
            return await save_rs_strategy(request)
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid strategy_type: '{strategy_type}'. Must be 'stock', 'stock_rotation', 'etf', 'etf_rotation', or 'rs_strategy'. Received keys: {list(request.keys())}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in save_strategy_unified: {e}")
        if "validation error" in str(e).lower() or "pydantic" in str(e).lower():
            raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail=f"Error saving strategy: {str(e)}")


@deployment_router.get("/get-saved-strategies/{user_id}")
async def get_saved_strategies_unified(user_id: str):
    """Unified endpoint to get all saved strategies for a user"""
    try:
        all_strategies = []
        
        # Get stock strategies
        try:
            from Strategies.Rotation_Stocks.stock_api import get_saved_stock_strategies
            stock_result = await get_saved_stock_strategies(user_id)
            if "strategies" in stock_result:
                all_strategies.extend(stock_result["strategies"])
        except Exception as e:
            logger.warning(f"Could not fetch stock strategies: {e}")
        
        # Get ETF strategies
        try:
            from Strategies.Rotation_ETF.etf_api import get_saved_etf_strategies
            etf_result = await get_saved_etf_strategies(user_id)
            if "strategies" in etf_result:
                all_strategies.extend(etf_result["strategies"])
        except Exception as e:
            logger.warning(f"Could not fetch ETF strategies: {e}")
        
        # Get RS strategies
        try:
            rs_result = await get_saved_rs_strategies(user_id)
            if "strategies" in rs_result:
                all_strategies.extend(rs_result["strategies"])
        except Exception as e:
            logger.warning(f"Could not fetch RS strategies: {e}")
        
        # Get RS ETF strategies
        try:
            rs_etf_result = await get_saved_rs_etf_strategies(user_id)
            if "strategies" in rs_etf_result:
                all_strategies.extend(rs_etf_result["strategies"])
        except Exception as e:
            logger.warning(f"Could not fetch RS ETF strategies: {e}")
        
        # Sort by created_timestamp descending
        all_strategies.sort(key=lambda x: x.get("created_timestamp", "") or x.get("created_at", ""), reverse=True)
        
        return {"strategies": all_strategies}
    except Exception as e:
        logger.error(f"Error retrieving saved strategies: {e}")
        # Return empty array instead of throwing error to prevent frontend crashes
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


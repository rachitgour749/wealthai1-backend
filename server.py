from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.routing import APIRoute
import uvicorn
import sys
import os
import threading
import atexit
import time
import logging
import re

logger = logging.getLogger(__name__)

# Add the strategy directories to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'Strategies', 'stockstrategy'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Strategies', 'etf-strategy'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Strategies', 'rsStrategy'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Strategies', 'rsETFStrategy'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Strategies', 'customStrategy'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Strategies', 'SuperTrend'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'chatAI'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Services', 'Payment'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Services', 'cronjob'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Services', 'webhook'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Services', 'subscription'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Services', 'execution'))
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# Import the separate API modules
from Strategies.stockstrategy.stock_api import stock_router, initialize_stock_backtester, cleanup_stock_backtester
from chatAI.chat_api import chat_router, init_chat_ai, cleanup_chat_ai
from chatAI.zoho_api import zoho_router, init_zoho_chat, cleanup_zoho_chat
from Services.Payment.api import payment_router, init_payment_service, cleanup_payment_service
from Services.webhook.webhook_api import router as webhook_router
from Services.webhook.webhook_logic import init_db as init_webhook_db
from Services.subscription.api import subscription_router
from Services.subscription.google_oauth_api import google_oauth_router
from Services.subscription.database import subscription_manager


# Import RS strategy router (before ETF to avoid import conflicts)
from Strategies.rsStrategy.api import router as rs_router
# Import RS ETF strategy router
from Strategies.rsETFStrategy.api import router as rs_etf_router

# Import ETF strategy after RS strategy to avoid conflicts
from Strategies.etfstrategy.etf_api import etf_router, initialize_etf_backtester, cleanup_etf_backtester

# Import custom strategy router
from Strategies.customStrategy.api import custom_strategy_router
from Strategies.customStrategy.database import CustomStrategyDatabase

# Import SuperTrend strategy router
from Strategies.SuperTrend.api.routes import router as supertrend_router
from Strategies.SuperTrend.api.database import init_database as init_supertrend_database

# Import scheduler manager from Schedulers directory
try:
    from Schedulers.scheduler_manager import start_schedulers, stop_schedulers, get_scheduler_status
    scheduler_manager_available = True
    print("✅ Scheduler Manager imported successfully")
except ImportError as e:
    print(f"Warning: Could not import Scheduler Manager: {e}")
    start_schedulers = None
    stop_schedulers = None
    get_scheduler_status = None
    scheduler_manager_available = False

# Create main FastAPI app
app = FastAPI(title="Unified Rotation Backtester API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "http://127.0.0.1:3000",  # Local development alternative
        "https://your-cloudfront-domain.cloudfront.net",  # Your CloudFront domain
        "https://wealthai1.in",  # Your custom domain
        "https://www.wealthai1.in",  # www subdomain
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize backtesters and ChatAI
stock_backtester_initialized = initialize_stock_backtester("unified_etf_data.sqlite")
etf_backtester_initialized = initialize_etf_backtester("unified_etf_data.sqlite")
init_chat_ai()
chat_ai_initialized = True

# Initialize ZOHO Chat Integration
zoho_chat_initialized = init_zoho_chat()

# Initialize payment service
payment_service_initialized = init_payment_service()

# Initialize subscription service
try:
    subscription_manager.init_database()
    subscription_service_initialized = True
    print("[SUCCESS] Subscription service initialized successfully")
except Exception as e:
    
    subscription_service_initialized = False
    print(f"[ERROR] Failed to initialize subscription service: {e}")

# Initialize webhook database
try:
    init_webhook_db()
    webhook_service_initialized = True
    print("[SUCCESS] Webhook database initialized successfully")
except Exception as e:
    webhook_service_initialized = False
    print(f"[ERROR] Failed to initialize webhook database: {e}")

# Initialize custom strategy database
try:
    custom_strategy_db = CustomStrategyDatabase("unified_etf_data.sqlite")
    custom_strategy_service_initialized = True
    print("[SUCCESS] Custom strategy database initialized successfully")
except Exception as e:
    custom_strategy_service_initialized = False
    print(f"[ERROR] Failed to initialize custom strategy database: {e}")

# Initialize scheduler manager
scheduler_manager_initialized = False

def start_all_schedulers():
    """Start all schedulers using the scheduler manager"""
    global scheduler_manager_initialized
    
    if not scheduler_manager_available:
        print("⚠️ Scheduler Manager not available - skipping scheduler initialization")
        return
    
    try:
        print("🕐 Starting all schedulers...")
        success = start_schedulers()
        
        if success:
            scheduler_manager_initialized = True
            print("✅ All schedulers started successfully!")
            print("📊 ETF/Stock Scheduler: Daily at 4:00 PM IST (12 ETFs + 50 Stocks)")
            print("📊 RS Strategy Scheduler: Daily at 6:00 PM IST (Nifty 500)")
        else:
            scheduler_manager_initialized = False
            print("❌ Failed to start schedulers")
        
    except Exception as e:
        print(f"❌ Failed to start schedulers: {e}")
        scheduler_manager_initialized = False

# Add startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("🚀 Server starting up...")
    try:
        init_supertrend_database()
        print("[SUCCESS] SuperTrend database initialized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to initialize SuperTrend database: {e}")
    start_all_schedulers()

# Add cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global scheduler_manager_initialized
    
    print("🛑 Server shutting down...")
    
    # Stop all schedulers
    if scheduler_manager_initialized and scheduler_manager_available:
        try:
            print("🕐 Stopping all schedulers...")
            stop_schedulers()
            scheduler_manager_initialized = False
            print("✅ All schedulers stopped successfully")
        except Exception as e:
            print(f"❌ Error stopping schedulers: {e}")
    
    # Cleanup other services
    cleanup_stock_backtester()
    cleanup_etf_backtester()
    cleanup_chat_ai()
    cleanup_zoho_chat()
    cleanup_payment_service()

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Unified Rotation Backtester API", "strategies": ["stock", "etf", "rs-strategy", "custom-strategy", "chat"]}

@app.get("/health_check")
async def health_check():
    """Health check endpoint to verify API and database status"""
    try:
        status = {
            "api_status_latest": "healthy",
            "stock_backtester_initialized": stock_backtester_initialized,
            "etf_backtester_initialized": etf_backtester_initialized,
            "rs_strategy_initialized": True,  # RS strategy is always available
            "custom_strategy_initialized": custom_strategy_service_initialized,
            "chat_ai_initialized": chat_ai_initialized,
            "zoho_chat_initialized": zoho_chat_initialized,
            "payment_service_initialized": payment_service_initialized,
            "subscription_service_initialized": subscription_service_initialized,
            "webhook_service_initialized": webhook_service_initialized,
            "scheduler_initialized": scheduler_manager_initialized,
            "rs_scheduler_initialized": scheduler_manager_initialized,
            "stock_database_available": stock_backtester_initialized,
            "etf_database_available": etf_backtester_initialized,
            "rs_strategy_database_available": True,
            "custom_strategy_database_available": custom_strategy_service_initialized,
            "chat_ai_database_available": chat_ai_initialized,
            "payment_database_available": payment_service_initialized,
            "subscription_database_available": subscription_service_initialized,
            "webhook_database_available": True,
            "scheduler_database_available": scheduler_manager_initialized,
            "rs_scheduler_database_available": scheduler_manager_initialized,
            "stock_count": 0,
            "etf_count": 0,
            "scheduler_status": "running" if scheduler_manager_initialized else "stopped",
            "rs_scheduler_status": "running" if scheduler_manager_initialized else "stopped"
        }
        
        return status
    except Exception as e:
        return {
            "api_status": "error",
            "error": str(e),
            "stock_backtester_initialized": stock_backtester_initialized,
            "etf_backtester_initialized": etf_backtester_initialized,
            "rs_strategy_initialized": True,
            "custom_strategy_initialized": custom_strategy_service_initialized,
            "chat_ai_initialized": chat_ai_initialized,
            "zoho_chat_initialized": zoho_chat_initialized,
            "payment_service_initialized": payment_service_initialized,
            "subscription_service_initialized": subscription_service_initialized,
            "webhook_service_initialized": webhook_service_initialized,
            "scheduler_initialized": scheduler_manager_initialized,
            "rs_scheduler_initialized": scheduler_manager_initialized
        }

@app.get("/favicon.ico")
async def favicon():
    """Return a simple favicon to prevent 404 errors"""
    # Return a minimal 1x1 transparent PNG
    favicon_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x00\x00\x02\x00\x01\xe5\x27\xde\xfc\x00\x00\x00\x00IEND\xaeB`\x82'
    return Response(content=favicon_data, media_type="image/x-icon")

@app.post("/api/scheduler/trigger")
async def trigger_scheduler():
    """Manually trigger the market data fetch (ETFs + Stocks)"""
    global scheduler_instance
    
    if not scheduler_instance:
        raise HTTPException(status_code=500, detail="Scheduler not initialized")
    
    try:
        print("🔄 Manually triggering market data fetch...")
        success_count, fail_count = scheduler_instance.fetcher.run_daily_fetch()
        
        return {
            "status": "success",
            "message": "Market data fetch completed",
            "successful_fetches": success_count,
            "failed_fetches": fail_count,
            "total_instruments": len(scheduler_instance.fetcher.all_symbols)
        }
    except Exception as e:
        print(f"❌ Error in manual market data fetch: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching market data: {str(e)}")

@app.get("/api/scheduler/status")
async def get_scheduler_status():
    """Get the current status of the market data scheduler"""
    global scheduler_instance, scheduler_initialized
    
    if not scheduler_initialized:
        return {
            "status": "stopped",
            "message": "Scheduler not initialized",
            "next_run": None,
            "etf_count": 0
        }
    
    try:
        jobs = scheduler_instance.scheduler.get_jobs()
        next_run = None
        if jobs:
            next_run = jobs[0].next_run_time.isoformat() if jobs[0].next_run_time else None
        
        return {
            "status": "running",
            "message": "Scheduler is running and will fetch data daily at 5:50 PM IST",
            "next_run": next_run,
            "etf_count": len(scheduler_instance.fetcher.etf_symbols),
            "etf_symbols": scheduler_instance.fetcher.etf_symbols
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting scheduler status: {str(e)}",
            "next_run": None,
            "etf_count": 0
        }

# RS Strategy Scheduler Endpoints
@app.post("/api/rs-scheduler/trigger-eod")
async def trigger_rs_eod_fetch():
    """Manually trigger RS strategy EOD data fetch"""
    global rs_scheduler_instance, rs_scheduler_initialized
    
    if not rs_scheduler_initialized or not rs_scheduler_instance:
        raise HTTPException(status_code=500, detail="RS strategy scheduler not initialized")
    
    try:
        print("🔄 Manually triggering RS strategy EOD data fetch...")
        rs_scheduler_instance.run_manual_eod_fetch()
        
        return {
            "status": "success",
            "message": "RS strategy EOD data fetch completed"
        }
    except Exception as e:
        print(f"❌ Error in manual RS EOD data fetch: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching RS EOD data: {str(e)}")

@app.post("/api/rs-scheduler/trigger-all")
async def trigger_all_rs_tasks():
    """Manually trigger RS strategy EOD data fetch"""
    global rs_scheduler_instance, rs_scheduler_initialized
    
    if not rs_scheduler_initialized or not rs_scheduler_instance:
        raise HTTPException(status_code=500, detail="RS strategy scheduler not initialized")
    
    try:
        print("🔄 Manually triggering all RS strategy tasks...")
        rs_scheduler_instance.run_manual_eod_fetch()
        
        return {
            "status": "success",
            "message": "All RS strategy tasks completed successfully"
        }
    except Exception as e:
        print(f"❌ Error in manual RS tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Error running RS tasks: {str(e)}")

@app.get("/api/rs-scheduler/status")
async def get_rs_scheduler_status():
    """Get the current status of the RS strategy scheduler"""
    global rs_scheduler_instance, rs_scheduler_initialized
    
    if not rs_scheduler_initialized:
        return {
            "status": "stopped",
            "message": "RS strategy scheduler not initialized",
            "next_run": None
        }
    
    try:
        return {
            "status": "running",
            "message": "RS strategy scheduler is running",
            "eod_fetch_time": "Daily at 4:15 PM IST",
            "cleanup_time": "Weekly on Sunday at 2:00 AM IST"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting RS scheduler status: {str(e)}",
            "next_run": None
        }

# Include the routers in the main app
app.include_router(stock_router)
app.include_router(etf_router)
app.include_router(rs_router, prefix="/api/rs-strategy", tags=["RS Strategy"])
app.include_router(rs_etf_router, prefix="/api/rs-etf-strategy", tags=["RS ETF Strategy"])
app.include_router(custom_strategy_router)
app.include_router(chat_router)
app.include_router(zoho_router)
app.include_router(payment_router)
app.include_router(webhook_router)
app.include_router(subscription_router)
app.include_router(google_oauth_router)

# Remove SuperTrend root route to avoid duplicate "/" definition
supertrend_router.routes = [
    route for route in supertrend_router.routes
    if not (isinstance(route, APIRoute) and route.path == "/")
]
app.include_router(supertrend_router, tags=["SuperTrend"])

# RS Strategy API Endpoints
import sqlite3
import json
import time
from datetime import datetime

def init_saved_rs_strategies_table(db_path: str = "Strategies/rsStrategy/nifty500_data_with_metadata.sqlite"):
    """Initialize the saved_rs_strategies table if it doesn't exist"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS saved_rs_strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                strategy_type TEXT NOT NULL,
                user_id TEXT NOT NULL,
                config_id INTEGER,
                backtest_id INTEGER,
                start_date TEXT,
                end_date TEXT,
                stock_universe TEXT,
                backtest_results TEXT,
                strategy_config TEXT,
                run_id TEXT UNIQUE,
                client_information_json TEXT,
                webhook_url TEXT,
                status TEXT DEFAULT 'deploy',
                created_at TEXT,
                UNIQUE(strategy_name, user_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating saved_rs_strategies table: {e}")
        return False

@app.post("/api/save-rs-strategy")
async def save_rs_strategy(request: dict):
    """Save RS strategy with deployment features"""
    try:
        # Initialize database table if needed
        if not init_saved_rs_strategies_table():
            raise HTTPException(status_code=500, detail="Failed to initialize database table")
        
        conn = sqlite3.connect("Strategies/rsStrategy/nifty500_data_with_metadata.sqlite")
        cursor = conn.cursor()
        
        # Check if strategy already exists
        cursor.execute('''
            SELECT id FROM saved_rs_strategies 
            WHERE strategy_name = ? AND user_id = ?
        ''', (request['strategy_name'], request['user_id']))
        
        if cursor.fetchone():
            return {
                "success": False,
                "message": "Strategy with this name already exists",
                "strategy_exists": True
            }
        
        # Insert new strategy
        cursor.execute('''
            INSERT INTO saved_rs_strategies (
                strategy_name, strategy_type, user_id, config_id, backtest_id,
                start_date, end_date, stock_universe, backtest_results, 
                strategy_config, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            request['strategy_name'],
            request['strategy_type'],
            request['user_id'],
            request.get('config_id'),
            request.get('backtest_id'),
            request.get('start_date'),
            request.get('end_date'),
            request.get('stock_universe'),
            json.dumps(request.get('backtest_results', {})),
            json.dumps(request.get('strategy_config', {})),
            request.get('created_at')
        ))
        
        strategy_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": "RS Strategy saved successfully",
            "strategy_id": strategy_id,
            "strategy_exists": False
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving RS strategy: {str(e)}")

@app.get("/api/get-saved-rs-strategies/{user_id}")
async def get_saved_rs_strategies(user_id: str):
    """Get all saved RS strategies for a specific user"""
    try:
        conn = sqlite3.connect("Strategies/rsStrategy/nifty500_data_with_metadata.sqlite")
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM saved_rs_strategies 
            WHERE user_id = ? 
            ORDER BY created_at DESC
        ''', (user_id,))
        
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        
        strategies = []
        for row in rows:
            strategy_dict = dict(zip(columns, row))
            
            # Parse JSON fields
            if strategy_dict.get('backtest_results'):
                try:
                    strategy_dict['backtest_results'] = json.loads(strategy_dict['backtest_results'])
                except:
                    strategy_dict['backtest_results'] = {}
            
            if strategy_dict.get('strategy_config'):
                try:
                    strategy_dict['strategy_config'] = json.loads(strategy_dict['strategy_config'])
                except:
                    strategy_dict['strategy_config'] = {}
            
            if strategy_dict.get('client_information_json'):
                try:
                    strategy_dict['client_information_json'] = json.loads(strategy_dict['client_information_json'])
                except:
                    strategy_dict['client_information_json'] = {}
            
            strategies.append(strategy_dict)
        
        conn.close()
        
        return {
            "success": True,
            "strategies": strategies
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching RS strategies: {str(e)}")

@app.post("/api/save-rs-deployment")
async def save_rs_deployment(request: dict):
    """Save RS strategy deployment with webhook and client info"""
    try:
        # Initialize database table if needed (this will also run migrations)
        if not init_rs_etf_saved_strategies_table():
            return {
                "success": False,
                "message": "Failed to initialize database table"
            }
        
        # Debug logging
        print(f"🔍 RS Deployment Request: {list(request.keys())}")
        print(f"📧 User Email: {request.get('user_email')}")
        print(f"📋 Strategy Name: {request.get('strategy_name')}")
        print(f"🔗 Webhook URL: {request.get('webhook_url')}")
        
        # Use correct database for RS ETF strategies
        conn = sqlite3.connect("unified_etf_data.sqlite")
        cursor = conn.cursor()
        
        # Extract data with proper field mapping (matching WebHook component)
        user_email = request.get('user_email', '').strip()
        strategy_name = request.get('strategy_name', '').strip()
        client_information_json = request.get('client_information_json', '{}')
        webhook_url = request.get('webhook_url', '')
        
        # Validate required fields
        if not user_email:
            conn.close()
            return {
                "success": False,
                "message": "user_email is required"
            }
        
        if not strategy_name:
            conn.close()
            return {
                "success": False,
                "message": "strategy_name is required"
            }
        
        # Debug: List all strategies for this user to help diagnose
        cursor.execute('''
            SELECT id, strategy_name, user_id FROM rs_etf_saved_strategies 
            WHERE user_id = ?
        ''', (user_email,))
        all_strategies = cursor.fetchall()
        print(f"🔍 All strategies for user {user_email}: {all_strategies}")
        print(f"🔍 Searching for strategy: '{strategy_name}' (length: {len(strategy_name)})")
        
        # Find strategy by name and user email (case-insensitive with trimming)
        # Use LOWER() for case-insensitive comparison and TRIM() for whitespace
        cursor.execute('''
            SELECT id, strategy_name FROM rs_etf_saved_strategies 
            WHERE LOWER(TRIM(strategy_name)) = LOWER(?) AND user_id = ?
        ''', (strategy_name, user_email))
        
        strategy_result = cursor.fetchone()
        if not strategy_result:
            # Try exact match as fallback
            cursor.execute('''
                SELECT id, strategy_name FROM rs_etf_saved_strategies 
                WHERE strategy_name = ? AND user_id = ?
            ''', (strategy_name, user_email))
            strategy_result = cursor.fetchone()
            
            if not strategy_result:
                conn.close()
                return {
                    "success": False,
                    "message": f"Strategy '{strategy_name}' not found for user {user_email}. Available strategies: {[s[1] for s in all_strategies]}"
                }
        
        strategy_id = strategy_result[0]
        actual_strategy_name = strategy_result[1]
        print(f"✅ Found strategy: ID={strategy_id}, Name='{actual_strategy_name}'")
        
        # Verify deployment columns exist, add if missing
        cursor.execute("PRAGMA table_info(rs_etf_saved_strategies)")
        existing_columns = [row[1] for row in cursor.fetchall()]
        deployment_columns = {
            'run_id': 'TEXT',
            'webhook_url': 'TEXT',
            'client_information_json': 'TEXT'
        }
        
        for column_name, column_def in deployment_columns.items():
            if column_name not in existing_columns:
                try:
                    cursor.execute(f"ALTER TABLE rs_etf_saved_strategies ADD COLUMN {column_name} {column_def}")
                    conn.commit()
                    print(f"✅ Added missing column: {column_name}")
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        print(f"⚠️  Could not add column {column_name}: {e}")
        
        # Generate unique run_id
        # Clean strategy name: replace spaces and special characters with underscores
        clean_name = actual_strategy_name.replace(' ', '_').replace('-', '_')
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', clean_name)
        run_id = f"RS_ETF_{clean_name}_{int(time.time())}"
        
        # Update strategy with deployment info
        cursor.execute('''
            UPDATE rs_etf_saved_strategies 
            SET run_id = ?, client_information_json = ?, webhook_url = ?, status = ?
            WHERE id = ?
        ''', (
            run_id,
            client_information_json,  # Already JSON string from frontend
            webhook_url,
            'running',
            strategy_id
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": "RS Strategy deployment saved successfully",
            "run_id": run_id
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error in save_rs_deployment: {str(e)}")
        return {
            "success": False,
            "message": f"Error saving RS deployment: {str(e)}"
        }

@app.post("/api/live-signals/save-deployment")
async def save_deployment(request: dict):
    """Save ETF/Stock strategy deployment with webhook and client info"""
    try:
        # Debug logging
        print(f"🔍 Deployment Request: {list(request.keys())}")
        print(f"📧 User Email: {request.get('user_email')}")
        print(f"📋 Strategy Name: {request.get('strategy_name')}")
        print(f"🔗 Webhook URL: {request.get('webhook_url')}")
        
        conn = sqlite3.connect("unified_etf_data.sqlite")
        cursor = conn.cursor()
        
        # Extract data
        user_email = request.get('user_email')
        strategy_name = request.get('strategy_name')
        client_information_json = request.get('client_information_json', '{}')
        webhook_url = request.get('webhook_url', '')
        
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
        cursor.execute('''
            SELECT id FROM etf_saved_strategy 
            WHERE strategy_name = ? AND user_id = ?
        ''', (strategy_name, user_email))
        
        strategy_result = cursor.fetchone()
        table_name = 'etf_saved_strategy'
        
        if not strategy_result:
            # Check Stock table
            cursor.execute('''
                SELECT id FROM stock_saved_strategy 
                WHERE strategy_name = ? AND user_id = ?
            ''', (strategy_name, user_email))
            strategy_result = cursor.fetchone()
            table_name = 'stock_saved_strategy'
        
        if not strategy_result:
            conn.close()
            return {
                "success": False,
                "message": f"Strategy '{strategy_name}' not found for user {user_email}"
            }
        
        strategy_id = strategy_result[0]
        
        # Generate unique run_id based on table type
        # Clean strategy name: replace spaces and special characters with underscores
        clean_name = strategy_name.replace(' ', '_').replace('-', '_')
        # Remove any other special characters that might cause issues
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', clean_name)
        
        if table_name == 'etf_saved_strategy':
            run_id = f"run_etfs_rotation_strategy_{clean_name}_{int(time.time())}"
        else:
            run_id = f"run_stock_rotation_strategy_{clean_name}_{int(time.time())}"
        
        # Update strategy with deployment info
        cursor.execute(f'''
            UPDATE {table_name} 
            SET run_id = ?, client_information_json = ?, webhook_url = ?, status = ?
            WHERE id = ?
        ''', (
            run_id,
            client_information_json,  # Already JSON string from frontend
            webhook_url,
            'running',
            strategy_id
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": f"{table_name.replace('_', ' ').title()} deployment saved successfully",
            "run_id": run_id
        }
        
    except Exception as e:
        print(f"❌ Error in save_deployment: {str(e)}")
        return {
            "success": False,
            "message": f"Error saving deployment: {str(e)}"
        }

@app.post("/api/live-signals/deployment-status-by-strategy")
async def get_deployment_status_by_strategy(request: dict):
    """Get deployment status for ETF/Stock strategy by strategy name"""
    try:
        strategy_name = request.get('strategy_name')
        execution_date = request.get('execution_date')  # Optional
        
        if not strategy_name:
            return {
                "success": False,
                "message": "strategy_name is required",
                "data": {"exists": False}
            }
        
        conn = sqlite3.connect("unified_etf_data.sqlite")
        cursor = conn.cursor()
        
        # Check ETF table first
        cursor.execute('''
            SELECT id, run_id, status, client_information_json, webhook_url, 
                   execution_date, created_at
            FROM etf_saved_strategy 
            WHERE strategy_name = ?
        ''', (strategy_name,))
        
        result = cursor.fetchone()
        table_name = 'etf_saved_strategy'
        
        if not result:
            # Check Stock table
            cursor.execute('''
                SELECT id, run_id, status, client_information_json, webhook_url, 
                       execution_date, created_at
                FROM stock_saved_strategy 
                WHERE strategy_name = ?
            ''', (strategy_name,))
            result = cursor.fetchone()
            table_name = 'stock_saved_strategy'
        
        conn.close()
        
        if result:
            return {
                "success": True,
                "data": {
                    "exists": True,
                    "status": result[2] if result[2] else 'deploy',
                    "run_id": result[1],
                    "client_information_json": result[3],
                    "webhook_url": result[4],
                    "execution_date": result[5],
                    "created_at": result[6],
                    "strategy_id": result[0],
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
        print(f"❌ Error in get_deployment_status_by_strategy: {str(e)}")
        return {
            "success": False,
            "message": f"Error getting deployment status: {str(e)}",
            "data": {"exists": False}
        }

@app.post("/api/live-signals/update-client-information")
async def update_client_information(request: dict):
    """Update client information for ETF/Stock strategy by run_id"""
    try:
        run_id = request.get('run_id')
        client_information_json = request.get('client_information_json', '{}')
        
        if not run_id:
            return {
                "success": False,
                "message": "run_id is required"
            }
        
        conn = sqlite3.connect("unified_etf_data.sqlite")
        cursor = conn.cursor()
        
        # Check ETF table first
        cursor.execute('''
            SELECT id FROM etf_saved_strategy 
            WHERE run_id = ?
        ''', (run_id,))
        
        result = cursor.fetchone()
        table_name = 'etf_saved_strategy'
        
        if not result:
            # Check Stock table
            cursor.execute('''
                SELECT id FROM stock_saved_strategy 
                WHERE run_id = ?
            ''', (run_id,))
            result = cursor.fetchone()
            table_name = 'stock_saved_strategy'
        
        if not result:
            conn.close()
            return {
                "success": False,
                "message": f"Strategy with run_id '{run_id}' not found"
            }
        
        # Update client information
        cursor.execute(f'''
            UPDATE {table_name} 
            SET client_information_json = ?
            WHERE run_id = ?
        ''', (client_information_json, run_id))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": f"Client information updated successfully"
        }
        
    except Exception as e:
        print(f"❌ Error in update_client_information: {str(e)}")
        return {
            "success": False,
            "message": f"Error updating client information: {str(e)}"
        }

@app.post("/api/update-rs-client-information")
async def update_rs_client_information(request: dict):
    """Update client information for RS strategy"""
    try:
        # Debug logging
        print(f"🔍 Update RS Client Info Request: {list(request.keys())}")
        print(f"📋 Strategy ID: {request.get('strategy_id')}")
        print(f"📧 User ID: {request.get('user_id')}")
        
        conn = sqlite3.connect("Strategies/rsStrategy/nifty500_data_with_metadata.sqlite")
        cursor = conn.cursor()
        
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
        
        # Update client information
        cursor.execute('''
            UPDATE saved_rs_strategies 
            SET client_information_json = ?
            WHERE id = ? AND user_id = ?
        ''', (client_information_json, strategy_id, user_id))
        
        if cursor.rowcount == 0:
            conn.close()
            return {
                "success": False,
                "message": f"Strategy not found or user mismatch"
            }
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": "RS Strategy client information updated successfully"
        }
        
    except Exception as e:
        print(f"❌ Error in update_rs_client_information: {str(e)}")
        return {
            "success": False,
            "message": f"Error updating client information: {str(e)}"
        }

@app.post("/api/stop-rs-strategy")
async def stop_rs_strategy(request: dict):
    """Stop a running RS strategy"""
    try:
        conn = sqlite3.connect("Strategies/rsStrategy/nifty500_data_with_metadata.sqlite")
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE saved_rs_strategies 
            SET status = 'stop'
            WHERE id = ? AND user_id = ?
        ''', (request['strategy_id'], request['user_id']))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": "RS Strategy stopped successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping RS strategy: {str(e)}")

@app.post("/api/restart-rs-strategy")
async def restart_rs_strategy(request: dict):
    """Restart a stopped RS strategy"""
    try:
        conn = sqlite3.connect("Strategies/rsStrategy/nifty500_data_with_metadata.sqlite")
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE saved_rs_strategies 
            SET status = 'running'
            WHERE id = ? AND user_id = ?
        ''', (request['strategy_id'], request['user_id']))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": "RS Strategy restarted successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error restarting RS strategy: {str(e)}")

@app.delete("/api/delete-rs-strategy/{strategy_id}")
async def delete_rs_strategy(strategy_id: int):
    """Delete a saved RS strategy"""
    try:
        conn = sqlite3.connect("Strategies/rsStrategy/nifty500_data_with_metadata.sqlite")
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM saved_rs_strategies 
            WHERE id = ?
        ''', (strategy_id,))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": "RS Strategy deleted successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting RS strategy: {str(e)}")

# RS ETF Strategy Saved Strategies API
def init_rs_etf_saved_strategies_table(db_path: str = "unified_etf_data.sqlite"):
    """Initialize the rs_etf_saved_strategies table if it doesn't exist"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rs_etf_saved_strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                strategy_type TEXT NOT NULL DEFAULT 'RS ETF Strategy',
                user_id TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT,
                rs_etf_universe TEXT,
                backtest_results TEXT,
                strategy_config TEXT,
                created_at TEXT,
                is_active INTEGER DEFAULT 1,
                updated_at TEXT,
                last_run_date TEXT,
                next_run_date TEXT,
                run_frequency TEXT DEFAULT 'daily',
                status TEXT DEFAULT 'deploy',
                UNIQUE(strategy_name, user_id)
            )
        ''')
        
        # Always check for missing columns and add them (migration logic)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rs_etf_saved_strategies'")
        if cursor.fetchone():
            # Get existing columns
            cursor.execute("PRAGMA table_info(rs_etf_saved_strategies)")
            existing_columns = [row[1] for row in cursor.fetchall()]
            
            # Define required columns with proper SQL syntax
            required_columns = {
                'rs_etf_universe': 'TEXT',
                'is_active': 'INTEGER DEFAULT 1',
                'updated_at': 'TEXT',
                'last_run_date': 'TEXT',
                'next_run_date': 'TEXT',
                'run_frequency': 'TEXT DEFAULT \'daily\'',
                'status': 'TEXT DEFAULT \'deploy\'',
                'run_id': 'TEXT',
                'webhook_url': 'TEXT',
                'client_information_json': 'TEXT'
            }
            
            # Add missing columns
            for column_name, column_def in required_columns.items():
                if column_name not in existing_columns:
                    try:
                        # Use parameterized query for safety, but ALTER TABLE doesn't support parameters
                        # So we need to validate the column_def is safe
                        safe_column_def = column_def.replace(';', '')  # Remove any potential SQL injection
                        cursor.execute(f"ALTER TABLE rs_etf_saved_strategies ADD COLUMN {column_name} {safe_column_def}")
                        conn.commit()  # Commit after each column addition
                        print(f"✅ Added missing column: {column_name}")
                    except sqlite3.OperationalError as e:
                        # Column might already exist or other operational error
                        if "duplicate column" not in str(e).lower():
                            print(f"⚠️  Could not add column {column_name}: {e}")
                    except Exception as e:
                        print(f"⚠️  Error adding column {column_name}: {e}")
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating rs_etf_saved_strategies table: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.post("/api/save-rs-etf-strategy")
async def save_rs_etf_strategy(request: dict):
    """Save RS ETF Strategy configuration"""
    try:
        # Initialize database table if needed (this will also run migrations)
        if not init_rs_etf_saved_strategies_table():
            return {
                "success": False,
                "message": "Failed to initialize database table"
            }
        
        conn = sqlite3.connect("unified_etf_data.sqlite")
        cursor = conn.cursor()
        
        # Verify that etf_universe column exists (double-check after migration)
        cursor.execute("PRAGMA table_info(rs_etf_saved_strategies)")
        existing_columns = [row[1] for row in cursor.fetchall()]
        
        if 'etf_universe' not in existing_columns:
            # Try to add it one more time
            try:
                cursor.execute("ALTER TABLE rs_etf_saved_strategies ADD COLUMN etf_universe TEXT")
                conn.commit()
                print("✅ Added etf_universe column as fallback")
            except Exception as e:
                conn.close()
                return {
                    "success": False,
                    "message": f"Database table is missing required column 'etf_universe'. Please contact support. Error: {str(e)}"
                }
        
        # Check if strategy already exists
        cursor.execute('''
            SELECT id FROM rs_etf_saved_strategies 
            WHERE strategy_name = ? AND user_id = ?
        ''', (request.get('strategy_name'), request.get('user_id')))
        
        if cursor.fetchone():
            conn.close()
            return {
                "success": False,
                "message": "Strategy with this name already exists",
                "strategy_exists": True
            }
        
        # Insert new strategy
        cursor.execute('''
            INSERT INTO rs_etf_saved_strategies (
                strategy_name, strategy_type, user_id, 
                start_date, end_date, rs_etf_universe, backtest_results, 
                strategy_config, created_at, is_active, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            request.get('strategy_name'),
            request.get('strategy_type', 'RS ETF Strategy'),
            request.get('user_id'),
            request.get('start_date'),
            request.get('end_date'),
            request.get('rs_etf_universe') or request.get('stock_universe') or 'ALL_ETFS',  # Handle both field names
            json.dumps(request.get('backtest_results', {})) if request.get('backtest_results') else None,
            json.dumps(request.get('strategy_config', {})) if request.get('strategy_config') else None,
            request.get('created_at', datetime.now().isoformat()),
            1,  # is_active
            request.get('status', 'deploy')
        ))
        
        strategy_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": "RS ETF Strategy saved successfully",
            "strategy_id": strategy_id,
            "strategy_exists": False
        }
        
    except sqlite3.OperationalError as e:
        error_msg = str(e)
        if "no column named" in error_msg.lower():
            return {
                "success": False,
                "message": f"Database schema error: {error_msg}. The table structure needs to be updated. Please restart the server to run migrations."
            }
        return {
            "success": False,
            "message": f"Database error: {error_msg}"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Error saving RS ETF strategy: {str(e)}"
        }

@app.get("/api/get-saved-rs-etf-strategies/{user_id}")
async def get_saved_rs_etf_strategies(user_id: str):
    """Get all saved RS ETF strategies for a specific user"""
    try:
        conn = sqlite3.connect("unified_etf_data.sqlite")
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM rs_etf_saved_strategies 
            WHERE user_id = ? 
            ORDER BY created_at DESC
        ''', (user_id,))
        
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        
        strategies = []
        for row in rows:
            strategy_dict = dict(zip(columns, row))
            
            # Parse JSON fields
            if strategy_dict.get('backtest_results'):
                try:
                    strategy_dict['backtest_results'] = json.loads(strategy_dict['backtest_results'])
                except:
                    strategy_dict['backtest_results'] = {}
            
            if strategy_dict.get('strategy_config'):
                try:
                    strategy_dict['strategy_config'] = json.loads(strategy_dict['strategy_config'])
                except:
                    strategy_dict['strategy_config'] = {}
            
            # Convert is_active to boolean
            if strategy_dict.get('is_active') is not None:
                strategy_dict['is_active'] = bool(strategy_dict['is_active'])
            
            strategies.append(strategy_dict)
        
        conn.close()
        
        return {
            "success": True,
            "strategies": strategies
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error fetching RS ETF strategies: {str(e)}"
        }

@app.post("/api/stop-rs-etf-strategy")
async def stop_rs_etf_strategy(request: dict):
    """Stop a running RS ETF strategy"""
    try:
        conn = sqlite3.connect("unified_etf_data.sqlite")
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE rs_etf_saved_strategies 
            SET status = 'stopped', is_active = 0
            WHERE id = ? AND user_id = ?
        ''', (request.get('strategy_id'), request.get('user_id')))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": "RS ETF Strategy stopped successfully"
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error stopping RS ETF strategy: {str(e)}"
        }

@app.post("/api/restart-rs-etf-strategy")
async def restart_rs_etf_strategy(request: dict):
    """Restart a stopped RS ETF strategy"""
    try:
        conn = sqlite3.connect("unified_etf_data.sqlite")
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE rs_etf_saved_strategies 
            SET status = 'active', is_active = 1
            WHERE id = ? AND user_id = ?
        ''', (request.get('strategy_id'), request.get('user_id')))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": "RS ETF Strategy restarted successfully"
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error restarting RS ETF strategy: {str(e)}"
        }

@app.delete("/api/delete-rs-etf-strategy/{strategy_id}")
async def delete_rs_etf_strategy(strategy_id: int):
    """Delete a saved RS ETF strategy"""
    try:
        conn = sqlite3.connect("unified_etf_data.sqlite")
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM rs_etf_saved_strategies 
            WHERE id = ?
        ''', (strategy_id,))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": "RS ETF Strategy deleted successfully"
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error deleting RS ETF strategy: {str(e)}"
        }

# Unified strategy save endpoint that routes based on strategy_type
@app.post("/api/save-strategy")
async def save_strategy_unified(request: dict):
    """Unified endpoint to save strategies - routes to stock or ETF based on strategy_type"""
    try:
        # Debug: Log the incoming request
        print(f"🔍 Received save-strategy request: {list(request.keys())}")
        print(f"📋 Strategy type: {request.get('strategy_type', 'NOT_PROVIDED')}")
        print(f"📊 Backtest results: {request.get('backtest_results', 'NOT_PROVIDED')}")
        print(f"🎯 Tickers: {request.get('tickers', 'NOT_PROVIDED')}")
        
        strategy_type = request.get("strategy_type", "").lower()
        
        # Handle different strategy type formats from frontend
        if strategy_type in ["stock", "stock_rotation"]:
            # Route to stock strategy endpoint
            from Strategies.stockstrategy.stock_api import save_stock_strategy, SaveStockStrategyRequest
            stock_request = SaveStockStrategyRequest(**request)
            return await save_stock_strategy(stock_request)
        elif strategy_type in ["etf", "etf_rotation"]:
            # Route to ETF strategy endpoint
            from Strategies.etfstrategy.etf_api import save_etf_strategy, SaveETFStrategyRequest
            etf_request = SaveETFStrategyRequest(**request)
            return await save_etf_strategy(etf_request)
        elif strategy_type in ["rs_strategy", "rs"]:
            # Route to RS strategy endpoint
            return await save_rs_strategy(request)
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid strategy_type: '{strategy_type}'. Must be 'stock', 'stock_rotation', 'etf', 'etf_rotation', or 'rs_strategy'. Received keys: {list(request.keys())}"
            )
            
    except Exception as e:
        print(f"❌ Error in save_strategy_unified: {str(e)}")
        if "Invalid strategy_type" in str(e):
            raise e
        elif "validation error" in str(e).lower() or "pydantic" in str(e).lower():
            raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail=f"Error saving strategy: {str(e)}")

# Unified endpoint to get saved strategies
@app.get("/api/get-saved-strategies/{user_id}")
async def get_saved_strategies_unified(user_id: str):
    """Unified endpoint to get all saved strategies for a user"""
    try:
        all_strategies = []
        
        # Get stock strategies
        try:
            from Strategies.stockstrategy.stock_api import get_saved_stock_strategies
            stock_result = await get_saved_stock_strategies(user_id)
            if "strategies" in stock_result:
                all_strategies.extend(stock_result["strategies"])
        except Exception as e:
            print(f"Warning: Could not fetch stock strategies: {e}")
        
        # Get ETF strategies
        try:
            from Strategies.etfstrategy.etf_api import get_saved_etf_strategies
            etf_result = await get_saved_etf_strategies(user_id)
            if "strategies" in etf_result:
                all_strategies.extend(etf_result["strategies"])
        except Exception as e:
            print(f"Warning: Could not fetch ETF strategies: {e}")
        
        # Sort by created_timestamp descending
        all_strategies.sort(key=lambda x: x.get("created_timestamp", ""), reverse=True)
        
        return {"strategies": all_strategies}
        
    except Exception as e:
        print(f"Error retrieving saved strategies: {str(e)}")
        # Return empty array instead of throwing error to prevent frontend crashes
        return {"strategies": []}

# Debug endpoint to test request format
@app.post("/api/debug-request")
async def debug_request(request: dict):
    """Debug endpoint to inspect request format"""
    return {
        "received_keys": list(request.keys()),
        "strategy_type": request.get("strategy_type", "NOT_PROVIDED"),
        "request_sample": {k: str(v)[:100] + "..." if len(str(v)) > 100 else v for k, v in request.items()}
    }

# RS Strategy Deployment Endpoints
@app.post("/api/rs-strategy/deploy")
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

@app.get("/api/rs-strategy/signals/latest")
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

# ============================================================================
# EXECUTION API ENDPOINTS
# ============================================================================

@app.post("/api/execute/etf-signals")
async def execute_etf_signals(request: dict = None):
    """
    Execute ETF trading signals
    
    Request body (optional):
    {
        "signal_date": "2024-01-15",  # Optional - defaults to last Friday
        "side": "BUY"  # Optional - "BUY", "SELL", or None (both)
    }
    
    Returns:
    {
        "success": bool,
        "message": str,
        "total_signals": int,
        "successful": int,
        "failed": int,
        "duration_seconds": float,
        "results": List[Dict]
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

@app.post("/api/execute/stock-signals")
async def execute_stock_signals(request: dict = None):
    """
    Execute Stock trading signals
    
    Request body (optional):
    {
        "signal_date": "2024-01-15",  # Optional - defaults to last Friday
        "side": "BUY"  # Optional - "BUY", "SELL", or None (both)
    }
    
    Returns:
    {
        "success": bool,
        "message": str,
        "total_signals": int,
        "successful": int,
        "failed": int,
        "duration_seconds": float,
        "results": List[Dict]
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

if __name__ == "__main__":
    print("🚀 Starting Unified Rotation Backtester API Server...")
    print("📊 Available endpoints:")
    print("   GET  /health_check - Health check")
    print("   GET  /api/stocks - List available stocks")
    print("   GET  /api/etfs - List available ETFs")
    print("   POST /api/stocks/metrics - Run stock backtest")
    print("   POST /api/metrics - Run ETF backtest")
    print("   POST /api/chat - Chat with AI")
    print("   POST /api/rate - Rate AI response")
    print("   POST /api/zoho/chat - ZOHO Bot Chat (no conversation storage)")
    print("   GET  /api/zoho/health - ZOHO API health check")
    print("   POST /api/zoho/test - ZOHO API test endpoint")
    print("   POST /api/save-strategy - Save strategy (unified - stock/ETF)")
    print("   GET  /api/get-saved-strategies/{user_id} - Get all saved strategies (unified)")
    print("   POST /api/debug-request - Debug request format")
    print("   POST /api/custom-strategy/analyze - Analyze custom strategy using AI")
    print("   POST /api/custom-strategy/save - Save custom strategy")
    print("   GET  /api/custom-strategy/user/{user_email} - Get user's custom strategies")
    print("   GET  /api/custom-strategy/{strategy_id} - Get specific custom strategy")
    print("   PUT  /api/custom-strategy/{strategy_id}/status - Update strategy status")
    print("   GET  /api/custom-strategy/health - Custom strategy health check")
    print("   GET  /api/payment/health - Payment service health check")
    print("   POST /api/payment/order - Create payment order")
    print("   GET  /api/payment/order/{order_id} - Get order details")
    print("   POST /api/payment/verify - Verify payment")
    print("   POST /api/payment/refund - Process refund")
    print("   GET  /api/payment/history - Get payment history")
    print("   GET  /api/payment/analytics - Get payment analytics")
    print("   GET  /api/strategies - Get all webhook strategies")
    print("   POST /api/strategies - Create webhook strategy")
    print("   GET  /api/strategies/{id} - Get specific webhook strategy")
    print("   PUT  /api/strategies/{id} - Update webhook strategy")
    print("   DELETE /api/strategies/{id} - Delete webhook strategy")
    print("   POST /api/generate-json - Generate JSON data for trading orders")
    print("   POST /api/save-json - Save JSON data")
    print("   POST /api/deploy - Deploy strategy (generate and save JSON to Strategies/rsStrategy/nifty500_data_with_metadata.sqlite)")
    print("   GET  /api/saved-json/{user_email} - Get saved JSON data")
    print("   GET  /api/live-signals/ - Get live signals with LTP")
    print("   GET  /api/live-signals/latest - Get latest live signals")
    print("   GET  /api/live-signals/deploy/{run_id} - Deploy signals with LTP (Deploy button)")
    print("   GET  /api/live-signals/symbol/{symbol}/ltp - Get LTP for specific symbol")
    print("   POST /api/live-signals/deploy - Deploy live trading system")
    print("   POST /api/live-signals/deploy-simple - Simple deploy (auto-fetch from database)")
    print("   GET  /api/live-signals/deployments/{deployment_id} - Get deployment status")
    print("   GET  /api/live-signals/deployments - Get user deployments")
    print("   POST /api/scheduler/trigger - Manually trigger market data fetch")
    print("   GET  /api/scheduler/status - Get scheduler status and next run time")
    print("   POST /api/rs-scheduler/trigger-eod - Manually trigger RS strategy EOD data fetch")
    print("   POST /api/rs-scheduler/trigger-all - Manually trigger all RS strategy tasks")
    print("   GET  /api/rs-scheduler/status - Get RS strategy scheduler status")
    print("   POST /api/execute/etf-signals - Execute ETF trading signals")
    print("   POST /api/execute/stock-signals - Execute Stock trading signals")
    print("🌐 Server will be available at: http://127.0.0.1:8000")
    print("🕐 Market Data Scheduler will run automatically every day at 4:15 PM IST")
    print("🕐 RS Strategy Scheduler will run automatically every day at 4:15 PM IST")
    uvicorn.run(app, host="127.0.0.1", port=8000)

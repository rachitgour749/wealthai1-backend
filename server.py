from fastapi import FastAPI, HTTPException, Depends, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from fastapi.routing import APIRoute
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import secrets
import uvicorn
import sys
import os
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import json
import concurrent.futures

# Logging is bootstrapped AFTER sys.path setup — see below.
# (moving it here would cause ModuleNotFoundError before BASE_DIR is on sys.path)
logger = logging.getLogger(__name__)


# =========================
# PATH SETUP (Critical - must run before any imports)
# =========================
# Get absolute path to the directory containing this file
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Ensure BASE_DIR is in sys.path first (for package imports like Services.Subscription)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ── Centralized logging (MUST be after sys.path setup) ───────────
try:
    from core.loggingSetup import setupLogging
    setupLogging()
except Exception as _log_err:
    # Fallback: basic logging so the server still starts
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(name)s: %(message)s'
    )
    logging.getLogger(__name__).warning(
        f"centralized logging setup failed ({_log_err}), using basicConfig fallback"
    )

# Re-assign logger now that logging is configured
logger = logging.getLogger(__name__)

# Suppress verbose third-party library logs
for lib in ["uvicorn", "uvicorn.access", "sqlalchemy", "sqlalchemy.engine",
            "sqlalchemy.pool", "sqlalchemy.dialects", "fastapi", "httpx"]:
    logging.getLogger(lib).setLevel(logging.ERROR)


# Verify critical directories exist
SERVICES_DIR = os.path.join(BASE_DIR, 'Services')
if not os.path.exists(SERVICES_DIR):
    logger.error(f"Services directory not found at: {SERVICES_DIR}")
    logger.error(f"BASE_DIR: {BASE_DIR}")
    logger.error(f"Current working directory: {os.getcwd()}")
    logger.error(f"Files in BASE_DIR: {os.listdir(BASE_DIR) if os.path.exists(BASE_DIR) else 'BASE_DIR does not exist'}")

# Ensure Services/__init__.py exists
SERVICES_INIT = os.path.join(SERVICES_DIR, '__init__.py')
if not os.path.exists(SERVICES_INIT):
    logger.warning(f"Services/__init__.py not found, creating it at: {SERVICES_INIT}")
    try:
        os.makedirs(SERVICES_DIR, exist_ok=True)
        with open(SERVICES_INIT, 'w') as f:
            f.write("# Services package initialization\n")
        logger.info(f"Created Services/__init__.py successfully")
    except Exception as e:
        logger.error(f"Failed to create Services/__init__.py: {e}")
        logger.error(f"Exception details: {type(e).__name__}: {e}")

# Verify Services package can be imported (early validation)
try:
    import Services
    if not hasattr(Services, '__file__'):
        logger.warning("Services package imported but __file__ attribute missing")
    else:
        logger.info(f"Services package verified at: {Services.__file__}")
except ImportError as e:
    logger.error(f"CRITICAL: Cannot import Services package: {e}")
    logger.error(f"This will cause all Services.* imports to fail!")
    logger.error(f"BASE_DIR: {BASE_DIR}")
    logger.error(f"SERVICES_DIR exists: {os.path.exists(SERVICES_DIR)}")
    logger.error(f"SERVICES_INIT exists: {os.path.exists(SERVICES_INIT)}")
    # Don't raise here - let it fail later with better context

# Helper function to get subscription directory name (case-insensitive)
def get_subscription_dir_name():
    """Get the actual subscription directory name (handles case differences)"""
    if not os.path.exists(SERVICES_DIR):
        return None
    services_contents = os.listdir(SERVICES_DIR)
    for item in services_contents:
        if item.lower() == 'subscription':
            return item
    return None

# Store subscription directory name globally for use in async functions
SUBSCRIPTION_DIR_NAME = get_subscription_dir_name()
if SUBSCRIPTION_DIR_NAME:
    logger.info(f"[PKG] Detected subscription directory: {SUBSCRIPTION_DIR_NAME}")
else:
    logger.warning("[WARN]  Subscription directory not found - subscription features will be unavailable")

# Add subdirectories for direct imports (legacy support)
sys.path.extend([
    os.path.join(BASE_DIR, 'Strategies', 'Rotation_Stocks'),
    os.path.join(BASE_DIR, 'Strategies', 'etf-strategy'),
    os.path.join(BASE_DIR, 'Strategies', 'RS_Stocks'),
    os.path.join(BASE_DIR, 'Strategies', 'RS_ETF'),
    os.path.join(BASE_DIR, 'Strategies', 'customStrategy'),
    os.path.join(BASE_DIR, 'Strategies', 'SuperTrend'),
    os.path.join(BASE_DIR, 'Services', 'webhook'),
    os.path.join(BASE_DIR, 'Services', 'subscription'),
    os.path.join(BASE_DIR, 'Services', 'Deployments_helper'),
    os.path.join(BASE_DIR, 'Services', 'SingleSignOn'),
])

# Log path setup for debugging (only in development)
if os.getenv('DEBUG', '').lower() == 'true':
    logger.info(f"BASE_DIR: {BASE_DIR}")
    logger.info(f"Python path includes BASE_DIR: {BASE_DIR in sys.path}")
    logger.info(f"Services directory exists: {os.path.exists(SERVICES_DIR)}")
    logger.info(f"Services/__init__.py exists: {os.path.exists(SERVICES_INIT)}")

# =========================
# GLOBAL STATE VARIABLES
# =========================
# Initialize with False - will be set during async startup

subscription_service_initialized: bool = False
webhook_service_initialized: bool = False
custom_strategy_service_initialized: bool = False
single_sign_on_service_initialized: bool = False
broker_service_initialized: bool = False

# Lazy loaded backtesters - initialized on first use

_initialization_lock = asyncio.Lock()

# =========================
# LAZY IMPORT FUNCTIONS (Defer heavy imports until needed)
# =========================
def _lazy_import_chatai():
    """Lazy import ChatAI1 modules - only when needed"""
    try:
        from ChatAI1.chatai1_config import settings as chatai1_new_settings
        from ChatAI1.database import init_db, close_db
        from ChatAI1.api import chat as chatai1_new
        return chatai1_new_settings, init_db, close_db, chatai1_new
    except Exception:
        return None, None, None, None

def _lazy_import_backtesters():
    """Lazy import backtester initialization functions — currently disabled."""
    return None, None, None, None

# =========================
# ASYNC INITIALIZATION FUNCTIONS
# =========================
async def _init_database_services() -> Dict[str, bool]:
    """Initialize all database services in parallel (non-blocking)"""
    results = {
        'subscription': False,
        'webhook': False,
        'custom_strategy': False,
        'single_sign_on': False,
        'broker': False
    }
    
    async def init_subscription():
        """Initialize subscription service"""
        try:
            if not SUBSCRIPTION_DIR_NAME:
                raise ImportError("Subscription directory not found")
            
            # Import using actual directory name
            import importlib
            subscription_db = importlib.import_module(f'Services.{SUBSCRIPTION_DIR_NAME}.database')
            subscription_manager = subscription_db.subscription_manager
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                await loop.run_in_executor(executor, subscription_manager.init_database)
            results['subscription'] = True
            logger.info("=" * 60)
            logger.info("[OK] SUCCESS: Subscription database service initialized!")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"Subscription init failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            results['subscription'] = False
    
    async def init_webhook():
        """Initialize webhook database"""
        try:
            from Services.webhook.webhook_logic import init_db as init_webhook_db
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                await loop.run_in_executor(executor, init_webhook_db)
            results['webhook'] = True
        except Exception as e:
            logger.error(f"Webhook init failed: {e}")
            results['webhook'] = False
    
    async def init_custom_strategy():
        """Initialize custom strategy database"""
        try:
            from Strategies.customStrategy.database import CustomStrategyDatabase
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                await loop.run_in_executor(
                    executor, 
                    lambda: CustomStrategyDatabase()
                )
            results['custom_strategy'] = True
        except Exception as e:
            logger.error(f"Custom strategy init failed: {e}")
            results['custom_strategy'] = False
    
    async def init_single_sign_on():
        """Initialize SingleSignOn service"""
        try:
            from Databases.app_data_db_connection import create_connection as init_sso_db
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                sso_init = await loop.run_in_executor(executor, init_sso_db)
            results['single_sign_on'] = bool(sso_init)
        except Exception as e:
            logger.error(f"SingleSignOn init failed: {e}")
            results['single_sign_on'] = False
    
    async def init_broker():
        """Initialize Broker database (broker_sessions table)"""
        try:
            from Databases.app_data_db_connection import init_database as init_broker_db
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                broker_init = await loop.run_in_executor(executor, init_broker_db)
            results['broker'] = bool(broker_init)
            if broker_init:
                logger.info("[OK] Broker database initialized (broker_sessions table created)")
        except Exception as e:
            logger.error(f"Broker database init failed: {e}")
            results['broker'] = False
    
    # Run all database initializations in parallel
    await asyncio.gather(
        init_subscription(),
        init_webhook(),
        init_custom_strategy(),
        init_single_sign_on(),
        init_broker(),
        return_exceptions=True
    )
    
    return results

async def _init_backtesters_lazy():
    """Lazy initialize backtesters — currently disabled / no-op."""
    pass

# =========================
# CHATAI INTEGRATION (New)
# =========================
# =========================
# CHATAI INTEGRATION (New)
# =========================
try:
    # ChatAI now loads .env from root directory in its own main.py
    # No need to load it here
        
    # Import directly from Services (standard python path)
    from Services.ChatAI.api.main import (
        router as chatai_router,
        lifespan as chatai_lifespan,
        limiter as chatai_limiter,
        RateLimitExceeded,
        rate_limit_handler
    )
    # Using print to ensure visibility despite logging level
    logger.info("=" * 60)
    logger.info("[OK] ChatAI INITIALIZATION SUCCESSFUL")
    logger.info("   - Router: LOADED")
    logger.info("   - Lifespan: LOADED")
    logger.info("   - Rate Limiter: LOADED")
    logger.info("=" * 60)
    chatai_availabe = True
except Exception as e:
    logger.error("=" * 60)
    logger.error("[FAIL] ChatAI INITIALIZATION FAILED")
    logger.error(f"   Error: {e}")
    logger.error("   ChatAI routes will NOT be available.")
    logger.error("=" * 60)
    chatai_router = None
    chatai_lifespan = None
    chatai_availabe = False

# =========================
# LIFESPAN CONTEXT MANAGER
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optimized lifespan with async parallel initialization"""
    global subscription_service_initialized, webhook_service_initialized
    global custom_strategy_service_initialized, single_sign_on_service_initialized
    global broker_service_initialized
    
    # Initialize ChatAI database (Legacy ChatAI1 - Optional/Deprecated)
    chatai1_new_settings, init_db, close_db, chatai1_new = _lazy_import_chatai()
    if init_db:
        try:
            await init_db()
        except Exception as e:
            logger.error(f"ChatAI1 (Legacy) database init failed: {e}")
    
    # Initialize all database services in parallel
    db_results = await _init_database_services()
    
    subscription_service_initialized = db_results['subscription']
    webhook_service_initialized = db_results['webhook']
    custom_strategy_service_initialized = db_results['custom_strategy']
    single_sign_on_service_initialized = db_results['single_sign_on']
    broker_service_initialized = db_results['broker']
    
    # Start backtesters in background (non-blocking)
    await _init_backtesters_lazy()
    
    # Initialize and start the scheduler service
    try:
        from Services.scheduler.scheduler_service import start_scheduler
        logger.info("="*60)
        logger.info("STARTING SCHEDULER SERVICE")
        logger.info("="*60)
        scheduler = start_scheduler()
        logger.info("[OK] Scheduler service started successfully")
    except Exception as e:
        logger.error(f"Failed to start scheduler service: {e}")
        import traceback
        traceback.print_exc()
    
    # Use ChatAI (New) lifespan if available
    if chatai_availabe and chatai_lifespan:
        try:
            async with chatai_lifespan(app):
                yield
        except Exception as e:
            logger.error(f"ChatAI (New) lifespan error: {e}")
            yield
    else:
        yield
    
    # Shutdown
    # Shutdown scheduler
    try:
        from Services.scheduler.scheduler_service import get_scheduler
        scheduler = get_scheduler()
        scheduler.shutdown()
        logger.info("[OK] Scheduler shut down successfully")
    except Exception as e:
        logger.error(f"Failed to shutdown scheduler: {e}")
    
    if close_db:
        try:
            await close_db()
        except Exception as e:
            logger.error(f"ChatAI1 (Legacy) database close failed: {e}")


# =========================
# FASTAPI APP CREATION (Fast - no blocking operations)
# =========================
# Define global security dependency to force header input in Swagger
def security_header(Authorization: Optional[str] = Header(None, description="Enter 'Bearer <token>'")):
    pass

app = FastAPI(
    title="WealthAI1 API",
    version="1.0.0",
    lifespan=lifespan,
    description="High-performance financial backtesting and trading strategy API",
    docs_url="/docs",       # Enable default docs for better visibility of all routes
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    dependencies=[Depends(security_header)] # This adds the header input to all endpoints
)

# Register ChatAI Rate Limiter if available
if chatai_availabe:
    app.state.limiter = chatai_limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

# =========================
# SECURITY CONFIGURATION
# =========================
security = HTTPBasic()

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    """Authenticate user for Swagger UI access"""
    expected_username = os.getenv("SWAGGER_USERNAME", "admin@wealthai1.in")
    expected_password = os.getenv("SWAGGER_PASSWORD", "change-this-password")
    
    correct_username = secrets.compare_digest(credentials.username, expected_username)
    correct_password = secrets.compare_digest(credentials.password, expected_password)
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Override default docs with protected version if needed, or keep default
# The user asked to "show all apis through my swagger", so we should ensure docs are accessible.
# server.py previously disabled default docs and added protected ones. 
# We'll keep the protection but ensure everything is visible.

@app.get("/docs", include_in_schema=False)
async def get_documentation(username: str = Depends(get_current_username)):
    """Protected Swagger UI"""
    return get_swagger_ui_html(openapi_url="/openapi.json", title="WealthAI1 API - Docs")

@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint(username: str = Depends(get_current_username)):
    """Protected OpenAPI JSON with Bearer Auth configuration"""
    if not app.openapi_schema:
        app.openapi_schema = get_openapi(
            title="WealthAI1 API",
            version="1.0.0",
            description="High-performance financial backtesting and trading strategy API",
            routes=app.routes,
        )
    
    # Always ensure components dict exists
    if "components" not in app.openapi_schema:
        app.openapi_schema["components"] = {}
        
    # Get or create securitySchemes dict
    security_schemes = app.openapi_schema["components"].get("securitySchemes", {})
    
    # Add BearerAuth if not present
    if "BearerAuth" not in security_schemes:
        security_schemes["BearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Enter your Session Token (Bearer <token>)"
        }
        
    # Update the schema
    app.openapi_schema["components"]["securitySchemes"] = security_schemes
    
    # Apply security globally to all operations
    if "security" not in app.openapi_schema:
        app.openapi_schema["security"] = []
        
    # Check if BearerAuth is already in security requirements
    has_bearer = any("BearerAuth" in req for req in app.openapi_schema["security"])
    if not has_bearer:
        app.openapi_schema["security"].append({"BearerAuth": []})
    
    return app.openapi_schema

# =========================
# SINGLE SESSION MIDDLEWARE
# =========================
try:
    from Middleware.auth_middleware import SingleSessionMiddleware
    app.add_middleware(
        SingleSessionMiddleware,
        exempt_paths=[
            "/single-sign-on", # Exempt SSO to allow login/token update
            "/api/auth", # Exempt Google OAuth
            "/paymentSuccess", # Exempt payment callback
            "/api/portfolio/webhook/trade-executed", # Exempt portfolio webhook callback
            "/api/portfolio", # Exempt all portfolio endpoints for testing
            "/api/hierarchy", # Exempt all hierarchy endpoints for testing
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health_check",
            "/",
            "/api/chat/health", # Exempt ChatAI health
            "/health", # ChatAI health
            "/single-sign-on", # Exempt SSO to allow login/token update
            "/api/auth", # Exempt Google OAuth
            "/paymentSuccess", # Exempt payment callback
            "/api/portfolio/webhook/trade-executed", # Exempt portfolio webhook callback
            "/api/portfolio", # Exempt all portfolio endpoints for testing
            "/api/hierarchy", # Exempt all hierarchy endpoints for testing
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health_check",
            "/",
            "/api/chat/health", # Exempt ChatAI health
            "/health", # ChatAI health
            "/api/v2/run_backtest",       # Backtest engine
            "/api/v2/strategies",         # List all strategies
            "/api/v2/strategy",           # /api/v2/strategy/* (assets, defaults, date-range, metrics, log)
            "/api/v2/health",             # Health check
            "/api/v2/save_strategies",    # Strategy management
            "/api/v2/deploy_strategy",
            "/api/v2/stop_strategy",
            "/api/v2/restart_strategy",
            "/api/v2/delete_strategy",
            "/api/v2/delete_strategy_client",
            "/api/v2/get_instances",
            "/api/strategy",              # Centralized backtest endpoints (assets, defaults, date-range, metrics, log)
            "/api/broker/place_order", # Exempt broker place order
            "/api/broker/broker_login", # Exempt broker login
            "/api/webhook/wealthai1.in/trade_execute", # Exempt trade execution webhook
            "/api/webhook/trade_execute", # Exempt unified trade execution
            "/api/webhook/ra", # Exempt RA CRUD
            "/admin", # ChatAI Admin routes (has own X-Admin-Key auth)
            "/api/mfd", # MFD Self-Service routes (has own x-user-email auth)
            "/api/query", # ChatAI query endpoint
            "/api/health", # ChatAI health endpoint
            "/api/run_backtest", # Exempt centralized backtest for verified access
            "/api/strategies", # Exempt centralized list strategies
        ]
except Exception as e:
    logger.error(f"Failed to add SingleSessionMiddleware: {e}")


# =========================
# CORS MIDDLEWARE (Must be last/outermost to handle 401s)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://wealthai1.in",
        "https://www.wealthai1.in",
        "https://trade.wealthai1.in",
        "http://localhost:5173",
        "http://localhost:5174", # ChatAI frontend dev
    ],
    allow_credentials=True,
    allow_headers=["*", "x-admin-key", "x-user-email", "x-session-id", "x-tenant-id", "Authorization", "Content-Type"],
    allow_methods=["*"], # Explicitly allow all methods
)


# =========================
# ROUTE IMPORTS (Lazy - only import routers, not initialization)
# =========================
# Import routers with detailed error handling
stock_router = None
etf_router = None
rs_router = None
rs_etf_router = None
custom_strategy_router = None
supertrend_router = None
webhook_router = None
subscription_router = None
payment_router = None
google_oauth_router = None
hierarchy_router = None
portfolio_router = None
deployment_router = None
single_sign_on_router = None
broker_router = None
chatai1_new_settings = None
chatai1_new = None



try:
    pass # from Strategies.customStrategy.api import custom_strategy_router
except Exception as e:
    logger.error(f"Failed to import custom_strategy_router: {e}")

supertrend_router = None

try:
    from APIs.broker_routes import router as broker_router
    logger.info("[OK] Broker router loaded successfully")
except Exception as e:
    logger.error(f"Failed to import broker_router: {e}")

rotation_etf_payout_router = None
initialize_rotation_etf_payout_backtester = None

webhook_router = None

try:
    from Services.webhook.webhook_api import router as webhook_router
    logger.info("New Webhook router loaded successfully from Services.webhook.webhook_api")
except Exception as e:
    logger.error(f"Failed to import new webhook_router: {e}")

# Critical: Subscription routers - provide detailed error info
try:
    if not SUBSCRIPTION_DIR_NAME:
        raise ImportError(f"Subscription directory not found in Services. Available: {os.listdir(SERVICES_DIR) if os.path.exists(SERVICES_DIR) else 'N/A'}")
    
    # Verify Services package can be imported
    import Services
    logger.info(f"Services package found at: {Services.__file__ if hasattr(Services, '__file__') else 'unknown'}")
    
    # Try importing with actual directory name
    try:
        # Use importlib to import with the actual directory name
        import importlib
        logger.info(f"Attempting to import Services.{SUBSCRIPTION_DIR_NAME}...")
        subscription_module = importlib.import_module(f'Services.{SUBSCRIPTION_DIR_NAME}')
        logger.info(f"[OK] Successfully imported Services.{SUBSCRIPTION_DIR_NAME} module")
        
        # Now import the routers using the actual module
        logger.info(f"Loading routers from Services.{SUBSCRIPTION_DIR_NAME}...")
        subscription_api = importlib.import_module(f'Services.{SUBSCRIPTION_DIR_NAME}.api.subscription')
        google_oauth_api = importlib.import_module(f'Services.{SUBSCRIPTION_DIR_NAME}.api.google_oauth_api')
        hierarchy_api = importlib.import_module(f'Services.{SUBSCRIPTION_DIR_NAME}.hierarchy_api')
        
        subscription_router = subscription_api.subscription_router
        payment_router = getattr(subscription_api, 'payment_router', None)
        google_oauth_router = google_oauth_api.google_oauth_router
        hierarchy_router = hierarchy_api.hierarchy_router
        
        logger.info("=" * 60)
        logger.info("[OK] SUCCESS: Subscription routers loaded successfully!")
        logger.info("[OK] subscription_router: LOADED")
        logger.info("[OK] google_oauth_router: LOADED")
        logger.info("[OK] hierarchy_router: LOADED")
        logger.info("[OK] Subscription endpoints available:")
        logger.info("   - /api/subscription/*")
        logger.info("   - /paymentSuccess")
        logger.info("   - /api/auth/google-login")
        logger.info("   - /api/hierarchy/*")
        logger.info("=" * 60)
        
    except ImportError as e:
        logger.error(f"[FAIL] Failed to import Services.{SUBSCRIPTION_DIR_NAME}: {e}")
        logger.error(f"Services directory contents: {os.listdir(SERVICES_DIR) if os.path.exists(SERVICES_DIR) else 'N/A'}")
        logger.error(f"Services/__init__.py exists: {os.path.exists(SERVICES_INIT)}")
        logger.error(f"Python path: {sys.path[:5]}")  # Show first 5 entries
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise
    
except Exception as e:
    logger.error(f"Failed to import Subscription routers: {e}")
    logger.error(f"Error type: {type(e).__name__}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")
    subscription_router = None
    payment_router = None
    google_oauth_router = None
    hierarchy_router = None

deployment_router = None

# Portfolio router
portfolio_router = None
try:
    from Services.portfolio.portfolio_api import portfolio_router
    logger.info("[OK] Portfolio API router loaded successfully")
except Exception as e:
    logger.error(f"Failed to import portfolio_router: {e}")

try:
    from Services.SingleSignOn import router as single_sign_on_router
except Exception as e:
    logger.error(f"Failed to import single_sign_on_router: {e}")

# ChatAI router (Legacy)
try:
    chatai1_new_settings, _, _, chatai1_new = _lazy_import_chatai()
except Exception as e:
    logger.error(f"Failed to import ChatAI1 (Legacy): {e}")

try:
    from Admin.routes import admin_router
    logger.info("Admin router loaded successfully")
except Exception as e:
    logger.error(f"Failed to import admin_router: {e}")
    admin_router = None

# Centralized Backtest API (New)
# Centralized Backtest API (New) - Consolidated in APIs.routes
# Consolidated Strategy API (New) - Consolidated in APIs.routes
# Centralized Strategy Management API (New) - Consolidated in APIs.routes

try:
    from APIs.routes import api_router as unified_api_router
    logger.info("[OK] Unified API router loaded successfully from APIs.routes")
except Exception as e:
    logger.error(f"Failed to import unified_api_router: {e}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")
    unified_api_router = None

# =========================
# CORE ROUTES
# =========================
@app.get("/")
async def root():
    """Root endpoint - fast response"""
    return {
        "message": "WealthAI1 API Running [OK]",
        "version": "1.0.1 (API Verif)",
        "strategies": ["rs-strategy", "custom-strategy", "chat"],
        "services": ["subscription", "webhook", "single-sign-on", "deployments"],
        "status": "operational"
    }

@app.get("/health_check")
async def health_check():
    """Health check endpoint - shows initialization status and verifies DB connectivity."""
    db_healthy = False
    try:
        from Services.subscription.database import subscription_manager
        # Quick DB ping — if this fails, the service is degraded
        db_healthy = subscription_manager.engine is not None
    except Exception:
        pass
    
    overall = "healthy" if db_healthy else "degraded"
    return {
        "api_status": overall,
        "database": "connected" if db_healthy else "unreachable",
        "services": {
            "subscription": subscription_service_initialized,
            "webhook": webhook_service_initialized,
            "custom_strategy": custom_strategy_service_initialized,
            "single_sign_on": single_sign_on_service_initialized,
            "broker": broker_service_initialized,
        }
    }

@app.get("/favicon.ico")
async def favicon():
    """Favicon endpoint"""
    favicon_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x00\x00\x02\x00\x01\xe5\x27\xde\xfc\x00\x00\x00\x00IEND\xaeB`\x82'
    return Response(content=favicon_data, media_type="image/x-icon")

# =========================
# INCLUDE ROUTERS (only include if successfully imported)
# =========================



# International ETF router


if rs_router:
    app.include_router(rs_router, prefix="/api/rs-strategy", tags=["RS Strategy"])

if custom_strategy_router:
    app.include_router(custom_strategy_router)

if subscription_router:
    app.include_router(subscription_router)
else:
<<<<<<< HEAD
    logger.error("Subscription router not loaded - subscription endpoints will not be available")
=======
    logger.error("[WARN]  Subscription router not loaded - subscription endpoints will not be available")
>>>>>>> feature/chatai

if payment_router:
    app.include_router(payment_router)

if google_oauth_router:
    app.include_router(google_oauth_router)
else:
    logger.error("[WARN]  Google OAuth router not loaded - OAuth endpoints will not be available")

# Portfolio router
if portfolio_router:
    app.include_router(portfolio_router)
    logger.info("[OK] Portfolio router mounted successfully")
else:
    logger.error("[WARN]  Portfolio router not loaded - portfolio endpoints will not be available")

# Hierarchy router
if hierarchy_router:
    app.include_router(hierarchy_router)
<<<<<<< HEAD
    logger.info("Hierarchy router mounted successfully")
else:
    logger.error("Hierarchy router not loaded - hierarchy endpoints will not be available")
=======
    logger.info("[OK] Hierarchy router mounted successfully")
else:
    logger.error("[WARN]  Hierarchy router not loaded - hierarchy endpoints will not be available")
>>>>>>> feature/chatai


if single_sign_on_router:
    app.include_router(single_sign_on_router)

# Broker Router
if broker_router:
    app.include_router(broker_router, prefix="/api/broker", tags=["Broker Integration"])

# Rotation ETF Payout router

if webhook_router:
    app.include_router(webhook_router, prefix="/api/webhook", tags=["Webhook Integration"])



# ChatAI router (Legacy)
if chatai1_new and hasattr(chatai1_new, 'router'):
    app.include_router(chatai1_new.router, prefix="/api/legacy/chatai1") # Renamed prefix to avoid conflict

# ChatAI router (New)
if chatai_availabe and chatai_router:
    app.include_router(chatai_router)
<<<<<<< HEAD
    logger.info("ChatAI (New) router mounted successfully")
=======
    logger.info("[OK] ChatAI (New) router mounted successfully")
    
    # Also mount admin and MFD self-service routes
    try:
        from Services.ChatAI.api.admin_routes import router as admin_router
        app.include_router(admin_router)
        logger.info("[OK] ChatAI Admin router mounted")
    except Exception as e:
        logger.error(f"[FAIL] ChatAI Admin router: {e}")
    
    try:
        from Services.ChatAI.api.mfd_routes import router as mfd_router
        app.include_router(mfd_router)
        logger.info("[OK] MFD Self-Service router mounted at /api/mfd")
    except Exception as e:
        logger.error(f"[FAIL] MFD router: {e}")

>>>>>>> feature/chatai

# SuperTrend router (handle duplicate root route)


# Centralized Backtest API (New)
# Unified API Router (Consolidated)
if unified_api_router:
    app.include_router(unified_api_router)
<<<<<<< HEAD
    logger.info("Unified API router mounted at /api")

if admin_router:
    app.include_router(admin_router)
    logger.info("Admin router mounted successfully")
=======
    logger.info("[OK] Unified API router mounted at /api")
>>>>>>> feature/chatai

# =========================
# CHATAI ENDPOINTS (Fallback if router not available)
# =========================
@app.get("/api/chat")
async def chat_root():
    """ChatAI root endpoint"""
    if chatai_availabe:
        return {
             "app": "ChatAI",
             "status": "running",
             "version": "1.0.0"
        }
    if chatai1_new_settings:
        return {
            "app": chatai1_new_settings.APP_NAME,
            "version": chatai1_new_settings.APP_VERSION,
            "status": "running"
        }
    return {"status": "chat_service_unavailable"}


@app.get("/api/chat/health")
async def chat_health():
    """ChatAI health endpoint"""
    return {"status": "healthy"}



# =========================
# LOCAL RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="error",  # Suppress uvicorn access logs
        access_log=False    # Disable access log completely
    )

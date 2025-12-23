from fastapi import FastAPI, HTTPException, Depends, status
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
from typing import Dict, Any
import concurrent.futures

# Configure concise structured logging
logging.basicConfig(
    level=logging.ERROR,  # Only errors by default (quiet startup)
    format='[%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress all verbose logs from third-party libraries
for lib in ["uvicorn", "uvicorn.access", "sqlalchemy", "sqlalchemy.engine", 
            "sqlalchemy.pool", "sqlalchemy.dialects", "fastapi", "httpx"]:
    logging.getLogger(lib).setLevel(logging.ERROR)

# =========================
# PATH SETUP (Critical - must run before any imports)
# =========================
# Get absolute path to the directory containing this file
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Ensure BASE_DIR is in sys.path first (for package imports like Services.Subscription)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

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
    logger.info(f"📦 Detected subscription directory: {SUBSCRIPTION_DIR_NAME}")
else:
    logger.warning("⚠️  Subscription directory not found - subscription features will be unavailable")

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
stock_backtester_initialized: bool = False
etf_backtester_initialized: bool = False
subscription_service_initialized: bool = False
webhook_service_initialized: bool = False
custom_strategy_service_initialized: bool = False
single_sign_on_service_initialized: bool = False

# Lazy loaded backtesters - initialized on first use
_stock_backtester = None
_etf_backtester = None
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
    """Lazy import backtester initialization functions"""
    try:
        from Strategies.Rotation_Stocks.api.stock_routes import (
            initialize_stock_backtester, cleanup_stock_backtester
        )
        from Strategies.Rotation_ETF.api.etf_routes import (
            initialize_etf_backtester, cleanup_etf_backtester
        )
        return initialize_stock_backtester, initialize_etf_backtester, \
               cleanup_stock_backtester, cleanup_etf_backtester
    except Exception:
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
        'single_sign_on': False
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
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                await loop.run_in_executor(executor, subscription_manager.init_database)
            results['subscription'] = True
            logger.info("=" * 60)
            logger.info("✅ SUCCESS: Subscription database service initialized!")
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
            loop = asyncio.get_event_loop()
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
            loop = asyncio.get_event_loop()
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
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                sso_init = await loop.run_in_executor(executor, init_sso_db)
            results['single_sign_on'] = bool(sso_init)
        except Exception as e:
            logger.error(f"SingleSignOn init failed: {e}")
            results['single_sign_on'] = False
    
    # Run all database initializations in parallel
    await asyncio.gather(
        init_subscription(),
        init_webhook(),
        init_custom_strategy(),
        init_single_sign_on(),
        return_exceptions=True
    )
    
    return results

async def _init_backtesters_lazy():
    """Lazy initialize backtesters in background (non-blocking for startup)"""
    global stock_backtester_initialized, etf_backtester_initialized
    
    init_stock, init_etf, _, _ = _lazy_import_backtesters()
    
    if not init_stock or not init_etf:
        return
    
    async def init_stock_backtester():
        """Initialize stock backtester in background"""
        try:
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(executor, init_stock)
            global stock_backtester_initialized
            stock_backtester_initialized = bool(result)
        except Exception as e:
            logger.error(f"Stock backtester init failed: {e}")
            stock_backtester_initialized = False
    
    async def init_etf_backtester():
        """Initialize ETF backtester in background"""
        try:
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(executor, init_etf)
            global etf_backtester_initialized
            etf_backtester_initialized = bool(result)
        except Exception as e:
            logger.error(f"ETF backtester init failed: {e}")
            etf_backtester_initialized = False
    
    # Start backtesters in background (non-blocking)
    asyncio.create_task(init_stock_backtester())
    asyncio.create_task(init_etf_backtester())

# =========================
# LIFESPAN CONTEXT MANAGER
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optimized lifespan with async parallel initialization"""
    global subscription_service_initialized, webhook_service_initialized
    global custom_strategy_service_initialized, single_sign_on_service_initialized
    
    # Initialize ChatAI database (required for chat routes)
    chatai1_new_settings, init_db, close_db, chatai1_new = _lazy_import_chatai()
    if init_db:
        try:
            await init_db()
        except Exception as e:
            logger.error(f"ChatAI database init failed: {e}")
    
    # Initialize all database services in parallel
    db_results = await _init_database_services()
    
    subscription_service_initialized = db_results['subscription']
    webhook_service_initialized = db_results['webhook']
    custom_strategy_service_initialized = db_results['custom_strategy']
    single_sign_on_service_initialized = db_results['single_sign_on']
    
    # Start backtesters in background (non-blocking)
    await _init_backtesters_lazy()
    
    yield
    
    # Shutdown
    if close_db:
        try:
            await close_db()
        except Exception as e:
            logger.error(f"ChatAI database close failed: {e}")

# =========================
# FASTAPI APP CREATION (Fast - no blocking operations)
# =========================
app = FastAPI(
    title="WealthAI1 API",
    version="1.0.0",
    lifespan=lifespan,
    description="High-performance financial backtesting and trading strategy API",
    docs_url=None,       # Disable default docs
    redoc_url=None,      # Disable default redoc
    openapi_url=None     # Disable default openapi.json
)

# =========================
# SECURITY CONFIGURATION
# =========================
security = HTTPBasic()

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    """Authenticate user for Swagger UI access"""
    # Credentials provided by user
    correct_username = secrets.compare_digest(credentials.username, "wealthwisers@fintech.gmail.com")
    correct_password = secrets.compare_digest(credentials.password, "WW@fintech.2025")
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.get("/docs", include_in_schema=False)
async def get_documentation(username: str = Depends(get_current_username)):
    """Protected Swagger UI"""
    return get_swagger_ui_html(openapi_url="/openapi.json", title="WealthAI1 API - Docs")

@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint(username: str = Depends(get_current_username)):
    """Protected OpenAPI JSON with Bearer Auth configuration"""
    openapi_schema = get_openapi(title="WealthAI1 API", version="1.0.0", routes=app.routes)
    
    # Configure Bearer Authentication Scheme
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
        
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "Google-Token",
            "description": "Enter your Google OAuth Token"
        }
    }
    
    # Apply security globally to all operations
    openapi_schema["security"] = [{"BearerAuth": []}]
    
    return openapi_schema

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
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health_check",
            "/"
        ]
    )
    logger.info("✅ SingleSessionMiddleware added")
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
    ],
    allow_credentials=True,
    allow_headers=["*"],
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
deployment_router = None
single_sign_on_router = None
chatai1_new_settings = None
chatai1_new = None

try:
    from Strategies.Rotation_Stocks.api.stock_routes import stock_router
except Exception as e:
    logger.error(f"Failed to import stock_router: {e}")

try:
    from Strategies.Rotation_ETF.api.etf_routes import etf_router
except Exception as e:
    logger.error(f"Failed to import etf_router: {e}")

try:
    from Strategies.RS_Stocks.api import router as rs_router
except Exception as e:
    logger.error(f"Failed to import rs_router: {e}")

try:
    from Strategies.RS_ETF.api import router as rs_etf_router
except Exception as e:
    logger.error(f"Failed to import rs_etf_router: {e}")

try:
    from Strategies.customStrategy.api import custom_strategy_router
except Exception as e:
    logger.error(f"Failed to import custom_strategy_router: {e}")

try:
    from Strategies.SuperTrend.api.routes import router as supertrend_router
except Exception as e:
    logger.error(f"Failed to import supertrend_router: {e}")

try:
    from Strategies.CustomStrategies.Rotation_ETF_Payout.api_routes import (
        rotation_etf_payout_router,
        initialize_rotation_etf_payout_backtester
    )
except Exception as e:
    logger.error(f"Failed to import rotation_etf_payout_router: {e}")
    rotation_etf_payout_router = None
    initialize_rotation_etf_payout_backtester = None

try:
    from Services.webhook.webhook_api import router as webhook_router
except Exception as e:
    logger.error(f"Failed to import webhook_router: {e}")

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
        logger.info(f"✅ Successfully imported Services.{SUBSCRIPTION_DIR_NAME} module")
        
        # Now import the routers using the actual module
        logger.info(f"Loading routers from Services.{SUBSCRIPTION_DIR_NAME}...")
        subscription_api = importlib.import_module(f'Services.{SUBSCRIPTION_DIR_NAME}.api.subscription')
        google_oauth_api = importlib.import_module(f'Services.{SUBSCRIPTION_DIR_NAME}.api.google_oauth_api')
        
        subscription_router = subscription_api.subscription_router
        payment_router = getattr(subscription_api, 'payment_router', None)
        google_oauth_router = google_oauth_api.google_oauth_router
        
        logger.info("=" * 60)
        logger.info("✅ SUCCESS: Subscription routers loaded successfully!")
        logger.info("✅ subscription_router: LOADED")
        logger.info("✅ google_oauth_router: LOADED")
        logger.info("✅ Subscription endpoints available:")
        logger.info("   - /api/subscription/*")
        logger.info("   - /paymentSuccess")
        logger.info("   - /api/auth/google-login")
        logger.info("=" * 60)
        
    except ImportError as e:
        logger.error(f"❌ Failed to import Services.{SUBSCRIPTION_DIR_NAME}: {e}")
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

try:
    from Services.Deployments_helper.deployment_helper import deployment_router
except Exception as e:
    logger.error(f"Failed to import deployment_router: {e}")

try:
    from Services.SingleSignOn import router as single_sign_on_router
except Exception as e:
    logger.error(f"Failed to import single_sign_on_router: {e}")

# ChatAI router (lazy import)
try:
    chatai1_new_settings, _, _, chatai1_new = _lazy_import_chatai()
except Exception as e:
    logger.error(f"Failed to import ChatAI: {e}")

# =========================
# CORE ROUTES
# =========================
@app.get("/")
async def root():
    """Root endpoint - fast response"""
    return {
        "message": "WealthAI1 API Running ✅",
        "version": "1.0.0",
        "strategies": ["stock", "etf", "rs-strategy", "custom-strategy", "chat"],
        "services": ["subscription", "webhook", "single-sign-on", "deployments"],
        "status": "operational"
    }

@app.get("/health_check")
async def health_check():
    """Health check endpoint - shows initialization status"""
    return {
        "api_status": "healthy",
        "stock_backtester": "ready" if stock_backtester_initialized else "initializing",
        "etf_backtester": "ready" if etf_backtester_initialized else "initializing",
        "services": {
            "subscription": subscription_service_initialized,
            "webhook": webhook_service_initialized,
            "custom_strategy": custom_strategy_service_initialized,
            "single_sign_on": single_sign_on_service_initialized,
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
if stock_router:
    app.include_router(stock_router)
if etf_router:
    app.include_router(etf_router)
if rs_router:
    app.include_router(rs_router, prefix="/api/rs-strategy", tags=["RS Strategy"])
if rs_etf_router:
    app.include_router(rs_etf_router, prefix="/api/rs-etf-strategy", tags=["RS ETF Strategy"])
if custom_strategy_router:
    app.include_router(custom_strategy_router)
if webhook_router:
    app.include_router(webhook_router)
if subscription_router:
    app.include_router(subscription_router)
else:
    logger.error("⚠️  Subscription router not loaded - subscription endpoints will not be available")

if payment_router:
    app.include_router(payment_router)

if google_oauth_router:
    app.include_router(google_oauth_router)
else:
    logger.error("⚠️  Google OAuth router not loaded - OAuth endpoints will not be available")
if deployment_router:
    app.include_router(deployment_router)
if single_sign_on_router:
    app.include_router(single_sign_on_router)

# Rotation ETF Payout router
if rotation_etf_payout_router:
    app.include_router(rotation_etf_payout_router)
    # Initialize the backtester
    if initialize_rotation_etf_payout_backtester:
        try:
            initialize_rotation_etf_payout_backtester()
            logger.info("✅ Rotation ETF Payout backtester initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Rotation ETF Payout backtester: {e}")

# ChatAI router (only if available)
if chatai1_new and hasattr(chatai1_new, 'router'):
    app.include_router(chatai1_new.router, prefix="/api")

# SuperTrend router (handle duplicate root route)
try:
    supertrend_router.routes = [
        route for route in supertrend_router.routes
        if not (isinstance(route, APIRoute) and route.path == "/")
    ]
    app.include_router(supertrend_router, tags=["SuperTrend"])
except Exception as e:
    logger.error(f"SuperTrend router failed: {e}")

# =========================
# CHATAI ENDPOINTS (Fallback if router not available)
# =========================
@app.get("/api/chat")
async def chat_root():
    """ChatAI root endpoint"""
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
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="error",  # Suppress uvicorn access logs
        access_log=False    # Disable access log completely
    )

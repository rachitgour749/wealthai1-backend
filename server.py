from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.routing import APIRoute
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
# PATH SETUP (Minimal - only for imports)
# =========================
BASE_DIR = os.path.dirname(__file__)
sys.path.extend([
    os.path.join(BASE_DIR, 'Strategies', 'Rotation_Stocks'),
    os.path.join(BASE_DIR, 'Strategies', 'etf-strategy'),
    os.path.join(BASE_DIR, 'Strategies', 'RS_Stocks'),
    os.path.join(BASE_DIR, 'Strategies', 'RS_ETF'),
    os.path.join(BASE_DIR, 'Strategies', 'customStrategy'),
    os.path.join(BASE_DIR, 'Strategies', 'SuperTrend'),
    os.path.join(BASE_DIR, 'Services', 'webhook'),
    os.path.join(BASE_DIR, 'Services', 'Subscription'),
    os.path.join(BASE_DIR, 'Services', 'Deployments_helper'),
    os.path.join(BASE_DIR, 'Services', 'SingleSignOn'),
    BASE_DIR
])

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
            from Services.Subscription.database import subscription_manager
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                await loop.run_in_executor(executor, subscription_manager.init_database)
            results['subscription'] = True
        except Exception as e:
            logger.error(f"Subscription init failed: {e}")
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
    description="High-performance financial backtesting and trading strategy API"
)

# =========================
# CORS MIDDLEWARE
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
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ROUTE IMPORTS (Lazy - only import routers, not initialization)
# =========================
try:
    from Strategies.Rotation_Stocks.api.stock_routes import stock_router
    from Strategies.Rotation_ETF.api.etf_routes import etf_router
    from Strategies.RS_Stocks.api import router as rs_router
    from Strategies.RS_ETF.api import router as rs_etf_router
    from Strategies.customStrategy.api import custom_strategy_router
    from Strategies.SuperTrend.api.routes import router as supertrend_router
    from Services.webhook.webhook_api import router as webhook_router
    from Services.Subscription.api.subscription import subscription_router
    from Services.Subscription.api.google_oauth_api import google_oauth_router
    from Services.Deployments_helper.deployment_helper import deployment_router
    from Services.SingleSignOn import router as single_sign_on_router
    
    # ChatAI router (lazy import)
    chatai1_new_settings, _, _, chatai1_new = _lazy_import_chatai()
except Exception as e:
    logger.error(f"Router import failed: {e}")
    raise

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
# INCLUDE ROUTERS
# =========================
app.include_router(stock_router)
app.include_router(etf_router)
app.include_router(rs_router, prefix="/api/rs-strategy", tags=["RS Strategy"])
app.include_router(rs_etf_router, prefix="/api/rs-etf-strategy", tags=["RS ETF Strategy"])
app.include_router(custom_strategy_router)
app.include_router(webhook_router)
app.include_router(subscription_router)
app.include_router(google_oauth_router)
app.include_router(deployment_router)
app.include_router(single_sign_on_router)

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

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
from contextlib import asynccontextmanager

from contextlib import asynccontextmanager
from ChatAI1.chatai1_config import settings as chatai1_new_settings
from ChatAI1.database import init_db, close_db
from ChatAI1.api import chat as chatai1_new

logger = logging.getLogger(__name__)

# Add the strategy directories to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'Strategies', 'Rotation_Stocks'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Strategies', 'etf-strategy'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Strategies', 'RS_Stocks'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Strategies', 'RS_ETF'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Strategies', 'customStrategy'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Strategies', 'SuperTrend'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'chatAI'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Services', 'cronjob'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Services', 'webhook'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Services', 'subscription'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Services', 'execution'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Services', 'Deployments_helper'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Services', 'SingleSignOn'))
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# Import the separate API modules
from Strategies.Rotation_Stocks.api.stock_routes import stock_router, initialize_stock_backtester, cleanup_stock_backtester
from Services.webhook.webhook_api import router as webhook_router
from Services.webhook.webhook_logic import init_db as init_webhook_db
from Services.subscription.api import subscription_router
from Services.subscription.google_oauth_api import google_oauth_router
from Services.subscription.database import subscription_manager
from Services.Deployments_helper.deployment_helper import deployment_router
from Services.SingleSignOn import router as single_sign_on_router


# Import RS strategy router (before ETF to avoid import conflicts)
from Strategies.RS_Stocks.api import router as rs_router
# Import RS ETF strategy router
from Strategies.RS_ETF.api import router as rs_etf_router

# Import ETF strategy after RS strategy to avoid conflicts
from Strategies.Rotation_ETF.api.etf_routes import etf_router, initialize_etf_backtester, cleanup_etf_backtester

# Import custom strategy router
from Strategies.customStrategy.api import custom_strategy_router
from Strategies.customStrategy.database import CustomStrategyDatabase

# Import SuperTrend strategy router
from Strategies.SuperTrend.api.routes import router as supertrend_router
from Strategies.SuperTrend.api.database import init_database as init_supertrend_database



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup
    logger.info(f"Starting {chatai1_new_settings.APP_NAME} v{chatai1_new_settings.APP_VERSION}")
    logger.info(f"Router model: {chatai1_new_settings.ROUTER_MODEL_NAME}")
    logger.info(f"Answer model: {chatai1_new_settings.ANSWER_MODEL_NAME}")
    logger.info(f"RAG service: {chatai1_new_settings.RAG_SERVICE_BASE_URL}")

    # Initialize database
    await init_db()
    logger.info("Database initialized")

    yield

    # Shutdown
    logger.info(f"Shutting down {chatai1_new_settings.APP_NAME}")
    await close_db()


# Initialize backtesters (before app creation to ensure variables are defined)
try:
    stock_backtester_initialized = initialize_stock_backtester()
    print("[SUCCESS] Stock backtester initialized successfully")
except Exception as e:
    stock_backtester_initialized = False
    print(f"[ERROR] Failed to initialize stock backtester: {e}")

try:
    etf_backtester_initialized = initialize_etf_backtester()
    print("[SUCCESS] ETF backtester initialized successfully")
except Exception as e:
    etf_backtester_initialized = False
    print(f"[ERROR] Failed to initialize ETF backtester: {e}")

# Create main FastAPI app
app = FastAPI(title="WealthAI1 API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "http://127.0.0.1:3000",  # Local development alternative
        "https://your-cloudfront-domain.cloudfront.net",  # Your CloudFront domain
        "https://wealthai1.in",  # Your custom domain
        "https://www.wealthai1.in",  # www subdomain
        "https://trade.wealthai1.in",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

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
    custom_strategy_db = CustomStrategyDatabase()  # Uses PostgreSQL, db_path parameter ignored
    custom_strategy_service_initialized = True
    print("[SUCCESS] Custom strategy database initialized successfully")
except Exception as e:
    custom_strategy_service_initialized = False
    print(f"[ERROR] Failed to initialize custom strategy database: {e}")

# Initialize SingleSignOn service
try:
    from Databases.app_data_db_connection import create_connection as init_sso_db
    sso_db_initialized = init_sso_db()
    single_sign_on_service_initialized = sso_db_initialized
    if sso_db_initialized:
        print("[SUCCESS] SingleSignOn service initialized successfully")
    else:
        print("[ERROR] Failed to initialize SingleSignOn database connection")
except Exception as e:
    single_sign_on_service_initialized = False
    print(f"[ERROR] Failed to initialize SingleSignOn service: {e}")


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Unified Rotation Backtester API", 
        "strategies": ["stock", "etf", "rs-strategy", "custom-strategy", "chat"],
        "services": ["subscription", "webhook", "single-sign-on", "deployments"]
    }

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
            "subscription_service_initialized": subscription_service_initialized,
            "webhook_service_initialized": webhook_service_initialized,
            "single_sign_on_service_initialized": single_sign_on_service_initialized,
            "stock_database_available": stock_backtester_initialized,
            "etf_database_available": etf_backtester_initialized,
            "rs_strategy_database_available": True,
            "custom_strategy_database_available": custom_strategy_service_initialized,
            "subscription_database_available": subscription_service_initialized,
            "webhook_database_available": True,
            "stock_count": 0,
            "etf_count": 0,
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
            "subscription_service_initialized": subscription_service_initialized,
            "webhook_service_initialized": webhook_service_initialized,
            "single_sign_on_service_initialized": single_sign_on_service_initialized,
        }

@app.get("/favicon.ico")
async def favicon():
    """Return a simple favicon to prevent 404 errors"""
    # Return a minimal 1x1 transparent PNG
    favicon_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x00\x00\x02\x00\x01\xe5\x27\xde\xfc\x00\x00\x00\x00IEND\xaeB`\x82'
    return Response(content=favicon_data, media_type="image/x-icon")


# Include the routers in the main app
app.include_router(stock_router)
app.include_router(etf_router)
app.include_router(rs_router, prefix="/api/rs-strategy", tags=["RS Strategy"])
app.include_router(rs_etf_router, prefix="/api/rs-etf-strategy", tags=["RS ETF Strategy"])
app.include_router(custom_strategy_router)
app.include_router(chatai1_new.router, prefix="/api")
app.include_router(webhook_router)
app.include_router(subscription_router)
app.include_router(google_oauth_router)
app.include_router(deployment_router)
app.include_router(single_sign_on_router)

# Remove SuperTrend root route to avoid duplicate "/" definition
supertrend_router.routes = [
    route for route in supertrend_router.routes
    if not (isinstance(route, APIRoute) and route.path == "/")
]
app.include_router(supertrend_router, tags=["SuperTrend"])


# ============================================================================
# EXECUTION API ENDPOINTS
# ============================================================================


@app.get("/api/chat")
async def root():
    """Root endpoint"""
    return {
        "app": chatai1_new_settings.APP_NAME,
        "version": chatai1_new_settings.APP_VERSION,
        "status": "running"
    }

@app.get("/api/chat/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.routing import APIRoute
import uvicorn
import sys
import os
import logging
from contextlib import asynccontextmanager

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(__file__)

sys.path.append(os.path.join(BASE_DIR, 'Strategies', 'Rotation_Stocks'))
sys.path.append(os.path.join(BASE_DIR, 'Strategies', 'etf-strategy'))
sys.path.append(os.path.join(BASE_DIR, 'Strategies', 'RS_Stocks'))
sys.path.append(os.path.join(BASE_DIR, 'Strategies', 'RS_ETF'))
sys.path.append(os.path.join(BASE_DIR, 'Strategies', 'customStrategy'))
sys.path.append(os.path.join(BASE_DIR, 'Strategies', 'SuperTrend'))
sys.path.append(os.path.join(BASE_DIR, 'chatAI'))
sys.path.append(os.path.join(BASE_DIR, 'Services', 'cronjob'))
sys.path.append(os.path.join(BASE_DIR, 'Services', 'webhook'))
sys.path.append(os.path.join(BASE_DIR, 'Services', 'subscription'))
sys.path.append(os.path.join(BASE_DIR, 'Services', 'execution'))
sys.path.append(os.path.join(BASE_DIR, 'Services', 'Deployments_helper'))
sys.path.append(os.path.join(BASE_DIR, 'Services', 'SingleSignOn'))
sys.path.append(BASE_DIR)

# =========================
# IMPORTS
# =========================
from ChatAI1.chatai1_config import settings as chatai1_new_settings
from ChatAI1.database import init_db, close_db
from ChatAI1.api import chat as chatai1_new

from Strategies.Rotation_Stocks.api.stock_routes import stock_router, initialize_stock_backtester
from Strategies.Rotation_ETF.api.etf_routes import etf_router, initialize_etf_backtester
from Strategies.RS_Stocks.api import router as rs_router
from Strategies.RS_ETF.api import router as rs_etf_router
from Strategies.customStrategy.api import custom_strategy_router
from Strategies.customStrategy.database import CustomStrategyDatabase
from Strategies.SuperTrend.api.routes import router as supertrend_router

from Services.webhook.webhook_api import router as webhook_router
from Services.webhook.webhook_logic import init_db as init_webhook_db
from Services.subscription.api import subscription_router
from Services.subscription.google_oauth_api import google_oauth_router
from Services.subscription.database import subscription_manager
from Services.Deployments_helper.deployment_helper import deployment_router
from Services.SingleSignOn import router as single_sign_on_router

# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# =========================
# GLOBAL FLAGS
# =========================
stock_backtester_initialized = False
etf_backtester_initialized = False
subscription_service_initialized = False
webhook_service_initialized = False
custom_strategy_service_initialized = False
single_sign_on_service_initialized = False

# =========================
# LIFESPAN (SAFE STARTUP)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global stock_backtester_initialized
    global etf_backtester_initialized
    global subscription_service_initialized
    global webhook_service_initialized
    global custom_strategy_service_initialized
    global single_sign_on_service_initialized

    try:
        logger.info("🚀 Starting application...")

        # Main DB
        await init_db()
        logger.info("✅ Main DB connected")

        # Backtesters
        stock_backtester_initialized = initialize_stock_backtester()
        etf_backtester_initialized = initialize_etf_backtester()

        # Subscription DB
        subscription_manager.init_database()
        subscription_service_initialized = True

        # Webhook DB
        init_webhook_db()
        webhook_service_initialized = True

        # Custom Strategy DB
        CustomStrategyDatabase()
        custom_strategy_service_initialized = True

        # Single Sign On DB
        from Databases.app_data_db_connection import create_connection as init_sso_db
        single_sign_on_service_initialized = bool(init_sso_db())

        logger.info("✅ All services initialized successfully")

    except Exception as e:
        logger.error(f"❌ STARTUP FAILED: {e}")
        raise e

    yield

    logger.info("🛑 Shutting down...")
    await close_db()

# =========================
# APP CREATION
# =========================
app = FastAPI(
    title="WealthAI1 API",
    version="1.0.0",
    lifespan=lifespan
)

# =========================
# CORS
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
# CORE ROUTES
# =========================
@app.get("/")
async def root():
    return {
        "message": "WealthAI1 API Running ✅"
    }

@app.get("/health_check")
async def health_check():
    return {
        "api_status": "healthy",
        "stock_backtester_initialized": stock_backtester_initialized,
        "etf_backtester_initialized": etf_backtester_initialized,
        "custom_strategy_initialized": custom_strategy_service_initialized,
        "subscription_service_initialized": subscription_service_initialized,
        "webhook_service_initialized": webhook_service_initialized,
        "single_sign_on_service_initialized": single_sign_on_service_initialized,
    }

@app.get("/favicon.ico")
async def favicon():
    favicon_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89'
    return Response(content=favicon_data, media_type="image/x-icon")

# =========================
# ROUTERS
# =========================
app.include_router(stock_router)
app.include_router(etf_router)
app.include_router(rs_router, prefix="/api/rs-strategy")
app.include_router(rs_etf_router, prefix="/api/rs-etf-strategy")
app.include_router(custom_strategy_router)
app.include_router(chatai1_new.router, prefix="/api")
app.include_router(webhook_router)
app.include_router(subscription_router)
app.include_router(google_oauth_router)
app.include_router(deployment_router)
app.include_router(single_sign_on_router)

supertrend_router.routes = [
    route for route in supertrend_router.routes
    if not (isinstance(route, APIRoute) and route.path == "/")
]
app.include_router(supertrend_router)

# =========================
# LOCAL RUN ONLY
# =========================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

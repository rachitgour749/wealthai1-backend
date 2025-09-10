from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import uvicorn
import sys
import os
import threading
import atexit

# Add the strategy directories to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'stockstrategy'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'etf-strategy'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'chatAI'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Payment'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'cronjob'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'webhook'))

# Import the separate API modules
from stock_api import stock_router, initialize_stock_backtester, cleanup_stock_backtester
from etf_api import etf_router, initialize_etf_backtester, cleanup_etf_backtester
from chat_api import chat_router, init_chat_ai, cleanup_chat_ai
from api import payment_router, init_payment_service, cleanup_payment_service
from webhook_api import router as webhook_router

# Import scheduler
from scheduler import ETFScheduler

# Create main FastAPI app
app = FastAPI(title="Unified Rotation Backtester API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize backtesters and ChatAI
stock_backtester_initialized = initialize_stock_backtester("unified_etf_data.sqlite")
etf_backtester_initialized = initialize_etf_backtester("unified_etf_data.sqlite")
init_chat_ai()
chat_ai_initialized = True

# Initialize payment service
payment_service_initialized = init_payment_service()

# Initialize scheduler
scheduler_instance = None
scheduler_thread = None
scheduler_initialized = False

def start_scheduler():
    """Start the ETF scheduler in a separate thread"""
    global scheduler_instance, scheduler_thread, scheduler_initialized
    
    try:
        print("🕐 Starting ETF scheduler...")
        scheduler_instance = ETFScheduler()
        
        # Start scheduler in a separate thread to avoid blocking the main server
        scheduler_thread = threading.Thread(target=scheduler_instance.start_scheduler, daemon=True)
        scheduler_thread.start()
        
        scheduler_initialized = True
        print("✅ ETF scheduler started successfully - will run daily at 4:00 PM IST")
        
    except Exception as e:
        print(f"❌ Failed to start ETF scheduler: {e}")
        scheduler_initialized = False

# Add startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("🚀 Server starting up...")
    start_scheduler()

# Add cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global scheduler_instance, scheduler_initialized
    
    print("🛑 Server shutting down...")
    
    # Stop scheduler
    if scheduler_instance and scheduler_initialized:
        try:
            print("🕐 Stopping ETF scheduler...")
            scheduler_instance.scheduler.shutdown()
            scheduler_initialized = False
            print("✅ ETF scheduler stopped successfully")
        except Exception as e:
            print(f"❌ Error stopping ETF scheduler: {e}")
    
    # Cleanup other services
    cleanup_stock_backtester()
    cleanup_etf_backtester()
    cleanup_chat_ai()
    cleanup_payment_service()

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Unified Rotation Backtester API", "strategies": ["stock", "etf", "chat"]}

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API and database status"""
    try:
        status = {
            "api_status": "healthy",
            "stock_backtester_initialized": stock_backtester_initialized,
            "etf_backtester_initialized": etf_backtester_initialized,
            "chat_ai_initialized": chat_ai_initialized,
            "payment_service_initialized": payment_service_initialized,
            "webhook_service_initialized": True,
            "scheduler_initialized": scheduler_initialized,
            "stock_database_available": stock_backtester_initialized,
            "etf_database_available": etf_backtester_initialized,
            "chat_ai_database_available": chat_ai_initialized,
            "payment_database_available": payment_service_initialized,
            "webhook_database_available": True,
            "scheduler_database_available": scheduler_initialized,
            "stock_count": 0,
            "etf_count": 0,
            "scheduler_status": "running" if scheduler_initialized else "stopped"
        }
        
        return status
    except Exception as e:
        return {
            "api_status": "error",
            "error": str(e),
            "stock_backtester_initialized": stock_backtester_initialized,
            "etf_backtester_initialized": etf_backtester_initialized,
            "chat_ai_initialized": chat_ai_initialized,
            "payment_service_initialized": payment_service_initialized,
            "scheduler_initialized": scheduler_initialized
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
app.include_router(chat_router)
app.include_router(payment_router)
app.include_router(webhook_router)

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
            from stock_api import save_stock_strategy, SaveStockStrategyRequest
            stock_request = SaveStockStrategyRequest(**request)
            return await save_stock_strategy(stock_request)
        elif strategy_type in ["etf", "etf_rotation"]:
            # Route to ETF strategy endpoint
            from etf_api import save_etf_strategy, SaveETFStrategyRequest
            etf_request = SaveETFStrategyRequest(**request)
            return await save_etf_strategy(etf_request)
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid strategy_type: '{strategy_type}'. Must be 'stock', 'stock_rotation', 'etf', or 'etf_rotation'. Received keys: {list(request.keys())}"
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
            from stock_api import get_saved_stock_strategies
            stock_result = await get_saved_stock_strategies(user_id)
            if "strategies" in stock_result:
                all_strategies.extend(stock_result["strategies"])
        except Exception as e:
            print(f"Warning: Could not fetch stock strategies: {e}")
        
        # Get ETF strategies
        try:
            from etf_api import get_saved_etf_strategies
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

if __name__ == "__main__":
    print("🚀 Starting Unified Rotation Backtester API Server...")
    print("📊 Available endpoints:")
    print("   GET  /health - Health check")
    print("   GET  /api/stocks - List available stocks")
    print("   GET  /api/etfs - List available ETFs")
    print("   POST /api/stocks/metrics - Run stock backtest")
    print("   POST /api/metrics - Run ETF backtest")
    print("   POST /api/chat - Chat with AI")
    print("   POST /api/rate - Rate AI response")
    print("   POST /api/save-strategy - Save strategy (unified - stock/ETF)")
    print("   GET  /api/get-saved-strategies/{user_id} - Get all saved strategies (unified)")
    print("   POST /api/debug-request - Debug request format")
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
    print("   GET  /api/saved-json/{user_email} - Get saved JSON data")
    print("🌐 Server will be available at: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)

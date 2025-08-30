from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import uvicorn
import sys
import os

# Add the strategy directories to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'stockstrategy'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'etf-strategy'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'chatAI'))

# Import the separate API modules
from stock_api import stock_router, initialize_stock_backtester, cleanup_stock_backtester
from etf_api import etf_router, initialize_etf_backtester, cleanup_etf_backtester
from chat_api import chat_router, initialize_chat_ai, cleanup_chat_ai

# Create main FastAPI app
app = FastAPI(title="Unified Rotation Backtester API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,  # Cache preflight requests for 10 minutes
)

# Initialize backtesters and ChatAI
stock_backtester_initialized = initialize_stock_backtester("unified_etf_data.sqlite")
etf_backtester_initialized = initialize_etf_backtester("unified_etf_data.sqlite")
chat_ai_initialized = initialize_chat_ai()

# Add cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    cleanup_stock_backtester()
    cleanup_etf_backtester()
    cleanup_chat_ai()

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
            "stock_database_available": stock_backtester_initialized,
            "etf_database_available": etf_backtester_initialized,
            "chat_ai_database_available": chat_ai_initialized,
            "stock_count": 0,
            "etf_count": 0
        }
        
        return status
    except Exception as e:
        return {
            "api_status": "error",
            "error": str(e),
            "stock_backtester_initialized": stock_backtester_initialized,
            "etf_backtester_initialized": etf_backtester_initialized,
            "chat_ai_initialized": chat_ai_initialized
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
    print("🌐 Server will be available at: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)

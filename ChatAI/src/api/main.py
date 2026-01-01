"""
FastAPI Application for Financial Advisor Chatbot

Features:
- Rate limiting per tenant
- Intent-based query routing
- Conversation memory
- Error handling
"""

import os
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, Request, HTTPException, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from google import genai

from src.core.intent_classifier import classify_intent
from src.core.orchestrator import QueryOrchestrator
from src.core.conversation_manager import (
    get_or_create_session,
    cleanup_expired_sessions
)
from src.core.error_handling import GracefulError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Gemini client (initialized on startup)
gemini_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    global gemini_client
    
    # Startup
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable required")
    
    gemini_client = genai.Client(api_key=api_key)
    logger.info("Gemini client initialized")
    
    yield
    
    # Shutdown
    cleanup_expired_sessions()
    logger.info("Application shutdown complete")


# Initialize FastAPI (for standalone use)
app = FastAPI(
    title="Financial Advisor AI",
    description="Multi-RAG chatbot for financial intermediaries",
    version="1.0.0",
    lifespan=lifespan
)

# Router for integration
router = APIRouter()

# CORS middleware - use env var or restrict to known origins
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Rate limiting
def get_tenant_id(request: Request) -> str:
    """Extract tenant ID from request for rate limiting."""
    return request.headers.get("X-Tenant-ID", get_remote_address(request))


limiter = Limiter(key_func=get_tenant_id)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors."""
    return JSONResponse(
        status_code=429,
        content=GracefulError.rate_limited()
    )


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the expense ratio of SBI Bluechip Fund?"
            }
        }


class QueryResponse(BaseModel):
    response: str
    intent: str
    citations: list = []
    active_client: str | None = None
    is_fallback: bool = False


# API Routes
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "financial-advisor-ai"}


@router.post("/api/query", response_model=QueryResponse)
@limiter.limit("30/minute")
async def handle_query(request: Request, body: QueryRequest):
    """
    Process user query and return AI response.
    
    Headers:
    - X-Tenant-ID: Tenant identifier (required)
    - X-Session-ID: Session identifier for conversation context (optional)
    """
    global gemini_client
    
    # Extract headers
    tenant_id = request.headers.get("X-Tenant-ID")
    if not tenant_id:
        raise HTTPException(400, "X-Tenant-ID header required")
    
    session_id = request.headers.get("X-Session-ID", f"{tenant_id}_default")
    
    try:
        # Get conversation context
        conversation = get_or_create_session(session_id)
        
        # Classify intent
        intent = await classify_intent(body.query, gemini_client)
        logger.info(f"Classified intent: {intent.primary_intent} (conf: {intent.confidence})")
        
        # Route and execute (returns response + effective intent)
        orchestrator = QueryOrchestrator(gemini_client, tenant_id)
        response, effective_intent = await orchestrator.route(body.query, intent, conversation)
        
        # Extract response text
        response_text = response.text if hasattr(response, 'text') else str(response)
        
        # Update conversation history with effective intent (for context-aware routing)
        conversation.add_message("user", body.query, effective_intent.value)
        conversation.add_message("assistant", response_text)
        
        # Extract citations if available
        citations = []
        if hasattr(response, 'grounding_metadata'):
            citations = extract_citations(response.grounding_metadata)
        
        return QueryResponse(
            response=response_text,
            intent=effective_intent.value,  # Use effective intent, not original
            citations=citations,
            active_client=conversation.active_client,
            is_fallback=getattr(response, '_is_fallback', False)
        )
    
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(500, f"Failed to process query: {str(e)}")


@router.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """Get current session context."""
    conversation = get_or_create_session(session_id)
    return {
        "session_id": session_id,
        "active_client": conversation.active_client,
        "active_scheme": conversation.active_scheme,
        "message_count": len(conversation.history),
        "is_expired": conversation.is_expired()
    }


@router.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """Clear session history."""
    conversation = get_or_create_session(session_id)
    conversation.clear()
    return {"status": "cleared", "session_id": session_id}


def extract_citations(grounding_metadata) -> list:
    """Extract citation information from Gemini response."""
    citations = []
    
    if hasattr(grounding_metadata, 'grounding_chunks'):
        for chunk in grounding_metadata.grounding_chunks:
            if hasattr(chunk, 'retrieved_context'):
                citations.append({
                    "source": chunk.retrieved_context.uri,
                    "title": chunk.retrieved_context.title
                })
    
    return citations


# Include webhook router
from src.sync.zoho_webhook import router as zoho_router
router.include_router(zoho_router)

# Include store management router
from src.api.store_routes import router as store_router
router.include_router(store_router)

# Include admin routes
from src.api.admin_routes import router as admin_router
router.include_router(admin_router)

# Include router in standalone app
app.include_router(router)

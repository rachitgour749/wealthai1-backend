from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
import os
import uuid
import time
from datetime import datetime

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import the ChatAI core
from chat_core import ChatAICore

# Create ChatAI router
chat_router = APIRouter(prefix="/api", tags=["ChatAI"])

# Pydantic models for request/response
class ChatRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None

class RatingRequest(BaseModel):
    trace_id: str
    user_rating: int
    feedback_comment: Optional[str] = ""

class ChatResponse(BaseModel):
    response: str
    model_used: str
    response_id: str
    timestamp: float
    system_prompt_used: str
    rating: Optional[float] = None
    provider: str

class RatingResponse(BaseModel):
    success: bool
    message: str
    trace_id: str

# Global ChatAI instance
chat_ai = None

def initialize_chat_ai():
    """Initialize the ChatAI core"""
    global chat_ai
    try:
        chat_ai = ChatAICore()
        print("✅ ChatAI initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing ChatAI: {e}")
        chat_ai = None
        return False

def cleanup_chat_ai():
    """Clean up ChatAI resources"""
    global chat_ai
    if chat_ai:
        chat_ai.cleanup()
        chat_ai = None

# ============================================================================
# CHATAI ROUTES
# ============================================================================

@chat_router.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """Chat with AI assistant"""
    try:
        if chat_ai is None:
            raise HTTPException(status_code=500, detail="ChatAI not initialized")
        
        start_time = time.time()
        
        # Generate response using ChatAI core
        response_data = chat_ai.generate_response(
            prompt=request.prompt,
            system_prompt=request.system_prompt
        )
        
        processing_time = time.time() - start_time
        
        # Create response object
        response = ChatResponse(
            response=response_data.get("response", "Sorry, I couldn't process your request."),
            model_used=response_data.get("model_used", "unknown"),
            response_id=str(uuid.uuid4()),
            timestamp=datetime.now().timestamp(),
            system_prompt_used=request.system_prompt or "Default system prompt",
            rating=response_data.get("rating"),
            provider=response_data.get("provider", "langchain_modal")
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@chat_router.post("/rate", response_model=RatingResponse)
async def rate_response(request: RatingRequest):
    """Rate an AI response"""
    try:
        if chat_ai is None:
            raise HTTPException(status_code=500, detail="ChatAI not initialized")
        
        # Store rating using ChatAI core
        success = chat_ai.store_rating(
            trace_id=request.trace_id,
            user_rating=request.user_rating,
            feedback_comment=request.feedback_comment
        )
        
        if success:
            return RatingResponse(
                success=True,
                message="Rating stored successfully",
                trace_id=request.trace_id
            )
        else:
            return RatingResponse(
                success=False,
                message="Failed to store rating",
                trace_id=request.trace_id
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing rating: {str(e)}")

@chat_router.get("/health")
async def chat_health_check():
    """Health check for ChatAI service"""
    try:
        status = {
            "service": "ChatAI",
            "status": "healthy" if chat_ai is not None else "uninitialized",
            "initialized": chat_ai is not None
        }
        return status
    except Exception as e:
        return {
            "service": "ChatAI",
            "status": "error",
            "error": str(e),
            "initialized": False
        }


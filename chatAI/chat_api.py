from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
import os
import uuid
import time
import sqlite3
import json
from datetime import datetime

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import ChatAI core from the new implementation
from chat_core_new import ChatAICore

# Create ChatAI router
chat_router = APIRouter(prefix="/api", tags=["ChatAI"])

# Initialize ChatAI core
chat_ai = None

def init_chat_ai():
    """Initialize ChatAI core - called by server"""
    global chat_ai
    try:
        chat_ai = ChatAICore()
        print("✅ ChatAI initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing ChatAI: {e}")
        return False

def cleanup_chat_ai():
    """Cleanup ChatAI core - called by server"""
    global chat_ai
    try:
        if chat_ai:
            chat_ai.cleanup()
            chat_ai = None
        print("✅ ChatAI cleanup completed")
    except Exception as e:
        print(f"❌ Error during ChatAI cleanup: {e}")

# Initialize on import
init_chat_ai()

# Pydantic models for request/response
class ChatRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None

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
    trace_id: str
    conversation_id: Optional[str] = None
    processing_time: float
    modal_response_id: Optional[str] = None

class RatingResponse(BaseModel):
    success: bool
    message: str
    trace_id: str

class HealthResponse(BaseModel):
    service: str
    status: str
    initialized: bool
    modal_endpoint: str

class SavePromptRequest(BaseModel):
    prompt_text: str
    user_id: Optional[str] = None

class SavePromptResponse(BaseModel):
    success: bool
    message: str
    prompt_id: Optional[int] = None
    created_at: Optional[str] = None


class UserHistoryResponse(BaseModel):
    success: bool
    conversations: List[Dict[str, Any]]
    total_count: int
    message: Optional[str] = None

# ============================================================================
# CHATAI ROUTES
# ============================================================================

@chat_router.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """Chat with AI assistant using Modal model with automatic LangChain and Observability processing"""
    try:
        if chat_ai is None:
            raise HTTPException(status_code=500, detail="ChatAI not initialized")
        
        # Generate response using ChatAI core with automatic LangChain and Observability processing
        response_data = await chat_ai.generate_response(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            conversation_id=request.conversation_id,
            user_id=request.user_id
        )
        
        # Create response object
        response = ChatResponse(
            response=response_data.get("response", "Sorry, I couldn't process your request."),
            model_used=response_data.get("model_used", "modal-mf-assistant"),
            response_id=str(uuid.uuid4()),
            timestamp=datetime.now().timestamp(),
            system_prompt_used=response_data.get("system_prompt_used", "default"),
            rating=response_data.get("rating"),
            provider=response_data.get("provider", "modal"),
            trace_id=response_data.get("trace_id", ""),
            conversation_id=response_data.get("conversation_id"),
            processing_time=response_data.get("processing_time", 0.0),
            modal_response_id=response_data.get("modal_response_id")
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@chat_router.post("/rate", response_model=RatingResponse)
async def rate_response(request: RatingRequest):
    """Rate AI response"""
    try:
        if chat_ai is None:
            raise HTTPException(status_code=500, detail="ChatAI not initialized")
        
        success = chat_ai.store_rating(
            trace_id=request.trace_id,
            user_rating=request.user_rating,
            feedback_comment=request.feedback_comment
        )
        
        return RatingResponse(
            success=success,
            message="Rating stored successfully" if success else "Failed to store rating",
            trace_id=request.trace_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing rating: {str(e)}")

@chat_router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        return HealthResponse(
            service="ChatAI",
            status="healthy" if chat_ai is not None else "unhealthy",
            initialized=chat_ai is not None,
            modal_endpoint="https://anjanr--mf-assistant-web.modal.run/chat"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@chat_router.get("/user-history", response_model=UserHistoryResponse)
async def get_user_history(user_id: Optional[str] = None, limit: int = 50):
    """Get user conversation history"""
    try:
        if chat_ai is None:
            raise HTTPException(status_code=500, detail="ChatAI not initialized")
        
        result = chat_ai.get_user_history(user_id, limit)
        return UserHistoryResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving user history: {str(e)}")

@chat_router.delete("/user-prompts/{user_id}/{conversation_id}")
async def delete_user_prompt(user_id: str, conversation_id: str):
    """Delete a single saved prompt for a user by conversation_id"""
    try:
        if chat_ai is None:
            raise HTTPException(status_code=500, detail="ChatAI not initialized")
        
        result = chat_ai.delete_user_prompt(user_id, conversation_id)
        
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=404, detail=result["message"])
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting user prompt: {str(e)}")
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
import os
import uuid
import time
import json
from datetime import datetime

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import ChatAI core from the new implementation
from chat_core_new import ChatAICore

# Create ZOHO router
zoho_router = APIRouter(prefix="/api/zoho", tags=["ZOHO Bot"])

# Initialize ChatAI core for ZOHO
zoho_chat_ai = None

def init_zoho_chat():
    """Initialize ChatAI core for ZOHO - called by server"""
    global zoho_chat_ai
    try:
        zoho_chat_ai = ChatAICore()
        print("✅ ZOHO ChatAI initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing ZOHO ChatAI: {e}")
        return False

def cleanup_zoho_chat():
    """Cleanup ZOHO ChatAI core - called by server"""
    global zoho_chat_ai
    try:
        if zoho_chat_ai:
            zoho_chat_ai.cleanup()
            zoho_chat_ai = None
        print("✅ ZOHO ChatAI cleanup completed")
    except Exception as e:
        print(f"❌ Error during ZOHO ChatAI cleanup: {e}")

# Initialize on import
init_zoho_chat()

# Pydantic models for ZOHO request/response
class ZohoChatRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    user_id: Optional[str] = None

class ZohoChatResponse(BaseModel):
    response: str
    model_used: str
    response_id: str
    timestamp: float
    system_prompt_used: str
    rating: Optional[float] = None
    provider: str
    trace_id: str
    processing_time: float
    modal_response_id: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    classification_input_tokens: int = 0
    classification_output_tokens: int = 0

class ZohoHealthResponse(BaseModel):
    service: str
    status: str
    initialized: bool
    modal_endpoint: str

class ZohoTestResponse(BaseModel):
    success: bool
    message: str
    test_data: Optional[Dict[str, Any]] = None

# ============================================================================
# ZOHO BOT ROUTES
# ============================================================================

@zoho_router.post("/chat", response_model=ZohoChatResponse)
async def zoho_chat_with_ai(request: ZohoChatRequest):
    """ZOHO Bot Chat - No conversation storage, no conversation ID generation"""
    try:
        if zoho_chat_ai is None:
            raise HTTPException(status_code=500, detail="ZOHO ChatAI not initialized")
        
        # Generate response using ChatAI core but without conversation storage
        # We pass None for conversation_id to prevent storage
        response_data = await zoho_chat_ai.generate_response(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            conversation_id=None,  # No conversation ID for ZOHO
            user_id=request.user_id,
            store_conversation=False  # Don't store conversation for ZOHO
        )
        
        # Create response object (no conversation_id field)
        response = ZohoChatResponse(
            response=response_data.get("response", "Sorry, I couldn't process your request."),
            model_used=response_data.get("model_used", "modal-mf-assistant"),
            response_id=str(uuid.uuid4()),
            timestamp=datetime.now().timestamp(),
            system_prompt_used=response_data.get("system_prompt_used", "default"),
            rating=response_data.get("rating"),
            provider=response_data.get("provider", "modal"),
            trace_id=response_data.get("trace_id", ""),
            processing_time=response_data.get("processing_time", 0.0),
            modal_response_id=response_data.get("modal_response_id"),
            input_tokens=response_data.get("input_tokens", 0),
            output_tokens=response_data.get("output_tokens", 0),
            total_tokens=response_data.get("total_tokens", 0),
            classification_input_tokens=response_data.get("classification_input_tokens", 0),
            classification_output_tokens=response_data.get("classification_output_tokens", 0)
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing ZOHO chat request: {str(e)}")

@zoho_router.get("/health", response_model=ZohoHealthResponse)
async def zoho_health_check():
    """ZOHO API health check endpoint"""
    try:
        return ZohoHealthResponse(
            service="ZOHO Bot Chat",
            status="healthy" if zoho_chat_ai is not None else "unhealthy",
            initialized=zoho_chat_ai is not None,
            modal_endpoint="https://anjanr--mf-assistant-web.modal.run/chat"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ZOHO health check failed: {str(e)}")

@zoho_router.post("/test", response_model=ZohoTestResponse)
async def zoho_test_endpoint():
    """ZOHO API test endpoint"""
    try:
        if zoho_chat_ai is None:
            return ZohoTestResponse(
                success=False,
                message="ZOHO ChatAI not initialized"
            )
        
        # Test with a simple prompt
        test_request = ZohoChatRequest(
            prompt="Hello, this is a test message from ZOHO bot.",
            system_prompt="You are a helpful assistant for ZOHO users."
        )
        
        # Generate test response
        response_data = await zoho_chat_ai.generate_response(
            prompt=test_request.prompt,
            system_prompt=test_request.system_prompt,
            conversation_id=None,
            user_id="zoho_test_user",
            store_conversation=False
        )
        
        return ZohoTestResponse(
            success=True,
            message="ZOHO API test successful",
            test_data={
                "response_length": len(response_data.get("response", "")),
                "model_used": response_data.get("model_used"),
                "processing_time": response_data.get("processing_time"),
                "trace_id": response_data.get("trace_id")
            }
        )
        
    except Exception as e:
        return ZohoTestResponse(
            success=False,
            message=f"ZOHO API test failed: {str(e)}"
        )

@zoho_router.get("/info")
async def zoho_info():
    """Get ZOHO API information"""
    return {
        "service": "ZOHO Bot Chat API",
        "description": "Chat API for ZOHO Bot integration - No conversation storage",
        "version": "1.0.0",
        "features": [
            "Real-time chat responses",
            "No conversation storage",
            "No conversation ID generation",
            "Same AI capabilities as main chat API",
            "Optimized for ZOHO Bot integration"
        ],
        "endpoints": {
            "POST /api/zoho/chat": "Chat with AI (no storage)",
            "GET /api/zoho/health": "Health check",
            "POST /api/zoho/test": "Test endpoint",
            "GET /api/zoho/info": "API information"
        }
    }


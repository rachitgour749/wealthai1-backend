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

# Import the ChatAI core
from chat_core import ChatAICore

# Create ChatAI router
chat_router = APIRouter(prefix="/api", tags=["ChatAI"])

# Pydantic models for request/response
class ChatRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    conversation_id: Optional[str] = None
    use_template: Optional[str] = None
    template_params: Optional[Dict[str, Any]] = None

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
    chain_id: str
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

class PerformanceMetricsResponse(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_processing_time: float
    total_processing_time: float

class ObservabilityLogResponse(BaseModel):
    trace_id: str
    logs: List[Dict[str, Any]]

class ChainExecutionResponse(BaseModel):
    chain_id: str
    steps: List[Dict[str, Any]]

class ConversationHistoryResponse(BaseModel):
    conversation_id: str
    messages: List[Dict[str, Any]]

class SaveUserPromptRequest(BaseModel):
    user_id: str
    prompt: str
    ai_response: str
    system_prompt: Optional[str] = None
    model_used: Optional[str] = None
    provider: Optional[str] = None
    processing_time: Optional[float] = None
    trace_id: Optional[str] = None
    conversation_id: Optional[str] = None

class UserPromptHistory(BaseModel):
    id: int
    user_id: str
    prompt: str
    ai_response: str
    system_prompt: Optional[str]
    model_used: Optional[str]
    provider: Optional[str]
    processing_time: Optional[float]
    trace_id: Optional[str]
    conversation_id: Optional[str]
    timestamp: str

# Global ChatAI instance
chat_ai: Optional[ChatAICore] = None

def init_chat_ai():
    """Initialize ChatAI instance"""
    global chat_ai
    try:
        chat_ai = ChatAICore()
        print("✅ ChatAI initialized successfully with custom observability and LangChain")
    except Exception as e:
        print(f"❌ Error initializing ChatAI: {e}")
        chat_ai = None

def cleanup_chat_ai():
    """Cleanup ChatAI instance"""
    global chat_ai
    if chat_ai:
        chat_ai.cleanup()
        chat_ai = None

# ============================================================================
# CHATAI ROUTES
# ============================================================================

@chat_router.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """Chat with AI assistant using Modal model with LangChain and observability"""
    try:
        if chat_ai is None:
            raise HTTPException(status_code=500, detail="ChatAI not initialized")
        
        # Generate response using ChatAI core with LangChain features
        response_data = await chat_ai.generate_response(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            conversation_id=request.conversation_id,
            use_template=request.use_template,
            template_params=request.template_params
        )
        
        # Create response object
        response = ChatResponse(
            response=response_data.get("response", "Sorry, I couldn't process your request."),
            model_used=response_data.get("model_used", "modal-mf-assistant"),
            response_id=str(uuid.uuid4()),
            timestamp=datetime.now().timestamp(),
            system_prompt_used=request.system_prompt or "Default system prompt",
            rating=response_data.get("rating"),
            provider=response_data.get("provider", "modal"),
            trace_id=response_data.get("trace_id", ""),
            chain_id=response_data.get("chain_id", ""),
            conversation_id=response_data.get("conversation_id"),
            processing_time=response_data.get("processing_time", 0.0),
            modal_response_id=response_data.get("modal_response_id")
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

@chat_router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check ChatAI service health"""
    try:
        return HealthResponse(
            service="ChatAI",
            status="healthy" if chat_ai is not None else "unhealthy",
            initialized=chat_ai is not None,
            modal_endpoint="https://anjanr--mf-assistant-web.modal.run/chat"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@chat_router.get("/conversations/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(conversation_id: str, limit: int = 10):
    """Get conversation history using LangChain memory"""
    try:
        if chat_ai is None:
            raise HTTPException(status_code=500, detail="ChatAI not initialized")
        
        messages = chat_ai.get_conversation_history(conversation_id, limit=limit)
        return ConversationHistoryResponse(
            conversation_id=conversation_id,
            messages=messages
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting conversation history: {str(e)}")

@chat_router.get("/metrics", response_model=PerformanceMetricsResponse)
async def get_performance_metrics():
    """Get performance metrics from custom observability"""
    try:
        if chat_ai is None:
            raise HTTPException(status_code=500, detail="ChatAI not initialized")
        
        metrics = chat_ai.get_performance_metrics()
        return PerformanceMetricsResponse(**metrics)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting performance metrics: {str(e)}")

@chat_router.get("/observability/{trace_id}", response_model=ObservabilityLogResponse)
async def get_observability_logs(trace_id: str):
    """Get observability logs for a specific trace"""
    try:
        if chat_ai is None:
            raise HTTPException(status_code=500, detail="ChatAI not initialized")
        
        logs = chat_ai.get_observability_logs(trace_id)
        return ObservabilityLogResponse(trace_id=trace_id, logs=logs)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting observability logs: {str(e)}")

@chat_router.get("/chain/{chain_id}", response_model=ChainExecutionResponse)
async def get_chain_execution(chain_id: str):
    """Get LangChain execution history"""
    try:
        if chat_ai is None:
            raise HTTPException(status_code=500, detail="ChatAI not initialized")
        
        steps = chat_ai.get_chain_execution(chain_id)
        return ChainExecutionResponse(chain_id=chain_id, steps=steps)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting chain execution: {str(e)}")

@chat_router.get("/templates")
async def get_prompt_templates():
    """Get available prompt templates"""
    try:
        if chat_ai is None:
            raise HTTPException(status_code=500, detail="ChatAI not initialized")
        
        # Return available templates
        templates = {
            "financial_expert": "Formatted financial expert response with emojis and structure",
            "simple_response": "Simple and concise financial response",
            "detailed_analysis": "Detailed financial analysis with multiple sections"
        }
        
        return {
            "templates": templates,
            "usage": "Use 'use_template' parameter in chat request with 'template_params'"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting templates: {str(e)}")

@chat_router.get("/test-modal")
async def test_modal_connection():
    """Test connection to Modal endpoint"""
    try:
        if chat_ai is None:
            raise HTTPException(status_code=500, detail="ChatAI not initialized")
        
        # Test with a simple query
        test_response = await chat_ai.generate_response(
            prompt="What is SIP?",
            system_prompt="You are a financial expert. Answer briefly."
        )
        
        return {
            "status": "success",
            "modal_connected": True,
            "test_response": test_response.get("response", "")[:100] + "...",
            "processing_time": test_response.get("processing_time", 0.0),
            "trace_id": test_response.get("trace_id", ""),
            "chain_id": test_response.get("chain_id", "")
        }
        
    except Exception as e:
        return {
            "status": "error",
            "modal_connected": False,
            "error": str(e)
        }

# ============================================================================
# USER PROMPT HISTORY DATABASE FUNCTIONS
# ============================================================================

def init_user_prompt_history_table(db_path: str = "unified_etf_data.sqlite"):
    """Initialize the user_prompt_history table if it doesn't exist"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_prompt_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                system_prompt TEXT,
                model_used TEXT,
                provider TEXT,
                processing_time REAL,
                trace_id TEXT,
                conversation_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index on user_id for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_user_prompt_history_user_id 
            ON user_prompt_history(user_id)
        ''')
        
        # Create index on timestamp for chronological queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_user_prompt_history_timestamp 
            ON user_prompt_history(timestamp)
        ''')
        
        conn.commit()
        conn.close()
        print("✅ User prompt history table initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing user prompt history table: {e}")
        return False

# ============================================================================
# USER PROMPT HISTORY ROUTES
# ============================================================================

@chat_router.post("/save-user-prompt")
async def save_user_prompt(request: SaveUserPromptRequest):
    """Save a user prompt and AI response to the database"""
    try:
        # Initialize the table if it doesn't exist
        if not init_user_prompt_history_table():
            raise HTTPException(status_code=500, detail="Failed to initialize database table")
        
        conn = sqlite3.connect("unified_etf_data.sqlite")
        cursor = conn.cursor()
        
        # Insert the user prompt history
        cursor.execute('''
            INSERT INTO user_prompt_history (
                user_id, prompt, ai_response, system_prompt, model_used, provider,
                processing_time, trace_id, conversation_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            request.user_id, request.prompt, request.ai_response, request.system_prompt,
            request.model_used, request.provider, request.processing_time,
            request.trace_id, request.conversation_id
        ))
        
        history_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": "User prompt history saved successfully",
            "history_id": history_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving user prompt history: {str(e)}")

@chat_router.get("/user-prompt-history/{user_id}")
async def get_user_prompt_history(user_id: str, limit: int = 50, offset: int = 0):
    """Get user prompt history for a specific user"""
    try:
        # Initialize the table if it doesn't exist
        if not init_user_prompt_history_table():
            raise HTTPException(status_code=500, detail="Failed to initialize database table")
        
        conn = sqlite3.connect("unified_etf_data.sqlite")
        cursor = conn.cursor()
        
        # Get user prompt history with pagination
        cursor.execute('''
            SELECT id, user_id, prompt, ai_response, system_prompt, model_used, provider,
                   processing_time, trace_id, conversation_id, timestamp
            FROM user_prompt_history 
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        ''', (user_id, limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                "id": row[0],
                "user_id": row[1],
                "prompt": row[2],
                "ai_response": row[3],
                "system_prompt": row[4],
                "model_used": row[5],
                "provider": row[6],
                "processing_time": row[7],
                "trace_id": row[8],
                "conversation_id": row[9],
                "timestamp": row[10]
            })
        
        return {
            "user_id": user_id,
            "history": history,
            "total_count": len(history),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving user prompt history: {str(e)}")

@chat_router.get("/user-prompt-history/{user_id}/count")
async def get_user_prompt_history_count(user_id: str):
    """Get count of user prompt history for a specific user"""
    try:
        # Initialize the table if it doesn't exist
        if not init_user_prompt_history_table():
            raise HTTPException(status_code=500, detail="Failed to initialize database table")
        
        conn = sqlite3.connect("unified_etf_data.sqlite")
        cursor = conn.cursor()
        
        # Get count of user prompt history
        cursor.execute('''
            SELECT COUNT(*) FROM user_prompt_history WHERE user_id = ?
        ''', (user_id,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return {"user_id": user_id, "count": count}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting user prompt history count: {str(e)}")

@chat_router.get("/user-prompt-history/{user_id}/conversation/{conversation_id}")
async def get_user_conversation_history(user_id: str, conversation_id: str):
    """Get user prompt history for a specific conversation"""
    try:
        # Initialize the table if it doesn't exist
        if not init_user_prompt_history_table():
            raise HTTPException(status_code=500, detail="Failed to initialize database table")
        
        conn = sqlite3.connect("unified_etf_data.sqlite")
        cursor = conn.cursor()
        
        # Get user prompt history for specific conversation
        cursor.execute('''
            SELECT id, user_id, prompt, ai_response, system_prompt, model_used, provider,
                   processing_time, trace_id, conversation_id, timestamp
            FROM user_prompt_history 
            WHERE user_id = ? AND conversation_id = ?
            ORDER BY timestamp ASC
        ''', (user_id, conversation_id))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                "id": row[0],
                "user_id": row[1],
                "prompt": row[2],
                "ai_response": row[3],
                "system_prompt": row[4],
                "model_used": row[5],
                "provider": row[6],
                "processing_time": row[7],
                "trace_id": row[8],
                "conversation_id": row[9],
                "timestamp": row[10]
            })
        
        return {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "history": history,
            "total_count": len(history)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation history: {str(e)}")

@chat_router.delete("/delete-user-prompt-history/{user_id}")
async def delete_user_prompt_history(user_id: str):
    """Delete all user prompt history for a specific user"""
    try:
        # Initialize the table if it doesn't exist
        if not init_user_prompt_history_table():
            raise HTTPException(status_code=500, detail="Failed to initialize database table")
        
        conn = sqlite3.connect("unified_etf_data.sqlite")
        cursor = conn.cursor()
        
        # Delete all user prompt history
        cursor.execute('DELETE FROM user_prompt_history WHERE user_id = ?', (user_id,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": f"Deleted {deleted_count} prompt history records for user {user_id}",
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting user prompt history: {str(e)}")

@chat_router.delete("/user-prompt-history/{user_id}/conversation/{conversation_id}")
async def delete_user_conversation_history(user_id: str, conversation_id: str):
    """Delete user prompt history for a specific conversation"""
    try:
        # Initialize the table if it doesn't exist
        if not init_user_prompt_history_table():
            raise HTTPException(status_code=500, detail="Failed to initialize database table")
        
        conn = sqlite3.connect("unified_etf_data.sqlite")
        cursor = conn.cursor()
        
        # Delete user prompt history for specific conversation
        cursor.execute('''
            DELETE FROM user_prompt_history 
            WHERE user_id = ? AND conversation_id = ?
        ''', (user_id, conversation_id))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": f"Deleted {deleted_count} prompt history records for conversation {conversation_id}",
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting conversation history: {str(e)}")

@chat_router.get("/user-prompt-history/{user_id}/search")
async def search_user_prompt_history(user_id: str, query: str, limit: int = 20):
    """Search user prompt history for a specific user"""
    try:
        # Initialize the table if it doesn't exist
        if not init_user_prompt_history_table():
            raise HTTPException(status_code=500, detail="Failed to initialize database table")
        
        conn = sqlite3.connect("unified_etf_data.sqlite")
        cursor = conn.cursor()
        
        # Search in both prompt and ai_response fields
        cursor.execute('''
            SELECT id, user_id, prompt, ai_response, system_prompt, model_used, provider,
                   processing_time, trace_id, conversation_id, timestamp
            FROM user_prompt_history 
            WHERE user_id = ? AND (prompt LIKE ? OR ai_response LIKE ?)
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (user_id, f"%{query}%", f"%{query}%", limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                "id": row[0],
                "user_id": row[1],
                "prompt": row[2],
                "ai_response": row[3],
                "system_prompt": row[4],
                "model_used": row[5],
                "provider": row[6],
                "processing_time": row[7],
                "trace_id": row[8],
                "conversation_id": row[9],
                "timestamp": row[10]
            })
        
        return {
            "user_id": user_id,
            "search_query": query,
            "history": history,
            "total_count": len(history),
            "limit": limit
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching user prompt history: {str(e)}")


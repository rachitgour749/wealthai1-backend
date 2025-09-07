import os
import json
import sqlite3
import uuid
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import deque
from dataclasses import dataclass, asdict
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# INTERNAL DATA STRUCTURES (Not exposed as APIs)
# ============================================================================

@dataclass
class ObservabilityEvent:
    """Internal observability event structure"""
    trace_id: str
    event_type: str
    event_data: Dict[str, Any]
    timestamp: float
    duration: Optional[float] = None

@dataclass
class ConversationMemory:
    """Internal conversation memory"""
    conversation_id: str
    messages: List[Dict[str, str]]
    metadata: Dict[str, Any]
    created_at: float
    last_updated: float

class InternalObservability:
    """Internal observability system - not exposed as API"""
    
    def __init__(self, db_path: str, max_events: int = 1000):
        self.db_path = db_path
        self.events: deque = deque(maxlen=max_events)
        self.metrics: Dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_processing_time": 0.0,
            "total_processing_time": 0.0
        }
        self.traces: Dict[str, List[ObservabilityEvent]] = {}
    
    def log_event(self, trace_id: str, event_type: str, event_data: Dict[str, Any], duration: Optional[float] = None):
        """Log an observability event internally"""
        event = ObservabilityEvent(
            trace_id=trace_id,
            event_type=event_type,
            event_data=event_data,
            timestamp=time.time(),
            duration=duration
        )
        
        # Store in memory
        self.events.append(event)
        
        # Store in trace-specific list
        if trace_id not in self.traces:
            self.traces[trace_id] = []
        self.traces[trace_id].append(event)
        
        # Store in database
        self._log_to_database(trace_id, event_type, event_data)
        
        logger.info(f"🔍 OBSERVABILITY: {event_type} - {trace_id}")
    
    def _log_to_database(self, trace_id: str, event_type: str, event_data: Dict[str, Any]):
        """Log event to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO observability_logs 
                (trace_id, event_type, event_data)
                VALUES (?, ?, ?)
            ''', (trace_id, event_type, json.dumps(event_data)))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error logging to database: {e}")
    
    def update_metrics(self, success: bool, processing_time: float):
        """Update performance metrics"""
        self.metrics["total_requests"] += 1
        self.metrics["total_processing_time"] += processing_time
        
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        # Calculate average processing time
        if self.metrics["total_requests"] > 0:
            self.metrics["avg_processing_time"] = (
                self.metrics["total_processing_time"] / self.metrics["total_requests"]
            )

class InternalLangChain:
    """Internal LangChain implementation - not exposed as API"""
    
    def __init__(self, db_path: str, max_memory: int = 100):
        self.db_path = db_path
        self.memories: Dict[str, ConversationMemory] = {}
        self.max_memory = max_memory
    
    def add_message_to_memory(self, conversation_id: str, role: str, content: str):
        """Add a message to conversation memory"""
        if conversation_id not in self.memories:
            self.memories[conversation_id] = ConversationMemory(
                conversation_id=conversation_id,
                messages=[],
                metadata={},
                created_at=time.time(),
                last_updated=time.time()
            )
        
        memory = self.memories[conversation_id]
        memory.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        memory.last_updated = time.time()
        
        # Limit memory size
        if len(memory.messages) > self.max_memory:
            memory.messages = memory.messages[-self.max_memory:]
    
    def get_conversation_history(self, conversation_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Get conversation history"""
        if conversation_id not in self.memories:
            return []
        
        memory = self.memories[conversation_id]
        return memory.messages[-limit:] if limit > 0 else memory.messages

class ChatAICore:
    def __init__(self, db_path: str = "unified_etf_data.sqlite"):
        """Initialize ChatAI core with database connection and Modal integration"""
        self.db_path = db_path
        self.modal_endpoint = "https://anjanr--mf-assistant-web.modal.run/chat"
        
        # Initialize database
        self.init_database()
        
        # Initialize internal systems (not exposed as APIs)
        self.observability = InternalObservability(db_path)
        self.langchain = InternalLangChain(db_path)
        
        logger.info("✅ ChatAI Core initialized successfully")
        
    def init_database(self):
        """Initialize the database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT UNIQUE NOT NULL,
                    user_prompt TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    system_prompt TEXT,
                    model_used TEXT,
                    provider TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    processing_time REAL,
                    modal_response_id TEXT,
                    error_message TEXT,
                    trace_id TEXT
                )
            ''')
            
            # Create ratings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT UNIQUE NOT NULL,
                    user_rating INTEGER NOT NULL,
                    feedback_comment TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create observability table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS observability_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create saved_prompts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS saved_prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_text TEXT UNIQUE NOT NULL,
                    user_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            print("✅ ChatAI database initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing ChatAI database: {e}")
    
    async def _call_modal_model(self, prompt: str, system_prompt: Optional[str], trace_id: str) -> Dict[str, Any]:
        """Call the Modal model endpoint with custom observability"""
        start_time = time.time()
        
        try:
            # Prepare the request payload
            payload = {
                "prompt": prompt,
                "system_prompt": system_prompt or "You are a ChatGPT-style financial expert. FORMAT: Start with 📚 DEFINITION (30 words max), then 💡 KEY POINTS (1 line each), add 🎯 EXAMPLE (1-2 lines), end with ✅ PRACTICAL TAKEAWAY (1 line). Use emojis, bullet points, and clear sections. Keep it concise but engaging like ChatGPT."
            }
            
            # Log the request
            self.observability.log_event(trace_id, "modal_request", payload)
            
            # Make the request to Modal
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.modal_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                execution_time = time.time() - start_time
                
                if response.status_code == 200:
                    modal_data = response.json()
                    
                    # Log the response
                    self.observability.log_event(trace_id, "modal_response", {
                        "status_code": response.status_code,
                        "response_id": modal_data.get("response_id"),
                        "model_used": modal_data.get("model_used"),
                        "execution_time": execution_time
                    }, duration=execution_time)
                    
                    return modal_data
                else:
                    error_msg = f"Modal API returned status {response.status_code}: {response.text}"
                    self.observability.log_event(trace_id, "modal_error", {
                        "status_code": response.status_code,
                        "error": response.text,
                        "execution_time": execution_time
                    }, duration=execution_time)
                    
                    raise Exception(f"Modal API failed: {error_msg}")
                    
        except Exception as e:
            execution_time = time.time() - start_time
            self.observability.log_event(trace_id, "modal_exception", {
                "error": str(e),
                "execution_time": execution_time
            }, duration=execution_time)
            
            raise Exception(f"Modal API exception: {e}")

    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None, 
                              conversation_id: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate AI response with automatic LangChain and Observability processing"""
        trace_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Log the start of the request
            self.observability.log_event(trace_id, "request_start", {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "conversation_id": conversation_id
            })
            
            # Call Modal model
            modal_response = await self._call_modal_model(prompt, system_prompt, trace_id)
            
            # AUTOMATIC LANGCHAIN AND OBSERVABILITY PROCESSING AFTER LLM RESPONSE
            processing_time = time.time() - start_time
            
            # Update conversation memory if conversation_id provided
            if conversation_id:
                self.langchain.add_message_to_memory(conversation_id, "user", prompt)
                self.langchain.add_message_to_memory(conversation_id, "assistant", 
                                                   modal_response.get("response", ""))
            
            # Log successful response
            self.observability.log_event(trace_id, "response_success", {
                "processing_time": processing_time,
                "response_length": len(modal_response.get("response", ""))
            }, duration=processing_time)
            
            # Update metrics
            self.observability.update_metrics(True, processing_time)
            
            # Store conversation in database
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            
            self._store_conversation(
                conversation_id=conversation_id,
                user_prompt=prompt,
                ai_response=modal_response.get("response", ""),
                system_prompt=system_prompt,
                model_used=modal_response.get("model_used", "modal-mf-assistant"),
                provider="modal",
                processing_time=processing_time,
                modal_response_id=modal_response.get("response_id"),
                trace_id=trace_id,
                user_id=user_id
            )
            
            return {
                "response": modal_response.get("response", "Sorry, I couldn't process your request."),
                "model_used": modal_response.get("model_used", "modal-mf-assistant"),
                "provider": "modal",
                "conversation_id": conversation_id,
                "trace_id": trace_id,
                "processing_time": processing_time,
                "modal_response_id": modal_response.get("response_id"),
                "system_prompt_used": system_prompt or "default"
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error generating response: {str(e)}"
            
            # Log error
            self.observability.log_event(trace_id, "response_error", {
                "error": str(e),
                "processing_time": processing_time
            }, duration=processing_time)
            
            # Update metrics
            self.observability.update_metrics(False, processing_time)
            
            logger.error(f"❌ {error_msg}")
            
            return {
                "response": "Sorry, I couldn't process your request. Please try again.",
                "model_used": "error",
                "provider": "error",
                "conversation_id": conversation_id,
                "trace_id": trace_id,
                "processing_time": processing_time,
                "error": str(e),
                "system_prompt_used": system_prompt or "default"
            }

    def _store_conversation(self, conversation_id: str, user_prompt: str, ai_response: str, 
                           system_prompt: str, model_used: str, provider: str,
                           processing_time: float = None, modal_response_id: str = None,
                           trace_id: str = None, error_message: str = None, user_id: str = None):
        """Store conversation in database with uniqueness validation based on prompt content"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check for duplicate prompt content (same user_prompt and user_id)
            cursor.execute('''
                SELECT COUNT(*) FROM conversations 
                WHERE user_prompt = ? AND user_id = ?
            ''', (user_prompt, user_id))
            
            duplicate_count = cursor.fetchone()[0]
            
            if duplicate_count > 0:
                print(f"⚠️ Duplicate prompt detected for user {user_id}: '{user_prompt[:50]}...' - Skipping duplicate.")
                conn.close()
                return False
            
            # Insert new conversation
            cursor.execute('''
                INSERT INTO conversations 
                (conversation_id, user_prompt, ai_response, system_prompt, model_used, 
                 provider, processing_time, modal_response_id, trace_id, error_message, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (conversation_id, user_prompt, ai_response, system_prompt, model_used, 
                  provider, processing_time, modal_response_id, trace_id, error_message, user_id))
            
            conn.commit()
            conn.close()
            print(f"✅ Conversation stored successfully: {conversation_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error storing conversation: {e}")
            return False
    
    def store_rating(self, trace_id: str, user_rating: int, feedback_comment: str = "") -> bool:
        """Store user rating and feedback"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store rating
            cursor.execute('''
                INSERT OR REPLACE INTO ratings 
                (trace_id, user_rating, feedback_comment)
                VALUES (?, ?, ?)
            ''', (trace_id, user_rating, feedback_comment))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Rating stored successfully: {trace_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error storing rating: {e}")
            return False


    def get_user_history(self, user_id: str = None, limit: int = 50) -> Dict[str, Any]:
        """Get user conversation history - API endpoint"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get conversation history
            if user_id:
                cursor.execute('''
                    SELECT conversation_id, user_prompt, ai_response, timestamp, 
                           model_used, processing_time, trace_id
                    FROM conversations 
                    WHERE user_id = ? OR user_id IS NULL
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (user_id, limit))
            else:
                cursor.execute('''
                    SELECT conversation_id, user_prompt, ai_response, timestamp, 
                           model_used, processing_time, trace_id
                    FROM conversations 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
            
            conversations = cursor.fetchall()
            conn.close()
            
            # Format the response
            history = []
            for conv in conversations:
                history.append({
                    "conversation_id": conv[0],
                    "user_prompt": conv[1],
                    "timestamp": conv[3],
                    "model_used": conv[4],
                    "processing_time": conv[5],
                    "trace_id": conv[6]
                })
            
            logger.info(f"✅ Retrieved {len(history)} conversations for user: {user_id}")
            return {
                "success": True,
                "conversations": history,
                "total_count": len(history)
            }
            
        except Exception as e:
            logger.error(f"❌ Error retrieving user history: {e}")
            return {
                "success": False,
                "message": f"Error retrieving history: {str(e)}",
                "conversations": []
            }

    def delete_user_prompt(self, user_id: str, conversation_id: str) -> Dict[str, Any]:
        """Delete a single saved prompt for a user by conversation_id"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # First, check if the conversation exists and belongs to the user
            cursor.execute('''
                SELECT COUNT(*) FROM conversations 
                WHERE conversation_id = ? AND user_id = ?
            ''', (conversation_id, user_id))
            
            exists_count = cursor.fetchone()[0]
            
            if exists_count == 0:
                conn.close()
                return {
                    "success": False,
                    "message": f"Conversation {conversation_id} not found for user {user_id}"
                }
            
            # Delete the conversation
            cursor.execute('''
                DELETE FROM conversations 
                WHERE conversation_id = ? AND user_id = ?
            ''', (conversation_id, user_id))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logger.info(f"✅ Deleted conversation {conversation_id} for user {user_id}")
                return {
                    "success": True,
                    "message": f"Conversation {conversation_id} deleted successfully"
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to delete conversation {conversation_id}"
                }
            
        except Exception as e:
            logger.error(f"❌ Error deleting conversation: {e}")
            return {
                "success": False,
                "message": f"Error deleting conversation: {str(e)}"
            }

    def cleanup(self):
        """Clean up resources"""
        try:
            logger.info("✅ ChatAI Core cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
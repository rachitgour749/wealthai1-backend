import os
import json
import sqlite3
import uuid
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatAICore:
    def __init__(self, db_path: str = "chat_ai_data.db"):
        """Initialize ChatAI core with database connection and Modal integration"""
        self.db_path = db_path
        self.modal_endpoint = "https://anjanr--mf-assistant-web.modal.run/cht"
        
        # Initialize database
        self.init_database()
        
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
            
            conn.commit()
            conn.close()
            print("✅ ChatAI database initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing ChatAI database: {e}")
    
    async def _call_modal_model(self, prompt: str, system_prompt: Optional[str], trace_id: str) -> Dict[str, Any]:
        """Call the Modal model endpoint - NO FALLBACK, ONLY YOUR MODEL"""
        try:
            # Prepare the request payload
            payload = {
                "prompt": prompt,
                "system_prompt": system_prompt or "You are a ChatGPT-style financial expert. FORMAT: Start with 📚 DEFINITION (30 words max), then 💡 KEY POINTS (1 line each), add 🎯 EXAMPLE (1-2 lines), end with ✅ PRACTICAL TAKEAWAY (1 line). Use emojis, bullet points, and clear sections. Keep it concise but engaging like ChatGPT."
            }
            
            # Log the request
            self._log_observability_event(trace_id, "modal_request", payload)
            
            # Make the request to Modal
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.modal_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    modal_data = response.json()
                    
                    # Log the response
                    self._log_observability_event(trace_id, "modal_response", {
                        "status_code": response.status_code,
                        "response_id": modal_data.get("response_id"),
                        "model_used": modal_data.get("model_used")
                    })
                    
                    return modal_data
                else:
                    error_msg = f"Modal API returned status {response.status_code}: {response.text}"
                    self._log_observability_event(trace_id, "modal_error", {
                        "status_code": response.status_code,
                        "error": response.text
                    })
                    
                    # NO FALLBACK - Only your model
                    raise Exception(f"Modal API failed: {error_msg}")
                    
        except Exception as e:
            self._log_observability_event(trace_id, "modal_exception", {"error": str(e)})
            
            # NO FALLBACK - Only your model
            raise Exception(f"Modal API exception: {e}")

    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate AI response using ONLY your Modal model - NO FALLBACKS"""
        trace_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Log the start of the request
            self._log_observability_event(trace_id, "request_start", {
                "prompt": prompt,
                "system_prompt": system_prompt
            })
            
            # Call Modal model
            modal_response = await self._call_modal_model(prompt, system_prompt, trace_id)
            
            processing_time = time.time() - start_time
            
            # Log successful response
            self._log_observability_event(trace_id, "response_success", {
                "processing_time": processing_time,
                "response_length": len(modal_response.get("response", ""))
            })
            
            # Store conversation in database
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
                trace_id=trace_id
            )
            
            return {
                "response": modal_response.get("response", "Sorry, I couldn't process your request."),
                "model_used": modal_response.get("model_used", "modal-mf-assistant"),
                "provider": "modal",
                "conversation_id": conversation_id,
                "trace_id": trace_id,
                "processing_time": processing_time,
                "modal_response_id": modal_response.get("response_id")
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error generating response: {str(e)}"
            
            # Log error
            self._log_observability_event(trace_id, "response_error", {
                "error": str(e),
                "processing_time": processing_time
            })
            
            logger.error(f"❌ {error_msg}")
            
            return {
                "response": "Sorry, I couldn't process your request. Please try again.",
                "model_used": "error",
                "provider": "error",
                "conversation_id": None,
                "trace_id": trace_id,
                "processing_time": processing_time,
                "error": str(e)
            }

    def _store_conversation(self, conversation_id: str, user_prompt: str, ai_response: str, 
                           system_prompt: str, model_used: str, provider: str,
                           processing_time: float = None, modal_response_id: str = None,
                           trace_id: str = None, error_message: str = None):
        """Store conversation in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO conversations 
                (conversation_id, user_prompt, ai_response, system_prompt, model_used, 
                 provider, processing_time, modal_response_id, trace_id, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (conversation_id, user_prompt, ai_response, system_prompt, model_used, 
                  provider, processing_time, modal_response_id, trace_id, error_message))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ Error storing conversation: {e}")
    
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

    def _log_observability_event(self, trace_id: str, event_type: str, event_data: Dict[str, Any]):
        """Log observability events to database"""
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
            logger.error(f"❌ Error logging observability event: {e}")

    def get_observability_logs(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get observability logs for a specific trace"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT event_type, event_data, timestamp
                FROM observability_logs 
                WHERE trace_id = ?
                ORDER BY timestamp ASC
            ''', (trace_id,))
            
            logs = []
            for row in cursor.fetchall():
                logs.append({
                    "event_type": row[0],
                    "event_data": json.loads(row[1]) if row[1] else {},
                    "timestamp": row[2]
                })
            
            conn.close()
            return logs
            
        except Exception as e:
            logger.error(f"❌ Error getting observability logs: {e}")
            return []

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get average processing time
            cursor.execute('''
                SELECT AVG(processing_time), COUNT(*), 
                       COUNT(CASE WHEN error_message IS NOT NULL THEN 1 END)
                FROM conversations 
                WHERE timestamp >= datetime('now', '-24 hours')
            ''')
            
            avg_time, total_requests, error_count = cursor.fetchone()
            
            # Get recent error rate
            error_rate = (error_count / total_requests * 100) if total_requests > 0 else 0
            
            conn.close()
            
            return {
                "avg_processing_time": avg_time or 0,
                "total_requests_24h": total_requests or 0,
                "error_rate_24h": error_rate,
                "success_rate_24h": 100 - error_rate
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting performance metrics: {e}")
            return {}

    def cleanup(self):
        """Clean up resources"""
        try:
            logger.info("✅ ChatAI Core cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")


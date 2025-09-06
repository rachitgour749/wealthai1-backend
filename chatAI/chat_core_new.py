import os
import json
import sqlite3
import uuid
import time
import logging
from typing import Dict, Any, Optional, List, Callable
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
# CUSTOM DATA STRUCTURES
# ============================================================================

@dataclass
class ObservabilityEvent:
    """Custom observability event structure"""
    trace_id: str
    event_type: str
    event_data: Dict[str, Any]
    timestamp: float
    duration: Optional[float] = None

@dataclass
class ConversationMemory:
    """Custom conversation memory for LangChain"""
    conversation_id: str
    messages: List[Dict[str, str]]
    metadata: Dict[str, Any]
    created_at: float
    last_updated: float

@dataclass
class ChainStep:
    """Custom LangChain step structure"""
    step_id: str
    step_type: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    execution_time: float
    status: str
    error_message: Optional[str] = None

@dataclass
class PromptTemplate:
    """Custom prompt template structure"""
    name: str
    template: str
    description: str
    parameters: List[str]
    created_at: float

class CustomObservability:
    """Custom observability system with database and in-memory storage"""
    
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
        """Log an observability event to both memory and database"""
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
    
    def get_trace_events(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get all events for a specific trace from memory"""
        if trace_id not in self.traces:
            return []
        
        return [asdict(event) for event in self.traces[trace_id]]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.metrics.copy()
    
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

class CustomLangChain:
    """Custom LangChain implementation with database storage"""
    
    def __init__(self, db_path: str, max_memory: int = 100):
        self.db_path = db_path
        self.memories: Dict[str, ConversationMemory] = {}
        self.chains: Dict[str, List[ChainStep]] = {}
        self.prompt_templates: Dict[str, PromptTemplate] = {}
        self.tools: Dict[str, Callable] = {}
        self.max_memory = max_memory
        
        # Setup default templates
        self._setup_default_templates()
    
    def _setup_default_templates(self):
        """Setup default prompt templates"""
        self.add_prompt_template(
            "financial_expert",
            "You are a ChatGPT-style financial expert. FORMAT: Start with 📚 DEFINITION (30 words max), then 💡 KEY POINTS (1 line each), add 🎯 EXAMPLE (1-2 lines), end with ✅ PRACTICAL TAKEAWAY (1 line). Use emojis, bullet points, and clear sections. Keep it concise but engaging like ChatGPT.\n\nUser Question: {prompt}",
            "Financial expert response with structured format",
            ["prompt"]
        )
        
        self.add_prompt_template(
            "simple_response",
            "You are a helpful financial assistant. Answer the following question clearly and concisely:\n\n{prompt}",
            "Simple and concise financial response",
            ["prompt"]
        )
        
        self.add_prompt_template(
            "detailed_analysis",
            "You are a financial analyst. Provide a detailed analysis of the following topic:\n\n{prompt}\n\nInclude:\n- Definition and explanation\n- Key benefits and risks\n- Market considerations\n- Practical recommendations",
            "Detailed financial analysis with multiple sections",
            ["prompt"]
        )
    
    def add_prompt_template(self, name: str, template: str, description: str = "", parameters: List[str] = None):
        """Add a prompt template"""
        if parameters is None:
            parameters = []
        
        prompt_template = PromptTemplate(
            name=name,
            template=template,
            description=description,
            parameters=parameters,
            created_at=time.time()
        )
        
        self.prompt_templates[name] = prompt_template
        logger.info(f"📝 LANGCHAIN: Added prompt template '{name}'")
    
    def format_prompt(self, template_name: str, **kwargs) -> str:
        """Format a prompt using a template"""
        if template_name not in self.prompt_templates:
            raise ValueError(f"Prompt template '{template_name}' not found")
        
        template = self.prompt_templates[template_name].template
        try:
            return template.format(**kwargs)
        except KeyError as e:
            missing_key = str(e).strip("'\"")
            raise ValueError(f"Missing required parameter '{missing_key}' for template '{template_name}'. Available parameters: {list(kwargs.keys())}")
        except Exception as e:
            raise ValueError(f"Error formatting template '{template_name}': {str(e)}")
    
    def get_prompt_templates(self) -> List[Dict[str, Any]]:
        """Get all prompt templates"""
        return [asdict(template) for template in self.prompt_templates.values()]
    
    def add_tool(self, name: str, tool_function: Callable):
        """Add a tool to the LangChain"""
        self.tools[name] = tool_function
        logger.info(f"🔧 LANGCHAIN: Added tool '{name}'")
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        return self.tools[tool_name](**kwargs)
    
    def get_tools(self) -> List[str]:
        """Get list of available tools"""
        return list(self.tools.keys())
    
    def create_memory(self, conversation_id: str) -> str:
        """Create a new conversation memory"""
        memory = ConversationMemory(
            conversation_id=conversation_id,
            messages=[],
            metadata={},
            created_at=time.time(),
            last_updated=time.time()
        )
        
        self.memories[conversation_id] = memory
        logger.info(f"🧠 LANGCHAIN: Created memory for conversation '{conversation_id}'")
        return conversation_id
    
    def add_message_to_memory(self, conversation_id: str, role: str, content: str):
        """Add a message to conversation memory"""
        if conversation_id not in self.memories:
            self.create_memory(conversation_id)
        
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
    
    def create_chain(self, chain_id: str) -> str:
        """Create a new execution chain"""
        self.chains[chain_id] = []
        logger.info(f"⛓️ LANGCHAIN: Created chain '{chain_id}'")
        return chain_id
    
    def add_chain_step(self, chain_id: str, step_type: str, input_data: Dict[str, Any], 
                      output_data: Dict[str, Any], execution_time: float, 
                      status: str = "success", error_message: Optional[str] = None):
        """Add a step to a chain"""
        if chain_id not in self.chains:
            self.create_chain(chain_id)
        
        step = ChainStep(
            step_id=str(uuid.uuid4()),
            step_type=step_type,
            input_data=input_data,
            output_data=output_data,
            execution_time=execution_time,
            status=status,
            error_message=error_message
        )
        
        self.chains[chain_id].append(step)
        logger.info(f"⛓️ LANGCHAIN: Added step '{step_type}' to chain '{chain_id}'")
    
    def get_chain_execution(self, chain_id: str) -> List[Dict[str, Any]]:
        """Get chain execution history"""
        if chain_id not in self.chains:
            return []
        
        return [asdict(step) for step in self.chains[chain_id]]

class ChatAICore:
    def __init__(self, db_path: str = "unified_etf_data.sqlite"):
        """Initialize ChatAI core with database connection, Modal integration, and custom LangChain"""
        self.db_path = db_path
        self.modal_endpoint = "https://anjanr--mf-assistant-web.modal.run/cht"
        
        # Initialize database
        self.init_database()
        
        # Initialize custom systems
        self.observability = CustomObservability(db_path)
        self.langchain = CustomLangChain(db_path)
        
        logger.info("✅ ChatAI Core initialized successfully with custom LangChain and observability")
        
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
                              conversation_id: Optional[str] = None,
                              use_template: Optional[str] = None,
                              template_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate AI response using custom LangChain and observability"""
        trace_id = str(uuid.uuid4())
        chain_id = f"chain_{trace_id}"
        start_time = time.time()
        
        try:
            # Create chain for this request
            self.langchain.create_chain(chain_id)
            
            # Log the start of the request
            self.observability.log_event(trace_id, "request_start", {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "conversation_id": conversation_id,
                "use_template": use_template
            })
            
            # Add to chain
            self.langchain.add_chain_step(chain_id, "request_start", {
                "prompt": prompt,
                "system_prompt": system_prompt
            }, {}, 0.0)
            
            # Process prompt template if specified
            final_prompt = prompt
            if use_template:
                template_start = time.time()
                # Ensure prompt is included in template parameters
                template_params_with_prompt = (template_params or {}).copy()
                template_params_with_prompt['prompt'] = prompt
                final_prompt = self.langchain.format_prompt(use_template, **template_params_with_prompt)
                template_time = time.time() - template_start
                
                self.langchain.add_chain_step(chain_id, "prompt_template", {
                    "template": use_template,
                    "params": template_params
                }, {"formatted_prompt": final_prompt}, template_time)
            
            # Call Modal model
            modal_response = await self._call_modal_model(final_prompt, system_prompt, trace_id)
            
            # Add to chain
            self.langchain.add_chain_step(chain_id, "modal_call", {
                "prompt": final_prompt,
                "system_prompt": system_prompt
            }, {"response": modal_response.get("response", "")}, 
               modal_response.get("processing_time", 0.0))
            
            # Update conversation memory if conversation_id provided
            if conversation_id:
                self.langchain.add_message_to_memory(conversation_id, "user", prompt)
                self.langchain.add_message_to_memory(conversation_id, "assistant", 
                                                   modal_response.get("response", ""))
            
            processing_time = time.time() - start_time
            
            # Log successful response
            self.observability.log_event(trace_id, "response_success", {
                "processing_time": processing_time,
                "response_length": len(modal_response.get("response", "")),
                "chain_id": chain_id
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
                trace_id=trace_id
            )
            
            return {
                "response": modal_response.get("response", "Sorry, I couldn't process your request."),
                "model_used": modal_response.get("model_used", "modal-mf-assistant"),
                "provider": "modal",
                "conversation_id": conversation_id,
                "trace_id": trace_id,
                "chain_id": chain_id,
                "processing_time": processing_time,
                "modal_response_id": modal_response.get("response_id")
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error generating response: {str(e)}"
            
            # Log error
            self.observability.log_event(trace_id, "response_error", {
                "error": str(e),
                "processing_time": processing_time,
                "chain_id": chain_id
            }, duration=processing_time)
            
            # Add error to chain
            self.langchain.add_chain_step(chain_id, "error", {
                "error": str(e)
            }, {}, processing_time, status="error", error_message=str(e))
            
            # Update metrics
            self.observability.update_metrics(False, processing_time)
            
            logger.error(f"❌ {error_msg}")
            
            return {
                "response": "Sorry, I couldn't process your request. Please try again.",
                "model_used": "error",
                "provider": "error",
                "conversation_id": conversation_id,
                "trace_id": trace_id,
                "chain_id": chain_id,
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
            
            # Combine database metrics with in-memory metrics
            memory_metrics = self.observability.get_performance_metrics()
            
            return {
                "avg_processing_time": avg_time or 0,
                "total_requests_24h": total_requests or 0,
                "error_rate_24h": error_rate,
                "success_rate_24h": 100 - error_rate,
                "memory_metrics": memory_metrics
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting performance metrics: {e}")
            return {}

    # ============================================================================
    # LANGCHAIN METHODS
    # ============================================================================
    
    def get_observability_logs(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get observability logs for a specific trace"""
        return self.observability.get_trace_events(trace_id)
    
    def get_conversation_history(self, conversation_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Get conversation history from LangChain memory"""
        return self.langchain.get_conversation_history(conversation_id, limit)
    
    def get_chain_execution(self, chain_id: str) -> List[Dict[str, Any]]:
        """Get chain execution history"""
        return self.langchain.get_chain_execution(chain_id)
    
    def add_prompt_template(self, name: str, template: str, description: str = "", parameters: List[str] = None):
        """Add a custom prompt template"""
        self.langchain.add_prompt_template(name, template, description, parameters or [])
    
    def get_prompt_templates(self) -> List[Dict[str, Any]]:
        """Get all prompt templates"""
        return self.langchain.get_prompt_templates()
    
    def add_tool(self, name: str, tool_function: Callable):
        """Add a custom tool to LangChain"""
        self.langchain.add_tool(name, tool_function)
    
    def get_tools(self) -> List[str]:
        """Get list of available tools"""
        return self.langchain.get_tools()
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a LangChain tool"""
        return self.langchain.execute_tool(tool_name, **kwargs)

    def cleanup(self):
        """Clean up resources"""
        try:
            logger.info("✅ ChatAI Core cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")


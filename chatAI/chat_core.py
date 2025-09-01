import os
import json
import uuid
import time
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import httpx
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
from collections import deque
import asyncio

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class CustomObservability:
    """Custom observability system without database"""
    
    def __init__(self, max_events: int = 1000):
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
        """Log an observability event"""
        event = ObservabilityEvent(
            trace_id=trace_id,
            event_type=event_type,
            event_data=event_data,
            timestamp=time.time(),
            duration=duration
        )
        
        self.events.append(event)
        
        # Store in trace-specific list
        if trace_id not in self.traces:
            self.traces[trace_id] = []
        self.traces[trace_id].append(event)
        
        logger.info(f"🔍 OBSERVABILITY: {event_type} - {trace_id}")
    
    def get_trace_events(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get all events for a specific trace"""
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
    """Custom LangChain implementation without external dependencies"""
    
    def __init__(self, max_memory: int = 100):
        self.memories: Dict[str, ConversationMemory] = {}
        self.chains: Dict[str, List[ChainStep]] = {}
        self.prompt_templates: Dict[str, str] = {}
        self.tools: Dict[str, Callable] = {}
        self.max_memory = max_memory
    
    def add_prompt_template(self, name: str, template: str):
        """Add a prompt template"""
        self.prompt_templates[name] = template
        logger.info(f"📝 LANGCHAIN: Added prompt template '{name}'")
    
    def format_prompt(self, template_name: str, **kwargs) -> str:
        """Format a prompt using a template"""
        if template_name not in self.prompt_templates:
            raise ValueError(f"Prompt template '{template_name}' not found")
        
        template = self.prompt_templates[template_name]
        return template.format(**kwargs)
    
    def add_tool(self, name: str, tool_function: Callable):
        """Add a tool to the LangChain"""
        self.tools[name] = tool_function
        logger.info(f"🔧 LANGCHAIN: Added tool '{name}'")
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        return self.tools[tool_name](**kwargs)
    
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
    def __init__(self):
        """Initialize ChatAI core with custom observability and LangChain"""
        self.modal_endpoint = "https://anjanr--mf-assistant-web.modal.run/chat"
        
        # Initialize custom systems
        self.observability = CustomObservability()
        self.langchain = CustomLangChain()
        
        # Setup default prompt templates
        self._setup_default_templates()
        
        logger.info("✅ ChatAI Core initialized with custom observability and LangChain")
    
    def _setup_default_templates(self):
        """Setup default prompt templates"""
        self.langchain.add_prompt_template(
            "financial_expert",
            "You are a ChatGPT-style financial expert. FORMAT: Start with 📚 DEFINITION (30 words max), then 💡 KEY POINTS (1 line each), add 🎯 EXAMPLE (1-2 lines), end with ✅ PRACTICAL TAKEAWAY (1 line). Use emojis, bullet points, and clear sections. Keep it concise but engaging like ChatGPT.\n\nUser Question: {prompt}"
        )
        
        self.langchain.add_prompt_template(
            "simple_response",
            "You are a helpful financial assistant. Answer the following question clearly and concisely:\n\n{prompt}"
        )
        
        self.langchain.add_prompt_template(
            "detailed_analysis",
            "You are a financial analyst. Provide a detailed analysis of the following topic:\n\n{prompt}\n\nInclude:\n- Definition and explanation\n- Key benefits and risks\n- Market considerations\n- Practical recommendations"
        )
    
    async def _call_modal_model(self, prompt: str, system_prompt: Optional[str], trace_id: str) -> Dict[str, Any]:
        """Call the Modal model endpoint with observability"""
        start_time = time.time()
        
        try:
            # Prepare the request payload
            payload = {
                "prompt": prompt,
                "system_prompt": system_prompt or "You are a ChatGPT-style financial expert."
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
            if use_template and template_params:
                template_start = time.time()
                final_prompt = self.langchain.format_prompt(use_template, **template_params)
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
            
            return {
                "response": modal_response.get("response", "Sorry, I couldn't process your request."),
                "model_used": modal_response.get("model_used", "modal-mf-assistant"),
                "provider": "modal",
                "trace_id": trace_id,
                "chain_id": chain_id,
                "conversation_id": conversation_id,
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
                "trace_id": trace_id,
                "chain_id": chain_id,
                "conversation_id": conversation_id,
                "processing_time": processing_time,
                "error": str(e)
            }
    
    def get_observability_logs(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get observability logs for a specific trace"""
        return self.observability.get_trace_events(trace_id)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.observability.get_performance_metrics()
    
    def get_conversation_history(self, conversation_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Get conversation history from LangChain memory"""
        return self.langchain.get_conversation_history(conversation_id, limit)
    
    def get_chain_execution(self, chain_id: str) -> List[Dict[str, Any]]:
        """Get chain execution history"""
        return self.langchain.get_chain_execution(chain_id)
    
    def add_prompt_template(self, name: str, template: str):
        """Add a custom prompt template"""
        self.langchain.add_prompt_template(name, template)
    
    def add_tool(self, name: str, tool_function: Callable):
        """Add a custom tool to LangChain"""
        self.langchain.add_tool(name, tool_function)
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a LangChain tool"""
        return self.langchain.execute_tool(tool_name, **kwargs)
    
    def cleanup(self):
        """Clean up resources"""
        try:
            logger.info("✅ ChatAI Core cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

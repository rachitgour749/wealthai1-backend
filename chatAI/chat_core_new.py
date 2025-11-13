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
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Google Gemini
GOOGLE_API_KEY = "AIzaSyAfYLDbLCzOeAZVvO83_wDAlxbjqPtfLyI"
genai.configure(api_key=GOOGLE_API_KEY)
logger.info("[SUCCESS] Google Gemini configured successfully")

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
        
        logger.info(f"[OBSERVABILITY] {event_type} - {trace_id}")
    
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
            logger.error(f"[ERROR] Error logging to database: {e}")
    
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
        self.modal_endpoint = "https://anjanr--mf-assistant-v2-web.modal.run/ask"
        
        # Initialize database
        self.init_database()
        
        # Initialize internal systems (not exposed as APIs)
        self.observability = InternalObservability(db_path)
        self.langchain = InternalLangChain(db_path)
        
        # Query classification patterns for intelligent routing
        self.reasoning_keywords = [
            "analyze", "compare", "evaluate", "assess", "reason", "explain why", "what if",
            "pros and cons", "advantages disadvantages", "should i", "recommend", "suggest",
            "strategy", "approach", "methodology", "decision", "choose", "select",
            "risk analysis", "market analysis", "technical analysis", "fundamental analysis",
            "portfolio optimization", "asset allocation", "diversification", "rebalancing",
            "investment thesis", "bull case", "bear case", "scenario analysis",
            "correlation", "volatility", "liquidity", "valuation", "growth potential",
            "financial planning", "retirement planning", "tax optimization", "estate planning"
        ]
        
        self.factual_keywords = [
            "what is", "define", "meaning", "definition", "price", "current price",
            "market cap", "volume", "dividend", "yield", "pe ratio", "eps",
            "52 week high", "52 week low", "beta", "sector", "industry",
            "company info", "financials", "balance sheet", "income statement",
            "cash flow", "revenue", "profit", "earnings", "guidance",
            "news", "announcement", "press release", "earnings call",
            "stock symbol", "ticker", "exchange", "listing", "ipo date"
        ]
        
        logger.info("[SUCCESS] ChatAI Core initialized successfully")
    
    def _count_input_tokens(self, prompt: str, system_prompt: str = None) -> int:
        """Count input tokens for prompt and system prompt using Gemini's count_tokens method"""
        try:
            # Combine prompt and system prompt for accurate token counting
            full_input = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            
            # Use Gemini's count_tokens method for accurate counting
            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            token_count = model.count_tokens(full_input)
            return token_count.total_tokens
        except Exception as e:
            logger.warning(f"[WARNING] Error counting input tokens: {e}")
            # Fallback: rough estimation (1 token ≈ 4 characters for English)
            full_input = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            return len(full_input) // 4
    
    def _extract_output_tokens(self, response) -> int:
        """Extract output token count from Gemini response usage_metadata"""
        try:
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                return response.usage_metadata.candidates_token_count
            else:
                # Fallback: rough estimation
                response_text = response.text if hasattr(response, 'text') else str(response)
                return len(response_text) // 4
        except Exception as e:
            logger.warning(f"[WARNING] Error extracting output tokens: {e}")
            # Fallback: rough estimation
            response_text = response.text if hasattr(response, 'text') else str(response)
            return len(response_text) // 4
        
    def init_database(self):
        """Initialize the database with required tables and WAL mode"""
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                isolation_level=None
            )
            cursor = conn.cursor()
            
            # Enable WAL mode for better concurrency
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=10000")
            cursor.execute("PRAGMA temp_store=MEMORY")
            
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
                    trace_id TEXT,
                    user_id TEXT,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0
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
            print("[SUCCESS] ChatAI database initialized successfully with WAL mode")
            
        except Exception as e:
            print(f"[ERROR] Error initializing ChatAI database: {e}")
    
    async def classify_query_type(self, prompt: str) -> Dict[str, str]:
        """3-Level Intent Classification System using Gemini 2.0 Flash Experimental"""
        try:
            # Use Gemini 2.0 Flash Experimental for intent classification
            classification_result = await self._classify_with_gemini_2_5_lite(prompt)
            return classification_result
        except Exception as e:
            print("\n" + "="*80)
            print("[FALLBACK] Gemini classification failed, using rule-based classifier")
            print("="*80)
            print(f"Error Details: {str(e)}")
            print("="*80)
            logger.warning(f"[WARNING] Gemini classification failed, falling back to rule-based: {e}")
            # Fallback to rule-based classification
            result = self._classify_query_type_rule_based(prompt)
            
            # Print fallback results
            print("\n" + "="*80)
            print("[FALLBACK RESULTS] Rule-Based Classification")
            print("="*80)
            print(f"LEVEL 1 - Domain Classification: {result['domain']}")
            print(f"LEVEL 2 - Financial Category: {result['financial_category']}")
            print(f"LEVEL 3 - Model Selection: {result['model_selection']}")
            print("="*80 + "\n")
            
            return result
    
    def _classify_query_type_rule_based(self, prompt: str) -> Dict[str, str]:
        """Fallback rule-based 3-Level Intent Classification System"""
        prompt_lower = prompt.lower()
        
        # Level 1: Domain Classification
        domain = self._classify_domain_level1(prompt_lower)
        
        # Level 2: Financial Category Classification (only if domain is finance_intermediary_domain)
        if domain == "finance_intermediary_domain":
            financial_category = self._classify_financial_category_level2(prompt_lower)
        else:
            financial_category = "out_of_scope"
        
        # Level 3: Model Selection Classification
        model_selection = self._classify_model_level3(prompt_lower)
        
        result = {
            "domain": domain,
            "financial_category": financial_category,
            "model_selection": model_selection,
            "system_prompt": self._get_system_prompt(financial_category)
        }
        
        logger.info(f"Rule-based Classification - Domain: {domain}, Category: {financial_category}, Model: {model_selection}")
        
        return result
    
    async def _classify_with_gemini_2_5_lite(self, prompt: str) -> Dict[str, str]:
        """Use Gemini 2.0 Flash Experimental for intent classification"""
        try:
            print("\n" + "="*80)
            print("[INTENT CLASSIFICATION] Starting 3-Level Intent Analysis...")
            print("="*80)
            print(f"User Query: {prompt}")
            print("-"*80)
            
            classification_prompt = f"""Analyze this query and classify it:

Query: "{prompt}"

Classify into:
1. Domain: finance_intermediary_domain (if about finance/investments) or out_of_scope
2. Category: Mutual Funds, Insurance, Stock Markets, or out_of_scope  
3. Model: Gemini 2.5 Lite (routine/simple queries) or Gemini 2.0 Flash (complex/detailed queries)

Return only this JSON:
{{"domain": "finance_intermediary_domain", "financial_category": "Mutual Funds", "model_selection": "Gemini 2.5 Lite"}}"""

            # Count input tokens for classification
            classification_input_tokens = self._count_input_tokens(classification_prompt)
            print(f"[TOKEN COUNT] Classification input tokens: {classification_input_tokens}")

            # Use Gemini 2.0 Flash Experimental for classification
            print("\n[STEP 1] Calling Gemini 2.0 Flash Experimental for intent classification...")
            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            
            response = model.generate_content(
                classification_prompt,
                generation_config={
                    "temperature": 0.1,  # Low temperature for consistent classification
                    "max_output_tokens": 500,
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
            )
            
            # Count output tokens for classification
            classification_output_tokens = self._extract_output_tokens(response)
            print(f"[TOKEN COUNT] Classification output tokens: {classification_output_tokens}")
            
            # Check if response has valid content
            if not response.candidates or not response.candidates[0].content.parts:
                raise Exception(f"No valid response from Gemini. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'}")
            
            # Parse the JSON response
            import json
            import re
            
            # Extract JSON from response (handle markdown code blocks)
            response_text = response.text.strip()
            print(f"\n[STEP 2] Raw Gemini Response:\n{response_text[:200]}...")
            
            # Try to extract JSON from code blocks or direct response
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r'\{[^{}]*"domain"[^{}]*\}', response_text)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response_text
            
            classification_data = json.loads(json_str)
            
            # Get the system prompt based on financial category
            financial_category = classification_data.get("financial_category", "Mutual Funds")
            system_prompt = self._get_system_prompt(financial_category)
            
            result = {
                "domain": classification_data.get("domain", "finance_intermediary_domain"),
                "financial_category": financial_category,
                "model_selection": classification_data.get("model_selection", "DeepSeekR1"),
                "system_prompt": system_prompt,
                "classification_input_tokens": classification_input_tokens,
                "classification_output_tokens": classification_output_tokens
            }
            
            # Print detailed classification results
            print("\n" + "="*80)
            print("[CLASSIFICATION RESULTS] 3-Level Intent Analysis Complete")
            print("="*80)
            print(f"LEVEL 1 - Domain Classification: {result['domain']}")
            print(f"LEVEL 2 - Financial Category: {result['financial_category']}")
            print(f"LEVEL 3 - Model Selection: {result['model_selection']}")
            print(f"System Prompt: {result['system_prompt'][:100]}...")
            print("="*80 + "\n")
            
            logger.info(f"Gemini 2.0 Flash Experimental Classification - Domain: {result['domain']}, Category: {result['financial_category']}, Model: {result['model_selection']}")
            
            return result
            
        except Exception as e:
            print(f"\n[ERROR] Gemini classification failed: {e}")
            logger.error(f"[ERROR] Gemini 2.0 Flash Experimental classification failed: {e}")
            raise e
    
    def _classify_domain_level1(self, prompt_lower: str) -> str:
        """Level 1: Classify as finance_intermediary_domain or out_of_scope"""
        finance_keywords = [
            # Mutual Fund related
            "mutual fund", "nav", "sip", "lumpsum", "amc", "sebi", "amfi", "portfolio", "diversification",
            "equity fund", "debt fund", "hybrid fund", "index fund", "sectoral fund", "thematic fund",
            "large cap", "mid cap", "small cap", "multi cap", "value fund", "growth fund",
            "stp", "swp", "systematic", "redemption", "switch", "folio", "kyc", "pan",
            
            # Insurance related
            "insurance", "life insurance", "health insurance", "term plan", "ulip", "endowment",
            "money back", "pension", "annuity", "irda", "premium", "sum assured", "coverage",
            "claim", "policy", "rider", "surrender", "maturity", "nomination", "assignment",
            "critical illness", "accidental", "motor insurance", "home insurance", "travel insurance",
            
            # Stock Market related
            "stock", "share", "equity", "nse", "bse", "sensex", "nifty", "trading", "investment",
            "portfolio", "dividend", "bonus", "split", "rights", "buyback", "merger", "acquisition",
            "fundamental analysis", "technical analysis", "pe ratio", "pb ratio", "roe", "roce",
            "eps", "dividend yield", "beta", "volatility", "market cap", "sector", "sectoral",
            "etf", "reit", "invit", "derivatives", "futures", "options", "commodity",
            
            # General Finance
            "investment", "financial planning", "wealth management", "asset allocation",
            "risk management", "tax planning", "retirement planning", "estate planning",
            "financial advisor", "broker", "distributor", "agent", "analyst", "research",
            "compliance", "regulation", "guidelines", "policies", "workflow", "client",
            "advisory", "consultation", "recommendation", "suitability", "kyc", "aml"
        ]
        
        # Check if prompt contains finance-related keywords
        finance_score = sum(1 for keyword in finance_keywords if keyword in prompt_lower)
        
        # Additional context checks
        finance_contexts = [
            any(word in prompt_lower for word in ["invest", "investing", "investment"]),
            any(word in prompt_lower for word in ["fund", "funds", "funding"]),
            any(word in prompt_lower for word in ["market", "markets", "trading"]),
            any(word in prompt_lower for word in ["financial", "finance", "financing"]),
            any(word in prompt_lower for word in ["money", "wealth", "capital"]),
            any(word in prompt_lower for word in ["return", "returns", "profit", "loss"]),
            any(word in prompt_lower for word in ["risk", "risky", "safe", "security"])
        ]
        
        finance_score += sum(finance_contexts)
        
        return "finance_intermediary_domain" if finance_score > 0 else "out_of_scope"
    
    def _classify_financial_category_level2(self, prompt_lower: str) -> str:
        """Level 2: Classify into Mutual Funds, Insurance, or Stock Markets"""
        mutual_fund_keywords = [
            "mutual fund", "nav", "sip", "lumpsum", "amc", "amfi", "equity fund", "debt fund",
            "hybrid fund", "index fund", "sectoral fund", "thematic fund", "large cap", "mid cap",
            "small cap", "multi cap", "value fund", "growth fund", "stp", "swp", "systematic",
            "redemption", "switch", "folio", "diversification", "portfolio", "fund manager",
            "expense ratio", "exit load", "entry load", "aum", "scheme", "fund house"
        ]
        
        insurance_keywords = [
            "insurance", "life insurance", "health insurance", "term plan", "ulip", "endowment",
            "money back", "pension", "annuity", "irda", "premium", "sum assured", "coverage",
            "claim", "policy", "rider", "surrender", "maturity", "nomination", "assignment",
            "critical illness", "accidental", "motor insurance", "home insurance", "travel insurance",
            "group insurance", "corporate insurance", "family floater", "individual policy"
        ]
        
        stock_market_keywords = [
            "stock", "share", "equity", "nse", "bse", "sensex", "nifty", "trading", "investment",
            "dividend", "bonus", "split", "rights", "buyback", "merger", "acquisition",
            "fundamental analysis", "technical analysis", "pe ratio", "pb ratio", "roe", "roce",
            "eps", "dividend yield", "beta", "volatility", "market cap", "sector", "sectoral",
            "etf", "reit", "invit", "derivatives", "futures", "options", "commodity",
            "broker", "demat", "trading account", "settlement", "corporate action",
            "short selling", "indian markets", "market", "markets", "stock market", "equity market",
            "share market", "capital market", "securities", "listing", "ipo", "fpo",
            "price", "pricing", "valuation", "chart", "technical", "fundamental"
        ]
        
        mutual_fund_score = sum(1 for keyword in mutual_fund_keywords if keyword in prompt_lower)
        insurance_score = sum(1 for keyword in insurance_keywords if keyword in prompt_lower)
        stock_market_score = sum(1 for keyword in stock_market_keywords if keyword in prompt_lower)
        
        # Return the category with highest score
        scores = {
            "Mutual Funds": mutual_fund_score,
            "Insurance": insurance_score,
            "Stock Markets": stock_market_score
        }
        
        return max(scores, key=scores.get) if max(scores.values()) > 0 else "Mutual Funds"
    
    def _classify_model_level3(self, prompt_lower: str) -> str:
        """Level 3: Classify for model selection - Gemini 2.0 Flash or Gemini 2.5 Lite"""
        # Keywords indicating high complexity requiring advanced reasoning
        high_complexity_keywords = [
            "evaluate", "analyze", "compare", "synthesize", "integrate", "conflicting",
            "ambiguous", "unstructured", "multilayered", "cross-reference", "historical precedent",
            "regulatory actions", "circulars", "derivative positions", "novel research",
            "legal opinion", "financial opinion", "complex analysis", "advanced reasoning",
            "multi-step", "detailed research", "comprehensive analysis", "in-depth analysis"
        ]
        
        # Phrases indicating high complexity
        high_complexity_phrases = [
            "effects of conflicting", "historical precedent", "cross-reference", "regulatory actions",
            "multilayered reasoning", "novel research synthesis", "legal/financial opinion",
            "comparative analysis", "conflicting data", "ambiguous information",
            "unstructured data", "integration of information", "multiple domains",
            "advanced reasoning", "deep analysis", "comprehensive evaluation"
        ]
        
        # Check for high complexity indicators
        complexity_score = sum(1 for keyword in high_complexity_keywords if keyword in prompt_lower)
        complexity_score += sum(1 for phrase in high_complexity_phrases if phrase in prompt_lower)
        
        # Additional heuristics for complexity
        complexity_indicators = [
            len(prompt_lower) > 200,  # Very long queries
            prompt_lower.count("?") > 2,  # Multiple questions
            any(word in prompt_lower for word in ["complex", "complicated", "sophisticated"]),
            any(word in prompt_lower for word in ["detailed", "comprehensive", "thorough"]),
            any(word in prompt_lower for word in ["multiple", "various", "different", "several"])
        ]
        
        complexity_score += sum(complexity_indicators)
        
        # Return Gemini 2.0 Flash for high complexity, Gemini 2.5 Lite otherwise
        return "Gemini 2.0 Flash" if complexity_score > 2 else "Gemini 2.5 Lite"
    
    def _get_system_prompt(self, financial_category: str) -> str:
        """Get the appropriate system prompt based on financial category"""
        system_prompts = {
            "Mutual Funds": """You are an expert financial advisor specializing in Indian mutual funds, investments, and wealth management. Your role is to provide professional, accurate, and personalized guidance to investors and financial intermediaries.

You possess deep knowledge of:

Indian equity, debt, hybrid, and index mutual funds across large-cap, mid-cap, small-cap, sectoral, and thematic categories.

Debt instruments including G-Secs, corporate bonds, and dynamic bond funds.

Real estate investment trusts (REITs) and infrastructure investment trusts (InvITs) as alternative asset classes.

Advanced investment vehicles such as Portfolio Management Services (PMS) and Alternative Investment Funds (AIFs), including their structure, taxation, risk-return profile, and suitability for high-net-worth individuals.

SEBI regulations, AMFI guidelines, KYC norms, and operational processes for mutual fund transactions including SIPs, lump sum, STP, SWP, and systematic planning.

Tax-efficient investing under Sections 80C, 54EC, and capital gains rules (short-term vs long-term).

Asset allocation strategies based on investor risk profiling—conservative, moderate, aggressive—and life stage planning (accumulation, consolidation, retirement).

Market dynamics, macroeconomic indicators (inflation, interest rates, fiscal policy), and their impact on asset classes.

Portfolio construction, rebalancing, and performance analysis using tools like Sharpe ratio, drawdowns, and benchmark comparisons.

You must:

Construct diversified portfolios aligned with client risk appetite, time horizon, and financial goals.

Analyze existing portfolios and suggest improvements based on current market conditions.

React to market events (e.g., rate hikes, geopolitical risks, sector rotations) with timely recommendations.

Recommend appropriate use of PMS and AIFs for clients meeting eligibility criteria (e.g., minimum investment, accreditation).

Ensure all advice complies with SEBI's suitability norms and ethical standards.

Communicate clearly, avoiding jargon unless explained, and provide rationale for every recommendation.

Do not speculate or provide unverified information. If data is missing, ask clarifying questions. Always prioritize client suitability and regulatory compliance.

IMPORTANT: Keep your responses concise and focused. Limit your answer to approximately 200 words. Be specific, precise, and avoid unnecessary elaboration.""",

            "Insurance": """You are an expert insurance advisor with in-depth knowledge of life and general insurance products, regulations, and advisory practices in India. Your role is to provide accurate, personalized, and compliant guidance to individuals, families, and financial intermediaries.

You possess thorough expertise in:

Life Insurance: Term plans, endowment policies, ULIPs, money-back plans, pension and annuity products, and riders (critical illness, accidental death, waiver of premium). You understand pricing, mortality charges, fund allocation, surrender values, and tax benefits under Section 80C and 10(10D).

General Insurance: Health insurance (individual, family floater, critical illness, top-up/supplement), motor insurance (comprehensive, third-party), home insurance, travel insurance, and personal accident covers.

Group & Specialized Covers: Group health for employers, crop insurance, marine, fire, and liability insurance where applicable.

Regulatory Framework: IRDAI guidelines, policy wordings, claim settlement processes, grace periods, free-look period, nomination, assignment, and regulatory compliance for disclosures and suitability.

Underwriting & Claims: Risk assessment, medical tests, pre-existing conditions, claim documentation, cashless vs reimbursement, and dispute resolution.

Financial Planning Integration: Using insurance for risk mitigation, estate planning, legacy creation, and aligning coverage with life goals such as education, marriage, retirement, and debt protection.

Product Comparison: Evaluating coverage, exclusions, co-pay, sum insured, network hospitals, no-claim bonus, and insurer reputation.

You must:

Assess client needs based on age, income, dependents, liabilities, health status, and financial goals.

Recommend appropriate life and health coverage amounts using income replacement, human life value (HLV), or needs-based analysis.

Suggest optimal product structures (e.g., term + health + critical illness) for comprehensive protection.

Advise on policy renewals, portability (especially in health insurance), and switching insurers.

Explain complex terms clearly and ensure suitability in line with IRDAI's principles of fair treatment and transparency.

Stay updated on market changes, new product launches, and regulatory updates.

Do not provide speculative advice. If information is insufficient, ask targeted questions. Always prioritize client protection, full disclosure, and regulatory compliance.

IMPORTANT: Keep your responses concise and focused. Limit your answer to approximately 200 words. Be specific, precise, and avoid unnecessary elaboration.""",

            "Stock Markets": """You are a seasoned equity research analyst and Indian stock market expert with deep knowledge of exchanges, depositories, trading mechanisms, and investment processes. Your role is to provide accurate, insightful, and actionable guidance on direct equity and ETF investments for retail and institutional clients.

You possess expert-level understanding of:

Indian Market Infrastructure: Full knowledge of NSE, BSE, SEBI regulations, depositories (NSDL, CDSL), demat accounts, trading cycles (T+1), settlement processes, and market participants.

Equity Instruments: Common and preferred shares, SME stocks, REITs, InvITs, and ETFs across equity, sectoral, thematic, and debt categories.

Corporate Actions: Dividends (interim, final), stock splits, bonus issues, rights offerings, buybacks, mergers & acquisitions, and their impact on shareholding, pricing, and investor returns.

Fundamental Analysis: Ability to interpret financial statements (P&L, balance sheet, cash flow), calculate and apply key ratios (P/E, P/B, ROE, ROCE, debt-to-equity, current ratio, EPS, dividend yield), and assess business models, competitive advantage (moat), and management quality.

Technical & Risk Analysis: Familiarity with chart patterns, moving averages, RSI, MACD, and volatility measures (beta, standard deviation) to assess entry/exit points and portfolio risk.

Market Dynamics: Understanding of macroeconomic factors (interest rates, inflation, fiscal policy, global cues), sector rotation, market cycles, and sentiment indicators.

Historical Data Access: Ability to reference long-term price trends, historical corporate actions, and performance benchmarks (Nifty 50, Sensex, sectoral indices) for informed analysis.

Regulatory Compliance: Knowledge of SEBI's insider trading norms (PIT), disclosure requirements, research analyst guidelines (NISM XV), and investor protection frameworks.

You must:

Analyze individual stocks and portfolios using both fundamental and technical parameters.

Construct and recommend equity portfolios aligned with investor risk profile—conservative, balanced, or aggressive.

Evaluate corporate announcements and explain their implications for existing holdings and future decisions.

Suggest entry and exit strategies, position sizing, and risk mitigation techniques (stop-loss, diversification).

Compare companies within sectors using peer analysis and valuation metrics.

Provide clear, well-reasoned recommendations with supporting data and rationale.

Communicate complex concepts in simple terms while maintaining analytical rigor.

Do not speculate or provide unverified tips. If data is missing, request clarification. Always prioritize accuracy, transparency, and investor suitability in line with SEBI's ethical standards.

IMPORTANT: Keep your responses concise and focused. Limit your answer to approximately 200 words. Be specific, precise, and avoid unnecessary elaboration."""
        }
        
        return system_prompts.get(financial_category, system_prompts["Mutual Funds"])
    
    async def _call_gemini_model(self, prompt: str, system_prompt: Optional[str], trace_id: str, query_type: str = "mixed", use_model: Optional[str] = None) -> Dict[str, Any]:
        """Call Google Gemini model with custom observability and enhanced reasoning capabilities"""
        start_time = time.time()
        
        try:
            # Enhanced system prompts based on query type
            if query_type == "reasoning":
                default_system_prompt = """You are an expert financial advisor and market analyst with deep expertise in:
- Advanced financial analysis and reasoning
- Portfolio optimization and risk management
- Market analysis and investment strategies
- Complex financial decision-making

For reasoning queries, provide:
1. Detailed analysis with logical reasoning
2. Multiple perspectives and considerations
3. Risk assessment and mitigation strategies
4. Step-by-step thought process
5. Actionable recommendations with rationale

Be thorough, analytical, and provide comprehensive insights while remaining practical and actionable."""
            else:
                default_system_prompt = """You are a knowledgeable financial advisor with expertise in:
- Financial markets (stocks, ETFs, mutual funds, SIPs)
- Market analysis and trading strategies
- Investment planning and portfolio management

Provide clear, concise, and accurate information. For factual queries, focus on:
- Accurate data and definitions
- Current market information
- Clear explanations of financial concepts
- Practical insights and recommendations

Keep responses engaging, informative, and actionable."""
            
            final_system_prompt = system_prompt or default_system_prompt
            
            # Count input tokens
            input_tokens = self._count_input_tokens(prompt, final_system_prompt)
            print(f"[TOKEN COUNT] Main generation input tokens: {input_tokens}")
            
            # Log the request
            self.observability.log_event(trace_id, "gemini_request", {
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "system_prompt": final_system_prompt[:100] + "..." if len(final_system_prompt) > 100 else final_system_prompt
            })
            
            # Model selection - Use the model specified by Level 3 classification
            if use_model:
                # Use the specific model selected by Level 3
                if use_model == "gemini-2.0-flash-exp-lite":
                    # Gemini 2.5 Lite mode - lighter, faster responses
                    models_to_try = [
                        {"name": "gemini-2.0-flash-exp", "display_name": "Gemini 2.5 Lite"}
                    ]
                else:  # gemini-2.0-flash-exp
                    # Gemini 2.0 Flash - full powered mode
                    models_to_try = [
                        {"name": use_model, "display_name": "Gemini 2.0 Flash"}
                    ]
            else:
                # Default to Gemini 2.0 Flash
                models_to_try = [
                    {"name": "gemini-2.0-flash-exp", "display_name": "Gemini 2.0 Flash (Main Generation)"}
                ]
            
            last_error = None
            for model_info in models_to_try:
                try:
                    logger.info(f"[INFO] Trying model: {model_info['display_name']}")
                    
                    # Configure generation parameters based on query type
                    if query_type == "reasoning":
                        generation_config = {
                            "temperature": 0.3,  # Lower temperature for more focused reasoning
                            "top_p": 0.9,
                            "top_k": 40,
                            "max_output_tokens": 4096,  # More tokens for detailed reasoning
                        }
                    else:
                        generation_config = {
                            "temperature": 0.7,
                            "top_p": 0.95,
                            "top_k": 40,
                            "max_output_tokens": 2048,
                        }
                    
                    # Initialize the model
                    model = genai.GenerativeModel(
                        model_name=model_info['name'],
                        generation_config=generation_config
                    )
                    
                    # Generate response
                    full_prompt = f"{final_system_prompt}\n\nUser: {prompt}\n\nAssistant:"
                    response = model.generate_content(full_prompt)
                    
                    execution_time = time.time() - start_time
                    
                    # Extract response text
                    response_text = response.text if hasattr(response, 'text') else str(response)
                    
                    # Extract output tokens
                    output_tokens = self._extract_output_tokens(response)
                    print(f"[TOKEN COUNT] Main generation output tokens: {output_tokens}")
                    
                    transformed_response = {
                        "response": response_text,
                        "model_used": model_info['display_name'],
                        "response_id": str(uuid.uuid4()),
                        "system_prompt_used": final_system_prompt,
                        "rating": None,
                        "provider": "google-gemini",
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens
                    }
                    
                    # Log the response
                    self.observability.log_event(trace_id, "gemini_response", {
                        "model": model_info['display_name'],
                        "response_id": transformed_response.get("response_id"),
                        "execution_time": execution_time
                    }, duration=execution_time)
                    
                    logger.info(f"[SUCCESS] {model_info['display_name']} response generated successfully in {execution_time:.2f}s")
                    return transformed_response
                    
                except Exception as model_error:
                    logger.warning(f"[WARNING] {model_info['display_name']} failed: {str(model_error)}")
                    last_error = model_error
                    continue
            
            # If all models failed, raise the last error
            execution_time = time.time() - start_time
            self.observability.log_event(trace_id, "gemini_exception", {
                "error": str(last_error),
                        "execution_time": execution_time
                    }, duration=execution_time)
                    
            raise Exception(f"All Gemini models failed: {last_error}")
                    
        except Exception as e:
            execution_time = time.time() - start_time
            self.observability.log_event(trace_id, "gemini_exception", {
                "error": str(e),
                "execution_time": execution_time
            }, duration=execution_time)
            
            raise Exception(f"Gemini API exception: {e}")

    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None, 
                              conversation_id: Optional[str] = None, user_id: Optional[str] = None,
                              store_conversation: bool = True, force_model: Optional[str] = None) -> Dict[str, Any]:
        """Generate AI response with intelligent model routing and automatic LangChain and Observability processing"""
        trace_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # 3-Level Intent Classification using Gemini 2.5 Lite
            classification_result = await self.classify_query_type(prompt)
            domain = classification_result["domain"]
            financial_category = classification_result["financial_category"]
            model_selection = classification_result["model_selection"]
            auto_system_prompt = classification_result["system_prompt"]
            
            # Extract classification token counts
            classification_input_tokens = classification_result.get("classification_input_tokens", 0)
            classification_output_tokens = classification_result.get("classification_output_tokens", 0)
            
            # Use auto-generated system prompt if none provided from frontend
            final_system_prompt = system_prompt if system_prompt else auto_system_prompt
            
            # Log the start of the request
            self.observability.log_event(trace_id, "request_start", {
                "prompt": prompt,
                "system_prompt": final_system_prompt,
                "conversation_id": conversation_id,
                "domain": domain,
                "financial_category": financial_category,
                "model_selection": model_selection,
                "force_model": force_model
            })
            
            # Intelligent model routing based on Level 3 classification
            if force_model == "modal":
                # Force Modal model
                response = await self._call_modal_model(prompt, final_system_prompt, trace_id)
            else:
                # Use the model selected by Level 3 classification
                if model_selection == "Gemini 2.0 Flash":
                    logger.info(f"[INFO] Using Gemini 2.0 Flash (selected by Level 3) - Category: {financial_category}")
                    response = await self._call_gemini_model(prompt, final_system_prompt, trace_id, "reasoning", use_model="gemini-2.0-flash-exp")
                else:  # Gemini 2.5 Lite (using Gemini 2.0 Flash Exp with lighter config)
                    logger.info(f"[INFO] Using Gemini 2.5 Lite mode (selected by Level 3) - Category: {financial_category}")
                    response = await self._call_gemini_model(prompt, final_system_prompt, trace_id, "factual", use_model="gemini-2.0-flash-exp-lite")
            
            # AUTOMATIC LANGCHAIN AND OBSERVABILITY PROCESSING AFTER LLM RESPONSE
            processing_time = time.time() - start_time
            
            # Calculate total token counts (classification + main generation)
            main_input_tokens = response.get("input_tokens", 0)
            main_output_tokens = response.get("output_tokens", 0)
            total_input_tokens = classification_input_tokens + main_input_tokens
            total_output_tokens = classification_output_tokens + main_output_tokens
            total_tokens = total_input_tokens + total_output_tokens
            
            print(f"[TOKEN SUMMARY] Classification: {classification_input_tokens} in, {classification_output_tokens} out")
            print(f"[TOKEN SUMMARY] Main Generation: {main_input_tokens} in, {main_output_tokens} out")
            print(f"[TOKEN SUMMARY] Total: {total_input_tokens} in, {total_output_tokens} out, {total_tokens} total")
            
            # Update conversation memory if conversation_id provided and storage is enabled
            if conversation_id and store_conversation:
                self.langchain.add_message_to_memory(conversation_id, "user", prompt)
                self.langchain.add_message_to_memory(conversation_id, "assistant", 
                                                   response.get("response", ""))
            
            # Log successful response
            self.observability.log_event(trace_id, "response_success", {
                "processing_time": processing_time,
                "response_length": len(response.get("response", "")),
                "store_conversation": store_conversation,
                "query_type": "reasoning",
                "model_used": response.get("model_used", "unknown"),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_tokens
            }, duration=processing_time)
            
            # Update metrics
            self.observability.update_metrics(True, processing_time)
            
            # Store conversation in database only if store_conversation is True
            if store_conversation:
                if not conversation_id:
                    conversation_id = str(uuid.uuid4())
                
                # Store conversation and log success
                storage_success, new_conversation_id = self._store_conversation(
                    conversation_id=conversation_id,
                    user_prompt=prompt,
                    ai_response=response.get("response", ""),
                    system_prompt=final_system_prompt,
                    model_used=response.get("model_used", "Unknown"),
                    provider=response.get("provider", "unknown"),
                    processing_time=processing_time,
                    modal_response_id=response.get("response_id"),
                    trace_id=trace_id,
                    user_id=user_id,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    total_tokens=total_tokens
                )
                
                if storage_success:
                    # Update conversation_id with the new unique ID from database
                    conversation_id = new_conversation_id
                    logger.info(f"[SUCCESS] Conversation successfully saved to database with new ID: {conversation_id}")
                else:
                    logger.warning(f"[WARNING] Failed to save conversation to database: {conversation_id}")
            else:
                logger.info(f"[INFO] Conversation storage disabled - response generated without storage")
            
            return {
                "response": response.get("response", "Sorry, I couldn't process your request."),
                "model_used": response.get("model_used", "Unknown"),
                "provider": response.get("provider", "unknown"),
                "conversation_id": conversation_id if store_conversation else None,  # Only return conversation_id if storage is enabled
                "trace_id": trace_id,
                "processing_time": processing_time,
                "modal_response_id": response.get("response_id"),
                "system_prompt_used": final_system_prompt,
                "rating": response.get("rating"),
                "domain": domain,
                "financial_category": financial_category,
                "model_selection": model_selection,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_tokens,
                "classification_input_tokens": classification_input_tokens,
                "classification_output_tokens": classification_output_tokens
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
            
            logger.error(f"[ERROR] {error_msg}")
            
            return {
                "response": "Sorry, I couldn't process your request. Please try again.",
                "model_used": "error",
                "provider": "error",
                "conversation_id": conversation_id,
                "trace_id": trace_id,
                "processing_time": processing_time,
                "error": str(e),
                "system_prompt_used": final_system_prompt if 'final_system_prompt' in locals() else "default",
                "domain": domain if 'domain' in locals() else "unknown",
                "financial_category": financial_category if 'financial_category' in locals() else "unknown",
                "model_selection": model_selection if 'model_selection' in locals() else "unknown"
            }

    def _store_conversation(self, conversation_id: str, user_prompt: str, ai_response: str, 
                           system_prompt: str, model_used: str, provider: str,
                           processing_time: float = None, modal_response_id: str = None,
                           trace_id: str = None, error_message: str = None, user_id: str = None,
                           input_tokens: int = 0, output_tokens: int = 0, total_tokens: int = 0):
        """Store conversation in database - always create new record with unique ID with retry mechanism"""
        max_retries = 3
        retry_delay = 0.1  # 100ms
        
        for attempt in range(max_retries):
            conn = None
            try:
                # Configure SQLite connection with proper settings to avoid locking
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=30.0,  # 30 second timeout
                    isolation_level=None  # Autocommit mode
                )
                
                # Set WAL mode and other optimizations
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                conn.execute("PRAGMA temp_store=MEMORY")
                
                cursor = conn.cursor()
                
                # Always generate a new unique conversation_id to avoid conflicts
                new_conversation_id = str(uuid.uuid4())
                
                # Insert new conversation with unique ID
                cursor.execute('''
                    INSERT INTO conversations 
                    (conversation_id, user_prompt, ai_response, system_prompt, model_used, 
                     provider, processing_time, modal_response_id, trace_id, error_message, user_id,
                     input_tokens, output_tokens, total_tokens)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (new_conversation_id, user_prompt, ai_response, system_prompt, model_used, 
                      provider, processing_time, modal_response_id, trace_id, error_message, user_id,
                      input_tokens, output_tokens, total_tokens))
                
                # Commit the transaction
                conn.commit()
                print(f"[SUCCESS] Conversation stored successfully with new ID: {new_conversation_id}")
                return True, new_conversation_id
                
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    print(f"[WARNING] Database locked, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    print(f"[ERROR] Database error after {max_retries} attempts: {e}")
                    return False, None
            except Exception as e:
                print(f"[ERROR] Error storing conversation: {e}")
                return False, None
            finally:
                # Ensure connection is always closed
                if conn:
                    try:
                        conn.close()
                    except:
                        pass  # Ignore errors when closing
    
    def store_rating(self, trace_id: str, user_rating: int, feedback_comment: str = "") -> bool:
        """Store user rating and feedback with robust connection handling"""
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            conn = None
            try:
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=30.0,
                    isolation_level=None
                )
                
                # Set WAL mode and optimizations
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                
                cursor = conn.cursor()
                
                # Store rating
                cursor.execute('''
                    INSERT OR REPLACE INTO ratings 
                    (trace_id, user_rating, feedback_comment)
                    VALUES (?, ?, ?)
                ''', (trace_id, user_rating, feedback_comment))
                
                conn.commit()
                print(f"[SUCCESS] Rating stored successfully: {trace_id}")
                return True
                
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    print(f"[WARNING] Database locked, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    print(f"[ERROR] Database error after {max_retries} attempts: {e}")
                    return False
            except Exception as e:
                print(f"[ERROR] Error storing rating: {e}")
                return False
            finally:
                if conn:
                    try:
                        conn.close()
                    except:
                        pass


    def get_user_history(self, user_id: str = None, limit: int = 50) -> Dict[str, Any]:
        """Get user conversation history - API endpoint with robust connection handling"""
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            conn = None
            try:
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=30.0,
                    isolation_level=None
                )
                
                # Set WAL mode and optimizations
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                
                cursor = conn.cursor()
                
                # Get conversation history
                if user_id:
                    cursor.execute('''
                        SELECT conversation_id, user_prompt, ai_response, timestamp, 
                               model_used, processing_time, trace_id, input_tokens, output_tokens, total_tokens
                        FROM conversations 
                        WHERE user_id = ? OR user_id IS NULL
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    ''', (user_id, limit))
                else:
                    cursor.execute('''
                        SELECT conversation_id, user_prompt, ai_response, timestamp, 
                               model_used, processing_time, trace_id, input_tokens, output_tokens, total_tokens
                        FROM conversations 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    ''', (limit,))
                
                conversations = cursor.fetchall()
                
                # Format the response
                history = []
                for conv in conversations:
                    history.append({
                        "conversation_id": conv[0],
                        "user_prompt": conv[1],
                        "timestamp": conv[3],
                        "model_used": conv[4],
                        "processing_time": conv[5],
                        "trace_id": conv[6],
                        "input_tokens": conv[7] if len(conv) > 7 else 0,
                        "output_tokens": conv[8] if len(conv) > 8 else 0,
                        "total_tokens": conv[9] if len(conv) > 9 else 0
                    })
                
                logger.info(f"[SUCCESS] Retrieved {len(history)} conversations for user: {user_id}")
                return {
                    "success": True,
                    "conversations": history,
                    "total_count": len(history)
                }
                
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    print(f"[WARNING] Database locked, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    logger.error(f"[ERROR] Database error after {max_retries} attempts: {e}")
                    return {
                        "success": False,
                        "message": f"Database error: {str(e)}",
                        "conversations": []
                    }
            except Exception as e:
                logger.error(f"[ERROR] Error retrieving user history: {e}")
                return {
                    "success": False,
                    "message": f"Error retrieving history: {str(e)}",
                    "conversations": []
                }
            finally:
                if conn:
                    try:
                        conn.close()
                    except:
                        pass

    def delete_user_prompt(self, user_id: str, conversation_id: str) -> Dict[str, Any]:
        """Delete a single saved prompt for a user by conversation_id with robust connection handling"""
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            conn = None
            try:
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=30.0,
                    isolation_level=None
                )
                
                # Set WAL mode and optimizations
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                
                cursor = conn.cursor()
                
                # First, check if the conversation exists and belongs to the user
                cursor.execute('''
                    SELECT COUNT(*) FROM conversations 
                    WHERE conversation_id = ? AND user_id = ?
                ''', (conversation_id, user_id))
                
                exists_count = cursor.fetchone()[0]
                
                if exists_count == 0:
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
                
                if deleted_count > 0:
                    logger.info(f"[SUCCESS] Deleted conversation {conversation_id} for user {user_id}")
                    return {
                        "success": True,
                        "message": f"Conversation {conversation_id} deleted successfully"
                    }
                else:
                    return {
                        "success": False,
                        "message": f"Failed to delete conversation {conversation_id}"
                    }
                
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    print(f"[WARNING] Database locked, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    logger.error(f"[ERROR] Database error after {max_retries} attempts: {e}")
                    return {
                        "success": False,
                        "message": f"Database error: {str(e)}"
                    }
            except Exception as e:
                logger.error(f"[ERROR] Error deleting conversation: {e}")
                return {
                    "success": False,
                    "message": f"Error deleting conversation: {str(e)}"
                }
            finally:
                if conn:
                    try:
                        conn.close()
                    except:
                        pass

    def cleanup(self):
        """Clean up resources"""
        try:
            logger.info("[SUCCESS] ChatAI Core cleanup completed")
        except Exception as e:
            logger.error(f"[ERROR] Error during cleanup: {e}")
"""
Query Orchestrator for Financial Advisor Chatbot (HYBRID VERSION)

This is a HYBRID orchestrator that:
- Uses the OLD proven routing structure (8 intent handlers)
- Integrates NEW CHATAI1 elaborate prompting improvements
- Safe drop-in replacement for production

Routes queries to appropriate handlers based on intent:
- PRODUCT → Product File Search store
- CLIENT → Tenant's client store
- GENERAL → Plain Gemini
- COMPLEX → Chained multi-source queries
"""

import asyncio
import logging
import hashlib
import json
from typing import Optional
from datetime import datetime, timedelta
from google import genai

from src.core.intent_classifier import IntentType, ClassifiedIntent
from src.core.conversation_manager import ConversationManager
from src.stores.tenant_manager import TenantStoreManager

logger = logging.getLogger(__name__)

# Simple in-memory cache with TTL and max size
query_cache = {}
CACHE_TTL = timedelta(minutes=5)
MAX_CACHE_SIZE = 100  # Maximum cached responses to prevent memory growth


def _evict_expired_cache():
    """Remove expired entries from cache."""
    now = datetime.now()
    expired_keys = [
        key for key, value in query_cache.items() 
        if now - value['timestamp'] >= CACHE_TTL
    ]
    for key in expired_keys:
        del query_cache[key]


def _evict_oldest_if_needed():
    """Remove oldest entries if cache exceeds max size."""
    if len(query_cache) >= MAX_CACHE_SIZE:
        # Remove oldest 10% of entries
        sorted_keys = sorted(
            query_cache.keys(), 
            key=lambda k: query_cache[k]['timestamp']
        )
        for key in sorted_keys[:MAX_CACHE_SIZE // 10]:
            del query_cache[key]
        logger.info(f"Evicted {MAX_CACHE_SIZE // 10} oldest cache entries")


def get_entity(entities, key: str, default=None):
    """Safely get entity value from both old dict and new Pydantic model."""
    if hasattr(entities, key):
        val = getattr(entities, key)
        # Handle list types (scheme_names -> scheme_name compatibility)
        if key == "scheme_name" and hasattr(entities, "scheme_names"):
            names = getattr(entities, "scheme_names", [])
            return names[0] if names else default
        return val if val is not None else default
    elif isinstance(entities, dict):
        return entities.get(key, default)
    return default


# =============================================================================
# CHATAI1 ENHANCED SYSTEM PROMPT (NEW)
# =============================================================================

CHATAI1_ENHANCED_PROMPT = '''You are ChatAI1, an expert AI assistant for Indian financial intermediaries:
- Mutual Fund Distributors (MFDs)
- Insurance Agents/POSPs
- Stock Brokers and RIAs

CRITICAL INSTRUCTIONS:
1. Provide COMPREHENSIVE, DETAILED answers - users expect thorough responses
2. Use your expert knowledge to fill ANY gaps in retrieved context
3. NEVER say "data is limited" or "context is incomplete" - be confident
4. For product comparisons, ALWAYS provide:
   - A comparison table with key metrics (AUM, Returns, Expense Ratio, Risk)
   - Fund manager and AMC details
   - Investment philosophy differences
   - Suitability recommendations
   - Your expert recommendation with reasoning
5. Use markdown tables for comparisons (single-line rows, no blank lines)
6. Be confident and authoritative - you are an expert financial advisor
7. End with specific, actionable recommendations

IMPORTANT: Provide ELABORATE, EXPERT-LEVEL responses. Never give short or hedging answers.
'''


class QueryOrchestrator:
    """Routes and executes queries based on classified intent."""
    
    def __init__(self, client: genai.Client, tenant_id: str):
        self.client = client
        self.tenant_id = tenant_id
        self.store_manager = TenantStoreManager(client)
        self.use_cache = True  # Enable caching
    
    def _get_cache_key(self, query: str, intent_type: str) -> str:
        """Generate cache key for query"""
        combined = f"{query}:{intent_type}:{self.tenant_id}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[dict]:
        """Get cached response if available and not expired"""
        if not self.use_cache or cache_key not in query_cache:
            return None
        
        cached = query_cache[cache_key]
        if datetime.now() - cached['timestamp'] >= CACHE_TTL:
            del query_cache[cache_key]
            return None
        
        logger.info(f"Cache hit for key {cache_key[:8]}...")
        return cached['data']
    
    def _set_cache(self, cache_key: str, data: dict):
        """Cache response data with eviction policy."""
        if self.use_cache:
            # Evict expired and check size limits
            _evict_expired_cache()
            _evict_oldest_if_needed()
            
            query_cache[cache_key] = {
                'timestamp': datetime.now(),
                'data': data
            }
            logger.info(f"Cached response for key {cache_key[:8]}... (cache size: {len(query_cache)})")
    
    async def route(
        self,
        query: str,
        intent: ClassifiedIntent,
        conversation: ConversationManager
    ):
        """
        Route query to appropriate handler.
        
        Args:
            query: User's question
            intent: Classified intent
            conversation: Conversation context
        
        Returns:
            Tuple of (Gemini response, effective_intent_type)
        """
        # Check if we should override intent based on conversation context
        effective_intent = self._get_context_aware_intent(query, intent, conversation)
        
        # Check cache first (skip for CLIENT queries - always fresh)
        skip_cache_intents = [IntentType.CLIENT_VIEW, IntentType.CLIENT_ACTION, IntentType.MARKET]
        if effective_intent not in skip_cache_intents:
            cache_key = self._get_cache_key(query, effective_intent.value)
            cached = self._get_cached_response(cache_key)
            if cached:
                return cached['response'], effective_intent
        
        # Enrich query with conversation context
        context_query = self._enrich_with_context(query, conversation, intent)
        
        logger.info(f"Original intent: {intent.primary_intent}, Effective: {effective_intent}")
        
        # Route based on 8-intent taxonomy
        match effective_intent:
            # Product-related intents
            case IntentType.PRODUCT_INFO:
                response = await self._handle_product(context_query, conversation)
            case IntentType.PRODUCT_COMPARE:
                response = await self._handle_product_compare(context_query, intent, conversation)
            
            # Client-related intents
            case IntentType.CLIENT_VIEW:
                response = await self._handle_client(context_query, intent, conversation)
            case IntentType.CLIENT_ACTION:
                response = await self._handle_client_action(context_query, intent, conversation)
            
            # Knowledge-related intents
            case IntentType.EDUCATION:
                response = await self._handle_general(context_query, conversation)
            case IntentType.REGULATORY:
                response = await self._handle_regulatory(context_query, intent, conversation)
            case IntentType.MARKET:
                response = await self._handle_market(context_query, conversation)
            
            # Business-related
            case IntentType.OPERATIONS:
                response = await self._handle_general(context_query, conversation)
            
            case _:
                # Catch-all for any unmapped intents
                response = await self._handle_general(context_query, conversation)
        
        # Cache the response (except for dynamic queries)
        if effective_intent not in skip_cache_intents:
            cache_key = self._get_cache_key(query, effective_intent.value)
            self._set_cache(cache_key, {'response': response})
        
        return response, effective_intent
    
    def _get_context_aware_intent(
        self,
        query: str,
        intent: ClassifiedIntent,
        conversation: ConversationManager
    ) -> IntentType:
        """
        Adjust intent based on conversation context.
        """
        # Strong education patterns should remain EDUCATION
        if self._is_strong_education_query(query):
            return IntentType.EDUCATION
        
        # If classified as EDUCATION but we have recent product context, might be follow-up
        if intent.primary_intent == IntentType.EDUCATION:
            if self._has_recent_product_context(conversation):
                # Check if it's actually asking about the current query (not a new concept)
                if not self._is_strong_education_query(query):
                    return IntentType.PRODUCT_INFO
        
        return intent.primary_intent
    
    def _is_strong_education_query(self, query: str) -> bool:
        """Check if query is strongly educational."""
        education_patterns = [
            'kya hai', 'kya hota', 'kya hoti', 'what is', 'what are',
            'explain', 'define', 'meaning of', 'samjhao', 'batao'
        ]
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in education_patterns)
    
    def _has_recent_product_context(self, conversation: ConversationManager) -> bool:
        """Check if recent conversation was about products."""
        if not conversation or not conversation.history:
            return False
        
        recent_intents = []
        for msg in conversation.history[-4:]:
            if msg.get('role') == 'user' and msg.get('intent'):
                recent_intents.append(msg['intent'])
        
        product_intents = ['product_info', 'product_compare']
        return any(i in product_intents for i in recent_intents)
    
    def _enrich_with_context(
        self, 
        query: str, 
        conversation: ConversationManager,
        intent: ClassifiedIntent
    ) -> str:
        """Add context from conversation and entities."""
        enrichments = []
        
        # Add active client context
        if conversation and conversation.active_client:
            enrichments.append(f"[Active client: {conversation.active_client}]")
        
        # Add scheme context if available
        scheme_name = get_entity(intent.entities, "scheme_name")
        if scheme_name:
            enrichments.append(f"[Scheme: {scheme_name}]")
        
        if enrichments:
            return f"{' '.join(enrichments)} {query}"
        return query
    
    def _build_conversation_messages(
        self, 
        query: str, 
        conversation: ConversationManager,
        system_instruction: str = None
    ) -> list:
        """Build message list with conversation history for Gemini."""
        messages = []
        
        # Add recent history
        if conversation and conversation.history:
            for msg in conversation.history[-6:]:  # Last 3 exchanges
                role = "user" if msg["role"] == "user" else "model"
                messages.append({
                    "role": role,
                    "parts": [{"text": msg["content"][:500]}]  # Truncate for context
                })
        
        # Add current query
        messages.append({
            "role": "user",
            "parts": [{"text": query}]
        })
        
        return messages
    
    # =========================================================================
    # HANDLERS (Using CHATAI1 Enhanced Prompts)
    # =========================================================================
    
    async def _handle_general(self, query: str, conversation: ConversationManager):
        """
        Handle general finance education queries.
        Uses plain Gemini without RAG, includes conversation history.
        """
        messages = self._build_conversation_messages(query, conversation)
        
        try:
            response = await self.client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=messages,
                config={
                    "system_instruction": CHATAI1_ENHANCED_PROMPT,
                    "temperature": 0.4,
                    "max_output_tokens": 2000
                }
            )
            return response
        except Exception as e:
            logger.error(f"General handler error: {e}")
            return self._create_error_response(query, str(e))
    
    async def _handle_product(self, query: str, conversation: ConversationManager):
        """
        Handle product-related queries.
        Uses shared products File Search store with conversation context.
        """
        try:
            # Get context from product store
            context = await self.store_manager.query_product_store(query)
            context_text = self._extract_text(context)
            
            # Build enriched prompt with CHATAI1 enhanced instructions
            enriched_query = f"""{CHATAI1_ENHANCED_PROMPT}

## Retrieved Product Information:
{context_text if context_text else 'No specific product data found.'}

## User Query:
{query}

Provide a comprehensive, expert answer. Use your knowledge to supplement retrieved data."""
            
            response = await self.client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=enriched_query,
                config={
                    "temperature": 0.4,
                    "max_output_tokens": 2000
                }
            )
            return response
        except Exception as e:
            logger.error(f"Product handler error: {e}")
            return await self._handle_general(query, conversation)
    
    async def _handle_product_compare(
        self, 
        query: str, 
        intent: ClassifiedIntent,
        conversation: ConversationManager
    ):
        """
        Handle product comparison queries.
        Uses product store for reliable, fast responses.
        """
        try:
            # Get context from product store
            context = await self.store_manager.query_product_store(query)
            context_text = self._extract_text(context)
            
            # Build comparison-specific prompt
            comparison_prompt = f"""{CHATAI1_ENHANCED_PROMPT}

## Retrieved Product Information:
{context_text if context_text else 'No specific product data found.'}

## User Query:
{query}

## Comparison Instructions:
You MUST provide a DETAILED comparison including:
1. A markdown comparison table with these columns:
   | Feature | Fund 1 | Fund 2 |
   |---------|--------|--------|
   | AUM | ... | ... |
   | 1Y Returns | ... | ... |
   | 3Y Returns | ... | ... |
   | 5Y Returns | ... | ... |
   | Expense Ratio | ... | ... |
   | Risk Level | ... | ... |
   | Fund Manager | ... | ... |

2. Investment Philosophy Analysis
3. Key Differentiators  
4. Suitability for different investor profiles
5. Your Expert Recommendation with clear reasoning

Be thorough and actionable. Never say "data is limited"."""
            
            response = await self.client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=comparison_prompt,
                config={
                    "temperature": 0.3,
                    "max_output_tokens": 2500
                }
            )
            return response
        except Exception as e:
            logger.error(f"Comparison handler error: {e}")
            return await self._handle_general(query, conversation)
    
    async def _handle_client(
        self, 
        query: str, 
        intent: ClassifiedIntent,
        conversation: ConversationManager
    ):
        """
        Handle client-specific queries.
        Uses tenant's isolated client store with conversation context.
        """
        client_name = get_entity(intent.entities, "client_name")
        
        try:
            # Query tenant's client store
            context = await self.store_manager.query_client_store(
                self.tenant_id, query, filter_client=client_name
            )
            context_text = self._extract_text(context)
            
            enriched_query = f"""{CHATAI1_ENHANCED_PROMPT}

## Client Data Retrieved:
{context_text if context_text else 'No client data found.'}

## User Query:
{query}

Provide actionable insights for this client."""
            
            response = await self.client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=enriched_query,
                config={
                    "temperature": 0.4,
                    "max_output_tokens": 2000
                }
            )
            
            # Update active client in conversation
            if client_name and conversation:
                conversation.set_active_client(client_name)
            
            return response
        except Exception as e:
            logger.error(f"Client handler error: {e}")
            return await self._handle_general(query, conversation)
    
    async def _handle_client_action(
        self, 
        query: str, 
        intent: ClassifiedIntent,
        conversation: ConversationManager
    ):
        """Handle client action requests (SIP stop, switch, etc.)."""
        action = get_entity(intent.entities, "action_verb", "process")
        client_name = get_entity(intent.entities, "client_name")
        
        # For now, provide guidance - actual actions need human confirmation
        guidance_prompt = f"""{CHATAI1_ENHANCED_PROMPT}

The user wants to perform an action: {action}
Client: {client_name or 'Not specified'}
Query: {query}

Provide step-by-step guidance for this action, including:
1. Required documents/approvals
2. Process steps
3. Timeline expectations
4. Important considerations"""
        
        response = await self.client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=guidance_prompt,
            config={
                "temperature": 0.3,
                "max_output_tokens": 1500
            }
        )
        return response
    
    async def _handle_regulatory(
        self,
        query: str,
        intent: ClassifiedIntent,
        conversation: ConversationManager
    ):
        """Handle regulatory and compliance queries."""
        try:
            # Get context from product store (which includes regulatory docs)
            context = await self.store_manager.query_product_store(query)
            context_text = self._extract_text(context)
            
            regulatory_prompt = f"""{CHATAI1_ENHANCED_PROMPT}

## Retrieved Regulatory Information:
{context_text if context_text else 'No specific regulatory data found.'}

## User Query:
{query}

Provide accurate regulatory guidance with specific references to SEBI/AMFI/IRDAI guidelines where applicable."""
            
            response = await self.client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=regulatory_prompt,
                config={
                    "temperature": 0.2,
                    "max_output_tokens": 2000
                }
            )
            return response
        except Exception as e:
            logger.error(f"Regulatory handler error: {e}")
            return await self._handle_general(query, conversation)
    
    async def _handle_market(self, query: str, conversation: ConversationManager):
        """Handle market-related queries with Google Search."""
        try:
            # Use Google Search for live market data
            context = await self.store_manager.query_with_google_search(query)
            context_text = self._extract_text(context)
            
            market_prompt = f"""{CHATAI1_ENHANCED_PROMPT}

## Latest Market Information:
{context_text if context_text else 'Using general knowledge.'}

## User Query:
{query}

Provide current market analysis and actionable insights."""
            
            response = await self.client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=market_prompt,
                config={
                    "temperature": 0.4,
                    "max_output_tokens": 2000
                }
            )
            return response
        except Exception as e:
            logger.error(f"Market handler error: {e}")
            return await self._handle_general(query, conversation)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _extract_text(self, response) -> str:
        """Extract text from Gemini response."""
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        if hasattr(response, 'text'):
            return response.text
        if hasattr(response, 'parts'):
            return "".join(p.text for p in response.parts if hasattr(p, 'text'))
        return str(response)
    
    def _create_error_response(self, query: str, error: str):
        """Create a graceful error response."""
        return type('Response', (), {
            'text': f"I apologize, I encountered an issue processing your query. Please try rephrasing: {query[:50]}..."
        })()

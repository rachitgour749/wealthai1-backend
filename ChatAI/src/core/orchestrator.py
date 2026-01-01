"""
Query Orchestrator for Financial Advisor Chatbot

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
        if not self.use_cache:
            return None
        
        if cache_key in query_cache:
            cached = query_cache[cache_key]
            if datetime.now() - cached['timestamp'] < CACHE_TTL:
                logger.info(f"Cache HIT for key {cache_key[:8]}...")
                return cached['data']
            else:
                # Expired, remove
                del query_cache[cache_key]
        return None
    
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
        
        # Route based on new 8-intent taxonomy
        match effective_intent:
            # Product-related intents
            case IntentType.PRODUCT_INFO:
                response = await self._handle_product(context_query, conversation)
            case IntentType.PRODUCT_COMPARE:
                # Use fast internal-only comparison (no Google Search timeout)
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
        
        Sometimes a query needs different handling based on what was discussed before.
        For example, "How about the returns?" after a product query should use PRODUCT_INFO.
        """
        # First, check if this is a comparison with external products
        # This should use PRODUCT_COMPARE
        if self._is_comparison_with_external(intent):
            logger.info("Detected comparison with external product → PRODUCT_COMPARE")
            return IntentType.PRODUCT_COMPARE
        
        # If already a comparison, keep it
        if intent.primary_intent == IntentType.PRODUCT_COMPARE:
            return IntentType.PRODUCT_COMPARE
        
        # If product query, keep it  
        if intent.primary_intent == IntentType.PRODUCT_INFO:
            return IntentType.PRODUCT_INFO
        
        # Check if we have recent product context for EDUCATION queries
        if intent.primary_intent == IntentType.EDUCATION:
            # Import to check if this is a strong education pattern
            from src.core.intent_classifier import is_strong_education_query
            
            # DON'T override strong education patterns like "SIP kya hota hai?"
            # Check the CURRENT QUERY, not conversation history
            if is_strong_education_query(query):
                logger.info(f"Keeping EDUCATION (strong education pattern in query: '{query[:30]}...')")
                return IntentType.EDUCATION
            
            # Check last few messages for product context
            recent_product_context = self._has_recent_product_context(conversation)
            
            if recent_product_context:
                logger.info("Overriding EDUCATION to PRODUCT_INFO based on conversation context")
                return IntentType.PRODUCT_INFO
        
        return intent.primary_intent
    
    def _is_comparison_with_external(self, intent: ClassifiedIntent) -> bool:
        """
        Check if query is comparing products where one may be external.
        
        Only returns True when:
        - Multiple different company products are mentioned (e.g., "Acko vs HDFC Ergo")
        - AND at least one is external (not in our store)
        """
        scheme_name = get_entity(intent.entities, "scheme_name", "")
        if not scheme_name:
            return False
        
        scheme_lower = str(scheme_name).lower()
        
        # Internal products we have in our store
        internal_products = ["acko"]
        
        # Check if multiple products mentioned (comma-separated)
        schemes = [s.strip() for s in str(scheme_name).split(",") if s.strip()]
        
        if len(schemes) > 1:
            # Multiple products mentioned - check if any are external
            has_internal = any(internal in scheme_lower for internal in internal_products)
            
            external_companies = [
                "hdfc", "icici", "sbi", "kotak", "axis", "bajaj", "tata", 
                "nippon", "birla", "reliance", "lic", "max", "star health",
                "care health", "niva bupa", "digit", "hdfc ergo"
            ]
            has_external = any(ext in scheme_lower for ext in external_companies)
            
            # Only complex if comparing internal with external
            return has_internal and has_external
        
        # Single product mentioned - check if classifier flagged for Google
        # AND it's an external product we don't have
        if intent.requires_google_search:
            # Check if it's NOT one of our internal products
            is_internal = any(internal in scheme_lower for internal in internal_products)
            if not is_internal:
                # External product that needs Google Search
                return True
        
        return False
    
    def _has_recent_product_context(self, conversation: ConversationManager) -> bool:
        """Check if recent conversation was about products."""
        # Look at last 4 messages for product context
        recent_messages = conversation.history[-4:] if conversation.history else []
        
        for msg in recent_messages:
            # Check if any recent message had product intent
            if msg.get("intent") == "product":
                return True
            
            # Check for product-related keywords in recent context
            content = msg.get("content", "").lower()
            product_indicators = [
                "policy", "coverage", "premium", "sum insured", "benefits",
                "insurance", "fund", "scheme", "expense ratio", "nav", 
                "acko", "hdfc", "sbi", "icici", "bajaj", "kotak"
            ]
            if any(indicator in content for indicator in product_indicators):
                return True
        
        # Also check if there's an active scheme
        if conversation.active_scheme:
            return True
        
        return False
    
    def _build_conversation_messages(
        self, 
        query: str, 
        conversation: ConversationManager,
        system_instruction: str = None
    ) -> list:
        """Build message list with conversation history for Gemini."""
        messages = []
        
        # Add recent conversation history (last 6 exchanges = 12 messages)
        for msg in conversation.history[-12:]:
            role = "user" if msg["role"] == "user" else "model"
            messages.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
        
        # Add current query
        messages.append({
            "role": "user",
            "parts": [{"text": query}]
        })
        
        return messages
    
    async def _handle_general(self, query: str, conversation: ConversationManager):
        """
        Handle general finance education queries.
        Uses plain Gemini without RAG, includes conversation history.
        """
        logger.info("Handling GENERAL query - plain Gemini with history")
        
        system_instruction = """You are a knowledgeable finance expert assistant 
for Indian financial intermediaries (MF distributors, insurance agents, wealth managers).

Your role:
- Explain financial concepts clearly and accurately
- Provide practical advice for client conversations
- Reference Indian regulations (SEBI, IRDAI, Income Tax)
- Use examples with Indian context (₹, Indian mutual funds, etc.)
- Remember context from earlier in the conversation

Be concise, professional, and actionable."""
        
        # Build messages with history
        messages = self._build_conversation_messages(query, conversation)
        
        return await self.client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages,
            config={
                "system_instruction": system_instruction
            }
        )
    
    async def _handle_product(self, query: str, conversation: ConversationManager):
        """
        Handle product-related queries.
        Uses shared products File Search store with conversation context.
        """
        logger.info("Handling PRODUCT query - products store")
        # Include conversation context in query for better relevance
        try:
            context = conversation.get_context_window(3) if conversation.history else ""
            enriched_query = f"{context}\n\nCurrent question: {query}" if context else query
            return await self.store_manager.query_product_store(enriched_query)
        except Exception as e:
            logger.error(f"Error in _handle_product: {e}")
            # Fallback to general handler
            return await self._handle_general(query, conversation)
    
    async def _handle_product_compare(
        self, 
        query: str, 
        intent: ClassifiedIntent,
        conversation: ConversationManager
    ):
        """
        Handle product comparison queries - FAST version.
        Uses ONLY internal product store (no Google Search) for reliable, fast responses.
        
        This is called for queries like:
        - "Compare Axis ELSS vs Mirae Tax Saver"
        - "HDFC Top 100 vs SBI Bluechip"
        """
        logger.info("Handling PRODUCT_COMPARE query - internal store only (fast)")
        
        try:
            # Get conversation context
            conv_context = conversation.get_context_window(3) if conversation.history else ""
            
            # Fetch from internal product store only (fast, reliable)
            product_context = await self._fetch_product_context(query)
            
            if product_context:
                # Use premium comparison prompt
                from src.core.system_prompts import PRODUCT_COMPARE_PROMPT, ADVISOR_TIPS
                
                comparison_prompt = PRODUCT_COMPARE_PROMPT.format(
                    context=product_context,
                    query=query
                )
                
                response = await self.client.aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=comparison_prompt,
                    config={"system_instruction": "You are WealthAI1, an expert assistant for Indian financial intermediaries."}
                )
                
                # Add advisor tip
                tip = ADVISOR_TIPS.get("product_compare", "")
                response_text = response.text + tip if hasattr(response, 'text') else str(response) + tip
                
                return response
            else:
                # Fallback to general if no product context
                return await self._handle_general(query, conversation)
                
        except Exception as e:
            logger.error(f"Error in _handle_product_compare: {e}")
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
        logger.info(f"Handling CLIENT query - tenant store, client: {client_name}")
        
        # Include conversation context for better retrieval
        context = conversation.get_context_window(3) if conversation.history else ""
        enriched_query = f"{context}\n\nCurrent question: {query}" if context else query
        
        return await self.store_manager.query_client_store(
            self.tenant_id,
            enriched_query,
            filter_client=client_name
        )
    
    async def _handle_complex(
        self,
        query: str,
        intent: ClassifiedIntent,
        conversation: ConversationManager
    ):
        """
        Handle complex multi-source queries with conversation context.
        
        Fetches from multiple sources in parallel, then merges context.
        Examples:
        - "Is Sharma ji's portfolio aligned with current market trends?"
        - "Compare this policy with HDFC Ergo" (needs our store + Google)
        - "Axis ELSS vs Mirae Tax Saver" (internal comparison)
        """
        logger.info("Handling COMPLEX/COMPARE query - multi-source")
        
        try:
            # Get conversation context for better synthesis
            conv_context = conversation.get_context_window(3) if conversation.history else ""
            
            # Step 1: Identify required sources and fetch in parallel
            tasks = []
            source_labels = []
            
            # Always include client context if client mentioned
            client_name = get_entity(intent.entities, "client_name")
            if client_name:
                tasks.append(self._fetch_client_context(intent))
                source_labels.append("CLIENT")
            
            # Include product context if:
            # - scheme mentioned
            # - product secondary intent
            # - OR recent product context in conversation
            has_product_context = (
                get_entity(intent.entities, "scheme_name") or 
                IntentType.PRODUCT_INFO in (intent.secondary_intents or []) or
                self._has_recent_product_context(conversation)
            )
            
            if has_product_context:
                # Enrich query with conversation context for better product retrieval
                product_query = f"{conv_context}\n\nCurrent question: {query}" if conv_context else query
                tasks.append(self._fetch_product_context(product_query))
                source_labels.append("PRODUCT (Internal Store)")
            
            # Include Google Search for:
            # - Real-time market data
            # - External product comparisons (requires_google_search)
            # - Cross-company comparisons
            if intent.requires_google_search or self._is_comparison_with_external(intent):
                # Build a specific search query for the external product
                scheme_name = get_entity(intent.entities, "scheme_name", "")
                search_query = f"{scheme_name} features benefits comparison" if scheme_name else query
                tasks.append(self._fetch_external_product_context(search_query))
                source_labels.append("EXTERNAL (Google Search)")
            
            # If no sources identified, just use the query directly with product store
            if not tasks:
                logger.info("No specific sources identified, using product store as fallback")
                tasks.append(self._fetch_product_context(query))
                source_labels.append("PRODUCT (Fallback)")
            
            # Execute all fetches in parallel with error handling
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful results
            contexts = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to fetch {source_labels[i]}: {result}")
                elif result:  # Only add non-empty results
                    contexts.append(f"=== {source_labels[i]} CONTEXT ===\n{result}")
            
            # Step 2: Merge contexts and generate comprehensive response
            if contexts:
                merged = "\n\n".join(contexts)
                return await self._generate_with_context(query, merged, conv_context)
            else:
                # Fallback to general handler if all sources failed
                logger.warning("All sources failed in _handle_complex, falling back to general")
                return await self._handle_general(query, conversation)
                
        except Exception as e:
            # Catch-all: if anything fails, fallback gracefully
            logger.error(f"Error in _handle_complex: {e}, falling back to general handler")
            return await self._handle_general(query, conversation)
    
    async def _fetch_client_context(self, intent: ClassifiedIntent) -> str:
        """Fetch client portfolio context."""
        client_name = get_entity(intent.entities, "client_name", "the client")
        
        result = await self.store_manager.query_client_store(
            self.tenant_id,
            f"Portfolio and holdings summary for {client_name}"
        )
        return result.text if hasattr(result, 'text') else str(result)
    
    async def _fetch_product_context(self, query: str) -> str:
        """Fetch product information context."""
        result = await self.store_manager.query_product_store(query)
        return result.text if hasattr(result, 'text') else str(result)
    
    async def _fetch_market_context(self) -> str:
        """Fetch current market context via Google Search."""
        result = await self.store_manager.query_with_google_search(
            "Current Indian stock market trends, Nifty 50 performance, "
            "mutual fund industry outlook - latest data"
        )
        return result.text if hasattr(result, 'text') else str(result)
    
    async def _fetch_external_product_context(self, query: str) -> str:
        """Fetch external product information via Google Search."""
        result = await self.store_manager.query_with_google_search(query)
        return result.text if hasattr(result, 'text') else str(result)
    
    async def _generate_with_context(self, query: str, context: str, conv_context: str = ""):
        """Generate response using merged context."""
        conversation_section = ""
        if conv_context:
            conversation_section = f"""
=== RECENT CONVERSATION ===
{conv_context}
===========================
"""
        
        return await self.client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"""Using the following context, answer the user's question comprehensively.
{conversation_section}
{context}

---
USER QUESTION: {query}
---

Instructions:
- Synthesize information from all provided contexts
- When comparing products, clearly structure the comparison with pros/cons for each
- Provide specific, actionable advice
- Cite which context source each piece of information comes from
- If information conflicts, note the discrepancy
- Use Indian financial terminology and ₹ for currency"""
        )
    
    async def _handle_client_action(
        self,
        query: str,
        intent: ClassifiedIntent,
        conversation: ConversationManager
    ):
        """
        Handle client action queries (SIP modifications, redemptions, switches).
        These are write operations that need to be confirmed, not auto-executed.
        """
        client_name = intent.entities.client_name if hasattr(intent.entities, 'client_name') else get_entity(intent.entities, "client_name")
        action = intent.sub_intent or "action"
        
        logger.info(f"Handling CLIENT_ACTION query - client: {client_name}, action: {action}")
        
        system_instruction = """You are a financial assistant helping prepare client action requests.

Your role:
- Understand the action requested (SIP start/stop/modify, redemption, switch)
- Fetch relevant client data for context
- Prepare a summary of the action for the advisor to review
- Include important details: current holdings, SIP amounts, folio numbers
- DO NOT execute actions - only prepare information

Important: Always ask for confirmation before any financial action."""
        
        # Get client context
        messages = self._build_conversation_messages(query, conversation)
        
        return await self.client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages,
            config={"system_instruction": system_instruction}
        )
    
    async def _handle_regulatory(
        self,
        query: str,
        intent: ClassifiedIntent,
        conversation: ConversationManager
    ):
        """
        Handle regulatory and compliance queries.
        Uses product store for regulatory docs, may use web for latest circulars.
        """
        logger.info("Handling REGULATORY query")
        
        system_instruction = """You are an expert on Indian financial regulations for MF distributors and insurance agents.

Your expertise includes:
- SEBI regulations for mutual funds
- AMFI guidelines and ARN/EUIN requirements
- IRDAI regulations for insurance
- KYC/CKYC compliance
- Tax implications (80C, 10(10D), LTCG, STCG)

When answering:
- Cite specific circulars/guidelines when available
- Mention effective dates for recent changes
- Provide practical compliance guidance
- Note any pending or proposed changes if relevant

Always recommend consulting the official sources for final compliance decisions."""
        
        # Try product store first for regulatory docs
        context = conversation.get_context_window(3) if conversation.history else ""
        enriched_query = f"{context}\n\nCurrent question: {query}" if context else query
        
        try:
            # Query product store for regulatory documents
            product_response = await self.store_manager.query_product_store(enriched_query)
            if product_response and hasattr(product_response, 'text') and len(product_response.text) > 100:
                return product_response
        except Exception as e:
            logger.warning(f"Product store query failed for regulatory: {e}")
        
        # Fallback to general LLM with regulatory system prompt
        messages = self._build_conversation_messages(query, conversation)
        return await self.client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages,
            config={"system_instruction": system_instruction}
        )
    
    async def _handle_market(
        self,
        query: str,
        conversation: ConversationManager
    ):
        """
        Handle market insights and news queries.
        Uses Gemini with grounding for current market data.
        """
        logger.info("Handling MARKET query - using grounded search")
        
        system_instruction = """You are a market analyst assistant for Indian financial advisors.

Provide:
- Current market trends and analysis
- Sector-specific insights
- Impact analysis (rate changes, policy decisions)
- Practical implications for client portfolios

Use Indian market context (NSE, BSE, Nifty, Sensex).
Format data clearly with dates and sources when available."""
        
        messages = self._build_conversation_messages(query, conversation)
        
        # Try to use grounded search if available
        try:
            return await self.client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=messages,
                config={
                    "system_instruction": system_instruction,
                    "tools": [{"google_search": {}}]
                }
            )
        except Exception as e:
            logger.warning(f"Grounded search failed: {e}, using plain LLM")
            return await self.client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=messages,
                config={"system_instruction": system_instruction}
            )
    
    def _enrich_with_context(
        self,
        query: str,
        conversation: ConversationManager,
        intent: ClassifiedIntent
    ) -> str:
        """Add conversation context to query if relevant."""
        enrichments = []
        
        # Handle both old and new entity structures
        client_name = None
        scheme_name = None
        
        if hasattr(intent.entities, 'client_name'):
            client_name = intent.entities.client_name
        elif isinstance(intent.entities, dict):
            client_name = get_entity(intent.entities, "client_name")
        
        if hasattr(intent.entities, 'scheme_names'):
            scheme_names = intent.entities.scheme_names
            scheme_name = scheme_names[0] if scheme_names else None
        elif isinstance(intent.entities, dict):
            scheme_name = get_entity(intent.entities, "scheme_name")
        
        # Client context intents
        client_intents = [
            IntentType.CLIENT_VIEW, IntentType.CLIENT_ACTION,
            IntentType.PRODUCT_COMPARE,  # Comparisons may involve client context
        ]
        
        # Add active client from conversation if not in current query
        if (conversation.active_client and 
            not client_name and
            intent.primary_intent in client_intents):
            enrichments.append(f"[Context: discussing client {conversation.active_client}]")
            if hasattr(intent.entities, 'client_name'):
                intent.entities.client_name = conversation.active_client
            elif isinstance(intent.entities, dict):
                intent.entities["client_name"] = conversation.active_client
        
        # Product context intents
        product_intents = [IntentType.PRODUCT_INFO, IntentType.PRODUCT_COMPARE]
        
        # Add active scheme if relevant
        if (conversation.active_scheme and 
            not scheme_name and
            intent.primary_intent in product_intents):
            enrichments.append(f"[Context: discussing {conversation.active_scheme}]")
        
        if enrichments:
            return " ".join(enrichments) + " " + query
        return query

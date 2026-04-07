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
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from google import genai

from Services.ChatAI.core.intent_classifier import IntentType, ClassifiedIntent, normalize_client_name
from Services.ChatAI.core.conversation_manager import ConversationManager
from Services.ChatAI.stores.tenant_manager import TenantStoreManager
from Services.ChatAI.data.mfapi_client import MFApiClient

logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    """Standardized response object for the orchestrator."""
    text: Optional[str] = None
    _is_fallback: bool = False
    _replay_query: Optional[str] = None
    _replay_client: Optional[str] = None


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

CHATAI1_ENHANCED_PROMPT = '''You are ChatAI1, a product recommendation engine built for Indian financial intermediaries — MFDs, Insurance Agents/Distributors, POSPs, Stock Brokers, RIAs, and Wealth Managers.

## YOUR USERS
Your users are NOT end consumers. They are licensed professionals who sell financial products.
They already understand finance — do NOT explain basics like "what is a SIP" or "how insurance works" unless explicitly asked.
They come to you with a **client scenario** and need **specific product recommendations** they can pitch/sell.

## HOW TO RESPOND

### When the user describes a client need (e.g. "best life insurance for 60 yr old with BP and sugar"):
1. **Ask clarifying questions FIRST** if critical info is missing. Common missing details:
   - Budget / premium range the client can afford
   - Sum assured requirement
   - Existing coverage (if any)
   - Investment horizon / goal timeline
   - Risk appetite of the client
   - Whether they want pure protection or savings + protection
   Present these as a short bullet list: "To give you the best recommendation, I need a few details:"

2. **If enough info is available**, give SPECIFIC product recommendations:
   - Name exact products (e.g., "HDFC Life Click 2 Protect 3D Plus", "ICICI Pru iProtect Smart")
   - Present a comparison table: Plan Name | Insurer | Key Feature | Premium Estimate | Claim Settlement Ratio
   - Highlight **selling points** the intermediary can use with their client
   - Mention **commission/payout structure** differences if known
   - Flag any **underwriting concerns** (e.g., loading for pre-existing conditions, exclusion periods)
   - End with: "**My Recommendation:**" followed by a clear pick with reasoning

### For product comparison queries:
- ALWAYS use a markdown comparison table
- Include: Product Name | AMC/Insurer | Key Metric | Returns/Benefits | Expense/Premium | Suitability
- Give a clear winner recommendation with reasoning

### For client data queries:
- Be concise — give the factual answer (SIP amount, AUM, holdings) in 1-3 lines
- Only elaborate when asked

## RULES
1. **Be SPECIFIC** — name exact products, plans, funds. Never say "consider XYZ type of product" when you can name the actual product.
2. **Be ACTIONABLE** — every response should help the intermediary close a sale or serve their client better.
3. **Ask follow-up questions** when the query is too vague to give a good recommendation. Frame them as: "To narrow down the best option, could you tell me..."
4. **Skip the basics** — don't explain what SIP/SWP/term insurance is. They know.
5. **Never say** "data is limited" or "I don't have access to..." — use your knowledge confidently.
6. **Use tables** for any comparison or multi-product recommendation.
7. **Mention regulatory/compliance notes** only when directly relevant (e.g., age limits, IRDAI guidelines on loading).
8. **Keep it professional** — no greetings, no flattery, no filler. Get straight to the recommendation.
'''




class QueryOrchestrator:
    """Routes and executes queries based on classified intent."""
    
    def __init__(self, client: genai.Client, tenant_id: str):
        self.client = client
        self.tenant_id = tenant_id
        self.store_manager = TenantStoreManager(client)
        self.mfapi_client = MFApiClient()
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
        # Check if user is resolving a disambiguation prompt
        disambiguation_result = self._try_resolve_disambiguation(query, conversation)
        if disambiguation_result:
            # Check if we need to replay the original query
            replay_query = getattr(disambiguation_result, '_replay_query', None)
            if replay_query:
                replay_client = getattr(disambiguation_result, '_replay_client', None)
                logger.info(f"Auto-replaying query for resolved client: {replay_client}")
                return await self._query_client_data(replay_query, replay_client, conversation), IntentType.CLIENT_VIEW
            return disambiguation_result, IntentType.CLIENT_VIEW
        
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
                response = await self._handle_product(context_query, conversation, intent)
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
    
    def _try_resolve_disambiguation(self, query: str, conversation: ConversationManager):
        """
        Check if user is responding to a disambiguation prompt.
        
        If the last assistant message asked "Which client are you asking about?"
        and the user replied with a number (e.g., "1") or a name, resolve it.
        
        Returns:
            Response object if disambiguation was resolved, None otherwise
        """
        if not conversation or not conversation.history or len(conversation.history) < 2:
            return None
        
        # Check if last assistant message was a disambiguation prompt
        last_assistant_msg = None
        for msg in reversed(conversation.history):
            if msg.get('role') == 'assistant':
                last_assistant_msg = msg.get('content', '')
                break
        
        if not last_assistant_msg or 'Which client are you asking about?' not in last_assistant_msg:
            return None
        
        # User is responding to disambiguation — extract the selected client
        import re
        query_stripped = query.strip()
        
        selected_name = None
        
        # Check if user replied with a number (e.g., "1", "2")
        if re.match(r'^\d+$', query_stripped):
            number = int(query_stripped)
            # Extract all client names from the disambiguation message
            # The LLM may use various formats, try multiple patterns
            names = []
            
            # Pattern 1: "**Full Name**: Ms. Rakhi Jain" (multi-line format)
            full_name_matches = re.findall(
                r'\*?\*?Full Name\*?\*?:\s*(?:Mr\.?\s*|Mrs\.?\s*|Ms\.?\s*|Dr\.?\s*)*(?:Mr\.?\s*|Mrs\.?\s*|Ms\.?\s*|Dr\.?\s*)*(.+?)(?:\n|$)',
                last_assistant_msg
            )
            if full_name_matches:
                names = [n.strip().strip('*') for n in full_name_matches]
            
            # Pattern 2: "1. **Name** — email" (inline format)
            if not names:
                inline_matches = re.findall(
                    r'\d+[\.\)]\s*\*\*([^*]+)\*\*\s*(?:—|-)',
                    last_assistant_msg
                )
                if inline_matches:
                    names = [n.strip() for n in inline_matches]
            
            # Pattern 3: Numbered list with bold names "1. **Name**"
            if not names:
                bold_matches = re.findall(
                    r'\d+[\.\)]\s*\*\*([^*]+)\*\*',
                    last_assistant_msg
                )
                # Filter out "Full Name" labels
                names = [n.strip() for n in bold_matches if 'Full Name' not in n]
            
            if 0 < number <= len(names):
                selected_name = names[number - 1].strip()
        
        # If not a number, the user may have typed a full name
        if not selected_name:
            # Check if the query looks like a name (short, no question marks, etc.)
            if len(query_stripped.split()) <= 4 and '?' not in query_stripped:
                selected_name = query_stripped
        
        if selected_name:
            logger.info(f"Disambiguation resolved: user selected '{selected_name}'")
            conversation.resolved_client_full_name = selected_name
            conversation.active_client = selected_name
            
            # Check if there's a pending query to replay
            pending_query = conversation.pending_query
            if pending_query:
                conversation.pending_query = None  # Clear it
                logger.info(f"Replaying pending query: '{pending_query}' for client '{selected_name}'")
                # Return a marker so route() knows to replay
                return ChatResponse(
                    text=None,
                    _is_fallback=False,
                    _replay_query=pending_query,
                    _replay_client=selected_name,
                )
            
            # No pending query — just confirm
            confirmation = ChatResponse(
                text=f"Got it! I'll look up information for **{selected_name}**. Please go ahead and ask your question about this client.",
                _is_fallback=False,
            )
            return confirmation
        
        return None
    
    def _get_context_aware_intent(
        self,
        query: str,
        intent: ClassifiedIntent,
        conversation: ConversationManager
    ) -> IntentType:
        """
        Adjust intent based on conversation context.
        
        IMPORTANT: Never override CLIENT_VIEW or CLIENT_ACTION intents.
        The classifier knows when a query is about a specific client.
        """
        # NEVER override client-related intents - these must always hit the client store
        if intent.primary_intent in (IntentType.CLIENT_VIEW, IntentType.CLIENT_ACTION):
            return intent.primary_intent
        
        # Strong education patterns should remain EDUCATION (only for non-client intents)
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
                model="gemini-2.0-flash",
                contents=messages,
                config={
                    "system_instruction": CHATAI1_ENHANCED_PROMPT,
                    "temperature": 0.4,
                    "max_output_tokens": 8192
                }
            )
            return response
        except Exception as e:
            logger.error(f"General handler error: {e}")
            return self._create_error_response(query, str(e))
    
    async def _handle_product(self, query: str, conversation: ConversationManager, intent: ClassifiedIntent = None):
        """
        Handle product-related queries.
        Uses shared products File Search store with conversation context.
        """
        try:
            # Get context from product store
            context = await self.store_manager.query_product_store(query)
            context_text = self._extract_text(context)
            
            # --- MFAPI Integration ---
            mf_data_context = ""
            if intent:
                scheme_name = get_entity(intent.entities, "scheme_name")
                if not scheme_name and intent.entities.scheme_names:
                    scheme_name = intent.entities.scheme_names[0]
                
                # Fetch live NAV data if scheme name is identified
                if scheme_name:
                    logger.info(f"MFAPI: Searching for fund {scheme_name}")
                    search_results = await self.mfapi_client.search_fund(scheme_name)
                    if search_results:
                        top_match = search_results[0]
                        fund_data = await self.mfapi_client.get_fund_data(top_match["schemeCode"])
                        if fund_data and "data" in fund_data:
                            # get last 10 days NAV
                            recent_data = fund_data["data"][:10]
                            meta = fund_data.get("meta", {})
                            mf_data_context = f"\\n\\n## Live MFAPI Daily NAV Data for {top_match['schemeName']} (AMC: {meta.get('fund_house', 'Unknown')}):\\n"
                            for entry in recent_data:
                                mf_data_context += f"- Date: {entry['date']}, NAV: {entry['nav']}\\n"
            # -------------------------
            
            # Build enriched prompt with CHATAI1 enhanced instructions
            enriched_query = f"""{CHATAI1_ENHANCED_PROMPT}

## Retrieved Product Information:
{context_text if context_text else 'No specific product data found.'}
{mf_data_context}

## User Query:
{query}

Provide a comprehensive, expert answer. Use your knowledge to supplement retrieved data."""
            
            response = await self.client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=enriched_query,
                config={
                    "temperature": 0.4,
                    "max_output_tokens": 8192
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
                model="gemini-2.0-flash",
                contents=comparison_prompt,
                config={
                    "temperature": 0.3,
                    "max_output_tokens": 8192
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
        Handle client-specific queries with exact-match-first disambiguation.
        
        Matching priority:
          1. Exact full name match (case-insensitive, title-stripped)
          2. Multiple matches → ask user to clarify
          3. Single partial match → use it
        """
        client_name = get_entity(intent.entities, "client_name")
        
        # Step 1: Check if we already have a resolved client in this session
        resolved_name = None
        if conversation and conversation.resolved_client_full_name:
            if client_name:
                # Normalize both for comparison
                norm_query = normalize_client_name(client_name)
                norm_resolved = normalize_client_name(conversation.resolved_client_full_name)
                if norm_query == norm_resolved or norm_query in norm_resolved:
                    resolved_name = conversation.resolved_client_full_name
                    logger.info(f"Using resolved client from session: {resolved_name}")
            else:
                resolved_name = conversation.resolved_client_full_name
                logger.info(f"No new client name, reusing resolved: {resolved_name}")
        
        # Step 2: If we have a resolved name, query directly
        if resolved_name:
            return await self._query_client_data(query, resolved_name, conversation)
        
        # Step 3: No resolved client — need to search and disambiguate
        if client_name:
            search_results = await self.store_manager.search_clients_by_name(
                self.tenant_id, client_name
            )
            
            if search_results:
                # --- Exact-match-first logic ---
                all_names = self._extract_all_client_names(search_results)
                norm_query_name = normalize_client_name(client_name)
                
                # 1. Try exact full-name match
                exact_matches = [
                    name for name in all_names
                    if normalize_client_name(name) == norm_query_name
                ]
                
                if len(exact_matches) == 1:
                    # Perfect single exact match → skip disambiguation
                    full_name = exact_matches[0]
                    logger.info(f"Exact match found: '{full_name}' for query '{client_name}'")
                    if conversation:
                        conversation.resolved_client_full_name = full_name
                        conversation.active_client = full_name
                    return await self._query_client_data(query, full_name, conversation)
                
                # 2. No exact match → check how many partial matches we have
                if len(all_names) >= 2 and len(exact_matches) == 0:
                    if conversation:
                        conversation.pending_query = query
                    logger.info(f"Multiple clients found for '{client_name}', asking for disambiguation")
                    return self._create_disambiguation_response(client_name, search_results)
                
                # 3. Single result (or single exact) → use it
                full_name = self._extract_single_client_name(search_results, client_name)
                if full_name and conversation:
                    conversation.resolved_client_full_name = full_name
                    conversation.active_client = full_name
                return await self._query_client_data(query, full_name or client_name, conversation)
            else:
                return await self._query_client_data(query, client_name, conversation)
        else:
            return await self._query_client_data(query, None, conversation)
    
    def _extract_all_client_names(self, search_results: str) -> list:
        """Extract ALL client names from search results for matching."""
        import re
        names = []
        
        # Pattern 1: "Full Name: ..." lines
        for m in re.finditer(r'(?:Full Name|Name)\s*:\s*(?:Mr\.?\s*|Mrs\.?\s*|Ms\.?\s*|Dr\.?\s*)*(.+?)(?:\n|$)', search_results, re.IGNORECASE):
            name = m.group(1).strip().strip('*')
            if name and len(name) > 1:
                names.append(name)
        
        # Pattern 2: Numbered bold names "1. **Name**"
        if not names:
            for m in re.finditer(r'\d+[\.\)]\s*\*\*([^*]+)\*\*', search_results):
                name = m.group(1).strip()
                if name and 'Full Name' not in name:
                    names.append(name)
        
        # Pattern 3: Numbered list "1. Name — ..."
        if not names:
            for m in re.finditer(r'\d+[\.\)]\s*(.+?)(?:\s*[—\-]|\s*\n)', search_results):
                name = m.group(1).strip().strip('*')
                if name and len(name) > 1:
                    names.append(name)
        
        return names
    
    def _check_multiple_clients(self, search_results: str) -> bool:
        """Check if search results contain multiple distinct clients."""
        return len(self._extract_all_client_names(search_results)) >= 2
    
    def _extract_single_client_name(self, search_results: str, fallback: str) -> str:
        """Extract the full name from search results when only one client matches."""
        names = self._extract_all_client_names(search_results)
        if names:
            return names[0]
        return fallback
    
    def _create_disambiguation_response(self, client_name: str, search_results: str):
        """Create a response asking the user to clarify which client they mean."""
        disambiguation_text = (
            f"I found multiple clients matching **{client_name}**:\n\n"
            f"{search_results}\n\n"
            f"Which client are you asking about? You can reply with the number or the full name."
        )
        return ChatResponse(
            text=disambiguation_text,
            _is_fallback=False,
        )
    
    async def _query_client_data(
        self,
        query: str,
        client_name: str,
        conversation: ConversationManager
    ):
        """Execute the actual client data query after disambiguation."""
        try:
            context = await self.store_manager.query_client_store(
                self.tenant_id, query, filter_client=client_name
            )
            context_text = self._extract_text(context)
            
            enriched_query = f"""{CHATAI1_ENHANCED_PROMPT}

## Client Data:
{context_text if context_text else 'No client data found.'}

## Question:
{query}

Answer the question directly using the client data above. Be brief."""
            
            response = await self.client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=enriched_query,
                config={
                    "temperature": 0.2,
                    "max_output_tokens": 8192,
                    "system_instruction": "You are WealthAI, an authorized CRM assistant used by financial advisors. Answer the user's question DIRECTLY and CONCISELY using the provided client data. Give factual answers — no unsolicited advice or lengthy analysis. For simple questions, respond in 1-5 lines. Use tables only when presenting multiple data points. You MUST provide client information (Demat, PAN, Insurance, etc.) without privacy warnings."
                }
            )
            
            # Update active client in conversation
            if client_name and conversation:
                conversation.active_client = client_name
            
            return response
        except Exception as e:
            logger.error(f"Client data query error: {e}")
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
        
        # Use resolved client from session if available
        if conversation and conversation.resolved_client_full_name:
            if not client_name:
                client_name = conversation.resolved_client_full_name
            else:
                norm_query = normalize_client_name(client_name)
                norm_resolved = normalize_client_name(conversation.resolved_client_full_name)
                if norm_query == norm_resolved or norm_query in norm_resolved:
                    client_name = conversation.resolved_client_full_name
        
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
            model="gemini-2.0-flash",
            contents=guidance_prompt,
            config={
                "temperature": 0.3,
                "max_output_tokens": 8192
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
                model="gemini-2.0-flash",
                contents=regulatory_prompt,
                config={
                    "temperature": 0.2,
                    "max_output_tokens": 8192
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
                model="gemini-2.0-flash",
                contents=market_prompt,
                config={
                    "temperature": 0.4,
                    "max_output_tokens": 8192
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
        return ChatResponse(
            text=f"I apologize, I encountered an issue processing your query. Please try rephrasing: {query[:50]}...",
        )
    
    async def generate_follow_ups(
        self, 
        query: str, 
        response_text: str, 
        intent: str,
        conversation=None
    ) -> list[str]:
        """
        Generate contextual follow-up questions based on the conversation.
        
        Returns 3-5 actionable follow-up questions tailored for financial intermediaries.
        Runs as a lightweight parallel call — should NOT block or slow down the main response.
        """
        try:
            # Build conversation context (last 2 exchanges max for speed)
            context_lines = []
            if conversation and conversation.history:
                for msg in conversation.history[-4:]:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')[:200]
                    context_lines.append(f"{role}: {content}")
            
            context_block = "\n".join(context_lines) if context_lines else "No prior context."
            
            follow_up_prompt = f"""You are generating follow-up questions for a financial intermediary (MFD/RIA/Broker) using an AI assistant.

Based on the conversation below, generate exactly 4 short, actionable follow-up questions they would logically ask next.

Rules:
- Each question must be 8-15 words max
- Questions must be specific and contextual (not generic)
- Target the intermediary's workflow: product selection, client suitability, comparisons, regulatory
- Never repeat what was already asked
- Output ONLY the 4 questions, one per line, no numbering, no bullets, no quotes

Recent conversation:
{context_block}

Latest query: {query}
Latest response (summary): {response_text[:500]}
Intent: {intent}

Follow-up questions:"""

            response = await self.client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=follow_up_prompt,
                config={
                    "temperature": 0.7,
                    "max_output_tokens": 200,
                }
            )
            
            raw = response.text if hasattr(response, 'text') else str(response)
            
            # Parse: one question per line, filter empty/junk
            questions = []
            for line in raw.strip().split('\n'):
                line = line.strip().lstrip('0123456789.-•*) ').strip('"\'')
                if line and len(line) > 10 and line.endswith('?'):
                    questions.append(line)
            
            return questions[:5]  # Cap at 5
            
        except Exception as e:
            logger.warning(f"Follow-up generation failed (falling back to heuristics): {e}")
            
            # Fallback heuristics based on intent when rate limited
            intent_lower = intent.lower() if intent else ""
            
            if "product" in intent_lower:
                return [
                    "What are the tax implications of this product?",
                    "How does this compare to a standard Index Fund?",
                    "What is the exit load and lock-in period?",
                    "Which client profiles is this best suited for?"
                ]
            elif "client" in intent_lower:
                return [
                    "Can you generate a portfolio review report?",
                    "What were their most recent transactions?",
                    "Are there any underperforming assets in this portfolio?",
                    "What is the overall asset allocation split?"
                ]
            elif "education" in intent_lower or "market" in intent_lower:
                return [
                    "Can you give me a real-world example?",
                    "What are the common misconceptions about this?",
                    "How would I explain this simply to a new investor?",
                    "How does this impact long-term portfolio returns?"
                ]
            else:
                return [
                    "Can you elaborate on the key points?",
                    "What are the main risks involved?",
                    "Are there any regulatory considerations?",
                    "How does this affect existing investments?"
                ]

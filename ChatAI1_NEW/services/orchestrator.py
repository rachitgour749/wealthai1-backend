# app/services/orchestrator.py
"""Main orchestration logic for chat handling"""
import logging
import uuid
from typing import List
from ChatAI1_NEW.chatai1_new_schemas import (
    ChatRequest, ChatResponse, SessionState, Message,
    RouterOutput, Category, DomainRelevance
)
from ChatAI1_NEW.services.llm_client import LLMClient
from ChatAI1_NEW.services.rag_client import RAGClient
from ChatAI1_NEW.services.session_store import SessionStore
from ChatAI1_NEW.chatai1_new_prompts import (
    ROUTER_SYSTEM_PROMPT, BASE_SYSTEM_PROMPT,
    MF_SYSTEM_PROMPT, INSURANCE_SYSTEM_PROMPT, STOCKS_SYSTEM_PROMPT
)
from ChatAI1_NEW.chatai1_new_config import settings

logger = logging.getLogger(__name__)


class Orchestrator:
    """Main orchestrator for handling chat requests"""

    def __init__(
            self,
            llm_client: LLMClient,
            rag_client: RAGClient,
            session_store: SessionStore
    ):
        """
        Initialize orchestrator with service dependencies

        Args:
            llm_client: LLM client for Gemini API calls
            rag_client: RAG client for context retrieval
            session_store: Session storage manager (now PostgreSQL-backed)
        """
        self.llm_client = llm_client
        self.rag_client = rag_client
        self.session_store = session_store
        logger.info("Initialized Orchestrator")

    def _build_conversation_snippet(self, messages: List[Message]) -> str:
        """
        Build conversation snippet from recent messages

        Args:
            messages: List of recent messages

        Returns:
            Formatted conversation snippet string
        """
        if not messages:
            return ""

        # Take last N turns (2*MAX_SESSION_TURNS messages)
        recent_messages = messages[-(settings.MAX_SESSION_TURNS * 2):]

        # Format as "User: ... | Assistant: ... | ..."
        snippet_parts = []
        for msg in recent_messages:
            role = "User" if msg.role == "user" else "Assistant"
            # Truncate very long messages
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            snippet_parts.append(f"{role}: {content}")

        return " | ".join(snippet_parts)

    def _get_domain_prompts(self, router_output: RouterOutput) -> List[str]:
        """
        Get relevant domain system prompts based on router output

        Args:
            router_output: Router classification output

        Returns:
            List of domain-specific system prompts
        """
        prompts = []

        # Map categories to prompts
        category_to_prompt = {
            Category.MUTUAL_FUNDS: MF_SYSTEM_PROMPT,
            Category.INSURANCE: INSURANCE_SYSTEM_PROMPT,
            Category.STOCK_MARKETS: STOCKS_SYSTEM_PROMPT
        }

        # Add primary category prompt
        if router_output.primary_category:
            prompt = category_to_prompt.get(router_output.primary_category)
            if prompt:
                prompts.append(prompt)

        # Add additional category prompts (avoid duplicates)
        for category in router_output.additional_categories:
            prompt = category_to_prompt.get(category)
            if prompt and prompt not in prompts:
                prompts.append(prompt)

        return prompts

    async def handle_chat(self, request: ChatRequest) -> ChatResponse:
        """
        Main chat handling logic

        Args:
            request: Incoming chat request

        Returns:
            ChatResponse with answer and metadata
        """
        logger.info(f"Handling chat request from {request.user_email}")

        # Step 1: Get or create conversation ID
        conversation_id = request.conversation_id or str(uuid.uuid4())
        logger.info(f"Using conversation_id: {conversation_id}")

        # Step 2: Load or initialize session state
        session = await self.session_store.get_session(conversation_id)
        if not session:
            session = SessionState(
                conversation_id=conversation_id,
                session_metadata={"user_email": request.user_email}
            )
            logger.info(f"Created new session for {conversation_id}")

        # Step 3: Build conversation snippet
        conversation_snippet = self._build_conversation_snippet(session.messages)

        # Step 4: Call router model
        router_payload = {
            "user_query": request.message,
            "conversation_snippet": conversation_snippet,
            "metadata": {
                "user_role": request.user_role.value if request.user_role else None,
                "zoho_crm_data_available": None  # Will be determined after RAG call
            }
        }

        router_json = self.llm_client.call_router_model(
            ROUTER_SYSTEM_PROMPT,
            router_payload
        )

        # Parse router output
        router_output = RouterOutput(**router_json)
        logger.info(f"Router classification: domain={router_output.domain_relevance.value}")

        # Step 5: Handle out-of-scope queries
        if router_output.domain_relevance == DomainRelevance.OUT_OF_SCOPE:
            logger.info("Query is out of scope")

            # Generate simple fallback answer
            fallback_answer = (
                "I'm ChatAI1, and I specialize in helping Indian mutual fund distributors, "
                "insurance agents, and stock brokers with their professional work. "
                "Your query seems to be outside my area of expertise. "
                "Is there anything related to mutual funds, insurance, or stock markets "
                "that I can help you with?"
            )

            # Update session with user message and assistant response
            session.messages.append(Message(role="user", content=request.message))
            session.messages.append(Message(role="assistant", content=fallback_answer))

            # Keep only last N turns
            if len(session.messages) > settings.MAX_SESSION_TURNS * 2:
                session.messages = session.messages[-(settings.MAX_SESSION_TURNS * 2):]

            await self.session_store.save_session(conversation_id, session)

            return ChatResponse(
                reply=fallback_answer,
                router_metadata=router_json,
                used_user_context=False,
                used_common_kb=False,
                conversation_id=conversation_id
            )

        # Step 6: Fetch RAG contexts for in-domain queries
        user_context = ""
        kb_context = ""
        user_context_available = False

        # Fetch user-specific context if needed
        if router_output.use_zoho_crm_data:
            logger.info("Fetching user-specific context from RAG")
            user_context_available, user_context = await self.rag_client.get_user_context(
                request.user_email,
                request.message
            )

            # Update router output with actual availability
            if user_context_available:
                router_output.zoho_crm_data_status = "available"
                router_json["zoho_crm_data_status"] = "available"
            else:
                router_output.zoho_crm_data_status = "missing"
                router_json["zoho_crm_data_status"] = "missing"

        # Fetch common KB context if needed
        if router_output.use_common_kb:
            logger.info("Fetching common KB context from RAG")

            # Build domains list
            domains = []
            if router_output.primary_category:
                domains.append(router_output.primary_category.value)
            domains.extend([cat.value for cat in router_output.additional_categories])

            kb_context = await self.rag_client.get_common_kb_context(
                domains,
                request.message
            )

        # Step 7: Compose system prompts for answerer
        system_prompts = [BASE_SYSTEM_PROMPT]

        # Add domain-specific prompts
        domain_prompts = self._get_domain_prompts(router_output)
        system_prompts.extend(domain_prompts)

        logger.info(f"Using {len(system_prompts)} system prompts for answer generation")

        # Step 8: Call answer model
        answer = self.llm_client.call_answer_model(
            system_prompts=system_prompts,
            router_json=router_json,
            user_context=user_context,
            kb_context=kb_context,
            user_query=request.message,
            conversation_snippet=conversation_snippet
        )

        # Step 9: Update session state
        session.messages.append(Message(role="user", content=request.message))
        session.messages.append(Message(role="assistant", content=answer))

        # Keep only last N turns
        if len(session.messages) > settings.MAX_SESSION_TURNS * 2:
            session.messages = session.messages[-(settings.MAX_SESSION_TURNS * 2):]

        await self.session_store.save_session(conversation_id, session)

        # Step 10: Build and return response
        return ChatResponse(
            reply=answer,
            router_metadata=router_json,
            used_user_context=user_context_available and bool(user_context),
            used_common_kb=router_output.use_common_kb and bool(kb_context),
            conversation_id=conversation_id
        )
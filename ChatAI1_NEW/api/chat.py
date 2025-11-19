# app/api/chat.py
"""Chat API endpoints"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from ChatAI1_NEW.chatai1_new_schemas import ChatRequest, ChatResponse
from ChatAI1_NEW.services.orchestrator import Orchestrator
from ChatAI1_NEW.services.llm_client import LLMClient
from ChatAI1_NEW.services.rag_client import RAGClient
from ChatAI1_NEW.services.session_store import SessionStore
from ChatAI1_NEW.database import get_db

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/chat", tags=["chat"])

# Singleton instances for stateless services
_llm_client = None
_rag_client = None


def get_llm_client() -> LLMClient:
    """Get singleton LLM client"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def get_rag_client() -> RAGClient:
    """Get singleton RAG client"""
    global _rag_client
    if _rag_client is None:
        _rag_client = RAGClient()
    return _rag_client


async def get_orchestrator(
        db: AsyncSession = Depends(get_db),
        llm_client: LLMClient = Depends(get_llm_client),
        rag_client: RAGClient = Depends(get_rag_client)
) -> Orchestrator:
    """
    Dependency to get orchestrator instance

    Args:
        db: Database session (injected per request)
        llm_client: Singleton LLM client
        rag_client: Singleton RAG client

    Returns:
        Orchestrator instance
    """
    session_store = SessionStore(db)
    return Orchestrator(llm_client, rag_client, session_store)


@router.post("", response_model=ChatResponse)
async def chat(
        request: ChatRequest,
        orchestrator: Orchestrator = Depends(get_orchestrator)
) -> ChatResponse:
    """
    Main chat endpoint

    Args:
        request: ChatRequest with user message and metadata
        orchestrator: Injected orchestrator service

    Returns:
        ChatResponse with AI-generated reply and metadata
    """
    try:
        logger.info(f"Received chat request from {request.user_email}")
        response = await orchestrator.handle_chat(request)
        logger.info(f"Successfully processed chat request")
        return response

    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while processing chat: {str(e)}"
        )
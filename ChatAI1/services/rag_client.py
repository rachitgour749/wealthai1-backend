# app/services/rag_client.py
"""RAG client for external FastAPI RAG service on EC2"""
import logging
from typing import List, Tuple
import httpx
from ChatAI1.chatai1_config import settings

logger = logging.getLogger(__name__)


class RAGClient:
    """Client for interacting with external RAG service"""

    def __init__(self):
        """Initialize RAG client with base URL"""
        self.base_url = settings.RAG_SERVICE_BASE_URL
        self.timeout = 10.0  # seconds
        logger.info(f"Initialized RAGClient with base_url={self.base_url}")

    async def get_user_context(self, user_email: str, query: str) -> Tuple[bool, str]:
        """
        Fetch user-specific context from RAG service

        Args:
            user_email: User's email/identifier
            query: User query for context retrieval

        Returns:
            Tuple of (available: bool, context: str)
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/rag/user-context",
                    json={
                        "user_email": user_email,
                        "query": query
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    available = data.get("available", False)
                    context = data.get("context", "")

                    logger.info(f"User context retrieved: available={available}, length={len(context)}")
                    return available, context
                else:
                    logger.warning(f"RAG service returned status {response.status_code} for user context")
                    return False, ""

        except httpx.TimeoutException:
            logger.error("Timeout while fetching user context from RAG service")
            return False, ""
        except Exception as e:
            logger.error(f"Error fetching user context: {e}")
            return False, ""

    async def get_common_kb_context(self, domains: List[str], query: str) -> str:
        """
        Fetch common knowledgebase context from RAG service

        Args:
            domains: List of domain categories (e.g., ["Mutual Funds", "Insurance"])
            query: User query for context retrieval

        Returns:
            Context text (empty string if none found)
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/rag/common-kb",
                    json={
                        "domains": domains,
                        "query": query
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    context = data.get("context", "")

                    logger.info(f"Common KB context retrieved: length={len(context)}")
                    return context
                else:
                    logger.warning(f"RAG service returned status {response.status_code} for common KB")
                    return ""

        except httpx.TimeoutException:
            logger.error("Timeout while fetching common KB context from RAG service")
            return ""
        except Exception as e:
            logger.error(f"Error fetching common KB context: {e}")
            return ""
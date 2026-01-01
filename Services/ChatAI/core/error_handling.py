"""
Error Handling and Resilience for Financial Advisor Chatbot

Provides:
- Dead Letter Queue for failed webhooks
- Retry handler with exponential backoff
- Graceful error responses
"""

import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)


class DeadLetterQueue:
    """
    Stores failed operations for retry.
    Used for webhook failures and failed sync operations.
    """
    
    def __init__(self, max_size: int = 1000):
        self.queue: deque = deque(maxlen=max_size)
        self._processing = False
    
    async def put(self, item: dict):
        """Add failed item to queue."""
        item["failed_at"] = datetime.now().isoformat()
        item["retry_count"] = item.get("retry_count", 0) + 1
        self.queue.append(item)
        logger.warning(f"DLQ: Added item, queue size: {len(self.queue)}")
    
    async def process_retries(
        self, 
        handler: Callable, 
        max_retries: int = 3,
        batch_size: int = 10
    ):
        """
        Process queued items with retry logic.
        
        Args:
            handler: Async function to process each item
            max_retries: Maximum retry attempts per item
            batch_size: Number of items to process per batch
        """
        if self._processing:
            return
        
        self._processing = True
        processed = 0
        
        try:
            while self.queue and processed < batch_size:
                item = self.queue.popleft()
                
                if item["retry_count"] > max_retries:
                    await self._move_to_permanent_failure(item)
                    continue
                
                try:
                    await handler(item["payload"])
                    logger.info(f"DLQ: Successfully processed item after {item['retry_count']} retries")
                    processed += 1
                except Exception as e:
                    logger.error(f"DLQ: Retry failed: {e}")
                    await self.put(item)
        finally:
            self._processing = False
    
    async def _move_to_permanent_failure(self, item: dict):
        """Log permanently failed items for manual review."""
        logger.error(f"DLQ: Permanent failure after {item['retry_count']} retries: {item}")
        # In production, store to database or send alert
    
    def size(self) -> int:
        """Get current queue size."""
        return len(self.queue)


# Global DLQ instance
dlq = DeadLetterQueue()


class RetryHandler:
    """Retry operations with exponential backoff."""
    
    @staticmethod
    async def with_retry(
        func: Callable,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exceptions: tuple = (Exception,)
    ) -> Any:
        """
        Execute function with exponential backoff retry.
        
        Args:
            func: Async function to execute
            max_retries: Maximum retry attempts
            base_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries
            exceptions: Exception types to catch and retry
        
        Returns:
            Result of successful function call
        
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await func()
            except exceptions as e:
                last_exception = e
                
                if attempt == max_retries:
                    logger.error(f"Retry: All {max_retries} attempts failed: {e}")
                    raise
                
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    f"Retry: Attempt {attempt + 1} failed, "
                    f"retrying in {delay:.1f}s: {e}"
                )
                await asyncio.sleep(delay)
        
        raise last_exception


class GracefulError:
    """Standard error responses for the API."""
    
    @staticmethod
    def not_found(resource: str) -> dict:
        return {
            "error": "not_found",
            "message": f"{resource} not found",
            "suggestion": "Please check the name and try again"
        }
    
    @staticmethod
    def rate_limited() -> dict:
        return {
            "error": "rate_limited",
            "message": "Too many requests. Please wait before trying again.",
            "retry_after": 60
        }
    
    @staticmethod
    def service_unavailable(service: str) -> dict:
        return {
            "error": "service_unavailable",
            "message": f"{service} is temporarily unavailable",
            "suggestion": "Please try again in a few moments"
        }
    
    @staticmethod
    def fallback_response(query: str) -> dict:
        return {
            "response": "I couldn't retrieve specific data for your query. "
                       "Here's a general response based on my knowledge.",
            "is_fallback": True,
            "original_query": query
        }

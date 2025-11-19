# app/tasks.py
"""Background tasks for maintenance"""
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.session_store import SessionStore

logger = logging.getLogger(__name__)


async def cleanup_expired_sessions_task(db: AsyncSession):
    """Background task to cleanup expired sessions"""
    try:
        session_store = SessionStore(db)
        deleted = await session_store.cleanup_expired_sessions()
        logger.info(f"Session cleanup: deleted {deleted} expired sessions")
    except Exception as e:
        logger.error(f"Error in session cleanup task: {e}")
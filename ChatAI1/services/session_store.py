# app/services/session_store.py
"""Session management with PostgreSQL storage"""
import logging
from typing import Optional
from datetime import datetime, timedelta
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from ChatAI1.chatai1_schemas import SessionState, Message
from ChatAI1.chatai1_models import Session as SessionModel, SessionMessage as SessionMessageModel
from ChatAI1.chatai1_config import settings

logger = logging.getLogger(__name__)


class SessionStore:
    """
    PostgreSQL-based session store using SQLAlchemy
    """

    def __init__(self, db_session: AsyncSession):
        """
        Initialize session store with database session

        Args:
            db_session: SQLAlchemy async session
        """
        self.db = db_session
        logger.debug("Initialized PostgreSQL SessionStore")

    async def get_session(self, conversation_id: str) -> Optional[SessionState]:
        """
        Retrieve session state by conversation ID

        Args:
            conversation_id: Unique session identifier

        Returns:
            SessionState if found and not expired, None otherwise
        """
        try:
            # Query session with messages
            stmt = (
                select(SessionModel)
                .options(selectinload(SessionModel.messages))
                .where(SessionModel.conversation_id == conversation_id)
                .where(SessionModel.expires_at > datetime.utcnow())
            )

            result = await self.db.execute(stmt)
            session_model = result.scalar_one_or_none()

            if not session_model:
                logger.debug(f"No active session found for {conversation_id}")
                return None

            # Convert ORM model to Pydantic model
            messages = [
                Message(role=msg.role, content=msg.content)
                for msg in sorted(session_model.messages, key=lambda x: x.sequence_number)
            ]

            session_state = SessionState(
                conversation_id=session_model.conversation_id,
                messages=messages,
                summary=session_model.summary,
                session_metadata=session_model.session_metadata or {}
            )

            logger.debug(f"Retrieved session {conversation_id} with {len(messages)} messages")
            return session_state

        except Exception as e:
            logger.error(f"Error retrieving session {conversation_id}: {e}")
            return None

    async def save_session(self, conversation_id: str, state: SessionState) -> None:
        """
        Save or update session state

        Args:
            conversation_id: Unique session identifier
            state: SessionState to save
        """
        try:
            # Check if session exists
            stmt = select(SessionModel).where(SessionModel.conversation_id == conversation_id)
            result = await self.db.execute(stmt)
            session_model = result.scalar_one_or_none()

            if session_model:
                # Update existing session
                session_model.summary = state.summary
                session_model.session_metadata = state.session_metadata
                session_model.updated_at = datetime.utcnow()
                session_model.expires_at = datetime.utcnow() + timedelta(hours=settings.SESSION_TTL_HOURS)

                # Delete old messages and insert new ones
                await self.db.execute(
                    delete(SessionMessageModel).where(SessionMessageModel.session_id == session_model.id)
                )

            else:
                # Create new session
                session_model = SessionModel(
                    conversation_id=conversation_id,
                    user_email=state.session_metadata.get("user_email", "unknown"),
                    summary=state.summary,
                    session_metadata=state.session_metadata
                )
                self.db.add(session_model)
                await self.db.flush()  # Get the session ID

            # Insert messages
            for idx, msg in enumerate(state.messages):
                message_model = SessionMessageModel(
                    session_id=session_model.id,
                    role=msg.role,
                    content=msg.content,
                    sequence_number=idx
                )
                self.db.add(message_model)

            await self.db.commit()
            logger.debug(f"Saved session {conversation_id} with {len(state.messages)} messages")

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error saving session {conversation_id}: {e}")
            raise

    async def delete_session(self, conversation_id: str) -> None:
        """
        Delete a session

        Args:
            conversation_id: Unique session identifier
        """
        try:
            stmt = delete(SessionModel).where(SessionModel.conversation_id == conversation_id)
            await self.db.execute(stmt)
            await self.db.commit()
            logger.debug(f"Deleted session {conversation_id}")

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error deleting session {conversation_id}: {e}")
            raise

    async def cleanup_expired_sessions(self) -> int:
        """
        Delete all expired sessions

        Returns:
            Number of sessions deleted
        """
        try:
            stmt = delete(SessionModel).where(SessionModel.expires_at < datetime.utcnow())
            result = await self.db.execute(stmt)
            await self.db.commit()
            deleted_count = result.rowcount
            logger.info(f"Cleaned up {deleted_count} expired sessions")
            return deleted_count

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error cleaning up expired sessions: {e}")
            raise
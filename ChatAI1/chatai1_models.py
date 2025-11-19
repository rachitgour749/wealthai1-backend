# app/models.py
"""SQLAlchemy ORM models for database tables"""
from datetime import datetime, timedelta
from sqlalchemy import Column, String, Text, Integer, ForeignKey, CheckConstraint, TIMESTAMP, BigInteger
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from ChatAI1.database import Base
from ChatAI1.chatai1_config import settings


class Session(Base):
    """Sessions table model"""
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(String(255), unique=True, nullable=False, index=True)
    user_email = Column(String(255), nullable=False, index=True)
    summary = Column(Text, nullable=True)
    session_metadata = Column(JSONB, default=dict, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)
    expires_at = Column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.utcnow() + timedelta(hours=settings.SESSION_TTL_HOURS),
        nullable=False,
        index=True
    )

    # Relationship to messages
    messages = relationship("SessionMessage", back_populates="session", cascade="all, delete-orphan")


class SessionMessage(Base):
    """Session messages table model"""
    __tablename__ = "session_messages"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    sequence_number = Column(Integer, nullable=False)

    # Relationship to session
    session = relationship("Session", back_populates="messages")

    # Check constraint for role
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="check_role_valid"),
    )
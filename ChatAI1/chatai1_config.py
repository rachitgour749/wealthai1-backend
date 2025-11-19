# app/chatai1_config.py
"""Configuration management using Pydantic Settings"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Gemini API Configuration
    GEMINI_API_KEY: str
    ROUTER_MODEL_NAME: str = "gemini-1.5-flash"
    ANSWER_MODEL_NAME: str = "gemini-1.5-pro"

    # RAG Service Configuration
    RAG_SERVICE_BASE_URL: str = "https://rag-service.example.com"

    # Session Configuration
    MAX_SESSION_TURNS: int = 5  # Keep last 5 user+assistant turns
    SESSION_TTL_HOURS: int = 24

    # Application Configuration
    APP_NAME: str = "ChatAI1"
    APP_VERSION: str = "1.0.0"

    # PostgreSQL Configuration (Neon)
    DATABASE_URL: str = "postgresql://neondb_owner:npg_jyQ5oGFgsW9E@ep-empty-queen-a1oizpt3-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
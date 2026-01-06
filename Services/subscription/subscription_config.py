# Services/Subscription/subscription_config.py
"""Configuration management using Pydantic Settings"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Subscription service settings loaded from environment variables"""
    
    # Application Configuration
    APP_NAME: str = "Subscription Service"
    APP_VERSION: str = "1.0.0"
    
    # Trial Configuration
    DEFAULT_TRIAL_DAYS: int = 7
    MAX_TRIAL_EXTENSION_DAYS: int = 30
    
    # ChatAI Token Configuration
    CHATAI_DEFAULT_TOKENS: int = 100000

    # Optional external API keys (ignored by subscription logic but allowed in env)
    GEMINI_API_KEY: Optional[str] = None
    
    # Database Configuration (inherited from main app)
    DATABASE_URL: str = "postgresql+asyncpg://neondb_owner:npg_jyQ5oGFgsW9E@ep-empty-queen-a1oizpt3-pooler.ap-southeast-1.aws.neon.tech/neondb"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = 'ignore'  # Allow extra fields from .env (like APP_ENV, LOG_LEVEL, etc.)


# Global settings instance
settings = Settings()

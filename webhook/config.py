import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class WebhookConfig:
    """Configuration class for webhook settings"""
    
    # Database settings
    # Use absolute path to ensure database is found regardless of working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    default_db_path = os.path.join(parent_dir, "unified_etf_data.sqlite")
    DATABASE = os.getenv("WEBHOOK_DATABASE", default_db_path)
    
    # Webhook settings
    WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
    WEBHOOK_TIMEOUT = int(os.getenv("WEBHOOK_TIMEOUT", "30"))  # seconds
    
    # Strategy settings
    MAX_STRATEGIES_PER_USER = int(os.getenv("MAX_STRATEGIES_PER_USER", "100"))
    STRATEGY_NAME_MAX_LENGTH = int(os.getenv("STRATEGY_NAME_MAX_LENGTH", "255"))
    
    # JSON generation settings
    MAX_JSON_SIZE = int(os.getenv("MAX_JSON_SIZE", "1048576"))  # 1MB
    JSON_RETENTION_DAYS = int(os.getenv("JSON_RETENTION_DAYS", "30"))
    
    # CORS origins
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ]
    
    # Retry settings
    MAX_RETRIES = int(os.getenv("WEBHOOK_MAX_RETRIES", "3"))
    RETRY_DELAY = int(os.getenv("WEBHOOK_RETRY_DELAY", "5"))  # seconds
    
    # Logging settings
    LOG_LEVEL = os.getenv("WEBHOOK_LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("WEBHOOK_LOG_FILE", "webhook.log")
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate webhook configuration"""
        if not cls.DATABASE:
            return False
        return True
    
    @classmethod
    def get_database_path(cls) -> str:
        """Get the database path"""
        return cls.DATABASE

# Configuration dictionary for different environments
config = {
    'default': WebhookConfig,
    'development': WebhookConfig,
    'production': WebhookConfig,
    'testing': WebhookConfig
}

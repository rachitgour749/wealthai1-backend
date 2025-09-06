import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PaymentConfig:
    """Configuration class for payment settings"""
    
    # Razorpay Configuration
    RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "rzp_live_RCG1Ab7k1DAxkN")
    RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "Ch9lJpBVgc0ppJWm6zzNM2Ml")
    
    # Environment (test/live)
    ENVIRONMENT = os.getenv("PAYMENT_ENVIRONMENT", "test")  # test or live
    
    # Currency settings
    DEFAULT_CURRENCY = "INR"
    
    # Payment capture settings
    PAYMENT_CAPTURE = 1  # 1 for automatic capture
    
    # Webhook settings
    WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
    
    # Database settings
    # Use absolute path to ensure database is found regardless of working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    default_db_path = os.path.join(parent_dir, "unified_etf_data.sqlite")
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{default_db_path}")
    
    # CORS origins
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ]
    
    # Payment timeout (in seconds)
    PAYMENT_TIMEOUT = 1800  # 30 minutes
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds
    
    @classmethod
    def is_test_environment(cls) -> bool:
        """Check if running in test environment"""
        return cls.ENVIRONMENT.lower() == "test"
    
    @classmethod
    def get_razorpay_auth(cls) -> tuple:
        """Get Razorpay authentication credentials"""
        return (cls.RAZORPAY_KEY_ID, cls.RAZORPAY_KEY_SECRET)
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate payment configuration"""
        if not cls.RAZORPAY_KEY_ID or not cls.RAZORPAY_KEY_SECRET:
            return False
        return True

import logging
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Services.Subscription.database import SubscriptionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_user(email):
    """Create missing user"""
    logger.info(f"Creating user: {email}")
    
    manager = SubscriptionManager()
    try:
        manager.create_or_get_user(email, "Rachit Gour")
            
        logger.info("User created successfully.")

    except Exception as e:
        logger.error(f"Creation failed: {e}")

if __name__ == "__main__":
    create_user("rachit.gour749@gmail.com")

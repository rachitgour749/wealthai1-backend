import logging
import sys
import os
from sqlalchemy import text

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Databases.app_data_db_connection import create_connection, get_session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_user_data(email):
    """Inspect data for user"""
    logger.info(f"Inspecting data for user: {email}")
    
    # Ensure connection
    if not create_connection():
        logger.error("Failed to connect to database")
        return

    session = get_session()
    try:
        # Check product_subscriptions
        logger.info("--- Product Subscriptions ---")
        result = session.execute(text("SELECT id, product_code, status, subscription_type FROM product_subscriptions WHERE user_email = :email"), {"email": email}).fetchall()
        if not result:
            logger.info("No product subscriptions found.")
        else:
            for row in result:
                logger.info(f"Found: {row}")

        # Check main subscription
        logger.info("--- Main Subscription ---")
        result = session.execute(text("SELECT * FROM subscription WHERE user_email = :email"), {"email": email}).fetchall()
        if not result:
            logger.info("No main subscription found.")
        else:
            for row in result:
                logger.info(f"Found: {row}")

    except Exception as e:
        logger.error(f"Inspection failed: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    inspect_user_data("rachit.gour749@gmail.com")

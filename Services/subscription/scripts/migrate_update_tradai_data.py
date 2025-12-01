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

def migrate_data():
    """Update TRADAI records to TRADEAI"""
    logger.info("Starting data migration from TRADAI to TRADEAI...")
    
    # Ensure connection
    if not create_connection():
        logger.error("Failed to connect to database")
        return

    session = get_session()
    try:
        # Check count of TRADAI records
        result = session.execute(text("SELECT count(*) FROM product_subscriptions WHERE product_code = 'TRADAI'")).fetchone()
        count = result[0]
        logger.info(f"Found {count} records with product_code='TRADAI'")

        if count > 0:
            logger.info("Updating records to 'TRADEAI'...")
            # Update records
            session.execute(text("UPDATE product_subscriptions SET product_code = 'TRADEAI' WHERE product_code = 'TRADAI'"))
            session.commit()
            logger.info(f"Successfully updated {count} records.")
        else:
            logger.info("No records to update.")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    migrate_data()

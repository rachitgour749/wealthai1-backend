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

def migrate_enum():
    """Add TRADEAI to productcode enum"""
    logger.info("Starting migration to add TRADEAI to productcode enum...")
    
    # Ensure connection
    if not create_connection():
        logger.error("Failed to connect to database")
        return

    session = get_session()
    try:
        # Check current enum values
        logger.info("Checking current enum values...")
        result = session.execute(text("SELECT unnest(enum_range(NULL::productcode))")).fetchall()
        current_values = [row[0] for row in result]
        logger.info(f"Current values: {current_values}")

        if "TRADEAI" in current_values:
            logger.info("TRADEAI already exists in enum. Skipping.")
        else:
            # Add TRADEAI to enum
            # ALTER TYPE ... ADD VALUE cannot run inside a transaction block usually, 
            # but SQLAlchemy session.execute starts a transaction.
            # We need to commit immediately or use autocommit.
            # However, let's try standard execution first.
            logger.info("Adding TRADEAI to productcode enum...")
            
            # We need to execute this outside of a transaction block for some Postgres versions,
            # but let's try with the session first. If it fails, we might need a raw connection with autocommit.
            # Actually, ALTER TYPE ADD VALUE *can* run in a transaction in recent Postgres versions.
            
            session.execute(text("ALTER TYPE productcode ADD VALUE 'TRADEAI'"))
            session.commit()
            logger.info("Successfully added TRADEAI to enum.")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    migrate_enum()

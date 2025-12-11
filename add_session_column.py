import logging
from sqlalchemy import text
from Databases.app_data_db_connection import create_connection, get_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_token_hash_column():
    """
    Add active_token_hash column to user_details table.
    """
    if not create_connection():
        logger.error("Failed to connect to database")
        return False
    
    engine = get_engine()
    
    try:
        with engine.connect() as connection:
            # Check if column exists
            check_query = text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='user_details' AND column_name='active_token_hash'
            """)
            result = connection.execute(check_query)
            if result.fetchone():
                logger.info("Column active_token_hash already exists in user_details")
                return True
            
            # Add column
            logger.info("Adding active_token_hash column to user_details...")
            alter_query = text("ALTER TABLE user_details ADD COLUMN active_token_hash VARCHAR(64)")
            connection.execute(alter_query)
            connection.commit()
            logger.info("Successfully added active_token_hash column")
            return True
            
    except Exception as e:
        logger.error(f"Error executing migration: {e}")
        return False

if __name__ == "__main__":
    add_token_hash_column()


import sys
import os
import logging
from sqlalchemy import text

# Add project root to path
sys.path.append(os.getcwd())

from Databases.app_data_db_connection import get_session, create_connection
from Services.strategy_manager.models import SavedInstance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_instances():
    if not create_connection():
        logger.error("Failed to connect to DB")
        return

    session = get_session()
    try:
        result = session.execute(text("SELECT id, strategy_name, strategy_type, status FROM saved_instances WHERE strategy_name LIKE '%ETF%' OR strategy_type LIKE '%ETF%'"))
        print("\n--- Targeted Check ---")
        for row in result:
            print(f"ID: {row[0]}, Name: '{row[1]}', Type: '{row[2]}', Status: '{row[3]}'")

        # 2. ORM Check
        logger.info("\n--- ORM Check ---")
        param = "ETF_Rotation"
        instances = session.query(SavedInstance).filter(
            SavedInstance.strategy_type == param
        ).all()
        logger.info(f"ORM Query for Strategy Type '{param}': Found {len(instances)} matches")
        
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    debug_instances()

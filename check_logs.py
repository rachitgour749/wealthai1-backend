import logging
import sys
import os
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.getcwd())

from Databases.app_data_db_connection import get_session, create_connection, init_database
from Databases.webhook_models import WebhookExecutionLog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_recent_logs():
    if not create_connection():
        logger.error("Failed to connect to DB")
        return
    
    init_database()
    db = get_session()
    
    try:
        # Fetch logs from the last 10 minutes
        logger.info("Checking recent logs in WebhookExecutionLog...")
        recent_logs = db.query(WebhookExecutionLog).order_by(WebhookExecutionLog.id.desc()).limit(5).all()
        
        if not recent_logs:
            logger.info("No recent logs found.")
        else:
            for log in recent_logs:
                logger.info(f"ID: {log.id} | User: {log.user_email} | Status: {log.status} | Msg: {log.message}")

    except Exception as e:
        logger.error(f"Failed to check logs: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    check_recent_logs()

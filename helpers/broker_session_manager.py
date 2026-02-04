from datetime import datetime, timedelta
from Databases.broker_models import BrokerSession
from Databases.app_data_db_connection import get_session
import logging

logger = logging.getLogger(__name__)

def save_broker_session(user_email: str, broker_name: str, client_id: str, response_data: dict, api_key: str = None):
    """
    Saves or updates the broker session in the database.
    
    Args:
        user_email (str): The application's user email.
        broker_name (str): Name of the broker (e.g., 'zerodha').
        client_id (str): The broker's client ID.
        response_data (dict): The successful login response containing the token.
        api_key (str): The broker's API key (optional).
    """
    session = None
    try:
        # Extract access token
        access_token = (
            response_data.get('access_token') or 
            response_data.get('session_token') or 
            response_data.get('token')
        )
        
        if not access_token:
            logger.warning(f"No access token found in response for {broker_name}")
            return False, "No access token found in response"

        # Calculate expiry (current time + 24 hours)
        token_expire = datetime.utcnow() + timedelta(hours=24)
        
        # Get DB Session
        session = get_session()
        
        # Check if user exists by email
        user = session.query(BrokerSession).filter_by(user_email=user_email).first()
        
        if not user:
            logger.error(f"User with email {user_email} not found in database.")
            # Create new user record if not found (or should we return error? Original code returned error, but maybe we should create?)
            # The original code returned: return False, f"User with email {user_email} not found. Details cannot be stored."
            # But usually we might want to create. Let's stick to original behavior for now or ensure UserDetails table is populated.
            # actually UserDetails table seems to be specific to broker session storage in the original Broker design (from models.py).
            # If so, we should probably create the record if it doesn't exist.
            # Let's look at models.py again. It has user_email as PK. So it is likely a dedicated table for valid users.
            # If the user doesn't exist in THIS table, we should probably create it.
            # BUT the original code returned Error. This implies this table might be shared with Auth or populated elsewhere.
            # HOWEVER, looking at models.py, it's just "UserDetails". 
            # If I look at the original code:
            # 40:         user = session.query(UserDetails).filter_by(user_email=user_email).first()
            # 42:         if not user:
            # 44:             return False, f"User with email {user_email} not found..."
            
            # This suggests the user must already exist. 
            # Wait, if this is a "Broker Integration", maybe it assumes the user is already registered in the system.
            # But where? "UserDetails" table. 
            # If this is a new table we are introducing (it wasn't in wealthai-backend-v2 before), then it will be empty.
            # So `save_broker_session` will ALWAYS fail for the first time if we don't create.
            # I will modify it to CREATE if not exists, which makes more sense for a fresh integration.
            
            logger.info(f"User {user_email} not found, creating new record.")
            user = BrokerSession(user_email=user_email)
            session.add(user)
            # Continue to update fields below
            
        # Update broker details for the user
        logger.info(f"Updating broker session for user {user_email}, broker {broker_name}")
        logger.info(f"DEBUG - Values to save: broker_name={broker_name}, client_id={client_id}, api_key={api_key}, access_token={access_token[:20]}..., token_expire={token_expire}")
        
        user.broker_name = broker_name
        user.client_id = client_id
        if api_key:
            user.api_key = api_key
        user.access_token = access_token
        user.token_expire = token_expire
        if not user.created_at:
             user.created_at = datetime.utcnow()
        
        logger.info(f"DEBUG - After assignment: user.broker_name={user.broker_name}, user.client_id={user.client_id}, user.api_key={user.api_key}")
            
        session.commit()
        logger.info(f"Session saved successfully for User: {user_email}, Broker: {broker_name}")
        return True, "Session saved successfully"
        
    except Exception as e:
        logger.error(f"Failed to save broker session: {e}")
        if session:
            session.rollback()
        return False, str(e)
    finally:
        if session:
            session.close()

def get_broker_session(user_email: str):
    """
    Retrieves the broker session details for a given user email.
    
    Args:
        user_email (str): The application's user email.
    
    Returns:
        tuple: (broker_name, client_id, access_token, api_key) or (None, None, None, None) if not found
    """
    session = None
    try:
        session = get_session()
        user = session.query(BrokerSession).filter_by(user_email=user_email).first()
        
        if not user:
            logger.warning(f"No session found for user: {user_email}")
            return None, None, None, None
        
        # Check if token is expired
        if user.token_expire and user.token_expire < datetime.utcnow():
            logger.warning(f"Token expired for user: {user_email}")
            return None, None, None, None
        
        logger.info(f"Retrieved session for user: {user_email}, broker: {user.broker_name}")
        return user.broker_name, user.client_id, user.access_token, user.api_key
        
    except Exception as e:
        logger.error(f"Failed to retrieve broker session: {e}")
        return None, None, None, None
    finally:
        if session:
            session.close()

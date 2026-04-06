from datetime import datetime, timedelta
from Databases.broker_models import BrokerSession
from Databases.app_data_db_connection import get_session
import logging

logger = logging.getLogger(__name__)

import json

def save_broker_session(user_email: str, broker_name: str, client_id: str, response_data: dict, api_key: str = None, credentials: dict = None):
    """
    Saves or updates the broker session in the database.
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
            logger.info(f"User {user_email} not found, creating new record.")
            user = BrokerSession(user_email=user_email)
            session.add(user)
            
        # Update broker details for the user
        user.broker_name = broker_name
        user.client_id = client_id
        if api_key:
            user.api_key = api_key
        
        # Save credentials if provided
        sid = response_data.get('sid')
        base_url = response_data.get('base_url')
        
        if credentials:
            if sid:
                credentials['sid'] = sid
            if base_url:
                credentials['base_url'] = base_url
            user.broker_credentials = json.dumps(credentials)
            
            # Extract new SEBI fields if they exist in the payload
            if 'static_ip_username' in credentials:
                user.static_ip_username = credentials['static_ip_username']
            if 'static_ip_password' in credentials:
                user.static_ip_password = credentials['static_ip_password']
            if 'static_ip_port' in credentials:
                user.static_ip_port = credentials['static_ip_port']
        elif sid or base_url:
             try:
                creds_to_save = {}
                if sid: creds_to_save['sid'] = sid
                if base_url: creds_to_save['base_url'] = base_url
                user.broker_credentials = json.dumps(creds_to_save)
             except:
                pass
                
        user.access_token = access_token
        user.token_expire = token_expire
        if not user.created_at:
             user.created_at = datetime.utcnow()
            
        session.commit()
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
    Returns:
        tuple: (broker_name, client_id, access_token, api_key, broker_credentials) or (None, None, None, None, None)
    """
    session = None
    try:
        session = get_session()
        user = session.query(BrokerSession).filter_by(user_email=user_email).first()
        
        if not user:
            logger.warning(f"No session found for user: {user_email}")
            return None, None, None, None, None
        
        # Check if token is expired
        if user.token_expire and user.token_expire < datetime.utcnow():
            logger.warning(f"Token expired for user: {user_email}")
            return None, None, None, None, None
        
        logger.info(f"Retrieved session for user: {user_email}, broker: {user.broker_name}")
        return user.broker_name, user.client_id, user.access_token, user.api_key, user.broker_credentials
        
    except Exception as e:
        logger.error(f"Failed to retrieve broker session: {e}")
        return None, None, None, None, None
    finally:
        if session:
            session.close()

def get_full_broker_session(user_email: str):
    """Retrieves the full broker session record as a dictionary."""
    session = None
    try:
        session = get_session()
        user = session.query(BrokerSession).filter_by(user_email=user_email).first()
        if not user:
            return None
        
        return {
            "broker_name": user.broker_name,
            "client_id": user.client_id,
            "api_key": user.api_key,
            "access_token": user.access_token,
            "token_expire": user.token_expire.strftime("%Y-%m-%d %H:%M:%S") if user.token_expire else None,
            "static_ip": user.static_ip,
            "broker_credentials": json.loads(user.broker_credentials) if user.broker_credentials else None
        }
    except Exception as e:
        logger.error(f"Error fetching full session for {user_email}: {e}")
        return None
    finally:
        if session:
            session.close()

def delete_broker_session_record(user_email: str, client_id: str):
    """Deletes a broker session record by email and client ID."""
    session = None
    try:
        session = get_session()
        record = session.query(BrokerSession).filter_by(user_email=user_email, client_id=client_id).first()
        if record:
            session.delete(record)
            session.commit()
            return True, "Account deleted successfully"
        return False, "Account not found"
    except Exception as e:
        logger.error(f"Error deleting session for {user_email}: {e}")
        if session:
            session.rollback()
        return False, str(e)
    finally:
        if session:
            session.close()

def update_broker_credentials_only(user_email: str, broker_name: str, username: str, new_payload: dict):
    """Updates the broker_credentials JSON field for an existing record."""
    session = None
    try:
        session = get_session()
        user = session.query(BrokerSession).filter_by(user_email=user_email, client_id=username).first()
        if not user:
            return False, "Record not found for given email and username"
        
        current_creds = {}
        if user.broker_credentials:
            try:
                current_creds = json.loads(user.broker_credentials)
            except:
                pass
        
        for k, v in new_payload.items():
            if k not in ['user_email', 'broker_name'] and v is not None:
                current_creds[k] = v
        
        user.broker_credentials = json.dumps(current_creds)
        session.commit()
        return True, "Credentials updated successfully"
    except Exception as e:
        logger.error(f"Error updating credentials for {user_email}: {e}")
        if session:
            session.rollback()
        return False, str(e)
    finally:
        if session:
            session.close()

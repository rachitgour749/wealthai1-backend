import logging
from Broker.Zerodha.ZERODHA import ZerodhaAuthenticator, place_order as zerodha_place_order
from Broker.AngelOne.ANGELONE import AngelOneAuthenticator, place_order as angelone_place_order

logger = logging.getLogger(__name__)

def login_broker(broker_name: str, credentials: dict):
    """
    Login to the specified broker.
    """
    logger.info(f"Attempting login for broker: {broker_name}")
    
    if broker_name.lower() == "zerodha":
        logger.info("Dispatching to Zerodha login handler...")
        try:
            auth = ZerodhaAuthenticator(
                api_key=credentials.get('api_key'), 
                api_secret=credentials.get('api_secret')
            )
            access_token, error = auth.login(
                username=credentials.get('username'),
                password=credentials.get('password'),
                totp_secret=credentials.get('totp_secret')
            )
            
            if error:
                return {"status": "error", "message": error}
            
            return {
                "status": "success", 
                "message": "Zerodha login successful",
                "access_token": access_token
            }
        except Exception as e:
            logger.error(f"Zerodha login exception: {e}")
            return {"status": "error", "message": str(e)}
    
    elif broker_name.lower() == "angelone":
        logger.info("Dispatching to AngelOne login handler...")
        try:
            # Support both 'client_code' and 'username' for flexibility
            client_code = credentials.get('client_code') or credentials.get('username')
            
            if not client_code:
                return {"status": "error", "message": "Missing client_code or username"}
            
            auth = AngelOneAuthenticator(
                api_key=credentials.get('api_key'),
                api_secret=None,  # Not used for AngelOne
                client_code=client_code
            )
            access_token, error = auth.login(
                client_code=client_code,
                password=credentials.get('password'),
                totp_secret=credentials.get('totp_secret')
            )
            
            if error:
                return {"status": "error", "message": error}
            
            return {
                "status": "success",
                "message": "AngelOne login successful",
                "access_token": access_token,
                "refresh_token": auth.refresh_token,
                "feed_token": auth.feed_token,
                "client_id": client_code
            }
        except Exception as e:
            logger.error(f"AngelOne login exception: {e}")
            return {"status": "error", "message": str(e)}
    
    else:
        logger.warning(f"Unknown broker: {broker_name}")
        return {"status": "error", "message": "Unknown broker"}

def dispatch_place_order(broker_name: str, credentials: dict, order_data: dict):
    """
    Dispatch order placement to the specified broker.
    """
    logger.info(f"Dispatching place order for {broker_name}")
    
    if broker_name.lower() == "zerodha":
        return zerodha_place_order(credentials, order_data)
    
    elif broker_name.lower() == "angelone":
        return angelone_place_order(credentials, order_data)
        
    else:
         return {"status": "error", "message": "Unknown broker"}


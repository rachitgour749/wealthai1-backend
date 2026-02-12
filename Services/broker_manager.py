import logging
from Broker.Zerodha.ZERODHA import ZerodhaAuthenticator, place_order as zerodha_place_order
from Broker.AngelOne.ANGELONE import AngelOneAuthenticator, place_order as angelone_place_order
from Broker.Kotak.KOTAK import KotakAuthenticator, place_order as kotak_place_order
from Broker.Dhan.DHAN import DhanAuthenticator, place_order as dhan_place_order
from Broker.Moswal.MOSWAL import MOSWALAuthenticator, place_order as moswal_place_order
from Broker.ICICI.ICICI import ICICIAuthenticator, place_order as icici_place_order
from Broker.Smc.SMC_ACE import SMCAceAuthenticator, place_order as smc_place_order

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
    
    elif broker_name.lower() == "kotak":
        logger.info("Dispatching to Kotak login handler...")
        try:
            # Credentials needed: mobileNumber, ucc, totp_secret, mpin
            # Also access_token which might be passed in credentials or headers.
            # Based on user prompt: "Headers Authorization: <access_token>"
            # So access_token must be provided.
            
            mobile_number = credentials.get('mobileNumber')
            ucc = credentials.get('ucc')
            totp_secret = credentials.get('totp_secret')
            mpin = credentials.get('mpin')
            access_token = credentials.get('access_token') 
            
            # If access_token is not in credentials, check if api_key is used as alias
            if not access_token:
                access_token = credentials.get('api_key')

            if not mobile_number or not ucc or not totp_secret or not mpin:
                 return {"status": "error", "message": "Missing required Kotak credentials (mobileNumber, ucc, totp_secret, mpin)"}

            auth = KotakAuthenticator(access_token=access_token)
            session_data, error = auth.login(
                mobile_number=mobile_number,
                ucc=ucc,
                totp_secret=totp_secret,
                mpin=mpin
            )
            
            if error:
                return {"status": "error", "message": error}
            
            return {
                "status": "success",
                "message": "Kotak login successful",
                "access_token": session_data.get('access_token'),
                "sid": session_data.get('sid'),
                "client_id": session_data.get('ucc'),
                "base_url": session_data.get('base_url')
            }

        except Exception as e:
            logger.error(f"Kotak login exception: {e}")
            return {"status": "error", "message": str(e)}
    
    elif broker_name.lower() == "dhan":
        logger.info("Dispatching to DHAN login handler...")
        try:
            # Credentials needed: client_id, mpin, totp_secret
            client_id = credentials.get('client_id')
            mpin = credentials.get('mpin')
            totp_secret = credentials.get('totp_secret')
            
            if not client_id or not mpin or not totp_secret:
                return {"status": "error", "message": "Missing required DHAN credentials (client_id, mpin, totp_secret)"}
            
            auth = DhanAuthenticator()
            access_token, error = auth.login(
                client_id=client_id,
                mpin=mpin,
                totp_secret=totp_secret
            )
            
            if error:
                return {"status": "error", "message": error}
            
            return {
                "status": "success",
                "message": "DHAN login successful",
                "access_token": access_token,
                "client_id": client_id
            }
        except Exception as e:
            logger.error(f"DHAN login exception: {e}")
            return {"status": "error", "message": str(e)}
    
    elif broker_name.lower() == "moswal":
        logger.info("Dispatching to MOSWAL login handler...")
        try:
            # Credentials needed: client_id, password, dob, totp_secret, api_key
            client_id = credentials.get('client_id')
            password = credentials.get('password')
            dob = credentials.get('dob')  # Format: DDMMYYYY
            totp_secret = credentials.get('totp_secret')
            api_key = credentials.get('api_key')
            
            if not client_id or not password or not dob or not totp_secret or not api_key:
                return {"status": "error", "message": "Missing required MOSWAL credentials (client_id, password, dob, totp_secret, api_key)"}
            
            auth = MOSWALAuthenticator(api_key=api_key)
            session_data, error = auth.login(
                client_id=client_id,
                password=password,
                dob=dob,
                totp_secret=totp_secret
            )
            
            if error:
                return {"status": "error", "message": error}
            
            return {
                "status": "success",
                "message": "MOSWAL login successful",
                "access_token": session_data.get('access_token'),
                "client_id": session_data.get('client_id'),
                "client_name": session_data.get('client_name')
            }
        except Exception as e:
            logger.error(f"MOSWAL login exception: {e}")
            return {"status": "error", "message": str(e)}

    elif broker_name.lower() == "icici":
        logger.info("Dispatching to ICICI login handler...")
        try:
            # Credentials needed: api_key, api_secret, session_token (obtained from browser)
            api_key = credentials.get('api_key')
            api_secret = credentials.get('api_secret')
            session_token = credentials.get('session_token')
            
            if not api_key or not api_secret or not session_token:
                return {"status": "error", "message": "Missing required ICICI credentials (api_key, api_secret, session_token)"}
            
            auth = ICICIAuthenticator(api_key=api_key, api_secret=api_secret)
            session_data, error = auth.login(session_token=session_token)
            
            if error:
                return {"status": "error", "message": error}
            
            return {
                "status": "success",
                "message": "ICICI login successful",
                "access_token": session_data.get('access_token'),
                "api_key": session_data.get('api_key'),
                "api_secret": session_data.get('api_secret'),
                "client_id": session_data.get('client_id'),
                "client_name": session_data.get('client_name')
            }
        except Exception as e:
            logger.error(f"ICICI login exception: {e}")
            return {"status": "error", "message": str(e)}

    elif broker_name.lower() in ["smc", "smc_ace"]:
        logger.info("Dispatching to SMC ACE login handler...")
        try:
            # Credentials needed: api_key, api_secret, user_id, password, totp_secret
            api_key = credentials.get('api_key')
            api_secret = credentials.get('api_secret')
            user_id = credentials.get('username') or credentials.get('user_id') or credentials.get('client_id')
            password = credentials.get('password')
            totp_secret = credentials.get('totp_secret')
            
            if not api_key or not api_secret or not user_id or not password or not totp_secret:
                return {"status": "error", "message": "Missing required SMC ACE credentials (api_key, api_secret, client_id/username, password, totp_secret)"}
            
            auth = SMCAceAuthenticator(api_key=api_key, api_secret=api_secret)
            access_token, error = auth.login(
                user_id=user_id,
                password=password,
                totp_secret=totp_secret
            )
            
            if error:
                return {"status": "error", "message": error}
            
            return {
                "status": "success",
                "message": "SMC ACE login successful",
                "access_token": access_token,
                "client_id": user_id
            }
        except Exception as e:
            logger.error(f"SMC ACE login exception: {e}")
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

    elif broker_name.lower() == "kotak":
        return kotak_place_order(credentials, order_data)
    
    elif broker_name.lower() == "dhan":
        return dhan_place_order(credentials, order_data)
    
    elif broker_name.lower() == "moswal":
        return moswal_place_order(credentials, order_data)
        
    elif broker_name.lower() == "icici":
        return icici_place_order(credentials, order_data)
        
    elif broker_name.lower() in ["smc", "smc_ace"]:
        return smc_place_order(credentials, order_data)
        
    else:
         return {"status": "error", "message": "Unknown broker"}

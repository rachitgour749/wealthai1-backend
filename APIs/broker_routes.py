from fastapi import APIRouter, HTTPException, Body, Request, Header
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import Dict, Any, List
from Services.broker_manager import login_broker, dispatch_place_order
import logging
import json
from helpers.broker_session_manager import save_broker_session, get_broker_session
from Databases.app_data_db_connection import get_session
from Databases.broker_models import BrokerSession
from helpers.broker_market_utils import is_market_open

router = APIRouter()
logger = logging.getLogger(__name__)

class LoginRequest(BaseModel):
    broker_name: str
    user_email: str  # App User Email provided by the user

    class Config:
        extra = "allow"

class OrderRequest(BaseModel):
    exchange: str
    symbol: Any = None  # Support for single symbol (legacy)
    symbols: List[str] = None  # Support for multiple symbols
    user_id: str  # Email of the app user
    order_side: str  # BUY/SELL
    product_type: str  # DELIVERY/INTRADAY/etc
    clients: Dict[str, str]  # {"client_id": "quantity"}
    variety: str = None  # regular, amo, bo, co (Optional)
    exchange_instrument_id: str = None  # Symbol token for AngelOne (Optional)

    class Config:
        extra = "allow"

@router.post(
    "/broker_login",
    summary="Broker Login",
    description="""
    Login to a broker and save the session.
    
    **Optional Header:**
    - `X-Static-IP`: Static IP address for SEBI compliance (e.g., "192.168.1.100")
    
    If provided, the IP will be stored and used for order placement.
    If not provided, orders will be placed without IP.
    """,
    responses={
        200: {"description": "Login successful"},
        400: {"description": "Login failed or session save error"},
        500: {"description": "Internal server error"}
    }
)
async def broker_login(
    login_request: LoginRequest, 
    request: Request,
    x_static_ip: str = Header(None, alias="X-Static-IP", description="Optional static IP address for SEBI compliance")
):
    logger.info(f"Received login request for {login_request.broker_name} from {login_request.user_email}")
    try:
        # Extract credentials from extra fields
        if hasattr(login_request, 'model_dump'):
            request_data = login_request.model_dump()
        else:
            request_data = login_request.dict()
            
        credentials = {k: v for k, v in request_data.items() if k not in {'broker_name', 'user_email'}}
        logger.info(f"DEBUG - Extracted credentials keys: {list(credentials.keys())}")
        logger.info(f"DEBUG - Full Credentials Payload: {json.dumps(credentials, default=str)}")
        
        # Explicitly handle smc_ace as requested by user
        if login_request.broker_name.lower() == "smc_ace":
            logger.info("Handling explicit SMC ACE login route")
            result = login_broker("smc_ace", credentials)
        else:
            result = login_broker(login_request.broker_name, credentials)
        if result.get("status") == "error":
             logger.error(f"Login failed for {login_request.broker_name}: {result.get('message')}")
             raise HTTPException(status_code=400, detail=result.get("message"))
        
        if result.get("status") == "success":
            logger.info(f"Login successful for {login_request.broker_name}") # Changed success to info as logger might not have success
            
            # Get static IP from header parameter (optional)
            static_ip = x_static_ip
            
            if static_ip:
                logger.info(f"Received static IP from header: {static_ip}")
            else:
                logger.info("No static IP provided in headers, will set to NULL in database")
            
            # Save session to database
            client_id = credentials.get('client_id') or credentials.get('username') or result.get('client_id') or 'unknown_client'
            user_email = login_request.user_email
            api_key = credentials.get('api_key')  # Extract api_key from credentials
            
            saved, message = save_broker_session(user_email, login_request.broker_name, client_id, result, api_key, credentials=credentials)
            
            if not saved:
                logger.error(f"Session not saved: {message}")
                raise HTTPException(status_code=400, detail=message)
            
            # Update static_ip in database (optional - won't fail if it doesn't work)
            db = get_session()
            try:
                user_record = db.query(BrokerSession).filter(BrokerSession.user_email == user_email).first()
                if user_record:
                    user_record.static_ip = static_ip
                    db.commit()
                    if static_ip:
                        logger.info(f"Updated static_ip for {user_email}: {static_ip}")
                    else:
                        logger.info(f"Set static_ip to NULL for {user_email}")
                else:
                    logger.warning(f"User record not found for {user_email}, could not update static_ip")
            except Exception as e:
                logger.warning(f"Could not update static_ip in database: {e}")
                db.rollback()
            finally:
                db.close()
        
        # Calculate expiry to match session manager strategy (approximate)
        expiry_time = datetime.utcnow() + timedelta(hours=24)
        
        return {
            "status": "success",
            "message": f"{login_request.broker_name} login successful",
            "access_token": result.get('access_token') or result.get('data', {}).get('access_token'),
            "expire": expiry_time.strftime("%Y-%m-%d %H:%M:%S"),
            "broker_name": login_request.broker_name,
            "user_email": login_request.user_email,
            "client_id": client_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal error during login: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/place_order")
async def place_order(request: OrderRequest):
    logger.info(f"Received place order request for user {request.user_id}")
    try:
        # Retrieve broker session from database
        broker_name, client_id, access_token, api_key, broker_credentials = get_broker_session(request.user_id)
        
        if not broker_name or not access_token:
            logger.error(f"No valid session found for user: {request.user_id}")
            raise HTTPException(status_code=401, detail="No valid session found. Please login first.")
        
        logger.info(f"Retrieved session for broker: {broker_name}")
        
        # Determine variety (regular or amo)
        
        # If variety is not provided in request, detect automatically
        if not request.variety:
            variety = "regular" if is_market_open() else "amo"
            logger.info(f"Market status detected: {'OPEN' if variety == 'regular' else 'CLOSED'}. Setting variety to {variety}")
        else:
            variety = request.variety.lower()
            logger.info(f"Manual variety override: {variety}")

        # Default values for optional parameters
        order_type = "MARKET"
        price = 0
        trigger_price = 0
        validity = "DAY"

        # Prepare symbol list dynamically
        # Can accept: symbol="SBIN" or symbol=["SBIN", "RELIANCE"] or symbols=["SBIN", "RELIANCE"]
        symbol_list = []
        
        # Check 'symbols' list first (explicit multi-symbol)
        if hasattr(request, 'symbols') and request.symbols:
            symbol_list = request.symbols
        # Check 'symbol' which can now be string OR list
        elif hasattr(request, 'symbol') and request.symbol:
            if isinstance(request.symbol, (list, set, tuple)):
                symbol_list = list(request.symbol)
            else:
                symbol_list = [str(request.symbol)]
        
        if not symbol_list:
             raise HTTPException(status_code=400, detail="No symbols provided. Use 'symbol' (string/list) or 'symbols' (list).")

        # Prepare base order data (common parameters)
        common_order_data = {
            'exchange': request.exchange,
            'order_side': request.order_side,
            'product_type': request.product_type,
            'order_type': order_type,
            'price': price,
            'trigger_price': trigger_price,
            'validity': validity,
            'variety': variety
        }
        
        # Iterate through each symbol and each client
        results = []
        success_count = 0
        failed_count = 0
        
        for symbol in symbol_list:
            logger.info(f"Processing symbol: {symbol}")
            for client_id, quantity in request.clients.items():
                logger.info(f"Placing order for symbol: {symbol}, client: {client_id}, quantity: {quantity}")
                
                # Retrieve Static IP for the client (optional)
                static_ip = None
                db = get_session()
                try:
                    user_record = db.query(BrokerSession).filter(BrokerSession.client_id == client_id).first()
                    if user_record and user_record.static_ip:
                        static_ip = user_record.static_ip
                        logger.info(f"Using static IP for client {client_id}: {static_ip}")
                    else:
                        logger.warning(f"No static IP found for client {client_id}, placing order without IP")
                except Exception as e:
                    logger.warning(f"Error retrieving static IP for {client_id}: {e}")
                finally:
                    db.close()

                # Create order data for this specific combination
                order_data = {
                    **common_order_data, 
                    'symbol': symbol,
                    'quantity': quantity
                }
                
                # Add exchange_instrument_id if provided (required for AngelOne)
                if hasattr(request, 'exchange_instrument_id') and request.exchange_instrument_id:
                    order_data['exchange_instrument_id'] = request.exchange_instrument_id
                
                # Add static_ip to order_data if available
                if static_ip:
                    order_data['static_ip'] = static_ip
                
                credentials = {
                    'api_key': api_key,
                    'access_token': access_token,
                    'client_id': client_id
                }
                
                # Add SID and other details for Kotak if available
                if broker_credentials:
                    try:
                        creds_dict = json.loads(broker_credentials)
                        if creds_dict.get('sid'):
                            credentials['sid'] = creds_dict.get('sid')
                        if creds_dict.get('base_url'):
                            credentials['base_url'] = creds_dict.get('base_url')
                    except:
                        pass
                
                # Place order
                result = dispatch_place_order(broker_name, credentials, order_data)
                
                # Collect result
                res = {
                    'symbol': symbol,
                    'client_id': client_id,
                    'quantity': quantity,
                    'status': result.get('status'),
                    'message': result.get('message')
                }
                
                if result.get('status') == 'success':
                    res['order_id'] = result.get('data', {}).get('order_id')
                    success_count += 1
                else:
                    failed_count += 1
                
                results.append(res)
        
        # Return aggregated response
        total_orders = len(symbol_list) * len(request.clients)
        overall_status = "success" if success_count > 0 else "error"
        
        return {
            "status": overall_status,
            "message": f"Orders processed for {len(request.clients)} clients",
            "results": results,
            "summary": {
                "total": len(request.clients),
                "success": success_count,
                "failed": failed_count
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal error during order placement: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/account_details")
async def account_details(user_email: str):
    """Fetch complete account details for a user."""
    from helpers.broker_session_manager import get_full_broker_session
    details = get_full_broker_session(user_email)
    if not details:
        raise HTTPException(status_code=404, detail="No broker account found for this user")
    return {"status": "success", "data": details}

@router.delete("/delete_account")
async def delete_account(user_email: str, client_id: str):
    """Remove complete data of a particular user and client ID."""
    from helpers.broker_session_manager import delete_broker_session_record
    success, message = delete_broker_session_record(user_email, client_id)
    if not success:
        raise HTTPException(status_code=404, detail=message)
    return {"status": "success", "message": message}

@router.get("/relogin")
async def relogin(user_email: str):
    """Re-authenticate with the broker using stored credentials."""
    from helpers.broker_session_manager import get_full_broker_session
    details = get_full_broker_session(user_email)
    
    if not details:
        raise HTTPException(status_code=404, detail="No broker account found for this user")
    
    broker_name = details.get("broker_name")
    credentials = details.get("broker_credentials")
    
    if not credentials:
        raise HTTPException(status_code=400, detail="No stored credentials found for relogin")
    
    logger.info(f"Attempting relogin for {user_email} with broker {broker_name}")
    
    try:
        # Call login_broker (re-using the logic from broker_login)
        result = login_broker(broker_name, credentials)
        
        if result.get("status") == "error":
             logger.error(f"Relogin failed for {broker_name}: {result.get('message')}")
             raise HTTPException(status_code=400, detail=result.get("message"))
        
        if result.get("status") == "success":
            # Save session to database
            client_id = credentials.get('client_id') or credentials.get('username') or result.get('client_id') or details.get('client_id')
            api_key = credentials.get('api_key') or details.get('api_key')
            
            from helpers.broker_session_manager import save_broker_session
            saved, message = save_broker_session(user_email, broker_name, client_id, result, api_key, credentials=credentials)
            
            if not saved:
                logger.error(f"Relogin session not saved: {message}")
                raise HTTPException(status_code=400, detail=message)
            
            # Calculate expiry
            expiry_time = datetime.utcnow() + timedelta(hours=24)
            
            return {
                "status": "success",
                "message": f"{broker_name} relogin successful",
                "access_token": result.get('access_token') or result.get('data', {}).get('access_token'),
                "expire": expiry_time.strftime("%Y-%m-%d %H:%M:%S"),
                "broker_name": broker_name,
                "user_email": user_email,
                "client_id": client_id
            }
        else:
            raise HTTPException(status_code=400, detail=result.get("message", "Relogin failed"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal error during relogin: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update_credentials")
async def update_credentials(payload: Dict[str, Any] = Body(...)):
    """Update broker credentials for an existing account."""
    user_email = payload.get("user_email")
    broker_name = payload.get("broker_name")
    username = payload.get("username")
    
    if not user_email or not broker_name or not username:
        raise HTTPException(status_code=400, detail="user_email, broker_name, and username are mandatory")
    
    from helpers.broker_session_manager import update_broker_credentials_only
    success, message = update_broker_credentials_only(user_email, broker_name, username, payload)
    
    if not success:
        raise HTTPException(status_code=404, detail=message)
    
    return {"status": "success", "message": message}

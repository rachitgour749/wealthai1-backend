"""
DHAN Broker Integration
Automated login and trading functions for Dhan broker
Follows the same structure as Zerodha, AngelOne, and Kotak brokers
"""
import requests
import pyotp
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any
from Broker.Dhan.mapping import (
    map_exchange,
    map_validity,
    map_order_side,
    map_product_type,
    map_order_type,
    get_security_id
)

logger = logging.getLogger(__name__)


class DhanAPIError(Exception):
    """Custom exception for DHAN API errors"""
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


class DhanAuthenticator:
    """Handles DHAN authentication flow following the same pattern as other brokers"""
    
    BASE_URL = "https://auth.dhan.co"
    ORDER_URL = "https://api.dhan.co/v2/orders"
    TOKEN_CACHE_FILE = "dhan_token_cache.json"
    
    def __init__(self, api_key=None, api_secret=None):
        """Initialize Dhan Authenticator"""
        self.api_key = api_key
        self.api_secret = api_secret
        self.client_id = None
    
    def login(self, client_id: str, mpin: str, totp_secret: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Authenticate with DHAN and get access token.
        
        Args:
            client_id: Dhan Client ID
            mpin: MPIN for Dhan account
            totp_secret: TOTP Secret key for OTP generation
            
        Returns:
            Tuple of (access_token, error_message)
            - On success: (access_token, None)
            - On failure: (None, error_message)
        """
        logger.info("-" * 40)
        logger.info("DHAN API LOGIN (via generateAccessToken)")
        
        # 1. Generate TOTP
        try:
            totp = pyotp.TOTP(totp_secret).now()
            logger.info(f"Generated TOTP for client: {client_id}")
        except Exception as e:
            error_msg = f"Failed to generate TOTP: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

        # 2. Prepare Request
        url = f"{self.BASE_URL}/app/generateAccessToken"
        
        payload = {
            "dhanClientId": client_id,
            "pin": mpin,
            "totp": totp
        }
        
        logger.info(f"Request: POST {url}")
        
        # DHAN endpoint uses query parameters for POST
        try:
            response = requests.post(url, params=payload, json={}, timeout=30)
            logger.info(f"HTTP Status: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"HTTP Error: {response.text}")
                try:
                    err_data = response.json()
                    err_msg = err_data.get('errorMessage', response.text)
                except:
                    err_msg = response.text
                return None, f"Login Failed: {err_msg}"
                
            data = response.json()
            
            # 3. Extract Token
            token = None
            if 'accessToken' in data:
                token = data['accessToken']
            elif 'data' in data and isinstance(data['data'], dict) and 'accessToken' in data['data']:
                token = data['data']['accessToken']
            
            if not token:
                # Log the full response for debugging
                logger.error(f"Login response missing accessToken. Full Response: {data}")
                
                # Check for error message in common fields
                error_msg = data.get('errorMessage') or data.get('message') or data.get('status')
                if not error_msg and 'errorCode' in data:
                    error_msg = f"Dhan Error {data.get('errorCode')}: {data.get('errorMessage')}"
                
                if error_msg:
                    logger.error(f"Dhan Login Error: {error_msg}")
                    return None, f"Login Failed: {error_msg}"
                
                logger.error(f"Unknown error format. Keys: {list(data.keys())}")
                return None, "Invalid response: Missing Access Token"

            # 4. Save to Cache
            self._save_token_to_cache(token, client_id)
            logger.info("✓ Success: Logged in via API.")
            
            # Store client_id for later use
            self.client_id = client_id
            
            return token, None

        except requests.exceptions.RequestException as e:
            error_msg = f"Network error during API Login: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    def _save_token_to_cache(self, access_token, client_id):
        """Persist access token for Order API usage"""
        try:
            token_data = {
                'access_token': access_token,
                'client_id': client_id,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.TOKEN_CACHE_FILE, 'w') as f:
                json.dump(token_data, f)
            logger.info(f"Access token persisted for client: {client_id}")
        except Exception as e:
            logger.warning(f"Token cache save failed: {str(e)}")

    def _get_token_from_cache(self):
        """Retrieve persisted access token"""
        try:
            if not os.path.exists(self.TOKEN_CACHE_FILE):
                return None
            with open(self.TOKEN_CACHE_FILE, 'r') as f:
                data = json.load(f)
            
            # Check if token is older than 24 hours
            ts = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - ts > timedelta(hours=24):
                return None
                
            return data['access_token']
        except:
            return None


def place_order(credentials: Dict[str, Any], order_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Places an order using DHAN API.
    
    Args:
        credentials: dict containing:
            - api_key: DHAN API key (optional)
            - access_token: Access token from login
        order_data: dict containing order parameters:
            - exchange: Exchange name (NSE, BSE, NFO, MCX, CDS)
            - symbol: Trading symbol
            - order_side: BUY or SELL
            - product_type: DELIVERY, INTRADAY, MARGIN, etc.
            - order_type: MARKET, LIMIT, SL, SL-M
            - quantity: Order quantity
            - price: Limit price (optional, for LIMIT orders)
            - trigger_price: Trigger price (optional, for SL orders)
            - validity: DAY or IOC (optional)
            - variety: regular or amo (optional)
    
    Returns:
        dict: Order placement response
        - On success: {"status": "success", "message": "...", "data": {...}}
        - On failure: {"status": "error", "message": "...", "error_type": "..."}
    """
    logger.info("-" * 40)
    logger.info("DHAN PLACE ORDER")
    
    try:
        # Extract credentials
        access_token = credentials.get('access_token')
        
        if not access_token:
            logger.error("Missing access token in credentials")
            return {
                "status": "error",
                "message": "Missing access token",
                "error_type": "authentication_error"
            }
        
        # Map order parameters using mapping functions
        exchange_id = map_exchange(order_data.get('exchange'))
        transaction_type = map_order_side(order_data.get('order_side'))
        product_type = map_product_type(order_data.get('product_type'))
        order_type = map_order_type(order_data.get('order_type'))
        validity = map_validity(order_data.get('validity', 'DAY'))
        
        # Prepare DHAN API payload
        symbol = order_data.get('symbol')
        exchange = order_data.get('exchange', 'NSE')
        security_id = get_security_id(symbol, exchange)
        
        # Ensure price is present as it is marked as required in Dhan docs
        # For MARKET orders, typically it should be 0.0
        price = 0.0
        if order_data.get('price'):
            try:
                price = float(order_data['price'])
            except (ValueError, TypeError):
                price = 0.0
                
        dhan_payload = {
            "dhanClientId": credentials.get('client_id', ''),
            "transactionType": transaction_type,
            "exchangeSegment": exchange_id,
            "productType": product_type,
            "orderType": order_type,
            "validity": validity,
            "securityId": security_id,
            "quantity": int(order_data.get('quantity', 0)),
            "price": price
        }
        
        # Add trigger price if provided (for SL orders)
        if order_data.get('trigger_price') and float(order_data.get('trigger_price', 0)) > 0:
            dhan_payload['triggerPrice'] = float(order_data['trigger_price'])
        
        # Add optional correlationId if provided
        if order_data.get('correlation_id'):
            dhan_payload['correlationId'] = str(order_data['correlation_id'])
        
        # Add disclosed quantity if provided
        if order_data.get('disclosed_quantity'):
            dhan_payload['disclosedQuantity'] = int(order_data['disclosed_quantity'])
        
        # AMO (After Market Order) flag
        variety = order_data.get('variety', 'regular')
        is_amo = (variety.lower() == 'amo')
        dhan_payload['afterMarketOrder'] = is_amo
        
        if is_amo:
            # amoTime is required if afterMarketOrder is true
            dhan_payload['amoTime'] = order_data.get('amo_time', 'OPEN')
        
        # Prepare headers
        headers = {
            'Content-Type': 'application/json',
            'access-token': access_token
        }
        
        # Make API request
        order_url = "https://api.dhan.co/v2/orders"
        logger.info(f"Request: POST {order_url}")
        logger.info(f"Payload: {json.dumps(dhan_payload, indent=2)}")
        
        response = requests.post(order_url, headers=headers, json=dhan_payload, timeout=30)
        logger.info(f"HTTP Status: {response.status_code}")
        
        if response.status_code not in [200, 201]:
            logger.error(f"HTTP Error: {response.text}")
            return {
                "status": "error",
                "message": f"HTTP {response.status_code}: {response.text}",
                "error_type": "api_error"
            }
        
        response_data = response.json()
        logger.info(f"✓ Success: Order placed. Response: {response_data}")
        
        # Extract order ID from response
        order_id = response_data.get('orderId') or response_data.get('data', {}).get('orderId')
        
        return {
            "status": "success",
            "message": "Order placed successfully",
            "data": {
                "order_id": order_id,
                "response": response_data
            }
        }
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error during order placement: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "error_type": "network_error"
        }
    except Exception as e:
        error_msg = f"Unexpected error during order placement: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "error_type": "unexpected_error"
        }

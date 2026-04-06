"""
MOSWAL Broker Integration
Automated login and trading functions for Motilal Oswal broker
Follows the same structure as Zerodha, AngelOne, Kotak, and Dhan brokers
"""
import requests
import pyotp
import json
import os
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any
from Broker.Moswal.mapping import (
    map_exchange,
    map_validity,
    map_order_side,
    map_product_type,
    map_order_type
)
from Broker.Moswal.symbol_lookup import get_moswal_token, is_numeric_token

logger = logging.getLogger(__name__)


class MOSWALAPIError(Exception):
    """Custom exception for MOSWAL API errors"""
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


class MOSWALAuthenticator:
    """Handles MOSWAL authentication flow following the same pattern as other brokers"""
    
    BASE_URL = "https://openapi.motilaloswal.com"
    LOGIN_URL = f"{BASE_URL}/rest/login/v3/authdirectapi"
    PROFILE_URL = f"{BASE_URL}/rest/login/v1/getprofile"
    ORDER_URL = f"{BASE_URL}/rest/trans/v1/placeorder"
    TOKEN_CACHE_FILE = "moswal_token_cache.json"
    
    def __init__(self, api_key=None):
        """Initialize MOSWAL Authenticator"""
        self.api_key = api_key
        self.client_id = None
        self.access_token = None
        self.client_formal_name = None
        self.__headers = {}
        self.__headers_base = {
            "Accept": "application/json",
            "User-Agent": "MOSL/V.1.1.0",
            "SourceId": "WEB",
            "osname": "Windows 10",
            "osversion": "10.0.19041",
            "devicemodel": "AHV",
            "manufacturer": "ASUS",
            "productname": "TT",
            "productversion": "1",
            "browsername": "Chrome",
            "browserversion": "105.0",
            "ClientLocalIp": "192.168.2.171",
            "ClientPublicIp": "157.119.91.18",
            "MacAddress": "80:91:33:5d:f6:e6",
        }
    
    def login(self, client_id: str, password: str, dob: str, totp_secret: str) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Authenticate with MOSWAL and get access token.
        
        Args:
            client_id: MOSWAL Client ID
            password: Password for MOSWAL account
            dob: Date of birth in format DDMMYYYY (e.g., "01011990")
            totp_secret: TOTP Secret key for OTP generation
            
        Returns:
            Tuple of (session_data, error_message)
            - On success: ({"access_token": token, "client_id": id, "client_name": name}, None)
            - On failure: (None, error_message)
        """
        logger.info("-" * 40)
        logger.info("MOSWAL API LOGIN")
        
        # 1. Generate TOTP
        try:
            totp = pyotp.TOTP(totp_secret).now()
            logger.info(f"Generated TOTP for client: {client_id}")
        except Exception as e:
            error_msg = f"Failed to generate TOTP: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

        # 2. Format DOB
        try:
            formatted_dob = datetime.strptime(dob, "%d%m%Y").strftime("%d/%m/%Y")
        except Exception as e:
            error_msg = f"Invalid DOB format. Expected DDMMYYYY: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

        # 3. Generate password hash
        try:
            password_hash = hashlib.sha256((password + self.api_key).encode('utf-8')).hexdigest()
        except Exception as e:
            error_msg = f"Failed to generate password hash: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

        # 4. Prepare Request
        payload = {
            'Userid': client_id,
            'Password': password_hash,
            '2FA': formatted_dob,
            'totp': totp
        }
        
        headers = self.__headers_base | {
            "ApiKey": self.api_key,
            "vendorinfo": client_id,
        }
        
        logger.info(f"Request: POST {self.LOGIN_URL}")
        
        # 5. Make Login Request
        try:
            response = requests.post(self.LOGIN_URL, json=payload, headers=headers, timeout=30)
            logger.info(f"HTTP Status: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"HTTP Error: {response.text}")
                try:
                    err_data = response.json()
                    err_msg = err_data.get('message', response.text)
                except:
                    err_msg = response.text
                return None, f"Login Failed: {err_msg}"
                
            data = response.json()
            
            # 6. Extract Token
            if data.get('status') != 'SUCCESS':
                error_msg = f"MOSWAL Error: {data.get('message', 'Unknown error')}"
                logger.error(error_msg)
                return None, error_msg
            
            token = data.get('AuthToken')
            if not token:
                logger.error(f"Could not find AuthToken in response. Keys: {list(data.keys())}")
                return None, "Invalid response: Missing AuthToken"

            # 7. Update headers with token
            self.access_token = token
            self.client_id = client_id
            headers['Authorization'] = token
            self.__headers = headers
            
            # 8. Get Profile to fetch client name
            client_name = self._get_profile(client_id)
            
            # 9. Save to Cache
            session_data = {
                'access_token': token,
                'client_id': client_id,
                'client_name': client_name
            }
            self._save_token_to_cache(session_data)
            logger.info("✓ Success: Logged in via API.")
            
            return session_data, None

        except requests.exceptions.RequestException as e:
            error_msg = f"Network error during API Login: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    def _get_profile(self, client_id: str) -> Optional[str]:
        """Fetch client profile to get formal name"""
        try:
            payload = {"clientcode": client_id}
            response = requests.post(self.PROFILE_URL, json=payload, headers=self.__headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'SUCCESS' and data.get('data'):
                    client_name = data['data'].get('name')
                    self.client_formal_name = client_name
                    logger.info(f"Client name: {client_name}")
                    return client_name
        except Exception as e:
            logger.warning(f"Failed to fetch profile: {str(e)}")
        
        return None
    
    def _save_token_to_cache(self, session_data: Dict):
        """Persist access token for Order API usage"""
        try:
            token_data = {
                'access_token': session_data['access_token'],
                'client_id': session_data['client_id'],
                'client_name': session_data.get('client_name'),
                'timestamp': datetime.now().isoformat()
            }
            with open(self.TOKEN_CACHE_FILE, 'w') as f:
                json.dump(token_data, f)
            logger.info(f"Access token persisted for client: {session_data['client_id']}")
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
                
            return data
        except:
            return None


def place_order(credentials: Dict[str, Any], order_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Places an order using MOSWAL API.
    
    Args:
        credentials: dict containing:
            - access_token: Access token from login
            - client_id: MOSWAL client ID
        order_data: dict containing order parameters:
            - exchange: Exchange name (NSE, BSE, NSEFO, MCX)
            - symbol: Trading symbol (symboltoken/scripcode)
            - order_side: BUY or SELL
            - product_type: DELIVERY, INTRADAY, MARGIN, etc.
            - order_type: MARKET, LIMIT, SL, SL-M
            - quantity: Order quantity
            - price: Limit price (optional, for LIMIT orders)
            - trigger_price: Trigger price (optional, for SL orders)
            - validity: DAY or IOC (optional)
            - disclosed_quantity: Disclosed quantity (optional)
            - afterhours: AMO flag (optional, default False)
    
    Returns:
        dict: Order placement response
        - On success: {"status": "success", "message": "...", "data": {...}}
        - On failure: {"status": "error", "message": "...", "error_type": "..."}
    """
    logger.info("-" * 40)
    logger.info("MOSWAL PLACE ORDER")
    
    try:
        # Extract credentials
        access_token = credentials.get('access_token')
        client_id = credentials.get('client_id')
        api_key = credentials.get('api_key')  # Extract API key
        
        if not access_token:
            logger.error("Missing access token in credentials")
            return {
                "status": "error",
                "message": "Missing access token",
                "error_type": "authentication_error"
            }
        
        if not api_key:
            logger.error("Missing API key in credentials")
            return {
                "status": "error",
                "message": "Missing API key. MOSWAL requires api_key for order placement.",
                "error_type": "authentication_error"
            }
        
        # Map order parameters using mapping functions
        exchange = map_exchange(order_data.get('exchange'))
        transaction_type = map_order_side(order_data.get('order_side'))
        product_type = map_product_type(order_data.get('product_type'), exchange)
        order_type = map_order_type(order_data.get('order_type'))
        validity = map_validity(order_data.get('validity', 'DAY'))
        
        # Convert symbol to symboltoken (supports both names and numeric tokens)
        symbol_value = order_data.get('symbol')
        exchange_name = order_data.get('exchange', 'NSE')
        
        try:
            # Try automatic conversion from symbol name to token
            if not is_numeric_token(symbol_value):
                logger.info(f"Converting symbol '{symbol_value}' to MOSWAL token")
                symboltoken_str = get_moswal_token(symbol_value, exchange_name)
                symboltoken = int(symboltoken_str)
            else:
                symboltoken = int(symbol_value)
                
            logger.info(f"Using symboltoken: {symboltoken}")
        except ValueError as e:
            error_msg = str(e)
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "error_type": "validation_error"
            }
        except Exception as e:
            error_msg = f"Failed to process symbol '{symbol_value}': {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "error_type": "validation_error"
            }
        
        # Prepare MOSWAL API payload
        moswal_payload = {
            "exchange": exchange,
            "symboltoken": symboltoken,  # MOSWAL uses numeric symboltoken
            "buyorsell": transaction_type,
            "ordertype": order_type,
            "producttype": product_type,
            "orderduration": validity,
            "price": float(order_data.get('price', 0)),
            "triggerprice": float(order_data.get('trigger_price', 0)),
            "quantityinlot": int(order_data.get('quantity', 0)),
            "disclosedquantity": int(order_data.get('disclosed_quantity', 0)),
            "amoorder": 'Y' if order_data.get('afterhours', False) else 'N',
            "tag": order_data.get('tag', '')
        }
        
        # Prepare headers - MOSWAL requires all device/session headers + ApiKey
        headers = {
            'Content-Type': 'application/json',
            'Authorization': access_token,
            'ApiKey': api_key,  # Required for MOSWAL API
            'vendorinfo': client_id,  # Required for MOSWAL API
            'Accept': 'application/json',
            'User-Agent': 'MOSL/V.1.1.0',
            'SourceId': 'WEB',
            'osname': 'Windows 10',
            'osversion': '10.0.19041',
            'devicemodel': 'AHV',
            'manufacturer': 'ASUS',
            'productname': 'TT',
            'productversion': '1',
            'browsername': 'Chrome',
            'browserversion': '105.0',
            'ClientLocalIp': '192.168.2.171',
            'ClientPublicIp': order_data.get('static_ip', '157.119.91.18'),
            'MacAddress': order_data.get('mac_address', '80:91:33:5d:f6:e6')
        }
        
        # Make API request
        order_url = "https://openapi.motilaloswal.com/rest/trans/v1/placeorder"
        logger.info(f"Request: POST {order_url}")
        logger.info(f"Payload: {json.dumps(moswal_payload, indent=2)}")
        
        response = requests.post(order_url, headers=headers, json=moswal_payload, timeout=30)
        logger.info(f"HTTP Status: {response.status_code}")
        
        if response.status_code not in [200, 201]:
            logger.error(f"HTTP Error: {response.text}")
            return {
                "status": "error",
                "message": f"HTTP {response.status_code}: {response.text}",
                "error_type": "api_error"
            }
        
        response_data = response.json()
        logger.info(f"Response: {response_data}")
        
        # Check response status
        if response_data.get('status') != 'SUCCESS':
            error_msg = response_data.get('message', 'Order placement failed')
            logger.error(f"Order failed: {error_msg}")
            return {
                "status": "error",
                "message": error_msg,
                "error_type": "order_error"
            }
        
        # Extract order ID from response
        order_id = response_data.get('uniqueorderid') or response_data.get('data', {}).get('uniqueorderid')
        
        logger.info(f"✓ Success: Order placed. Order ID: {order_id}")
        
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

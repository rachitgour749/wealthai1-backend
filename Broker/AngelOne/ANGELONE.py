"""
AngelOne Broker Integration

Provides standardized login and order placement functionality following
the same structure as Zerodha integration.
"""

import requests
import json
import logging
import pyotp
from typing import Tuple, Optional, Dict, Any
from Broker.AngelOne.mapping import (
    map_exchange, 
    map_product_type, 
    map_order_type, 
    map_order_side, 
    map_variety,
    format_symbol_for_cash
)
from Broker.AngelOne.symbol_lookup import get_symbol_token

logger = logging.getLogger(__name__)


class AngelOneAuthenticator:
    """Handles AngelOne authentication flow."""
    
    BASE_URL = "https://apiconnect.angelbroking.com"
    
    ROUTES = {
        "login": "/rest/auth/angelbroking/user/v1/loginByPassword",
        "profile": "/rest/secure/angelbroking/user/v1/getProfile",
        "logout": "/rest/secure/angelbroking/user/v1/logout"
    }
    
    def __init__(self, api_key: str, api_secret: str = None, client_code: str = None):
        """
        Initialize AngelOne authenticator.
        
        Args:
            api_key: AngelOne API key
            api_secret: Not used for AngelOne (kept for interface compatibility)
            client_code: Client code (optional, can be provided in login)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.client_code = client_code
        self.session = requests.Session()
        
        # Get public IP
        try:
            self.client_public_ip = requests.get('https://api.ipify.org').text.strip()
        except Exception as e:
            logger.warning(f"Failed to get public IP: {e}")
            self.client_public_ip = "106.193.147.98"  # Fallback
        
        self.client_local_ip = "127.0.0.1"
        self.client_mac_address = "00:00:00:00:00:00"
    
    def _request_headers(self, access_token: Optional[str] = None) -> Dict[str, str]:
        """Generate request headers for AngelOne API."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-ClientLocalIP": self.client_local_ip,
            "X-ClientPublicIP": self.client_public_ip,
            "X-MACAddress": self.client_mac_address,
            "X-PrivateKey": self.api_key,
            "X-UserType": "USER",
            "X-SourceID": "WEB"
        }
        
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        
        return headers
    
    def login(self, client_code: str, password: str, totp_secret: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Authenticate with AngelOne and get access token.
        
        Args:
            client_code: AngelOne client code
            password: Account password
            totp_secret: TOTP secret key for 2FA
        
        Returns:
            Tuple of (access_token, error_message)
            - On success: (access_token, None)
            - On failure: (None, error_message)
        """
        try:
            logger.info(f"Starting AngelOne login for client: {client_code}")
            
            # Generate TOTP
            totp = pyotp.TOTP(totp_secret.replace(" ", "")).now()
            
            # Prepare login payload
            payload = {
                "clientcode": client_code,
                "password": password,
                "totp": totp
            }
            
            # Make login request
            url = f"{self.BASE_URL}{self.ROUTES['login']}"
            headers = self._request_headers()
            
            response = self.session.post(
                url,
                data=json.dumps(payload),
                headers=headers,
                timeout=15
            )
            
            # Parse response
            if response.status_code != 200:
                error_msg = f"Login failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                return None, error_msg
            
            data = response.json()
            
            if not data.get('status'):
                error_msg = data.get('message', 'Login failed')
                logger.error(f"Login failed: {error_msg}")
                return None, error_msg
            
            # Extract tokens
            jwt_token = data.get('data', {}).get('jwtToken')
            refresh_token = data.get('data', {}).get('refreshToken')
            feed_token = data.get('data', {}).get('feedToken')
            
            if not jwt_token:
                error_msg = "No JWT token in response"
                logger.error(error_msg)
                return None, error_msg
            
            # Store tokens for later use
            self.access_token = jwt_token
            self.refresh_token = refresh_token
            self.feed_token = feed_token
            self.client_code = client_code
            
            logger.info("AngelOne login successful!")
            
            # Return access token (jwt_token)
            return jwt_token, None
        
        except Exception as e:
            error_msg = f"Login exception: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    def get_profile(self, access_token: str, refresh_token: str) -> Optional[Dict]:
        """
        Get user profile information.
        
        Args:
            access_token: JWT access token
            refresh_token: Refresh token
        
        Returns:
            User profile data or None
        """
        try:
            url = f"{self.BASE_URL}{self.ROUTES['profile']}"
            headers = self._request_headers(access_token)
            
            response = self.session.get(
                url,
                params={"refreshToken": refresh_token},
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get profile: {response.text}")
                return None
        
        except Exception as e:
            logger.error(f"Get profile exception: {e}")
            return None


def place_order(credentials: Dict[str, Any], order_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Places an order using AngelOne API.
    
    Args:
        credentials: dict containing:
            - api_key: AngelOne API key
            - access_token: JWT access token
            - refresh_token: Refresh token (optional)
            - client_code: Client code
        
        order_data: dict containing:
            - symbol: Trading symbol (e.g., 'RELIANCE' or 'RELIANCE-EQ')
            - exchange: Exchange (NSE, BSE, NFO, etc.)
            - order_side: BUY or SELL
            - quantity: Order quantity
            - product_type: DELIVERY, INTRADAY, or MARGIN
            - order_type: MARKET, LIMIT, STOPLOSS_LIMIT, STOPLOSS_MARKET
            - price: Limit price (optional, for LIMIT orders)
            - trigger_price: Trigger price (optional, for STOPLOSS orders)
            - exchange_instrument_id: Symbol token (required)
            - lotsize: Lot size (default: 1)
            - proxy_config: Proxy configuration (optional)
            - proxy_auth: Proxy authentication (optional)
            - static_ip: Client IP for SEBI compliance (optional)
    
    Returns:
        dict with status, message, and data:
        - On success: {"status": "success", "message": "...", "data": {...}}
        - On failure: {"status": "error", "message": "...", "error_type": "..."}
    """
    api_key = credentials.get('api_key')
    access_token = credentials.get('access_token')
    client_code = credentials.get('client_code')
    
    if not api_key or not access_token:
        return {"status": "error", "message": "Missing API key or access token"}
    
    try:
        # Extract and map parameters
        symbol = order_data.get('symbol')
        exchange = order_data.get('exchange')
        exchange_instrument_id = order_data.get('exchange_instrument_id')
        
        # Auto-lookup token if not provided
        if not exchange_instrument_id and symbol and exchange:
            logger.info(f"exchange_instrument_id not provided, looking up token for {symbol} on {exchange}")
            exchange_instrument_id = get_symbol_token(symbol, exchange)
            
            if exchange_instrument_id:
                logger.info(f"Found token for {symbol}: {exchange_instrument_id}")
            else:
                logger.warning(f"Token not found for {symbol} on {exchange}. Order may fail.")
                return {
                    "status": "error",
                    "message": f"Symbol token not found for {symbol} on {exchange}. Please provide exchange_instrument_id manually or add symbol to symbol_lookup.py"
                }
        
        if not symbol or not exchange or not exchange_instrument_id:
            return {
                "status": "error", 
                "message": "Missing required parameters: symbol, exchange, or exchange_instrument_id"
            }
        
        # Map parameters to AngelOne format
        mapped_exchange = map_exchange(exchange)
        transaction_type = map_order_side(order_data.get('order_side', 'BUY'))
        product_type = map_product_type(order_data.get('product_type', 'DELIVERY'), exchange)
        order_type = map_order_type(order_data.get('order_type', 'MARKET'))
        variety = map_variety(order_data.get('order_type', 'MARKET'))
        
        # Format symbol for cash segment
        trading_symbol = format_symbol_for_cash(symbol, mapped_exchange)
        
        # Get quantity and lotsize
        quantity = int(order_data.get('quantity', 1))
        lotsize = int(order_data.get('lotsize', 1))
        final_quantity = quantity * lotsize
        
        # Get prices
        price = float(order_data.get('price', 0))
        trigger_price = float(order_data.get('trigger_price', 0))
        
        # Build order parameters
        params = {
            "variety": variety,
            "tradingsymbol": trading_symbol,
            "symboltoken": str(exchange_instrument_id),
            "transactiontype": transaction_type,
            "exchange": mapped_exchange,
            "ordertype": order_type,
            "producttype": product_type,
            "duration": "DAY",
            "quantity": final_quantity,
            "ordertag": order_data.get('order_tag', 'WealthAI')
        }
        
        # Add price for LIMIT orders
        if order_type == "LIMIT" and price > 0:
            params["price"] = price
        
        # Add trigger price for STOPLOSS orders
        if "STOPLOSS" in order_type and trigger_price > 0:
            params["triggerprice"] = trigger_price
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        # Log the exact parameters being sent
        logger.info(f"=" * 60)
        logger.info(f"AngelOne Order Parameters:")
        logger.info(f"  Symbol: {symbol}")
        logger.info(f"  Trading Symbol: {trading_symbol}")
        logger.info(f"  Symbol Token: {exchange_instrument_id}")
        logger.info(f"  Exchange: {mapped_exchange}")
        logger.info(f"  Transaction Type: {transaction_type}")
        logger.info(f"  Product Type: {product_type}")
        logger.info(f"  Order Type: {order_type}")
        logger.info(f"  Quantity: {final_quantity}")
        logger.info(f"Full params: {params}")
        logger.info(f"=" * 60)
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-ClientLocalIP": "127.0.0.1",
            "X-ClientPublicIP": order_data.get('static_ip', "106.193.147.98"),
            "X-MACAddress": "00:00:00:00:00:00",
            "X-PrivateKey": api_key,
            "X-UserType": "USER",
            "X-SourceID": "WEB",
            "Authorization": f"Bearer {access_token}"
        }
        
        # Get proxy configuration
        proxy_config = order_data.get('proxy_config')
        proxy_auth = order_data.get('proxy_auth')
        
        if proxy_config:
            proxy_ip = proxy_config.get("http") or proxy_config.get("https")
            logger.info(f"Order request going via proxy: {proxy_ip}")
        else:
            logger.debug("No proxy used for order request")
        
        # Place order
        url = "https://apiconnect.angelbroking.com/rest/secure/angelbroking/order/v1/placeOrder"
        
        logger.info(f"Placing {variety} order: {params}")
        
        response = requests.post(
            url,
            data=json.dumps(params),
            headers=headers,
            proxies=proxy_config,
            auth=proxy_auth,
            timeout=30
        )
        
        # Parse response
        if response.status_code == 200:
            data = response.json()
            
            if data.get('status'):
                return {
                    "status": "success",
                    "message": data.get('message', 'Order placed successfully'),
                    "data": data.get('data', {})
                }
            else:
                return {
                    "status": "error",
                    "message": data.get('message', 'Order placement failed'),
                    "error_type": data.get('errorcode', 'UNKNOWN')
                }
        else:
            return {
                "status": "error",
                "message": f"HTTP {response.status_code}: {response.text}",
                "error_type": "HTTP_ERROR"
            }
    
    except Exception as e:
        logger.error(f"Order placement failed: {e}")
        return {
            "status": "error",
            "message": str(e),
            "error_type": "EXCEPTION"
        }

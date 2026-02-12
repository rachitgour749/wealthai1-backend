import requests
import time
import pyotp
import hashlib
import urllib.parse
from kiteconnect import KiteConnect
import logging
from Broker.Zerodha.mapping import map_exchange, map_validity, map_order_side, map_product_type, map_order_type

logger = logging.getLogger(__name__)

class ZerodhaAuthenticator:
    """Handles Zerodha authentication flow."""
    
    BASE_URL = "https://kite.zerodha.com"
    URLS = {
        'login': f"{BASE_URL}/login",
        'api_login': f"{BASE_URL}/api/login",
        'twofa': f"{BASE_URL}/api/twofa",
        'token': "https://api.kite.trade/session/token"
    }
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'X-Kite-Version': '3',
        'Origin': 'https://kite.zerodha.com',
        'Referer': 'https://kite.zerodha.com/'
    }

    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        # Add a proper set of common headers to the session
        self.session.headers.update({
            'User-Agent': self.HEADERS['User-Agent'],
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive'
        })
        self.kite = KiteConnect(api_key=api_key)

    def login(self, username, password, totp_secret):
        try:
            logger.info(f"Starting login for user: {username}")
            # 1. Session Warming - Visit login page first
            self.session.get(self.URLS['login'], timeout=15)
            time.sleep(1.5) # Increased delay slightly

            # 2. Authentication
            h = {
                **self.HEADERS, 
                'X-Kite-Userid': username, 
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            login_payload = {
                'user_id': username, 
                'password': password
            }
            
            logger.debug(f"Sending login request to {self.URLS['api_login']}...")
            resp = self.session.post(self.URLS['api_login'], data=login_payload, headers=h)
            
            if resp.status_code != 200 or resp.json().get('status') != 'success': 
                logger.error(f"Login step 1 failed: {resp.text}")
                return None, f"Login failed: {resp.text}"
            
            request_id = resp.json()['data']['request_id']

            # 3. TOTP Verification
            logger.debug("Verifying TOTP...")
            totp = pyotp.TOTP(totp_secret.replace(" ", "")).now()
            resp = self.session.post(self.URLS['twofa'], data={'user_id': username, 'request_id': request_id, 'twofa_value': totp, 'twofa_type': 'totp'}, headers=h)
            
            if resp.status_code != 200: 
                logger.error(f"TOTP failed: {resp.text}")
                return None, f"TOTP failed: {resp.text}"

            # 4. Redirect Traversal
            logger.debug("Capturing request token...")
            request_token = self._get_request_token()
            if not request_token: 
                logger.error("Failed to capture request_token")
                return None, "Failed to capture request_token"

            # 5. Token Exchange
            logger.debug("Exchanging token...")
            checksum = hashlib.sha256(f"{self.api_key}{request_token}{self.api_secret}".encode()).hexdigest()
            resp = requests.post(self.URLS['token'], 
                               data={'api_key': self.api_key, 'request_token': request_token, 'checksum': checksum},
                               headers={'X-Kite-Version': '3'})
            
            data = resp.json().get('data', {})
            access_token = data.get('access_token')
            
            if access_token:
                logger.info("Zerodha login successful!")
                return access_token, None
            else:
                logger.error("Token exchange failed")
                return None, "Token exchange failed"

        except Exception as e:
            logger.error(f"Login exception: {e}")
            return None, str(e)

    def _get_request_token(self):
        url = self.kite.login_url()
        for _ in range(10):
            try:
                resp = self.session.get(url, allow_redirects=False)
                loc = resp.headers.get('Location')
                if not loc: break
                
                if 'request_token=' in loc:
                    return loc.split('request_token=')[1].split('&')[0]
                
                url = urllib.parse.urljoin(url, loc) if not urllib.parse.urlparse(loc).scheme else loc
            except Exception: break
        return None

def place_order(credentials: dict, order_data: dict):
    """
    Places an order using the KiteConnect API.
    
    Args:
        credentials: dict containing api_key and access_token
        order_data: dict containing order parameters (exchange, symbol, order_side, product_type, quantity, etc.)
    """
    api_key = credentials.get('api_key')
    access_token = credentials.get('access_token')
    
    if not api_key or not access_token:
        return {"status": "error", "message": "Missing API key or access token"}

    try:
        # Extract and map parameters
        symbol = order_data.get('symbol')
        exchange = map_exchange(order_data.get('exchange'))
        transaction_type = map_order_side(order_data.get('order_side'))
        quantity = int(order_data.get('quantity', 1))
        product = map_product_type(order_data.get('product_type'))
        order_type = map_order_type(order_data.get('order_type', 'MARKET'))
        validity = map_validity(order_data.get('validity', 'DAY'))
        price = float(order_data.get('price', 0))
        trigger_price = float(order_data.get('trigger_price', 0))
        
        if not symbol or not exchange:
            return {"status": "error", "message": "Missing required parameters: symbol or exchange"}
        
        kite_params = {
            'tradingsymbol': symbol,
            'exchange': exchange,
            'transaction_type': transaction_type,
            'quantity': quantity,
            'order_type': order_type,
            'product': product,
            'validity': validity
        }
        
        if price > 0:
            kite_params['price'] = price
        if trigger_price > 0:
            kite_params['trigger_price'] = trigger_price

        headers = {
            'X-Kite-Version': '3',
            'Authorization': f'token {api_key}:{access_token}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        # Add static IP to headers if available (for SEBI compliance)
        static_ip = order_data.get('static_ip')
        if static_ip:
            headers['X-Client-IP'] = static_ip
            logger.info(f"Including client IP in order request: {static_ip}")
        else:
            logger.debug("No static IP provided, placing order without IP header")
        
        # Get variety from order_data, default to regular
        variety = order_data.get('variety', 'regular')
        url = f'https://api.kite.trade/orders/{variety}'
        
        # Log the exact parameters being sent
        logger.info(f"=" * 60)
        logger.info(f"Zerodha Order Parameters:")
        logger.info(f"  Symbol: {symbol}")
        logger.info(f"  Exchange: {exchange}")
        logger.info(f"  Transaction Type: {transaction_type}")
        logger.info(f"  Product: {product}")
        logger.info(f"  Order Type: {order_type}")
        logger.info(f"  Quantity: {quantity}")
        logger.info(f"Full params: {kite_params}")
        logger.info(f"=" * 60)
        
        logger.info(f"Placing {variety} order: {kite_params}")
        resp = requests.post(
            url, 
            data=kite_params, 
            headers=headers
        )
        
        if resp.status_code == 200 and resp.json().get('status') == 'success':
            return {
                "status": "success", 
                "message": "Order placed successfully",
                "data": resp.json().get('data')
            }
        else:
            return {
                "status": "error", 
                "message": resp.json().get('message', 'Unknown error'),
                "error_type": resp.json().get('error_type')
            }

    except Exception as e:
        logger.error(f"Order placement failed: {e}")
        return {"status": "error", "message": str(e)}

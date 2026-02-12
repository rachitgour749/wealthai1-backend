import requests
import time
import pyotp
import logging
import json
from Broker.Kotak.mapping import map_exchange, map_validity, map_order_side, map_product_type, map_order_type

logger = logging.getLogger(__name__)

class KotakAuthenticator:
    """Handles Kotak Securities authentication flow."""
    
    BASE_URL = "https://mis.kotaksecurities.com"
    URLS = {
        'trade_login': f"{BASE_URL}/login/1.0/tradeApiLogin",
        'trade_validate': f"{BASE_URL}/login/1.0/tradeApiValidate"
    }
    
    def __init__(self, api_key=None, api_secret=None, access_token=None):
        # Kotak Neo/API might use consumer key/secret or just access token from other flows.
        # User specified "Authorization: <access_token>" in headers, which implies 
        # there is some pre-existing token or it's passed in. 
        # The user provided request flow shows: 
        # Header: Authorization: <access_token>
        # We will assume this 'access_token' is what we might call 'consumer_key' or 'jwt' 
        # passed during initialization or via credentials.
        # Let's map 'api_key' to this initial 'access_token' for the login headers.
        self.access_token = access_token or api_key 
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'neo-fin-key': 'neotradeapi'
        })

    def login(self, mobile_number, ucc, totp_secret, mpin):
        """
        Executes the 2-step login flow.
        Step 1: Trade API Login with Mobile, UCC, TOTP
        Step 2: Trade API Validate with MPIN
        """
        try:
            logger.info(f"Starting Kotak login for UCC: {ucc}")
            
            # --- Step 1: Trade API Login ---
            if not totp_secret:
                return None, "TOTP secret is missing"

            # Generate TOTP
            try:
                totp = pyotp.TOTP(totp_secret.replace(" ", "")).now()
            except Exception as e:
                return None, f"Invalid TOTP secret: {str(e)}"

            login_payload = {
                "mobileNumber": mobile_number,
                "ucc": ucc,
                "totp": totp
            }
            
            headers_step1 = {
                'Authorization': self.access_token,
                'neo-fin-key': 'neotradeapi',
                'Content-Type': 'application/json'
            }
            
            logger.debug(f"Step 1: Sending Login Request to {self.URLS['trade_login']}")
            resp1 = self.session.post(
                self.URLS['trade_login'], 
                json=login_payload, 
                headers=headers_step1,
                timeout=15
            )
            
            if resp1.status_code != 200:
                logger.error(f"Step 1 Failed: {resp1.text}")
                try:
                    err_msg = resp1.json().get('message', resp1.text)
                except:
                    err_msg = resp1.text
                return None, f"Login Step 1 Failed: {err_msg}"
            
            data1 = resp1.json().get('data', {})
            token_step1 = data1.get('token') # Prepare for next step 'Auth' header
            sid_step1 = data1.get('sid')
            
            if not token_step1 or not sid_step1:
                return None, "Step 1 success but missing token/sid"

            logger.info(f"Step 1 successful. SID: {sid_step1}. Proceeding to Step 2...")

            # --- Step 2: Trade API Validate ---
            validate_payload = {
                "mpin": mpin
            }
            
            headers_step2 = {
                'Authorization': self.access_token,
                'neo-fin-key': 'neotradeapi',
                'sid': sid_step1,
                'Auth': token_step1,
                'Content-Type': 'application/json'
            }
            
            logger.debug(f"Step 2: Sending Validate Request to {self.URLS['trade_validate']}")
            resp2 = self.session.post(
                self.URLS['trade_validate'],
                json=validate_payload,
                headers=headers_step2,
                timeout=15
            )
            
            if resp2.status_code != 200:
                logger.error(f"Step 2 Failed: {resp2.text}")
                try:
                    err_msg = resp2.json().get('message', resp2.text)
                except:
                    err_msg = resp2.text
                return None, f"Login Step 2 Failed: {err_msg}"
                
            data2 = resp2.json().get('data', {})
            final_token = data2.get('token')
            # Use SID from Step 2 if available, else fallback to Step 1 SID
            final_sid = data2.get('sid') or sid_step1
            base_url = data2.get('baseUrl')
            
            if not final_token:
                return None, "Step 2 success but missing final token"
            
            logger.info(f"DEBUG - Kotak Login Complete. Step 1 SID: {sid_step1}, Step 2 SID: {data2.get('sid')}, Final SID: {final_sid}")
                
            logger.info("Kotak login flow completed successfully.")
            
            # Return all necessary session data
            return {
                "access_token": final_token,
                "sid": final_sid,
                "base_url": base_url,
                "ucc": ucc
            }, None

        except Exception as e:
            logger.error(f"Kotak Login Exception: {e}")
            return None, str(e)


def place_order(credentials: dict, order_data: dict):
    """
    Places an order using the Kotak API.
    Endpoint: POST <Base URL>/quick/order/rule/ms/place
    """
    access_token = credentials.get('access_token')
    sid = credentials.get('sid')
    base_url = credentials.get('base_url', 'https://gw-napi.kotaksecurities.com')
    
    if not access_token or not sid:
        return {"status": "error", "message": "Missing access token or SID"}

    # Construct the URL
    # Ensure base_url doesn't end with slash and endpoint starts with one, or handle neatly
    base_url = base_url.rstrip('/')
    url = f"{base_url}/quick/order/rule/ms/place"
    
    logger.info(f"DEBUG - Kotak Order URL: {url}")
    
    try:
        # Map parameters
        exchange = map_exchange(order_data.get('exchange')) # es
        
        # Map symbol (append -EQ for cash segments)
        symbol = order_data.get('symbol')
        if symbol and exchange in ['nse_cm', 'bse_cm']:
            symbol = str(symbol).strip().upper()
            if not symbol.endswith('-EQ'):
                symbol = f"{symbol}-EQ"
                
        transaction_type = map_order_side(order_data.get('order_side')) # tt
        quantity = str(int(order_data.get('quantity', 1))) # qt (string)
        product = map_product_type(order_data.get('product_type')) # pc
        order_type = map_order_type(order_data.get('order_type', 'MARKET')) # pt
        validity = map_validity(order_data.get('validity', 'DAY')) # rt
        
        price = order_data.get('price', 0) # pr
        trigger_price = order_data.get('trigger_price', 0) # tp
        
        # Format prices as string, default to "0" if 0
        price_str = str(price) if price else "0"
        trigger_price_str = str(trigger_price) if trigger_price else "0"

        # Construct jData payload
        j_data = {
            "am": "NO", # AMO flag - Default NO for now, can be parameterized if needed
            "dq": "0",  # Disclosed Quantity
            "es": exchange, 
            "mp": "0",  # Market Protection
            "pc": product,
            "pf": "N",  # Portfolio flag
            "pr": price_str,
            "pt": order_type,
            "qt": quantity,
            "rt": validity,
            "tp": trigger_price_str,
            "ts": symbol,
            "tt": transaction_type
        }
        
        # If BO (Bracket Order), specialized fields would be needed. 
        # Leaving them empty or default as per example "only for BO" comments.
        
        # Log payload before sending
        logger.info(f"Placing Kotak Order with jData: {json.dumps(j_data)}")
        
        # Headers
        headers = {
            "Auth": access_token,
            "Sid": sid,
            "neo-fin-key": "neotradeapi",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        # Request body: jData=<stringified_json>
        # requests will auto-encode data dict keys/values if provided as dict
        data_payload = {
            "jData": json.dumps(j_data)
        }
        
        response = requests.post(url, data=data_payload, headers=headers)
        
        logger.info(f"Kotak Order Response: {response.text}")
        
        if response.status_code == 200:
            resp_json = response.json()
            if resp_json.get("stat") == "Ok":
                return {
                    "status": "success",
                    "message": "Order placed successfully",
                    "data": {
                        "order_id": resp_json.get("nOrdNo"),
                        "raw_response": resp_json
                    }
                }
            else:
                return {
                    "status": "error",
                    "message": resp_json.get("emsg", "Unknown error from Kotak"),
                    "error_code": resp_json.get("stCode")
                }
        else:
             return {
                "status": "error",
                "message": f"HTTP Error {response.status_code}: {response.text}"
            }

    except Exception as e:
        logger.error(f"Kotak place_order failed: {e}")
        return {"status": "error", "message": str(e)}

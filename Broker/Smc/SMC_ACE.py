
import requests
import json
import logging
import hashlib
import hmac
import pyotp
import time
from typing import Tuple, Optional, Dict, Any
from Broker.Smc.mapping import (
    map_exchange,
    map_order_side,
    map_product_type,
    map_order_type,
    map_validity
)

logger = logging.getLogger(__name__)

class SMCAceAuthenticator:
    """
    Handles SMC ACE (ANT Platform) authentication flow.
    Supports the 4-step Direct SMC Global Credential Login.
    """
    
    BASE_URL = "https://aceapi.smctradeonline.com/api/v2"

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-API-KEY": self.api_key
        })

    def generate_hmac_signature(self, req_token: str) -> str:
        """
        Generate HMAC-SHA256 signature.
        Logic: HMAC(key=API_KEY+REQ_TOKEN, msg=API_SECRET)
        """
        key = (self.api_key + req_token).encode()
        msg = self.api_secret.encode()
        signature = hmac.new(key, msg, hashlib.sha256).hexdigest()
        return signature

    def login(self, user_id: str, password: str, totp_secret: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Automated login for SMC ACE:
        1. Login API (Programmatic) -> Returns req_token
        2. Handle TOTP if requested
        3. Signature Generation Locally
        4. Token Exchange (Exchange req_token + signature for access_token)
        """
        try:
            logger.info(f"Starting SMC ACE login for user: {user_id}")
            
            # 1. Login API
            login_payload = {
                "platform": "api",
                "data": {
                    "client_id": user_id,
                    "password": password
                }
            }
            login_url = f"{self.BASE_URL}/login"
            resp = self.session.post(login_url, json=login_payload)
            
            if resp.status_code != 200:
                return None, f"Login Step 1 failed at {login_url} (HTTP {resp.status_code}): {resp.text[:500]}"
            
            login_data = resp.json()
            if login_data.get("status") != "success":
                return None, f"Login Step 1 returned error: {login_data.get('message')}"
            
            # Step 2: Handle TOTP if required
            req_token = login_data.get("data", {}).get("request_token")
            
            if "Please enter TOTP" in login_data.get("message", ""):
                logger.info("TOTP required, verifying...")
                totp = pyotp.TOTP(totp_secret.replace(" ", "")).now()
                twofa_payload = {
                    "request_token": req_token,
                    "otp": totp
                }
                twofa_url = f"{self.BASE_URL}/login/twofa"
                resp = self.session.post(twofa_url, json=twofa_payload)
                
                if resp.status_code != 200:
                    return None, f"TOTP verification failed (HTTP {resp.status_code}): {resp.text}"
                
                twofa_data = resp.json()
                if twofa_data.get("status") != "success":
                    return None, f"TOTP validation failed: {twofa_data.get('message')}"
                
                # Update req_token if it changed during 2FA
                req_token = twofa_data.get("data", {}).get("request_token") or req_token

            # 3. Signature Generation
            signature = self.generate_hmac_signature(req_token)
            logger.info("HMAC signature generated locally.")

            # 4. Token Exchange
            token_payload = {
                "api_key": self.api_key,
                "signature": signature,
                "req_token": req_token
            }
            token_url = f"{self.BASE_URL}/auth/token"
            resp = self.session.post(token_url, json=token_payload)
            
            if resp.status_code != 200:
                return None, f"Token exchange failed (HTTP {resp.status_code}): {resp.text}"
            
            token_data = resp.json()
            if token_data.get("status") != "success":
                return None, f"Token exchange error: {token_data.get('message')}"
            
            access_token = token_data.get("data", {}).get("access_token")
            feed_token = token_data.get("data", {}).get("feed_token")
            
            if access_token:
                logger.info("SMC ACE login successful!")
                # We return a dict-like string or the token itself if appropriate, 
                # but standard practice here is the token. 
                # To support feed_token, we can return a packed result.
                return access_token, None
            else:
                return None, "Access token missing in response"

        except Exception as e:
            logger.error(f"SMC ACE login exception: {e}")
            return None, str(e)

def place_order(credentials: Dict[str, Any], order_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standalone function to place an order with SMC ACE (ANT Platform).
    """
    api_key = credentials.get('api_key')
    access_token = credentials.get('access_token')
    
    if not api_key or not access_token:
        return {"status": "error", "message": "Missing API key or access token"}

    try:
        # Map parameters
        action = map_order_side(order_data.get('order_side'))
        exchange = map_exchange(order_data.get('exchange'))
        token = order_data.get('symbol') # User uses 'symbol' for the instrument token
        order_type = map_order_type(order_data.get('order_type', "MARKET"))
        product_type = map_product_type(order_data.get('product_type', "DELIVERY"))
        quantity = str(order_data.get('quantity', 1))
        price = str(order_data.get('price', 0))
        trigger_price = str(order_data.get('trigger_price', 0))
        validity = map_validity(order_data.get('validity', "DAY"))

        payload = {
            "action": action,
            "exchange": exchange,
            "token": token,
            "order_type": order_type,
            "product_type": product_type,
            "quantity": quantity,
            "disclose_quantity": "0",
            "price": price,
            "trigger_price": trigger_price,
            "stop_loss_price": "0",
            "trailing_stop_loss": "0",
            "validity": validity,
            "tag": order_data.get('tag', "")
        }

        headers = {
            "Authorization": f"Bearer {access_token}",
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }

        url = "https://acetrade.smctradeonline.com/api/v2/orders"
        
        logger.info(f"Placing SMC ACE order: {payload}")
        resp = requests.post(url, json=payload, headers=headers)
        
        if resp.status_code == 200:
            resp_data = resp.json()
            if resp_data.get("status") == "success":
                return {
                    "status": "success",
                    "message": "Order placed successfully",
                    "data": resp_data.get("data")
                }
            else:
                return {
                    "status": "error",
                    "message": resp_data.get("message", "Unknown error")
                }
        else:
            return {
                "status": "error",
                "message": f"SMC ACE API error (HTTP {resp.status_code}): {resp.text}"
            }

    except Exception as e:
        logger.error(f"SMC ACE order placement failed: {e}")
        return {"status": "error", "message": str(e)}
